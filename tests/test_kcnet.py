"""Frozen encoder, random projection, checkpoint and initialization contracts."""

import io
from pathlib import Path
import tempfile
import unittest

import torch
from torch.nn import functional as F

from aig.experiment import LABELS, ROOT, build_model, initialize_kc_encoder
from aig.training import parser, resolve_model_config
from networks.kcnet import FixedFilterKCNet
from networks.learned_filters import LearnableFilterCNN128


class KCNetTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1234)

    def test_default_size_and_distinct_frozen_connections(self):
        model, _ = build_model("fixed_filter_kcnet")
        self.assertEqual(sum(p.numel() for p in model.parameters()), 97075)
        self.assertEqual(sum(p.numel() for p in model.parameters() if p.requires_grad), 4097)
        self.assertEqual(tuple(model.projection_indices.shape), (4096, 7))
        self.assertTrue(all(len(set(row)) == 7 for row in model.projection_indices.tolist()))
        self.assertGreaterEqual(model.projection_indices.min().item(), 0)
        self.assertLess(model.projection_indices.max().item(), 2048)
        with torch.no_grad():
            self.assertEqual(tuple(model(torch.rand(2, 1, 224, 224)).shape), (2, 1))

    def test_only_readout_updates_and_zero_tiny_features_stay_finite(self):
        model = FixedFilterKCNet(img_size=32, pool_size=2, kc_units=48)
        before = {name: value.clone() for name, value in model.state_dict().items()}
        optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=.01)
        labels = torch.tensor([[0.], [1.], [1.]])
        for images in (torch.zeros(3, 1, 32, 32), torch.full((3, 1, 32, 32), .5), torch.rand(3, 1, 32, 32)):
            model.train()
            self.assertTrue(model.readout.training)
            self.assertFalse(model.frontend.training)
            self.assertFalse(model.cnn_branch.training)
            optimizer.zero_grad(set_to_none=True)
            loss = F.binary_cross_entropy_with_logits(model(images), labels)
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            for name, parameter in model.named_parameters():
                if name.startswith("readout."):
                    self.assertIsNotNone(parameter.grad)
                    self.assertTrue(torch.isfinite(parameter.grad).all())
                else:
                    self.assertIsNone(parameter.grad)
            optimizer.step()
        for name, value in model.state_dict().items():
            if name.startswith("readout."):
                self.assertFalse(torch.equal(value, before[name]))
            else:
                self.assertTrue(torch.equal(value, before[name]), name)
        for value in (0., 1e-30):
            self.assertTrue(torch.isfinite(model.project_features(torch.full((2, 512), value))).all())
        features = torch.rand(2, 512)
        torch.testing.assert_close(model.project_features(features)[0], model.project_features(features[:1])[0], rtol=0, atol=0)
        model.half()
        self.assertEqual(model._feature_eps.dtype, torch.float32)
        self.assertTrue(torch.isfinite(model.project_features(torch.zeros(2, 512, dtype=torch.float16))).all())

    def test_checkpoint_restores_connections_epsilon_and_predictions(self):
        options = {"pool_size": 2, "kc_units": 31, "fan_in": 3, "projection_seed": 17, "feature_eps": 3e-5}
        model = FixedFilterKCNet(img_size=32, **options).eval()
        images = torch.rand(2, 1, 32, 32)
        with torch.no_grad():
            expected = model(images)
        handle = io.BytesIO()
        torch.save(model.state_dict(), handle)
        handle.seek(0)
        state = torch.load(handle, weights_only=True)
        restored = FixedFilterKCNet(img_size=32, **dict(options, projection_seed=19, feature_eps=1e-8)).eval()
        self.assertFalse(torch.equal(model.projection_indices, restored.projection_indices))
        restored.load_state_dict(state)
        with torch.no_grad():
            torch.testing.assert_close(expected, restored(images), rtol=0, atol=0)
        self.assertEqual(restored._feature_eps.item(), model._feature_eps.item())

    def test_projection_seed_is_separate_and_resume_uses_saved_architecture(self):
        torch.manual_seed(42)
        first = FixedFilterKCNet(img_size=32, kc_units=16, projection_seed=17)
        torch.manual_seed(42)
        second = FixedFilterKCNet(img_size=32, kc_units=16, projection_seed=18)
        self.assertFalse(torch.equal(first.projection_indices, second.projection_indices))
        torch.testing.assert_close(first.readout.weight, second.readout.weight, rtol=0, atol=0)
        args = parser().parse_args(["--model", "fixed_filter_kcnet", "--kc-units", "31", "--projection-seed", "17"])
        saved = resolve_model_config(args)
        defaults = parser().parse_args(["--model", "fixed_filter_kcnet"])
        self.assertEqual(resolve_model_config(defaults, saved), saved)
        mismatch = parser().parse_args(["--model", "fixed_filter_kcnet", "--projection-seed", "18"])
        with self.assertRaisesRegex(ValueError, "projection_seed"):
            resolve_model_config(mismatch, saved)
        with self.assertRaises(ValueError):
            resolve_model_config(parser().parse_args(["--kc-units", "31"]))
        for options in ({"pool_size": 5}, {"kc_units": 0}, {"fan_in": 2049},
                        {"projection_seed": -1}, {"feature_eps": 0.}, {"feature_eps": 1e-50}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                FixedFilterKCNet(img_size=32, **options)

    def test_finetuned_variant_has_identical_initial_state_and_more_trainable_parameters(self):
        torch.manual_seed(42)
        frozen, _ = build_model("fixed_filter_kcnet")
        torch.manual_seed(42)
        tuned, _ = build_model("finetuned_cnn_kcnet")
        self.assertEqual(sum(p.numel() for p in tuned.parameters() if p.requires_grad), 97057)
        for name, value in frozen.state_dict().items():
            torch.testing.assert_close(value, tuned.state_dict()[name], rtol=0, atol=0)
        with torch.no_grad():
            images = torch.rand(2, 1, 224, 224)
            torch.testing.assert_close(frozen(images), tuned(images), rtol=0, atol=0)
        self.assertTrue(tuned.cnn_branch.training)
        self.assertFalse(tuned.frontend.training)
        tuned.eval()
        self.assertFalse(tuned.cnn_branch.training)

    def test_finetuning_updates_cnn_and_readout_with_finite_gradients(self):
        model = FixedFilterKCNet(img_size=32, pool_size=2, kc_units=48, train_cnn=True)
        before = {name: value.clone() for name, value in model.state_dict().items()}
        optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=.001)
        for images in (torch.zeros(3, 1, 32, 32), torch.full((3, 1, 32, 32), .5), torch.rand(3, 1, 32, 32)):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            loss = F.binary_cross_entropy_with_logits(model(images), torch.tensor([[0.], [1.], [1.]]))
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            for name, parameter in model.named_parameters():
                if name.startswith("frontend."):
                    self.assertIsNone(parameter.grad)
                else:
                    self.assertIsNotNone(parameter.grad)
                    self.assertTrue(torch.isfinite(parameter.grad).all(), name)
            self.assertTrue(any(p.grad.abs().max() > 0 for p in model.cnn_branch.parameters()))
            optimizer.step()
        self.assertTrue(any(not torch.equal(value, before[name]) for name, value in model.state_dict().items() if name.startswith("cnn_branch.")))
        self.assertFalse(torch.equal(model.readout.weight, before["readout.weight"]))
        for name, value in model.state_dict().items():
            if not name.startswith(("cnn_branch.", "readout.")):
                self.assertTrue(torch.equal(value, before[name]), name)
        for value in (0., 1e-30):
            features = torch.full((2, 512), value, requires_grad=True)
            hidden = model.project_features(features)
            (hidden * torch.linspace(.01, 1., 48)).sum().backward()
            self.assertTrue(torch.isfinite(features.grad).all())

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
    def test_cuda_finetuning_reaches_the_cnn_and_preserves_frontend_freezing(self):
        model = FixedFilterKCNet(img_size=32, kc_units=48, train_cnn=True).cuda().train()
        logits = model(torch.rand(2, 1, 32, 32, device="cuda"))
        self.assertTrue(torch.isfinite(logits).all())
        F.binary_cross_entropy_with_logits(logits, torch.tensor([[0.], [1.]], device="cuda")).backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.cnn_branch.parameters()))
        self.assertTrue(all(p.grad is None for p in model.frontend.parameters()))

    def test_encoder_initialization_preserves_head_and_checks_provenance(self):
        source = LearnableFilterCNN128(trainable=False, img_size=32, angle_eps=.002)
        with torch.no_grad():
            source.cnn_branch.features[0].weight.add_(.02)
        signature = {"kind": "manifest_sha256", "sha256": "0" * 64}
        config = {"model": "fixed_filter_cnn", "dataset": "smoke", "dataset_signature": signature,
                  "input_size": 32, "transform": "gray_center_224", "labels": LABELS,
                  "selection": "minimum validation BCE", "seed": 42}
        checkpoint = {"format_version": 1, "config": config, "model_state": source.state_dict(),
                      "epoch": 2, "best_epoch": 2}
        temporary_root = ROOT / "tmp/tests"
        temporary_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temporary_root) as directory:
            path = Path(directory) / "best.pt"
            torch.save(checkpoint, path)
            model = FixedFilterKCNet(img_size=32, kc_units=16)
            head = model.readout.weight.clone()
            indices = model.projection_indices.clone()
            info = initialize_kc_encoder(model, path, "smoke", signature)
            self.assertEqual(info["epoch"], 2)
            self.assertEqual(len(info["sha256"]), 64)
            for name, value in source.frontend.state_dict().items():
                torch.testing.assert_close(value, model.frontend.state_dict()[name], rtol=0, atol=0)
            for name, value in source.cnn_branch.state_dict().items():
                torch.testing.assert_close(value, model.cnn_branch.state_dict()[name], rtol=0, atol=0)
            self.assertTrue(torch.equal(head, model.readout.weight))
            self.assertTrue(torch.equal(indices, model.projection_indices))
            with self.assertRaisesRegex(ValueError, "dataset/signature"):
                initialize_kc_encoder(model, path, "other", signature)
            with self.assertRaisesRegex(ValueError, "dataset/signature"):
                initialize_kc_encoder(model, path, "smoke", dict(signature, sha256="1" * 64))
            checkpoint["epoch"] = 3
            torch.save(checkpoint, path)
            with self.assertRaisesRegex(ValueError, "validation-selected"):
                initialize_kc_encoder(model, path, "smoke", signature)
            checkpoint["epoch"] = 2
            checkpoint["config"] = dict(config, model="learned_filter_cnn")
            torch.save(checkpoint, path)
            with self.assertRaisesRegex(ValueError, "fixed_filter_cnn"):
                initialize_kc_encoder(model, path, "smoke", signature)


if __name__ == "__main__":
    unittest.main()
