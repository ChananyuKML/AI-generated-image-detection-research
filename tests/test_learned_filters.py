"""Dataset-free checks for learned gradients and their fixed-filter control."""

import io
import unittest

import torch
import torch.nn.functional as F

from networks.learned_filters import LearnedGradientFrontend, LearnableFilterCNN128
from utils.custom_transforms import RawGradMagDir


class LearnedFilterTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1234)

    def assert_finite(self, tensor):
        self.assertTrue(torch.isfinite(tensor).all().item())

    def test_frontend_keeps_two_channels_at_full_resolution(self):
        images = torch.rand(2, 1, 224, 224)
        features = LearnedGradientFrontend()(images)
        self.assertEqual(features.shape, (2, 2, 224, 224))
        self.assert_finite(features)
        self.assertGreaterEqual(features.min().item(), 0.0)
        self.assertLessEqual(features.max().item(), 1.0)

    def test_features_are_independent_of_other_images_in_the_batch(self):
        frontend = LearnedGradientFrontend()
        image = torch.rand(1, 1, 17, 23)
        companions = torch.cat(
            [torch.zeros_like(image), torch.rand_like(image) * 7 + 4], dim=0
        )
        alone = frontend(image)
        together = frontend(torch.cat([companions[:1], image, companions[1:]], dim=0))
        torch.testing.assert_close(alone, together[1:2], rtol=1e-5, atol=1e-6)

    def test_fixed_and_trainable_models_share_initial_predictions(self):
        torch.manual_seed(42)
        fixed = LearnableFilterCNN128(trainable=False, img_size=32).eval()
        torch.manual_seed(42)
        learned = LearnableFilterCNN128(trainable=True, img_size=32).eval()
        self.assertEqual(set(fixed.state_dict()), set(learned.state_dict()))
        for name, value in fixed.state_dict().items():
            torch.testing.assert_close(value, learned.state_dict()[name], rtol=0, atol=0)
        for name in ("Gx", "Gy"):
            self.assertFalse(getattr(fixed.frontend, name).requires_grad)
            self.assertTrue(getattr(learned.frontend, name).requires_grad)
        self.assertTrue(any(p.requires_grad for p in fixed.cnn_branch.parameters()))
        images = torch.rand(2, 1, 32, 32)
        with torch.no_grad():
            fixed_logits, learned_logits = fixed(images), learned(images)
        self.assertEqual(learned_logits.shape, (2, 1))
        torch.testing.assert_close(fixed_logits, learned_logits, rtol=0, atol=0)

    def test_frozen_frontend_matches_legacy_on_ordinary_responses(self):
        # Tiny positive safety thresholds recover the legacy representation
        # away from its singular zero-response case, without reimplementing it.
        frontend = LearnedGradientFrontend(
            trainable=False, magnitude_eps=1e-12, angle_eps=1e-12
        )
        images = torch.rand(2, 1, 19, 21)
        legacy = RawGradMagDir()
        expected = torch.stack([legacy(image) for image in images])
        torch.testing.assert_close(frontend(images), expected, rtol=1e-5, atol=1e-6)

    def test_backward_is_finite_for_zero_constant_and_random_inputs(self):
        frontend = LearnedGradientFrontend()
        cases = {
            "zero": torch.zeros(2, 1, 16, 16),
            "constant": torch.full((2, 1, 16, 16), 0.5),
            "random": torch.rand(2, 1, 16, 16),
        }
        for name, images in cases.items():
            with self.subTest(input=name):
                frontend.zero_grad(set_to_none=True)
                images.requires_grad_()
                features = frontend(images)
                self.assert_finite(features)
                features.square().sum().backward()
                self.assert_finite(images.grad)
                for kernel in (frontend.Gx, frontend.Gy):
                    self.assertIsNotNone(kernel.grad)
                    self.assert_finite(kernel.grad)

    def test_origin_and_tiny_response_angles_have_finite_gradients(self):
        for response in (0.0, -1e-8):
            with self.subTest(response=response):
                frontend = LearnedGradientFrontend()
                with torch.no_grad():
                    frontend.Gx.zero_()
                    frontend.Gy.zero_()
                    frontend.Gx[0, 0, 1, 1] = response
                    frontend.Gy[0, 0, 1, 1] = -response
                images = torch.rand(2, 1, 13, 15, requires_grad=True)
                features = frontend(images)
                self.assert_finite(features)
                self.assertEqual(torch.count_nonzero(features[:, 1]).item(), 0)
                features[:, 1].sum().backward()
                self.assert_finite(images.grad)
                for kernel in (frontend.Gx, frontend.Gy):
                    self.assertIsNotNone(kernel.grad)
                    self.assert_finite(kernel.grad)

    def test_binary_training_updates_kernels_only_in_the_learned_control(self):
        images = torch.rand(3, 1, 32, 32)
        labels = torch.tensor([[0.0], [1.0], [1.0]])
        for trainable in (False, True):
            with self.subTest(trainable=trainable):
                torch.manual_seed(42)
                model = LearnableFilterCNN128(trainable=trainable, img_size=32)
                before = {name: getattr(model.frontend, name).detach().clone()
                          for name in ("Gx", "Gy")}
                optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
                loss = F.binary_cross_entropy_with_logits(model(images), labels)
                self.assert_finite(loss)
                loss.backward()
                for name in ("Gx", "Gy"):
                    kernel = getattr(model.frontend, name)
                    if trainable:
                        self.assertIsNotNone(kernel.grad)
                        self.assert_finite(kernel.grad)
                        self.assertGreater(kernel.grad.abs().max().item(), 0.0)
                    else:
                        self.assertIsNone(kernel.grad)
                optimizer.step()
                for name in ("Gx", "Gy"):
                    changed = not torch.equal(before[name], getattr(model.frontend, name))
                    self.assertEqual(changed, trainable)

    def test_weights_only_checkpoint_restores_filters_and_safety_settings(self):
        model = LearnableFilterCNN128(
            img_size=32, magnitude_eps=2e-5, angle_eps=2e-3, normalization_eps=2e-7
        ).eval()
        with torch.no_grad():
            model.frontend.Gx.add_(torch.rand_like(model.frontend.Gx) * 0.01)
            model.frontend.Gy.add_(torch.rand_like(model.frontend.Gy) * 0.01)
        images = torch.rand(2, 1, 32, 32)
        with torch.no_grad():
            expected = model(images)
        checkpoint_file = io.BytesIO()
        torch.save({"model_state": model.state_dict(), "img_size": 32}, checkpoint_file)
        checkpoint_file.seek(0)
        checkpoint = torch.load(checkpoint_file, weights_only=True, map_location="cpu")
        restored = LearnableFilterCNN128(img_size=checkpoint["img_size"]).eval()
        restored.load_state_dict(checkpoint["model_state"], strict=True)
        for name, value in model.frontend.state_dict().items():
            torch.testing.assert_close(value, restored.frontend.state_dict()[name], rtol=0, atol=0)
        with torch.no_grad():
            torch.testing.assert_close(expected, restored(images), rtol=0, atol=0)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
    def test_cuda_frontend_agrees_with_cpu_and_backpropagates(self):
        cpu_frontend = LearnedGradientFrontend()
        gpu_frontend = LearnedGradientFrontend().cuda()
        gpu_frontend.load_state_dict(cpu_frontend.state_dict())
        images = torch.rand(2, 1, 17, 23)
        gpu_images = images.cuda().requires_grad_()
        gpu_features = gpu_frontend(gpu_images)
        torch.testing.assert_close(cpu_frontend(images), gpu_features.cpu(), rtol=1e-4, atol=1e-5)
        gpu_features.square().sum().backward()
        self.assert_finite(gpu_images.grad)
        for kernel in (gpu_frontend.Gx, gpu_frontend.Gy):
            self.assert_finite(kernel.grad)


if __name__ == "__main__":
    unittest.main()
