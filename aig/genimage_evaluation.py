"""Evaluate a source-generator detector on all seven held-out Tiny generator tests."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .experiment import BinaryImages, binary_metrics, build_model, project_path, select_device, write_json
from .genimage_protocol import GENERATORS, PREFIX, PROTOCOL, REFERENCE, audit_manifest_roots
from .progress import TrainingProgress


def evaluate_matrix_row(checkpoint_path, data_root, device, batch_size=32, workers=0, output_dir=None):
    checkpoint_path, data_root = Path(checkpoint_path), Path(data_root)
    audit = audit_manifest_roots(data_root)
    checkpoint = torch.load(checkpoint_path, weights_only=True, map_location='cpu')
    config = checkpoint['config']
    source = config['dataset'].removeprefix(PREFIX)
    if (config['dataset'] != PREFIX + source or source not in GENERATORS
            or config.get('dataset_signature') != audit['signatures'][config['dataset']]):
        raise ValueError('The detector must be trained on the matching single-generator protocol split.')
    if (checkpoint.get('format_version') != 1 or config.get('labels') != {'nature': 0, 'ai': 1}
            or config.get('selection') != 'minimum validation BCE'
            or checkpoint['epoch'] != checkpoint['best_epoch']):
        raise ValueError('Cross-generator testing requires the validation-selected best.pt.')
    model, transform = build_model(config['model'], config.get('model_config'))
    model.load_state_dict(checkpoint['model_state'])
    model.to(device).eval()
    with checkpoint_path.open('rb') as handle:
        checkpoint_hash = hashlib.file_digest(handle, 'sha256').hexdigest()
    output_dir = Path(output_dir) if output_dir else checkpoint_path.parent / 'genimage_evaluation'
    output_dir.mkdir(parents=True, exist_ok=True)
    progress = TrainingProgress(output_dir, config['model'], config['dataset'], config['seed'], config['epochs'], checkpoint['epoch'])
    progress.path = checkpoint_path.parent / 'evaluation_progress.json'
    progress.emit(force=True, status='evaluating', checkpoint_epoch=checkpoint['epoch'])
    cells = []
    for index, target in enumerate(GENERATORS, 1):
        root = data_root / (PREFIX + target)
        dataset = BinaryImages(root / 'test', transform)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers,
                            pin_memory=device.type == 'cuda')
        progress.emit(force=True, testing_generator=target, target_index=index, target_count=len(GENERATORS))
        progress.start_phase('testing_' + target, len(loader))
        labels, scores, loss_sum, correct = [], [], 0.0, 0
        with torch.inference_mode():
            for batch, (inputs, targets) in enumerate(loader, 1):
                logits = model(inputs.to(device, non_blocking=True)).flatten()
                loss_sum += torch.nn.functional.binary_cross_entropy_with_logits(
                    logits, targets.to(device).float(), reduction='sum').item()
                probabilities = torch.sigmoid(logits).cpu()
                labels.extend(targets.tolist())
                scores.extend(probabilities.tolist())
                correct += int(((probabilities > .5) == targets.bool()).sum())
                progress.batch('testing_' + target, batch, len(loader), len(labels), loss_sum / len(labels), correct / len(labels))
        metrics = binary_metrics(labels, scores, .5)
        metrics.update(bce=loss_sum / len(dataset), model=config['model'], seed=config['seed'],
                       evaluation_protocol=PROTOCOL, protocol_reference=REFERENCE,
                       training_generator=source, testing_generator=target,
                       training_dataset=config['dataset'], evaluation_dataset=PREFIX + target,
                       evaluation_dataset_signature=audit['signatures'][PREFIX + target],
                       checkpoint_sha256=checkpoint_hash, checkpoint_epoch=checkpoint['epoch'],
                       checkpoint_selection=config['selection'], threshold_selection='fixed at 0.5; no test-set tuning')
        if 'encoder_source' in config:
            metrics.update(encoder_source=config['encoder_source'], model_config=config['model_config'])
        target_dir = output_dir / target
        write_json(target_dir / 'metrics.json', metrics)
        with (target_dir / 'predictions.csv').open('w', newline='', encoding='utf-8') as handle:
            writer = csv.writer(handle)
            writer.writerow(('image', 'label', 'probability_ai', 'prediction'))
            for (path, label), score in zip(dataset.samples, scores):
                writer.writerow((path.relative_to(root).as_posix(), label, score, int(score > .5)))
        cells.append(metrics)
        print(f"{source} -> {target}: accuracy={metrics['accuracy']:.4f}, ROC-AUC={metrics['roc_auc']:.4f}, images={len(dataset)}", flush=True)
    summary = {'evaluation_protocol': PROTOCOL, 'model': config['model'], 'seed': config['seed'],
               'training_generator': source, 'training_dataset': config['dataset'],
               'checkpoint_sha256': checkpoint_hash, 'checkpoint_epoch': checkpoint['epoch'],
               'results': cells, 'aggregation': 'equal-weight mean across seven test generators',
               'missing_official_generators': ['sdv4'], 'dataset_audit': audit,
               **{key: sum(cell[key] for cell in cells) / len(cells)
                  for key in ('accuracy', 'balanced_accuracy', 'roc_auc', 'average_precision', 'bce')}}
    write_json(output_dir / 'matrix_row.json', summary)
    progress.emit(force=True, status='completed', phase='completed')
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--data-root', default='dataset/prepared')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--device', choices=('auto', 'cuda', 'cpu'), default='auto')
    parser.add_argument('--output-dir')
    args = parser.parse_args(argv)
    if args.batch_size < 1 or args.workers < 0:
        parser.error('Use positive batch size and nonnegative workers.')
    report = evaluate_matrix_row(project_path(args.checkpoint), project_path(args.data_root),
                                 select_device(args.device), args.batch_size, args.workers,
                                 project_path(args.output_dir) if args.output_dir else None)
    print(json.dumps({key: report[key] for key in ('model', 'training_generator', 'accuracy', 'roc_auc')}, indent=2))


if __name__ == '__main__':
    main()
