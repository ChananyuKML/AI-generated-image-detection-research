"""Paper-style accuracy tables from recorded GenImage matrices; no inference."""

import math
from pathlib import Path
import tempfile
import time

from .genimage_protocol import GENERATORS, matrix_summary

MODEL_LABELS = {
    'fixed_filter_cnn': 'Fixed-filter CNN',
    'learned_filter_cnn': 'Learned-filter CNN',
    'dual_cnn_128': 'Original gradient CNN',
    'fixed_filter_kcnet': 'Frozen CNN + KCNet',
    'finetuned_cnn_kcnet': 'Fine-tuned CNN + KCNet',
}
GENERATOR_LABELS = dict(zip(GENERATORS, ('Midjourney', 'SD1.5', 'ADM', 'GLIDE', 'Wukong', 'VQDM', 'BigGAN')))


def normalize_matrices(record, models, seeds):
    """Recalculate means from the recorded cells, retaining missing values."""
    lookup = {}
    for matrix in record.get('matrices', []):
        key = (matrix['model'], matrix['seed'])
        if key in lookup:
            raise ValueError('Duplicate model/seed matrix.')
        if matrix['generators'] != list(GENERATORS):
            raise ValueError('Unexpected generator order; refusing to transpose results.')
        values = matrix['accuracy']
        if len(values) != 7 or any(len(row) != 7 for row in values):
            raise ValueError('Expected a 7 by 7 accuracy matrix.')
        cells = []
        for i, source in enumerate(GENERATORS):
            for j, target in enumerate(GENERATORS):
                accuracy = values[i][j]
                if accuracy is not None:
                    if isinstance(accuracy, bool) or not isinstance(accuracy, (int, float)) or not math.isfinite(accuracy):
                        raise ValueError('Accuracy must be a finite fraction or missing.')
                    cells.append(dict(training_generator=source, testing_generator=target, accuracy=accuracy))
        lookup[key] = matrix_summary(cells)
    return [dict(lookup.get((model, seed), matrix_summary([])), model=model, seed=seed)
            for seed in seeds for model in models]


def percent(value, missing='Pending'):
    return missing if value is None else f'{100 * value:.2f}'


def markdown_table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |',
                      '| ' + ' | '.join(['---'] + ['---:'] * (len(headers) - 1)) + ' |',
                      *['| ' + ' | '.join(map(str, row)) + ' |' for row in rows]])


def latex_table(caption, headers, rows):
    def escape(value):
        return str(value).replace('_', r'\_').replace('%', r'\%').replace('&', r'\&')
    lines = [r'\begin{table*}[t]', r'\centering', r'\small', '\\caption{' + escape(caption) + '}',
             '\\begin{tabular}{l' + 'r' * (len(headers) - 1) + '}', r'\hline',
             ' & '.join(map(escape, headers)) + r' \\', r'\hline']
    lines.extend(' & '.join(map(escape, row)) + r' \\' for row in rows)
    return '\n'.join(lines + [r'\hline', r'\end{tabular}', r'\end{table*}'])


def write_accuracy_tables(batch, record, models, seeds):
    """Write Markdown and LaTeX with complete-column and complete-matrix means."""
    matrices = normalize_matrices(record, models, seeds)
    output = Path(batch) / 'presentation'
    output.mkdir(parents=True, exist_ok=True)
    headers = ['Method', *GENERATOR_LABELS.values(), 'Avg.', 'Cells']
    total = sum(matrix['completed_cells'] for matrix in matrices)
    expected = len(matrices) * 49
    intro = [
        '# Tiny GenImage accuracy tables', '',
        f'Recorded test cells: **{total}/{expected}**. Accuracy values are percentages, displayed to two decimal places.', '',
        'Seven-generator Tiny GenImage subset adaptation. SD1.4 is absent. Each target test set contains 500 real and 500 AI images. '
        'Checkpoints are selected by minimum validation BCE; the AI threshold is strictly greater than 0.5.', '',
        'The summary follows the layout of Table III in the research paper: each generator column is the equal-weight mean '
        'over its seven training-source detectors. Avg. is the mean of all 49 train/test cells. '
        'A column mean stays Pending until all seven required cells exist; the overall mean requires all 49. '
        'Pending is an unmeasured result, not zero. Source rows and individual cells are shown below.', '',
        'Source: `genimage_matrices.json` in this batch, generated from held-out `genimage_evaluation/<target>/metrics.json`. '
        'Existing `genimage_table4.csv` stores fractions in [0, 1]; multiply by 100 for percent. '
        'Published full-GenImage scores are kept separate from these new subset measurements.', '',
    ]
    latex = ['% Generated from recorded Tiny GenImage tests. Accuracy (%); -- means pending.',
             '% SD1.4 is absent. Column averages require seven cells; overall average requires 49.', '']
    for seed in seeds:
        selected = [matrix for matrix in matrices if matrix['seed'] == seed]
        rows = [[MODEL_LABELS.get(matrix['model'], matrix['model']),
                 *[percent(matrix['column_means'][target]) for target in GENERATORS],
                 percent(matrix['macro_accuracy']), f"{matrix['completed_cells']}/49"] for matrix in selected]
        intro += [f'## Model-by-test-generator summary - seed {seed}', '', markdown_table(headers, rows), '']
        latex_rows = [[str(value).replace('Pending', '--') for value in row] for row in rows]
        latex += [latex_table(f'Tiny GenImage cross-subset accuracy (percent), seed {seed}. '
                              'Each test-generator column averages seven training sources; Avg. averages 49 pairs.',
                              headers, latex_rows), '']
        for matrix in selected:
            label = MODEL_LABELS.get(matrix['model'], matrix['model'])
            rows = [[GENERATOR_LABELS[source], *[percent(value) for value in matrix['accuracy'][i]],
                     percent(matrix['row_means'][source])]
                    for i, source in enumerate(GENERATORS)]
            matrix_headers = ['Train / test', *GENERATOR_LABELS.values(), 'Row avg.']
            intro += [f'### {label} - seed {seed}', '', markdown_table(matrix_headers, rows), '',
                      f"All 49 pairs: **{percent(matrix['macro_accuracy'])}**; "
                      f"same-generator mean (7 diagonal cells): **{percent(matrix['diagonal_accuracy'])}**; "
                      f"unseen-generator mean (42 off-diagonal cells): **{percent(matrix['off_diagonal_accuracy'])}**.", '']
            latex += [latex_table(f'{label}: training-source rows and test-generator columns, '
                                  f'Tiny GenImage accuracy (percent), seed {seed}.', matrix_headers,
                                  [[str(value).replace('Pending', '--') for value in row] for row in rows]), '']
    for filename, content in [('genimage_accuracy_tables.md', '\n'.join(intro)),
                              ('genimage_accuracy_tables.tex', '\n'.join(latex))]:
        path = output / filename
        if not path.exists() or path.read_text(encoding='utf-8') != content:
            # The optional observer and queue may export at the same boundary.
            with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=output,
                                             prefix=filename + '.', suffix='.part', delete=False) as handle:
                handle.write(content)
                temporary = Path(handle.name)
            try:
                for attempt in range(10):
                    try:
                        temporary.replace(path)
                        break
                    except PermissionError:
                        if attempt == 9:
                            raise
                        time.sleep(.05)
            finally:
                temporary.unlink(missing_ok=True)
    return {'completed_cells': total, 'expected_cells': expected, 'complete': total == expected,
            'markdown': str(output / 'genimage_accuracy_tables.md'),
            'latex': str(output / 'genimage_accuracy_tables.tex')}
