"""Build seven standalone lab notebooks and a small navigation notebook.

python experiments/build_notebook.py --execute --bundle
python experiments/build_notebook.py --part 03_inverse --execute

Each notebook includes its own setup and helpers. Interactive HTML is exported
separately so saved notebooks remain small enough for convenient GitHub previews.
"""
from __future__ import annotations
import argparse
import copy
import importlib.util
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
PARTS = [
    ('01_foundations', 'Membranes, PINNs, Deep Ritz and boundary conditions', '1–3'),
    ('02_poisson_and_limits', 'Poisson surfaces and limits of continuous losses', '4–5'),
    ('03_inverse', 'Inverse diffusion and sensor placement', '6'),
    ('04_dimensions_and_geometry', 'High dimensions and the annulus', '7–8'),
    ('05_reaction_diffusion', 'Reaction–diffusion patterns', '9'),
    ('06_features_and_sampling', 'Fourier features and adaptive collocation', '10–11'),
    ('07_time_and_parameters', 'Heat flow and parameter-conditioned PINNs', '12–13'),
]
MAX_NOTEBOOK_BYTES = 5_000_000  # Project budget, not a claimed GitHub service limit.


def filename(slug):
    return f'Neural_PDE_{slug}.ipynb'


def load_module(name):
    path = ROOT / 'experiments' / 'notebook_parts' / f'{name}.py'
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def md(source):
    return dict(cell_type='markdown', source=source)


def index_table():
    rows = ['| Notebook | Experiments |', '|---|---|']
    rows += [f'| [{slug[:2]} · {title}]({filename(slug)}) | {numbers} |'
             for slug, title, numbers in PARTS]
    return '\n'.join(rows)


def make_notebook(records, title):
    cells = []
    for i, record in enumerate(records):
        cell = dict(cell_type=record['cell_type'], id=f'cell-{i:02}',
                    metadata={}, source=record['source'].strip())
        if cell['cell_type'] == 'code':
            compile(cell['source'], f'{title}-cell-{i}', 'exec')
            cell.update(execution_count=None, outputs=[])
            cell['metadata']['scrolled'] = False
        cells.append(cell)
    return dict(cells=cells, nbformat=4, nbformat_minor=5,
                metadata=dict(kernelspec=dict(display_name='Python 3', language='python', name='python3'),
                              language_info=dict(name='python', file_extension='.py',
                                                 mimetype='text/x-python', pygments_lexer='ipython3'),
                              title=title))


def build_part(slug, title, numbers):
    core = load_module('core').CELLS
    # One original Markdown cell contains the end of experiment 3 and start of 4.
    end_three, start_four = core[13]['source'].split('## 4 ·', 1)
    sections = {
        '01_foundations': core[5:13] + [md(end_three)],
        '02_poisson_and_limits': [md('## 4 ·' + start_four)] + core[14:],
        '03_inverse': load_module('inverse').CELLS,
        '04_dimensions_and_geometry': load_module('extras').CELLS,
        '05_reaction_diffusion': load_module('bonus').CELLS,
        '06_features_and_sampling': load_module('multiscale').CELLS + load_module('adaptive').CELLS,
        '07_time_and_parameters': load_module('heat').CELLS + load_module('parametric').CELLS,
    }
    intro = f'''# Neural PDE Laboratory · {slug[:2]}
## {title}

Experiments **{numbers}** of the laboratory. This notebook is self-contained:
run it in a fresh kernel; no earlier notebook or saved result is required.

[All seven notebooks](Neural_PDE_Laboratory.ipynb) · [Setup and guide](NOTEBOOK_README.md)

Install `experiments/requirements-notebook.txt`, then select **Restart Kernel and
Run All Cells**. The supplied static plots and diagnostics can be read on GitHub.
Interactive explorers are separate, self-contained HTML files in `notebook_outputs/`;
download that folder and open `index.html` in a browser, or regenerate them locally.

Python 3.11+; CPU / float64. Set `RUN_MODE = "thorough"` for longer training runs.

**Notation:**''' + core[0]['source'].split('**Notation:**', 1)[1]
    records = [md(intro)] + copy.deepcopy(core[1:5]) + copy.deepcopy(sections[slug])
    records[3] = md('''## Reading and saving the plots

Static figures are saved as PNG and PDF and embedded in this notebook. Interactive
plots are saved as offline HTML files: open their links locally to rotate, zoom,
and animate them. GitHub displays the static figures without executing JavaScript.''')
    for record in records:
        record['source'] = record['source'].replace('## Inverse diffusion:', '## 6 · Inverse diffusion:')
        record['source'] = record['source'].replace('## Extra experiment — dimension, sampling, and the price of a grid',
                                                   '## 7 · High dimensions: sampling and the price of a grid')
        record['source'] = record['source'].replace('## Extra experiment — a neural Poisson solution around a hole',
                                                   '## 8 · A neural Poisson solution around a hole')
    records += load_module('conclusion').make_cells(slug)
    pos = [part[0] for part in PARTS].index(slug)
    links = ['[Laboratory index](Neural_PDE_Laboratory.ipynb)']
    if pos:
        links.insert(0, f'[Previous notebook]({filename(PARTS[pos-1][0])})')
    if pos + 1 < len(PARTS):
        links.append(f'[Next notebook]({filename(PARTS[pos+1][0])})')
    records.append(md(' · '.join(links)))
    return make_notebook(records, title)


def write_notebook(notebook, target):
    text = json.dumps(notebook, indent=1, ensure_ascii=False) + '\n'
    if len(text.encode()) > MAX_NOTEBOOK_BYTES:
        raise ValueError(f'{target.name} exceeds the 5 MB project size budget')
    target.write_text(text)
    print(f'Wrote {target.name}: {len(notebook["cells"])} cells, {target.stat().st_size/1e6:.2f} MB.', flush=True)


def bundle():
    paths = [ROOT / 'Neural_PDE_Laboratory.ipynb', ROOT / 'NOTEBOOK_README.md',
             ROOT / 'experiments' / 'requirements-notebook.txt', Path(__file__).resolve()]
    paths += [ROOT / filename(slug) for slug, _, _ in PARTS]
    paths += sorted((ROOT / 'experiments' / 'notebook_parts').glob('*.py'))
    paths += sorted(p for p in (ROOT / 'notebook_outputs').iterdir() if p.is_file())
    target = ROOT / 'Neural_PDE_Laboratory_bundle.zip'
    with zipfile.ZipFile(target, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for path in paths:
            archive.write(path, path.relative_to(ROOT))
    print(f'Bundled {len(paths)} files: {target.stat().st_size/1e6:.1f} MB.', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--bundle', action='store_true')
    parser.add_argument('--part', choices=[part[0] for part in PARTS])
    parser.add_argument('--workdir', type=Path, default=ROOT,
                        help='Kernel working directory; useful for isolated validation runs.')
    args = parser.parse_args()
    for slug, title, numbers in PARTS:
        if args.part and args.part != slug:
            continue
        notebook = build_part(slug, title, numbers)
        if args.execute:
            import nbformat
            from nbclient import NotebookClient
            notebook = nbformat.from_dict(notebook)
            print(f'Executing {slug} in a fresh kernel...', flush=True)
            client = NotebookClient(notebook, timeout=600, kernel_name='python3',
                                    resources={'metadata': {'path': str(args.workdir.resolve())}},
                                    allow_errors=False, record_timing=True)
            def report(cell, cell_index, **kwargs):
                if cell.cell_type == 'code':
                    print(f'  Cell {cell_index+1}: {cell.source.splitlines()[0][:85]}', flush=True)
            client.on_cell_start = report
            client.execute()
            nbformat.validate(notebook)
        write_notebook(notebook, ROOT / filename(slug))
    index = '# Neural PDE Laboratory\n\nThe laboratory is split into seven independently runnable notebooks with saved plots.\n\n'
    index += index_table() + '\n\n[Setup, outputs and rebuilding](NOTEBOOK_README.md)\n\n'
    index += 'Download `notebook_outputs/` and open `index.html` to browse the offline interactive explorers.'
    write_notebook(make_notebook([md(index)], 'Neural PDE Laboratory — index'),
                   ROOT / 'Neural_PDE_Laboratory.ipynb')
    if args.bundle:
        bundle()


if __name__ == '__main__':
    main()
