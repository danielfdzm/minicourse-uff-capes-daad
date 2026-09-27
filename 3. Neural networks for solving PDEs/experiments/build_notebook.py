"""Assemble the self-contained notebook; optionally execute it in a fresh kernel.

python experiments/build_notebook.py --execute

The cell-source modules are only needed to maintain/rebuild the notebook. The
resulting .ipynb runs by itself and does not import any of these source modules.
"""
from __future__ import annotations
import argparse
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / 'Neural_PDE_Laboratory.ipynb'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    cells = []
    for name in ['core', 'inverse', 'extras', 'bonus', 'multiscale', 'adaptive', 'heat', 'parametric', 'conclusion']:
        path = ROOT / 'experiments' / 'notebook_parts' / f'{name}.py'
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        for i, record in enumerate(module.CELLS):
            source = record['source'].strip()
            source = source.replace('## Inverse diffusion: where you measure matters',
                                    '## 6 · Inverse diffusion: where you measure matters')
            source = source.replace('## Extra experiment — dimension, sampling, and the price of a grid',
                                    '## 7 · High dimensions: sampling and the price of a grid')
            source = source.replace('## Extra experiment — a neural Poisson solution around a hole',
                                    '## 8 · A neural Poisson solution around a hole')
            cell = dict(cell_type=record['cell_type'], id=f'{name}-{i:02}',
                        metadata={}, source=source)
            if cell['cell_type'] == 'code':
                cell.update(execution_count=None, outputs=[])
                cell['metadata']['scrolled'] = False
                # Parse every code cell, including the commented installation cell.
                compile(source, f'{name}-cell-{i}', 'exec')
            cells.append(cell)
    notebook = dict(cells=cells, nbformat=4, nbformat_minor=5,
                    metadata=dict(kernelspec=dict(display_name='Python 3', language='python', name='python3'),
                                  language_info=dict(name='python', file_extension='.py', mimetype='text/x-python',
                                                     pygments_lexer='ipython3'),
                                  title='Neural PDE Laboratory'))
    TARGET.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + '\n')
    print(f'Built {TARGET.name}: {len(cells)} cells.', flush=True)
    if args.execute:
        import nbformat
        from nbclient import NotebookClient
        notebook = nbformat.read(TARGET, as_version=4)
        client = NotebookClient(notebook, timeout=600, kernel_name='python3',
                                resources={'metadata': {'path': str(ROOT)}},
                                allow_errors=False, record_timing=True)
        def report(cell, cell_index, **kwargs):
            if cell.cell_type == 'code':
                first = cell.source.splitlines()[0][:85]
                print(f'Cell {cell_index+1}/{len(cells)}: {first}', flush=True)
        client.on_cell_start = report
        try:
            client.execute()
        finally:
            nbformat.write(notebook, TARGET)
        nbformat.validate(notebook)
        errors = [output for cell in notebook.cells if cell.cell_type == 'code'
                  for output in cell.outputs if output.output_type == 'error']
        if errors:
            raise RuntimeError(f'{len(errors)} notebook execution errors')
        print(f'Validated executed notebook: {TARGET.stat().st_size/1e6:.1f} MB.', flush=True)

if __name__ == '__main__':
    main()
