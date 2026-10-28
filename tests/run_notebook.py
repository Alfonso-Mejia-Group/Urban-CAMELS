"""Executes every code cell of Urban_CAMELS_data_prep.ipynb in order (as a user would), headless.

usage: UCAMELS_DATA=/path/to/Urban-CAMELS python tests/run_notebook.py Urban_CAMELS_data_prep.ipynb [work_dir]
The package is written to work_dir (default: a new temporary folder)."""
import json, os, sys, tempfile, time
from pathlib import Path
nb_path = Path(sys.argv[1]).resolve()
work = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(tempfile.mkdtemp(prefix='ucamels_nb_'))
work.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(nb_path.parent)); os.chdir(work)
nb = json.load(open(nb_path))
g = {'display': lambda *a, **k: print(*a), '__name__': '__main__'}
import matplotlib; matplotlib.use('Agg')
t0 = time.time()
for i, c in enumerate(nb['cells']):
    if c['cell_type'] != 'code':
        continue
    src = ''.join(c['source']) if isinstance(c['source'], list) else c['source']
    print(f'\n===== cell {i} ({time.time() - t0:.0f} s)', flush=True)
    exec(compile(src, f'cell{i}', 'exec'), g)
print(f'\nNOTEBOOK RAN TO THE END in {time.time() - t0:.0f} s; output in {work}')
