# UFGTime+

The layout follows [UFGTime](https://github.com/WonderHeiYi/UFGTIME).

## Environment

Use the UFGTime environment: Python 3.10, PyTorch 2.1.1 with CUDA 12.1,
DGL 2.0.0 with the matching CUDA build, and PyG 2.5.3.
Install CUDA-matched PyTorch and DGL first, then the remaining dependencies:

```bash
pip install -r requirements.txt
```

## Reproduce

Put the original dataset CSV files in `data/`, then run:

```bash
python main.py --data_name ECG
python main.py --data_name ETTh1_96 --device cuda:0
```

The entry loads `config/ufgtime_plus_config.yaml` automatically. For this
folder inside the original project, reuse its existing data directly:

```bash
python main.py --data_name Covid --root_path ../data
```

Short-term datasets: `ECG`, `Covid`, `Electricity`, `Solar`, `Traffic`,
`Wiki500`, `exchange`, `fred_md`, `ili`, `nasdaq`, `nn5`, `nyse`.
ETT: `ETTh1` and `ETTm1`, each with suffix `_96`, `_192`, `_336`, or `_720`.

In YAML, `scheduler_patience: null` disables the scheduler. Short-term runs
without a recorded patience use this setting; Electricity, Wiki500, exchange,
and nn5 use patience 10, and ETT uses 30.

Results are printed and appended to `results/<dataset>.txt`.

Source: `main.py`, `argument.py`, `src/{model,data_loader,data_provider,utils}.py`.
