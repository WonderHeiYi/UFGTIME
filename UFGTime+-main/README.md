# UFGTime+ :hugs:

UFGTime+ extends [UFGTime](https://github.com/WonderHeiYi/UFGTIME) by connecting variables at the same frequency and incorporating a trend branch. It combines frequency-domain framelet message passing with trend forecasting to capture cross-variable dependencies and temporal patterns in multivariate time series.

## :hammer_and_wrench: Python Dependencies

Our UFGTime+ framework is implemented in Python 3.10. The major libraries include:

- **Python 3.10**
- **PyTorch 2.1.1 with CUDA 12.1**
- **NumPy 1.26.4**
- **PyG 2.5.3**
- **DGL 2.0.0 with CUDA 12.1**

Install PyTorch and DGL with matching CUDA builds, then install the remaining dependencies:

```bash
pip install -r requirements.txt
```

## :weight_lifting: To Train Model

To reproduce an experiment, run:

```bash
python main.py --data_name ECG
python main.py --data_name ETTh1_96
```

The training script automatically loads the dataset-specific parameters from `config/ufgtime_plus_config.yaml`. Replace `--data_name` with any dataset option listed below.

## :open_file_folder: Datasets

Place the original dataset CSV files in the `data/` folder at the project root. Use `--root_path` to specify another data directory.

- **Short-term datasets:** `ECG`, `Covid`, `Electricity`, `Solar`, `Traffic`, `Wiki500`, `exchange`, `fred_md`, `ili`, `nasdaq`, `nn5`, `nyse`.
- **ETT datasets:** `ETTh1` and `ETTm1`, each with prediction horizons `96`, `192`, `336`, and `720`. Use dataset options such as `ETTh1_96` or `ETTm1_720`.

ETT data files should be named `ETTh1.csv` and `ETTm1.csv`; the horizon suffix is only used in `--data_name`.

## :open_file_folder: File Specifications

- **data/**: Dataset CSV files.
- **src/**: Model, data loading, and utility functions.
- **main.py**: Training and evaluation entry point.
- **argument.py**: Command-line arguments and configuration loading.
- **config/ufgtime_plus_config.yaml**: Dataset-specific model and training parameters.
- **results/**: Evaluation results, saved as `<dataset>.txt`.

In the YAML configuration, `scheduler_patience: null` disables learning-rate scheduling. Evaluation metrics are printed and appended to the corresponding result file.
