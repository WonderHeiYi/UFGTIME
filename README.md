
# UFGTime: Reforming the Pure Graph Paradigm for Multivariate Time Series Forecasting in the Frequency Domain :hugs: 

UFGTtime transforms time series into frequency domain signals, preserving sequential information, and introduces a hyperspectral graph structure with sparse topological connections to enhance attention to cross-signal relationships, while mitigating smoothing effects through a global framelet message-passing operator.
<p>
  <img src="./flowchart.jpg" width="1000">
  <br />
</p>

## :hammer_and_wrench: Python Dependencies

Our proposed UFGTime framework is implemented in Python 3.10 and major libraries include:

### :warning: Requirement

- <code>Python 3.10</code>
- <code>PyTorch 2.1.1+cu121</code>
- <code>NumPy 1.26.4</code>
- <code>PyG 2.5.3</code>
- <code>DGL 2.0.0.cu121</code>

## 	:weight_lifting: To Train Model:

```bash
python main.py --data_name [dataset]
```

## :open_file_folder: File Specifications

- **data**: Dict of data sources
- **src**: Dict of source code
  - **data_loader**: Description for the dataset classes.
  - **data_provider.py**: Used functions for data dataloader.
  - **model.py**: Description for the model and relative functions.
  - **utils.py**: Description for the help functions.
- **main**: Main code

## Implementation
Datasets include `ECG`, `Covid`, `Electricity`, `Solar`, `Traffic`, `Wiki500`, `ETTh2_96`, `ETTh2_192`, `ETTh2_336`, `ETTh2_720`, `ETTm2_96`, `ETTm2_192`, `ETTm2_336`, `ETTm2_720`, `ETTm1_96`, `ETTm1_192`, `ETTm1_336`, `ETTm1_720`, `ETTh1_96`, `ETTh1_192`, `ETTh1_336`, `ETTh1_720`
### To reproduce our experiments, please run (example):
`python main.py --data_name ECG` 
Please substitute the `data_name` from the above dataset options to reproduce the experiment.
