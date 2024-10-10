from torch.utils.data import DataLoader
from src.data_loader import (Dataset_ETT_hour, Dataset_ETT_minute,
                            Dataset_Custom, Dataset_Pred, Dataset_PEMS,
                            Dataset_ECG, Dataset_Electricity, Dataset_Covid,
                            Dataset_Solar, Dataset_Traffic, Dataset_Wiki500)
from src.utils import set_seed

"""revised from https://github.com/YoZhibo/ForecastGrapher/
"""

data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'custom': Dataset_Custom,
    'PEMS': Dataset_PEMS,
    'ECG' : Dataset_ECG,
    'Electricity': Dataset_Electricity,
    'Covid': Dataset_Covid,
    'Solar': Dataset_Solar,
    'Traffic': Dataset_Traffic,
    'Wiki500': Dataset_Wiki500
}

def data_provider(
        root_path,
        dataset_name,
        flag,
        seq_len,
        pred_len,
        batch_size,
        features,
        target,
        seed,
        label_len=0,
        signal_len=16,
    ):
    # set_seed(seed)
    Data = data_dict[dataset_name]
    if dataset_name in ['PEMS03', 'PEMS04', 'PEMS07', 'PEMS08']:
        data_path = str(dataset_name) + str('.npz')
    else:
        data_path = str(dataset_name) + str('.csv')

    if flag == 'test':
        shuffle_flag = False
        drop_last = False
        batch_size = batch_size
    elif flag == 'pred':
        shuffle_flag = False
        drop_last = False
        batch_size = 1
        Data = Dataset_Pred
    else:
        shuffle_flag = True
        drop_last = False
        batch_size = batch_size

    data_set = Data(
        root_path=root_path,
        data_path=data_path,
        flag=flag,
        size=[seq_len, label_len, pred_len],
        features=features,
        target=target,
        signal_len=signal_len,
    )
    # print(flag, len(data_set))
    data_loader = DataLoader(
                    data_set,
                    batch_size=batch_size,
                    shuffle=shuffle_flag,
                    drop_last=drop_last
                )
    return data_set, data_loader