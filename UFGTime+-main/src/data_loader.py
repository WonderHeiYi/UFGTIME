"""Keep the short-term and ETT preprocessing used in the recorded runs."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.utils.data import Dataset


class ForecastDataset(Dataset):
    def __init__(self, root_path, data_name, flag, seq_len, pred_len):
        if flag not in {"train", "val", "test"}:
            raise ValueError("flag must be train, val, or test")
        self.seq_len, self.pred_len = seq_len, pred_len
        frame = pd.read_csv(Path(root_path) / f"{data_name}.csv")
        if data_name.startswith("ETT"):
            values = frame.iloc[:, 1:]
            factor = 4 if data_name.startswith("ETTm") else 1
            train_end, val_end, test_end = (8640 * factor, 11520 * factor, 14400 * factor)
            if len(frame) < test_end:
                raise ValueError(f"{data_name} needs at least {test_end} rows")
            self.scaler = StandardScaler().fit(values.iloc[:train_end].values)
            values = self.scaler.transform(values.values)
            bounds = {"train": (0, train_end),
                      "val": (train_end - seq_len, val_end),
                      "test": (val_end - seq_len, test_end)}
            self.extra_window = 1
        else:
            if data_name != "ECG":
                frame = frame.rename(columns={"Unnamed: 0": "date", "LocalTime": "date"}).set_index("date")
            train_end = int(len(frame) * 0.7)
            val_end = int(len(frame) * (0.7 + 0.2))
            self.scaler = MinMaxScaler().fit(frame.iloc[:train_end].values)
            values = self.scaler.transform(frame.values).astype(np.float32)
            bounds = {"train": (0, train_end), "val": (train_end, val_end),
                      "test": (val_end, len(values))}
            # Preserve the original short-term window count (no trailing +1).
            self.extra_window = 0
        begin, end = bounds[flag]
        self.data = values[begin:end]
        self.num_ts = self.data.shape[1]
        if len(self) < 1:
            raise ValueError(f"{data_name}/{flag} is too short for these window lengths")

    def __getitem__(self, index):
        end = index + self.seq_len
        return self.data[index:end], self.data[end:end + self.pred_len]

    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len + self.extra_window
