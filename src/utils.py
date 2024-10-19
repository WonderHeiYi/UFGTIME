# -*- coding:utf-8 -*-
import os
import math
import random
import yaml
import torch
import torch.nn as nn
import numpy as np 
import pandas as pd

from typing import List
from pandas.tseries import offsets
from pandas.tseries.frequencies import to_offset


def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    # print(f"Random seed set as {seed}")

def get_frame(frametype:str='Haar'):    
    if frametype == 'Haar':
        D1 = lambda x: np.cos(x / 2)
        D2 = lambda x: np.sin(x / 2)
        dfilters = [D1, D2]
    elif frametype == 'Linear':
        D1 = lambda x: np.square(np.cos(x / 2))
        D2 = lambda x: np.sin(x) / np.sqrt(2)
        D3 = lambda x: np.square(np.sin(x / 2))
        dfilters = [D1, D2, D3]
    else:
        raise Exception('Invalid FrameType')
    return dfilters

def cheb_approx(func, n=2):
    """Chebyshev polynomial approxmation for
    given frame type assuming f : [0, pi] -> R
    Param:
        frame type: frame functions
        n: degree of Chebyshev Poly approx
    Return:
        c: Chebyshev poly
    """  
    quad_points = 500
    c = np.zeros(n, dtype=np.float32)
    a = np.pi / 2
    for k in range(1, n + 1):
        Integrand = lambda x: np.cos((k - 1) * x)\
                    * func(a * (np.cos(x) + 1))
        x = np.linspace(0, np.pi, quad_points)
        y = Integrand(x)
        c[k - 1] = 2 / np.pi * np.trapz(y, x)
    return c


def sinusoidal_posemb(x, dim, theta=10000):
    """Sinusoidal postion embeding for sequantial time
    Param:
        x: int/pd.datatime64 time input
        dim: target dimision for time features
        theta: embedding theta
    Return:
        time embeding
    """
    emb = np.log(theta) / (dim // 2 - 1)
    emb = np.exp(np.arange(dim // 2) * -emb)
    emb = np.outer(x, emb)
    emb = np.concatenate((np.sin(emb), np.cos(emb)), axis=-1)
    return emb

def concat_fun(inputs, axis=-1):
    if len(inputs) == 1:
        return inputs[0]
    else:
        return torch.cat(inputs, dim=axis)

def slice_arrays(arrays, start=None, stop=None):
    """Slice an array or list of arrays.

    This takes an array-like, or a list of
    array-likes, and outputs:
        - arrays[start:stop] if `arrays` is an array-like
        - [x[start:stop] for x in arrays] if `arrays` is a list

    Can also work on list/array of indices: `slice_arrays(x, indices)`

    Arguments:
        arrays: Single array or list of arrays.
        start: can be an integer index (start index)
            or a list/array of indices
        stop: integer (stop index); should be None if
            `start` was a list.

    Returns:
        A slice of the array(s).

    Raises:
        ValueError: If the value of start is a list and stop is not None.
    """

    if arrays is None:
        return [None]

    if isinstance(arrays, np.ndarray):
        arrays = [arrays]

    if isinstance(start, list) and stop is not None:
        raise ValueError('The stop argument has to be None if the value of start '
                         'is a list.')
    elif isinstance(arrays, list):
        if hasattr(start, '__len__'):
            # hdf5 datasets only support list objects as indices
            if hasattr(start, 'shape'):
                start = start.tolist()
            return [None if x is None else x[start] for x in arrays]
        else:
            if len(arrays) == 1:
                return arrays[0][start:stop]
            return [None if x is None else x[start:stop] for x in arrays]
    else:
        if hasattr(start, '__len__'):
            if hasattr(start, 'shape'):
                start = start.tolist()
            return arrays[start]
        elif hasattr(start, '__getitem__'):
            return arrays[start:stop]
        else:
            return [None]

def save_model(model, model_dir, epoch=None):
    if model_dir is None:
        return
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    epoch = str(epoch) if epoch else ''
    file_name = os.path.join(model_dir, epoch + '_dhfm.pt')
    with open(file_name, 'wb') as f:
        torch.save(model, f)

def load_model(model_dir, epoch=None):
    if not model_dir:
        return
    epoch = str(epoch) if epoch else ''
    file_name = os.path.join(model_dir, epoch + '_dhfm.pt')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if not os.path.exists(file_name):
        return
    with open(file_name, 'rb') as f:
        model = torch.load(f)
    return model

def masked_MAPE(v, v_, axis=None):
    '''
    Mean absolute percentage error.
    :param v: np.ndarray or int, ground truth.
    :param v_: np.ndarray or int, prediction.
    :param axis: axis to do calculation.
    :return: int, MAPE averages on all elements of input.
    '''
    mask = (v == 0)
    percentage = np.abs(v_ - v) / np.abs(v)
    if np.any(mask):
        masked_array = np.ma.masked_array(percentage, mask=mask)
        result = masked_array.mean(axis=axis)
        if isinstance(result, np.ma.MaskedArray):
            return result.filled(np.nan)
        else:
            return result
    return np.mean(percentage, axis).astype(np.float64)

def MAPE(v, v_, axis=None):
    '''
    Mean absolute percentage error.
    :param v: np.ndarray or int, ground truth.
    :param v_: np.ndarray or int, prediction.
    :param axis: axis to do calculation.
    :return: int, MAPE averages on all elements of input.
    '''
    mape = (np.abs(v_ - v) / (np.abs(v)+1e-5)).astype(np.float64)
    mape = np.where(mape > 5, 5, mape)
    return np.mean(mape, axis)

def SMAPE(P, A):
    nz = np.where(A > 0)
    Pz = P[nz]
    Az = A[nz]
    return np.mean(2 * np.abs(Az - Pz) / (np.abs(Az) + np.abs(Pz)))

def RMSE(v, v_, axis=None):
    '''
    Mean squared error.
    :param v: np.ndarray or int, ground truth.
    :param v_: np.ndarray or int, prediction.
    :param axis: axis to do calculation.
    :return: int, RMSE averages on all elements of input.
    '''
    return np.sqrt(np.mean((v_ - v) ** 2, axis)).astype(np.float64)

def MAE(v, v_, axis=None):
    '''
    Mean absolute error.
    :param v: np.ndarray or int, ground truth.
    :param v_: np.ndarray or int, prediction.
    :param axis: axis to do calculation.
    :return: int, MAE averages on all elements of input.
    '''
    return np.mean(np.abs(v_ - v), axis).astype(np.float64)

def evaluate(y, y_hat, by_step=False, by_node=False):
    '''
    :param y: array in shape of [count, time_step, node].
    :param y_hat: in same shape with y.
    :param by_step: evaluate by time_step dim.
    :param by_node: evaluate by node dim.
    :return: array of mape, mae and rmse.
    '''
    if not by_step and not by_node:
        return MAPE(y, y_hat), MAE(y, y_hat), RMSE(y, y_hat)
    if by_step and by_node:
        return MAPE(y, y_hat, axis=0),\
               MAE(y, y_hat, axis=0), RMSE(y, y_hat, axis=0)
    if by_step:
        return MAPE(y, y_hat, axis=(0, 2)),\
               MAE(y, y_hat, axis=(0, 2)), RMSE(y, y_hat, axis=(0, 2))
    if by_node:
        return MAPE(y, y_hat, axis=(0, 1)),\
               MAE(y, y_hat, axis=(0, 1)), RMSE(y, y_hat, axis=(0, 1))


class CSiLU(nn.Module):
    def __init__(self, use_phase=False):
        super().__init__()
        self.use_phase = use_phase
        self.act = nn.SiLU()

    def forward(self, x):
        if self.use_phase:
            return self.act(torch.abs(x)) * torch.exp(1.j * torch.angle(x)) 
        else:
            return self.act(x.real) + 1.j * self.act(x.imag)
        

# class TimeFeature:
#     def __init__(self):
#         pass

#     def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
#         pass

#     def __repr__(self):
#         return self.__class__.__name__ + "()"


# class SecondOfMinute(TimeFeature):
#     """Minute of hour encoded as value between [-0.5, 0.5]"""

#     def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
#         return index.second / 59.0 - 0.5


# class MinuteOfHour(TimeFeature):
#     """Minute of hour encoded as value between [-0.5, 0.5]"""

#     def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
#         return index.minute / 59.0 - 0.5


# class HourOfDay(TimeFeature):
#     """Hour of day encoded as value between [-0.5, 0.5]"""

#     def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
#         return index.hour / 23.0 - 0.5


# class DayOfWeek(TimeFeature):
#     """Hour of day encoded as value between [-0.5, 0.5]"""

#     def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
#         return index.dayofweek / 6.0 - 0.5


# class DayOfMonth(TimeFeature):
#     """Day of month encoded as value between [-0.5, 0.5]"""

#     def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
#         return (index.day - 1) / 30.0 - 0.5


# class DayOfYear(TimeFeature):
#     """Day of year encoded as value between [-0.5, 0.5]"""

#     def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
#         return (index.dayofyear - 1) / 365.0 - 0.5


# class MonthOfYear(TimeFeature):
#     """Month of year encoded as value between [-0.5, 0.5]"""

#     def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
#         return (index.month - 1) / 11.0 - 0.5


# class WeekOfYear(TimeFeature):
#     """Week of year encoded as value between [-0.5, 0.5]"""

#     def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
#         return (index.isocalendar().week - 1) / 52.0 - 0.5


# def time_features_from_frequency_str(freq_str: str) -> List[TimeFeature]:
#     """
#     Returns a list of time features that will be appropriate for the given frequency string.
#     Parameters
#     ----------
#     freq_str
#         Frequency string of the form [multiple][granularity] such as "12H", "5min", "1D" etc.
#     """

#     features_by_offsets = {
#         offsets.YearEnd: [],
#         offsets.QuarterEnd: [MonthOfYear],
#         offsets.MonthEnd: [MonthOfYear],
#         offsets.Week: [DayOfMonth, WeekOfYear],
#         offsets.Day: [DayOfWeek, DayOfMonth, DayOfYear],
#         offsets.BusinessDay: [DayOfWeek, DayOfMonth, DayOfYear],
#         offsets.Hour: [HourOfDay, DayOfWeek, DayOfMonth, DayOfYear],
#         offsets.Minute: [
#             MinuteOfHour,
#             HourOfDay,
#             DayOfWeek,
#             DayOfMonth,
#             DayOfYear,
#         ],
#         offsets.Second: [
#             SecondOfMinute,
#             MinuteOfHour,
#             HourOfDay,
#             DayOfWeek,
#             DayOfMonth,
#             DayOfYear,
#         ],
#     }

#     offset = to_offset(freq_str)

#     for offset_type, feature_classes in features_by_offsets.items():
#         if isinstance(offset, offset_type):
#             return [cls() for cls in feature_classes]

#     supported_freq_msg = f"""
#     Unsupported frequency {freq_str}
#     The following frequencies are supported:
#         Y   - yearly
#             alias: A
#         M   - monthly
#         W   - weekly
#         D   - daily
#         B   - business days
#         H   - hourly
#         T   - minutely
#             alias: min
#         S   - secondly
#     """
#     raise RuntimeError(supported_freq_msg)


# def time_features(dates, freq='h'):
#     return np.vstack([feat(dates) for feat in time_features_from_frequency_str(freq)])

    

