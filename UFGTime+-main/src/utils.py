import os
import random

import numpy as np
import torch
import torch.nn as nn

def set_seed(seed: int=42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def get_frame(frametype: str='Haar'):
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
    quad_points = 500
    c = np.zeros(n, dtype=np.float32)
    a = np.pi / 2
    for k in range(1, n + 1):
        Integrand = lambda x: np.cos((k - 1) * x) * func(a * (np.cos(x) + 1))
        x = np.linspace(0, np.pi, quad_points)
        y = Integrand(x)
        c[k - 1] = 2 / np.pi * np.trapz(y, x)
    return c


def MAPE(v, v_, axis=None):
    mape = (np.abs(v_ - v) / (np.abs(v) + 1e-05)).astype(np.float64)
    mape = np.where(mape > 5, 5, mape)
    return np.mean(mape, axis)


def RMSE(v, v_, axis=None):
    return np.sqrt(np.mean((v_ - v) ** 2, axis)).astype(np.float64)


def MAE(v, v_, axis=None):
    return np.mean(np.abs(v_ - v), axis).astype(np.float64)


def evaluate(y, y_hat, by_step=False, by_node=False):
    if not by_step and (not by_node):
        return (MAPE(y, y_hat), MAE(y, y_hat), RMSE(y, y_hat))
    if by_step and by_node:
        return (MAPE(y, y_hat, axis=0), MAE(y, y_hat, axis=0), RMSE(y, y_hat, axis=0))
    if by_step:
        return (MAPE(y, y_hat, axis=(0, 2)), MAE(y, y_hat, axis=(0, 2)), RMSE(y, y_hat, axis=(0, 2)))
    if by_node:
        return (MAPE(y, y_hat, axis=(0, 1)), MAE(y, y_hat, axis=(0, 1)), RMSE(y, y_hat, axis=(0, 1)))


class CSiLU(nn.Module):

    def __init__(self, use_phase=False):
        super().__init__()
        self.use_phase = use_phase
        self.act = nn.SiLU()

    def forward(self, x):
        if self.use_phase:
            return self.act(torch.abs(x)) * torch.exp(1j * torch.angle(x))
        else:
            return self.act(x.real) + 1j * self.act(x.imag)
