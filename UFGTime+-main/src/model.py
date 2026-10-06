"""UFGTime+: same-frequency graph framelets with trend fusion."""

import numpy as np
import torch
import torch.nn as nn
import dgl.sparse as dglsp
from dgl import knn_graph
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.utils import get_laplacian, to_undirected

from src.utils import CSiLU


def create_filter(num_nodes):
    filter = nn.Parameter(torch.Tensor(num_nodes, 1))
    nn.init.normal_(filter, mean=1, std=0.1)
    return filter


class UFGConv(MessagePassing):

    def __init__(self, in_channels, out_channels, channel_mix=True, bias=False, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)
        self.channel_mix = channel_mix
        self.linear = Linear(in_channels, out_channels).to(torch.cfloat)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        self.linear.reset_parameters()

    def forward(self, x, edge_index, edge_attr):
        if self.channel_mix:
            x = self.linear(x)
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        if self.bias is not None:
            out = out + self.bias
        return out

    def message(self, x_j, edge_attr):
        return edge_attr.view(-1, 1) * x_j


class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """

    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    """

    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return (res, moving_mean)


class UFGTimePlus(nn.Module):

    def __init__(self, seq_length, signal_length, pred_length, hidden_size,
                 embed_size, num_ts, device, approx, s, lev, num_topk=2,
                 knn_dist='cosine', exclude_self=True, knn_algorithm=None,
                 window_centering=False):
        super().__init__()
        self.device = device
        self.embed_size = embed_size
        self.seq_len = seq_length
        self.pred_length = pred_length
        self.k = num_topk
        if knn_dist not in {'cosine', 'euclidean'}:
            raise ValueError("knn_dist must be 'cosine' or 'euclidean'.")
        self.knn_dist = knn_dist
        self.exclude_self = exclude_self
        self.knn_algorithm = knn_algorithm
        self.hidden_size = hidden_size
        self.approx = approx
        self.s = s
        self.lev = lev
        self.J = np.log(2 / np.pi) / np.log(s) + lev - 1
        self.num_ts = num_ts
        self.signal_len = signal_length
        # Retain the original initialization order, including unused parameters.
        self.embeddings = nn.Parameter(torch.randn(1, self.embed_size))
        kernel_size = 25
        self.decompsition = series_decomp(kernel_size)
        self.num_nodes = (self.signal_len // 2 + 1) * self.num_ts
        self.filters = nn.ParameterList(
            [create_filter(self.num_nodes) for i in range(2 * lev)])
        self.conv1_list = nn.ModuleList(
            [UFGConv(in_channels=1, out_channels=hidden_size)
             for i in range(0, self.lev + 1)])
        self.conv2_list = nn.ModuleList(
            [UFGConv(in_channels=hidden_size, out_channels=hidden_size)
             for i in range(0, self.lev + 1)])
        self.clin = nn.Linear(
            in_features=hidden_size * (signal_length // 2 + 1),
            out_features=signal_length // 2 + 1).to(torch.cfloat)
        self.lin2 = nn.Linear(in_features=seq_length, out_features=hidden_size)
        self.lin3 = nn.Linear(in_features=hidden_size, out_features=pred_length)
        self.lin4 = nn.Linear(in_features=pred_length * 2, out_features=pred_length)
        self.Linear_Trend = nn.Linear(self.seq_len, self.pred_length)
        self.isn = nn.InstanceNorm2d(self.num_ts)
        self.act_imag = CSiLU()
        self.act_real = nn.SiLU()
        self.window_centering = window_centering

    @torch.no_grad()
    def get_operator(self, L, approx, s, J, lev, device='cpu'):
        filter_len = approx.shape[0]
        a = np.pi / 2
        FD1 = dglsp.identity((L.shape[0], L.shape[0]), device=device)
        d_list = []
        for l in range(1, lev + 1):
            T0F = FD1
            if lev == 1:
                T1F = s ** (-J + l - 1) / a * L - T0F
                d_list.extend((0.5 * approx[:, 0:0 + 1] * T0F
                               + approx[:, 1:1 + 1] * T1F).flatten())
            else:
                T1F = s ** (-J + l - 1) / a * L @ T0F - T0F
                d_list.extend((0.5 * approx[:, 0:0 + 1] * T0F
                               + approx[:, 1:1 + 1] * T1F).flatten())
                FD1 = d_list[0 + (l - 1) * filter_len]
        return d_list

    @staticmethod
    def _empty_normalized_laplacian(num_nodes: int, device: torch.device):
        empty_edges = torch.empty((2, 0), dtype=torch.long, device=device)
        empty_weights = torch.empty((0,), dtype=torch.float32, device=device)
        lap_index, lap_weight = get_laplacian(
            empty_edges, empty_weights, normalization='sym', num_nodes=num_nodes)
        return dglsp.spmatrix(lap_index, lap_weight, shape=(num_nodes, num_nodes))

    @torch.no_grad()
    def construct_laplacian(self, x, k, num_nodes):
        if x.ndim != 3:
            raise ValueError(
                f'UFGTime+ KNN expects [B*C, N, D], but received shape {tuple(x.shape)}.')
        num_graphs, nodes_per_graph, _ = x.shape
        expected_nodes = num_graphs * nodes_per_graph
        if num_nodes != expected_nodes:
            raise ValueError(f'num_nodes={num_nodes} does not match B*C*N={expected_nodes}.')
        if k < 1:
            raise ValueError(f'k must be positive, got {k}.')
        max_neighbors = nodes_per_graph - 1 if self.exclude_self else nodes_per_graph
        if max_neighbors == 0:
            return self._empty_normalized_laplacian(num_nodes, x.device)
        k_eff = min(k, max_neighbors)
        algorithm = self.knn_algorithm
        if algorithm is None:
            algorithm = 'bruteforce-sharemem' if x.is_cuda else 'bruteforce'
        graph = knn_graph(x, k_eff, algorithm=algorithm,
                          dist=self.knn_dist, exclude_self=self.exclude_self)
        src, dst = graph.edges()
        directed_edges = torch.stack((src, dst), dim=0)
        edge_index, edge_weight = to_undirected(
            directed_edges,
            torch.ones(directed_edges.shape[1], device=x.device, dtype=x.dtype),
            num_nodes=num_nodes, reduce='min')
        lap_index, lap_weight = get_laplacian(
            edge_index, edge_weight, normalization='sym', num_nodes=num_nodes)
        return dglsp.spmatrix(lap_index, lap_weight, shape=(num_nodes, num_nodes))

    def forward(self, x):
        window_mean = None
        if self.window_centering:
            window_mean = x.mean(dim=1, keepdim=True).detach()
            x = x - window_mean
        if x.ndim != 3:
            raise ValueError(f'Expected input [B, T, N], but received shape {tuple(x.shape)}.')
        _, trend = self.decompsition(x)
        trend_output = self.Linear_Trend(trend.permute(0, 2, 1).contiguous())
        x_freq = torch.fft.rfft(x, n=self.signal_len, dim=1, norm='ortho')
        batch_size, num_freqs, num_variables = x_freq.shape
        expected_freqs = self.signal_len // 2 + 1
        if num_freqs != expected_freqs:
            raise RuntimeError(f'rFFT returned C={num_freqs}; expected {expected_freqs}.')
        if num_variables != self.num_ts:
            raise RuntimeError(f'Input has N={num_variables} variables; model expects {self.num_ts}.')
        if num_freqs * num_variables != self.num_nodes:
            raise RuntimeError('Frequency-variable node count is inconsistent with model setup.')
        # Each [batch, frequency] slice is an independent N-variable graph.
        knn_features = torch.view_as_real(x_freq).detach().contiguous()
        knn_features = knn_features.reshape(batch_size * num_freqs, num_variables, 2)
        total_nodes = batch_size * num_freqs * num_variables
        laplacian = self.construct_laplacian(knn_features, self.k, num_nodes=total_nodes)
        operators = self.get_operator(laplacian, self.approx, self.s,
                                      self.J, self.lev, self.device)
        # Graph nodes and features share frequency-major [B, C, N] order.
        framelet_x = x_freq.contiguous().reshape(total_nodes, 1)
        batch_filters = [node_filter.repeat(batch_size, 1) for node_filter in self.filters]
        framelet_x = sum(
            batch_filters[i] * self.conv1_list[i](
                framelet_x, operators[i].indices(), operators[i].val)
            for i in range(0, self.lev + 1))
        framelet_x = framelet_x.reshape(batch_size, num_freqs, num_variables, -1)
        framelet_x = framelet_x.permute(0, 2, 3, 1).contiguous()
        framelet_x = framelet_x.reshape(batch_size, num_variables, -1)
        x_freq_out = self.clin(framelet_x)
        x_time = torch.fft.irfft(x_freq_out, n=self.seq_len, dim=-1, norm='ortho')
        x_time = self.isn(x_time)
        x_time = self.act_real(x_time)
        x_time = self.lin2(x_time)
        x_time = self.isn(x_time)
        x_time = self.act_real(x_time)
        graph_output = self.lin3(x_time)
        output = self.lin4(torch.cat((graph_output, trend_output), dim=-1))
        if window_mean is not None:
            output = output + window_mean.transpose(1, 2)
        return output
