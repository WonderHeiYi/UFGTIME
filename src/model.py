import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import dgl.sparse as dglsp

from dgl import knn_graph, to_bidirected, add_self_loop
from dgl.nn.pytorch.conv.graphconv import EdgeWeightNorm 
from torch_geometric.utils import get_laplacian, from_dgl, to_undirected
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn import MessagePassing

from src.utils import CSiLU

warnings.filterwarnings("ignore")

class ParameterClamp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return input.clamp(min=1e-6, max=1e+6) 

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.clone()

def create_filter(num_nodes):
    # param_clamp = ParameterClamp()
    filter = nn.Parameter(torch.Tensor(num_nodes, 1))
    nn.init.normal_(filter, mean=1, std=0.1)
    # param_clamp.apply(filter)
    return filter


class UFGConv(MessagePassing):
    def __init__(
        self,
        in_channels,
        out_channels,
        channel_mix=True,
        bias=False,
        **kwargs,
    ):
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
    

class FreqTimeUFGV2(nn.Module):
    def __init__(
        self,
        seq_length,
        signal_length,
        pred_length,
        hidden_size,
        embed_size,
        num_ts,
        device, 
        approx,
        s,
        lev,
        num_topk=2
    ):
        super().__init__()
        self.device=device
        self.embed_size = embed_size
        self.seq_len = seq_length
        self.pred_length = pred_length
        self.k = num_topk
        self.hidden_size = hidden_size
        self.approx = approx
        self.s = s
        self.lev = lev
        self.J = np.log(2 / np.pi) / np.log(s) + lev - 1
        self.num_ts = num_ts
        self.signal_len = signal_length
        self.embeddings = nn.Parameter(torch.randn(1, self.embed_size))
        kernel_size = 25
        self.decompsition = series_decomp(kernel_size)

        self.num_nodes = (self.signal_len // 2 + 1) * self.num_ts 
        self.filters = nn.ParameterList(
                    [create_filter(self.num_nodes) for i in range(2 * lev)]
        )  # note 2 * lev is only for Haar, change 3 * lev if chose Linear

        self.conv1_list =  nn.ModuleList(
                    [UFGConv(in_channels=1, out_channels=hidden_size)
                    for i in range(0, self.lev + 1)]
        )
        self.conv2_list =  nn.ModuleList(
                    [UFGConv(in_channels=hidden_size, out_channels=hidden_size)
                    for i in range(0, self.lev + 1)]
        )

        self.clin = nn.Linear(in_features=hidden_size * (signal_length//2+1),
                              out_features=signal_length//2+1).to(torch.cfloat)   
        self.lin2 = nn.Linear(in_features=seq_length, 
                              out_features=hidden_size)
        self.lin3 = nn.Linear(in_features=hidden_size, 
                              out_features=pred_length)
        self.lin4 = nn.Linear(in_features=pred_length*2, 
                              out_features=pred_length)
        self.Linear_Trend = nn.Linear(self.seq_len, self.pred_length)
        self.isn = nn.InstanceNorm2d(self.num_ts)
        
        # self.embeddings = nn.Linear(1, embed_size).to(torch.cfloat)
        self.act_imag = CSiLU()
        self.act_real = nn.SiLU()

    @torch.no_grad()
    def construct_laplacian(self, x, k, num_nodes):
        """Construct sparse Laplacian matrix
        in torch.sparse_coo_tensor format
        Param:
            x: features (use real part if x in complex) [batch, n, d]
            k: The number of neighbors
        Return:
            Laplacian(dgl.sparse.SparseMatrix) 
        """
        # norm = EdgeWeightNorm(norm='both')
        # graph =  knn_graph(x, k, algorithm='bruteforce-sharemem').reverse().add_self_loop()
        # edge_weight = norm(
        #     graph, torch.ones(graph.num_edges(), device=self.device)
        # )
        # return dglsp.spmatrix(graph.adj().indices(), edge_weight)
        L = from_dgl(knn_graph(x, k, algorithm='bruteforce-sharemem'))
        L = to_undirected(
                L.edge_index.flip(0),
                torch.ones(len(L.edge_index[0]),
                        device=self.device),
                num_nodes=num_nodes,
                reduce='min'
        )
        L = get_laplacian(L[0], L[1], normalization='sym', 
                          num_nodes=num_nodes)
        return dglsp.spmatrix(L[0], L[1])

    @torch.no_grad()
    def get_operator(self, L, approx, s, J, lev, device='cpu'):
        """Get operators of fast tight frame decomposition (FTFD)
        adapted from https://github.com/YuGuangWang/UFG/blob/main
        Param:
            L[dglsp.SparseMatrix]: laplacian matrix
            approx[np.array]: Chebshev approxmation
            s: dilation scale
            J: dilation level to start decomp
            Lev: level of transformation
        Return:
            d[list]: list of matrices[torch.sparse_coo], row-by-row 
        """
        filter_len = approx.shape[0]
        a = np.pi / 2  # consider the domain of masks as [0, pi]
        FD1 = dglsp.identity((L.shape[0], L.shape[0]), device=device)
        d_list = []
        for l in range(1, lev + 1):
            T0F = FD1
            if lev == 1:
                T1F = ((s ** (-J + l - 1) / a) * L) - T0F
                d_list.extend(
                    (0.5 * approx[:,0:0+1] * T0F\
                     + approx[:,1:1+1] * T1F).flatten()
                )
            else:
                T1F = ((s ** (-J + l - 1) / a) * L) @ T0F - T0F
                d_list.extend(
                    (0.5 * approx[:,0:0+1] * T0F\
                     + approx[:,1:1+1] * T1F).flatten()
                )
                FD1 = d_list[0 + (l - 1) * filter_len]
        return d_list
    
    def forward(self, x):
        seasonal_init, trend_init = self.decompsition(x)
        trend_init = trend_init.permute(0,2,1)
        seasonal_init = seasonal_init.permute(0,2,1)
        trend_output = self.Linear_Trend(trend_init)
        x = torch.fft.rfft(x, n=self.signal_len, dim=1, norm='ortho')
        B, C, N = x.shape  # [B:batchsize, C:signal len, N:num of ts]

        x = x.permute(0, 2, 1).contiguous()
        x = x.reshape(B, N*C, 1)  # [B,N,C] => [B,NC,1]
        # x = self.embedding(x)

        batch_total_nodes = B * self.num_nodes

        # construct batch laplacians by x real part
        d_list = self.construct_laplacian(
                    torch.cat((x.real, x.imag), dim=-1),
                    self.k, num_nodes=batch_total_nodes
                ) 
        d_list = self.get_operator(d_list, self.approx, self.s, self.J,
                                   self.lev, self.device) 

        # x = x.reshape(B*N*C, self.embed_size)  
        x = x.reshape(B*N*C, 1) 

        # copy filters with num of batchsize
        batch_filters = []
        for filter in self.filters:
            batch_filters.append(filter.repeat(B, 1))

        # graph framelet convolutions
        x = sum(
                batch_filters[i] * self.conv1_list[i](
                    x, d_list[i].indices(), d_list[i].val
                ) for i in range(0, self.lev + 1)
            )
        # x = self.act_imag(x)

        # x = sum(
        #         batch_filters[i] * self.conv2_list[i](
        #             x, d_list[i].indices(), d_list[i].val
        #         ) for i in range(0, self.lev + 1)
        #     )

        x = x.reshape(B, N, C, -1)  # [BNC,#emb] => [B,N,C,#emb] 
        x = x.permute(0, 1, 3, 2).reshape(B, N, -1)  # [B,N,#embC]
        x = self.clin(x)
        # x = self.act_imag(x)
        x = torch.fft.irfft(x, n=self.seq_len, dim=-1, norm='ortho')

        # linear feedforwards
        x = self.isn(x)
        x = self.act_real(x)
        x = self.lin2(x)
        x = self.isn(x)
        x = self.act_real(x)
        x = self.lin3(x)
        x = torch.cat((x, trend_output), dim=-1)
        x = self.lin4(x)
        # x = self.act_real(x)

        return x
    

# class FreqTimeUFGV1(nn.Module):
#     def __init__(
#         self,
#         signal_length,
#         pred_length,
#         hidden_size,
#         embed_size,
#         num_nodes,
#         device, 
#         approx,
#         s,
#         lev,
#         num_topk=2
#     ):
#         super().__init__()
#         self.device = device
#         self.pred_length = pred_length
#         self.k = num_topk
#         self.hidden_size = hidden_size
#         self.approx = approx
#         self.s = s
#         self.lev = lev
#         self.J = np.log(2 / np.pi) / np.log(s) + lev - 1
#         self.embed_size = embed_size

#         self.filters = nn.ParameterList(
#                         [create_filter(num_nodes) for i in range(2 * lev - 1)])
#         self.conv1 = UFGConv(in_channels=embed_size, 
#                              out_channels=hidden_size)
#         self.conv2 = UFGConv(in_channels=hidden_size,
#                              out_channels=hidden_size)
        
#         self.lin1 = nn.Linear(in_features=hidden_size * (signal_length//2+1),
#                               out_features=hidden_size).to(torch.cfloat)   
#         self.lin2 = nn.Linear(in_features=hidden_size, 
#                               out_features=pred_length).to(torch.cfloat)
        
#         self.embeddings = nn.Linear(1, embed_size).to(torch.cfloat)
#         self.act = CSiLU()
    
#     def construct_laplacian(self, x, k, norm='sym'):
#         """Construct sparse Laplacian matrix
#         in torch.sparse_coo_tensor format
#         Param:
#             x: features (use real part if x in complex) [batch, n, d]
#             k: The number of neighbors
#             norm: normalization scheme, 'None', 'sym', 'rw'
#         Return:
#             Laplacian(dgl.sparse.SparseMatrix) 
#         """
#         L = knn_graph(x, k, algorithm='bruteforce-sharemem')
#         L = get_laplacian(
#                 to_undirected(L.edge_index.flip(0)), normalization=norm)
#         L = dglsp.spmatrix(L[0], L[1])
#         return L
    
#     def get_operator(self, L, approx, s, J, lev, device='cpu'):
#         """Get operators of fast tight frame decomposition (FTFD)
#         adapted from https://github.com/YuGuangWang/UFG/blob/main
#         Param:
#             L[dglsp.SparseMatrix]: laplacian matrix
#             approx[np.array]: Chebshev approxmation
#             s: dilation scale
#             J: dilation level to start decomp
#             Lev: level of transformation
#         Return:
#             d[list]: list of matrices[torch.sparse_coo], row-by-row 
#         """
#         filter_len = approx.shape[0]
#         a = np.pi / 2  # consider the domain of masks as [0, pi]
#         FD1 = dglsp.identity((L.shape[0], L.shape[0]), device=device)
#         d_list = []
#         for l in range(1, lev + 1):
#             T0F = FD1
#             if lev == 1:
#                 T1F = ((s ** (-J + l - 1) / a) * L) - T0F
#                 d_list.extend(
#                     (0.5 * approx[:,0:0+1] * T0F\
#                      + approx[:,1:1+1] * T1F).flatten()
#                 )
#             else:
#                 T1F = ((s ** (-J + l - 1) / a) * L) @ T0F - T0F
#                 d_list.extend(
#                     (0.5 * approx[:,0:0+1] * T0F\
#                      + approx[:,1:1+1] * T1F).flatten()
#                 )
#                 FD1 = d_list[0 + (l - 1) * filter_len]
#         return d_list
    
#     def forward(self, x):
#         B, C, N = x.shape 

#         x = x.permute(0, 2, 1).contiguous() 
#         x = x.reshape(B, N*C, 1) 

#         x = self.embeddings(x)  # [B,NC,1] => [B,NC,#emb]

#         # construct batch laplacians by x real part
#         d_list = self.construct_laplacian(x.real, self.k)  # d_list is L 
#         d_list = self.get_operator(d_list, self.approx, self.s, self.J,
#                                    self.lev, self.device) 

#         x = x.reshape(B*N*C, self.embed_size)  

#         # graph framelet convolutions
#         x = self.conv1(x, d_list[0].indices(), d_list[0].val)\
#             + sum(
#                 self.filters[i-1] * self.conv1(
#                     x, d_list[i].indices(), d_list[i].val
#                 ) for i in range(1, self.lev + 1)
#             )
#         x = self.act(x)

#         x = self.conv2(x, d_list[0].indices(), d_list[0].val)\
#             + sum(
#                 self.filters[i-1] * self.conv2(
#                     x, d_list[i].indices(), d_list[i].val
#                 ) for i in range(1, self.lev + 1)
#             )
#         x = self.act(x)

#         x = x.reshape(B, N, C, -1)  # [BNC,#emb] => [B,N,C,#emb] 
#         x = x.permute(0, 1, 3, 2).reshape(B, N, -1)  # [B,N,#embC]

#         # linear feedforwards
#         x = self.lin1(x)
#         x = self.act(x)
#         x = self.lin2(x)
        
#         # ifft transfer complex predict to real
#         x = torch.fft.irfft(x, n=self.pred_length, dim=-1, norm='ortho')
#         return x


class FGN(nn.Module):
    def __init__(self, pre_length, embed_size,
                 feature_size, seq_length, hidden_size, hard_thresholding_fraction=1, hidden_size_factor=1, sparsity_threshold=0.01):
        super().__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.number_frequency = 1
        self.pre_length = pre_length
        self.feature_size = feature_size
        self.seq_length = seq_length
        self.frequency_size = self.embed_size // self.number_frequency
        self.hidden_size_factor = hidden_size_factor
        self.sparsity_threshold = sparsity_threshold
        self.hard_thresholding_fraction = hard_thresholding_fraction
        self.scale = 0.02
        self.embeddings = nn.Parameter(torch.randn(1, self.embed_size))

        self.w1 = nn.Parameter(
            self.scale * torch.randn(2, self.frequency_size, self.frequency_size * self.hidden_size_factor))
        self.b1 = nn.Parameter(self.scale * torch.randn(2, self.frequency_size * self.hidden_size_factor))
        self.w2 = nn.Parameter(
            self.scale * torch.randn(2, self.frequency_size * self.hidden_size_factor, self.frequency_size))
        self.b2 = nn.Parameter(self.scale * torch.randn(2, self.frequency_size))
        self.w3 = nn.Parameter(
            self.scale * torch.randn(2, self.frequency_size,
                                     self.frequency_size * self.hidden_size_factor))
        self.b3 = nn.Parameter(
            self.scale * torch.randn(2, self.frequency_size * self.hidden_size_factor))
        self.embeddings_10 = nn.Parameter(torch.randn(self.seq_length, 8))
        self.fc = nn.Sequential(
            nn.Linear(self.embed_size * 8, 64),
            nn.LeakyReLU(),
            nn.Linear(64, self.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_size, self.pre_length)
        )
        self.to('cuda:0')

    def tokenEmb(self, x):
        x = x.unsqueeze(2)
        y = self.embeddings
        return x * y

    # FourierGNN
    def fourierGC(self, x, B, N, L):
        o1_real = torch.zeros([B, (N*L)//2 + 1, self.frequency_size * self.hidden_size_factor],
                              device=x.device)
        o1_imag = torch.zeros([B, (N*L)//2 + 1, self.frequency_size * self.hidden_size_factor],
                              device=x.device)
        o2_real = torch.zeros(x.shape, device=x.device)
        o2_imag = torch.zeros(x.shape, device=x.device)

        o3_real = torch.zeros(x.shape, device=x.device)
        o3_imag = torch.zeros(x.shape, device=x.device)

        o1_real = F.relu(
            torch.einsum('bli,ii->bli', x.real, self.w1[0]) - \
            torch.einsum('bli,ii->bli', x.imag, self.w1[1]) + \
            self.b1[0]
        )

        o1_imag = F.relu(
            torch.einsum('bli,ii->bli', x.imag, self.w1[0]) + \
            torch.einsum('bli,ii->bli', x.real, self.w1[1]) + \
            self.b1[1]
        )

        # 1 layer
        y = torch.stack([o1_real, o1_imag], dim=-1)
        y = F.softshrink(y, lambd=self.sparsity_threshold)

        o2_real = F.relu(
            torch.einsum('bli,ii->bli', o1_real, self.w2[0]) - \
            torch.einsum('bli,ii->bli', o1_imag, self.w2[1]) + \
            self.b2[0]
        )

        o2_imag = F.relu(
            torch.einsum('bli,ii->bli', o1_imag, self.w2[0]) + \
            torch.einsum('bli,ii->bli', o1_real, self.w2[1]) + \
            self.b2[1]
        )

        # 2 layer
        x = torch.stack([o2_real, o2_imag], dim=-1)
        x = F.softshrink(x, lambd=self.sparsity_threshold)
        x = x + y

        o3_real = F.relu(
                torch.einsum('bli,ii->bli', o2_real, self.w3[0]) - \
                torch.einsum('bli,ii->bli', o2_imag, self.w3[1]) + \
                self.b3[0]
        )

        o3_imag = F.relu(
                torch.einsum('bli,ii->bli', o2_imag, self.w3[0]) + \
                torch.einsum('bli,ii->bli', o2_real, self.w3[1]) + \
                self.b3[1]
        )

        # 3 layer
        z = torch.stack([o3_real, o3_imag], dim=-1)
        z = F.softshrink(z, lambd=self.sparsity_threshold)
        z = z + x
        z = torch.view_as_complex(z)
        return z

    def forward(self, x):
        x = x.permute(0, 2, 1).contiguous()
        B, N, L = x.shape
        # B*N*L ==> B*NL
        x = x.reshape(B, -1)
        # embedding B*NL ==> B*NL*D
        x = self.tokenEmb(x)

        # FFT B*NL*D ==> B*NT/2*D
        x = torch.fft.rfft(x, dim=1, norm='ortho')

        x = x.reshape(B, (N*L)//2+1, self.frequency_size)

        bias = x

        # FourierGNN
        x = self.fourierGC(x, B, N, L)

        x = x + bias

        x = x.reshape(B, (N*L)//2+1, self.embed_size)

        # ifft
        x = torch.fft.irfft(x, n=N*L, dim=1, norm="ortho")

        x = x.reshape(B, N, L, self.embed_size)
        x = x.permute(0, 1, 3, 2)  # B, N, D, L

        # projection
        x = torch.matmul(x, self.embeddings_10)
        x = x.reshape(B, N, -1)
        x = self.fc(x)

        return x


class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
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
        return res, moving_mean

class Model(nn.Module):
    """
    Decomposition-Linear
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = 12
        self.pred_len = 12

        # Decompsition Kernel Size
        kernel_size = 25
        self.decompsition = series_decomp(kernel_size)
        self.individual = False
        self.channels = 1

        if self.individual:
            self.Linear_Seasonal = nn.ModuleList()
            self.Linear_Trend = nn.ModuleList()
            
            for i in range(self.channels):
                self.Linear_Seasonal.append(nn.Linear(self.seq_len,self.pred_len))
                self.Linear_Trend.append(nn.Linear(self.seq_len,self.pred_len))

                # Use this two lines if you want to visualize the weights
                # self.Linear_Seasonal[i].weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
                # self.Linear_Trend[i].weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
        else:
            self.Linear_Seasonal = nn.Linear(self.seq_len,self.pred_len)
            self.Linear_Trend = nn.Linear(self.seq_len,self.pred_len)
            
            # Use this two lines if you want to visualize the weights
            # self.Linear_Seasonal.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
            # self.Linear_Trend.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))

    def forward(self, x):
        # x: [Batch, Input length, Channel]
        seasonal_init, trend_init = self.decompsition(x)
        seasonal_init, trend_init = seasonal_init.permute(0,2,1), trend_init.permute(0,2,1)
        if self.individual:
            seasonal_output = torch.zeros([seasonal_init.size(0),seasonal_init.size(1),self.pred_len],dtype=seasonal_init.dtype).to(seasonal_init.device)
            trend_output = torch.zeros([trend_init.size(0),trend_init.size(1),self.pred_len],dtype=trend_init.dtype).to(trend_init.device)
            for i in range(self.channels):
                seasonal_output[:,i,:] = self.Linear_Seasonal[i](seasonal_init[:,i,:])
                trend_output[:,i,:] = self.Linear_Trend[i](trend_init[:,i,:])
        else:
            seasonal_output = self.Linear_Seasonal(seasonal_init)
            trend_output = self.Linear_Trend(trend_init)

        x = seasonal_output + trend_output
        # return x.permute(0,2,1) # to [Batch, Output length, Channel]
        return x