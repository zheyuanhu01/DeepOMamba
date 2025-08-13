import copy
import logging
import math

import torch
import torch.nn as nn
from einops import rearrange


logger = logging.getLogger(__name__)


class SpectralConv2d(nn.Module):
    def __init__(self, in_dim, out_dim, n_modes):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_modes = n_modes

        self.fourier_weight = nn.ParameterList([])
        for _ in range(2):
            weight = torch.FloatTensor(in_dim, out_dim, n_modes, 2)
            param = nn.Parameter(weight)
            nn.init.xavier_normal_(param)
            self.fourier_weight.append(param)

    def forward(self, x):
        # x.shape == [batch_size, grid_size, grid_size, in_dim]
        x = self.forward_fourier(x)
        return x

    def forward_fourier(self, x):
        x = rearrange(x, 'b m n i -> b i m n')
        # x.shape == [batch_size, in_dim, grid_size, grid_size]

        B, I, M, N = x.shape

        # # # Dimesion Y # # #
        x_fty = torch.fft.rfft(x, dim=-1, norm='ortho')
        # x_ft.shape == [batch_size, in_dim, grid_size, grid_size // 2 + 1]

        out_ft = x_fty.new_zeros(B, I, M, N // 2 + 1)
        # out_ft.shape == [batch_size, in_dim, grid_size, grid_size // 2 + 1, 2]

        out_ft[:, :, :, :self.n_modes] = torch.einsum(
            "bixy,ioy->boxy",
            x_fty[:, :, :, :self.n_modes],
            torch.view_as_complex(self.fourier_weight[0]))

        xy = torch.fft.irfft(out_ft, n=N, dim=-1, norm='ortho')
        # x.shape == [batch_size, in_dim, grid_size, grid_size]

        # # # Dimesion X # # #
        x_ftx = torch.fft.rfft(x, dim=-2, norm='ortho')
        # x_ft.shape == [batch_size, in_dim, grid_size // 2 + 1, grid_size]

        out_ft = x_ftx.new_zeros(B, I, M // 2 + 1, N)
        # out_ft.shape == [batch_size, in_dim, grid_size // 2 + 1, grid_size, 2]

        out_ft[:, :, :self.n_modes, :] = torch.einsum(
            "bixy,iox->boxy",
            x_ftx[:, :, :self.n_modes, :],
            torch.view_as_complex(self.fourier_weight[1]))

        xx = torch.fft.irfft(out_ft, n=M, dim=-2, norm='ortho')
        # x.shape == [batch_size, in_dim, grid_size, grid_size]

        # # Combining Dimensions # #
        x = xx + xy

        x = rearrange(x, 'b i m n -> b m n i')
        # x.shape == [batch_size, grid_size, grid_size, out_dim]

        return x


class FFNO(nn.Module):
    def __init__(self, modes, width, input_dim, output_dim, n_layers=4):
        super().__init__()
        self.modes = modes
        self.width = width
        self.input_dim = input_dim
        self.in_proj = nn.Linear(input_dim, self.width)
        self.n_layers = n_layers

        self.sp_convs = nn.ModuleList([SpectralConv2d(in_dim=width, out_dim=width, n_modes=modes) for _ in range(n_layers)])
        self.ws = nn.ModuleList([nn.Conv1d(width, width, 1) for _ in range(n_layers)])

        self.fc1 = nn.Linear(width, width)
        self.fc2 = nn.Linear(width, width)
        self.fc3 = nn.Linear(width, output_dim)
        self.act = nn.GELU()

    def forward(self, x):
        size_x, size_y = x.shape[1], x.shape[2]

        x = self.in_proj(x)

        length = len(self.ws)
        batchsize = x.shape[0]

        for i, (speconv, w) in enumerate(zip(self.sp_convs, self.ws)):
            x1 = speconv(x)
            x2 = w(x.permute(0, 3, 1, 2).reshape(batchsize, self.width, -1))
            x2 = x2.view(batchsize, self.width, size_x, size_y).permute(0, 2, 3, 1)
            x = x1 + x2
            if i != length - 1:
                x = self.act(x)

        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.act(x)
        x = self.fc3(x)

        x = x.reshape(x.size(0), -1, x.size(-1))
        
        return x


if __name__ == "__main__":
    pass