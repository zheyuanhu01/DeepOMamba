import torch.nn as nn
import numpy as np
import torch
from time_model import GRU, LSTM, Mamba_NO, MambaConfig

@torch.jit.script
def compl_mul1d(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
    res = torch.einsum("bix,iox->box", a, b)
    return res

class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1):
        super(SpectralConv1d, self).__init__()

        """
        1D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes1 = modes1

        self.scale = (1 / (in_channels*out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.cfloat))

    def forward(self, x):
        batchsize = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfftn(x, dim=[2])

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.in_channels, x.size(-1)//2 + 1, device=x.device, dtype=torch.cfloat)
        out_ft[:, :, :self.modes1] = compl_mul1d(x_ft[:, :, :self.modes1], self.weights1)

        # Return to physical space
        x = torch.fft.irfftn(out_ft, s=[x.size(-1)], dim=[2])
        return x


class FNO_Jamba_Layer_1(nn.Module):
    def __init__(self, modes, width, model_t_type):
        super(FNO_Jamba_Layer_1, self).__init__()

        self.sp_conv = SpectralConv1d(width, width, modes)
        self.w = nn.Conv1d(width, width, 1)
        self.act = nn.GELU()
        
        if model_t_type == "GRU":
            self.model_t = GRU(
                input_dim=width, 
                output_dim=width, 
                hidden_dim=width, 
                num_layers=1)
        elif model_t_type == "LSTM":
            self.model_t = LSTM(
                input_dim=width, 
                output_dim=width, 
                hidden_dim=width, 
                num_layers=1)
        elif model_t_type == "Mamba":
            self.model_t = Mamba_NO(
                d_model=width,
                n_layer=1,
                d_intermediate=0,
                input_dim=width,
                output_dim=width,)

    def forward(self, x): # (batch, N_x, N_t, width)
        batch_size, N_x, N_t, width = x.size(0), x.size(1), x.size(2), x.size(3)

        ############## FNO1d ##############
        x = x.permute(0, 2, 3, 1) # (batch, N_t, width, N_x)
        x = x.reshape(batch_size * N_t, width, N_x) # (batch * N_t, width, N_x)

        x1 = self.sp_conv(x)
        x2 = self.w(x)
        x = x1 + x2
        x = self.act(x) # (batch * N_t, width, N_x)
        ############## FNO1d ##############

        ############## Temporal ##############
        x = x.reshape(batch_size, N_t, width, N_x) # (batch, N_t, width, N_x)
        x = x.permute(0, 3, 1, 2) # (batch, N_x, N_t, width)
        x = x.reshape(batch_size * N_x, N_t, -1) # (batch * N_x, N_t, width)
        x = self.model_t(x) # (batch * N_x, N_t, width)
        x = x.reshape(batch_size, N_x, N_t, -1) # (batch, N_x, N_t, width)
        ############## Temporal ##############
        
        return x

class FNO_Jamba_1(nn.Module):
    def __init__(self, input_dim, output_dim, modes, width, num_layers, model_t_type):
        super(FNO_Jamba_1, self).__init__()
        self.in_proj = nn.Linear(input_dim, width)
        self.blocks = nn.ModuleList([
            FNO_Jamba_Layer_1(modes, width, model_t_type) 
            for _ in range(num_layers)
            ])
        self.fc1 = nn.Linear(width, width)
        self.fc2 = nn.Linear(width, width)
        self.fc3 = nn.Linear(width, output_dim)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.in_proj(x)
        for layer in self.blocks:
            x = layer(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.act(x)
        x = self.fc3(x)

        x = x.reshape(x.size(0), -1, x.size(-1))
        
        return x

class FNO_Jamba_Layer_2(nn.Module):
    def __init__(self, modes, width, model_t_type):
        super(FNO_Jamba_Layer_2, self).__init__()

        self.sp_conv = SpectralConv1d(width, width, modes)
        self.w = nn.Conv1d(width, width, 1)
        self.act = nn.GELU()
        
        if model_t_type == "GRU":
            self.model_t = GRU(
                input_dim=width, 
                output_dim=width, 
                hidden_dim=width, 
                num_layers=1)
        elif model_t_type == "LSTM":
            self.model_t = LSTM(
                input_dim=width, 
                output_dim=width, 
                hidden_dim=width, 
                num_layers=1)
        elif model_t_type == "Mamba":
            self.model_t = Mamba_NO(
                d_model=width,
                n_layer=1,
                d_intermediate=0,
                input_dim=width,
                output_dim=width,)

    def forward(self, X): # (batch, N_x, N_t, width)
        batch_size, N_x, N_t, width = X.size(0), X.size(1), X.size(2), X.size(3)

        ############## FNO1d ##############
        x = X.permute(0, 2, 3, 1) # (batch, N_t, width, N_x)
        x = x.reshape(batch_size * N_t, width, N_x) # (batch * N_t, width, N_x)

        x1 = self.sp_conv(x)
        x2 = self.w(x)
        x = x1 + x2
        x = self.act(x) # (batch * N_t, width, N_x)

        x = x.reshape(batch_size, N_t, width, N_x) # (batch, N_t, width, N_x)
        x = x.permute(0, 3, 1, 2) # (batch, N_x, N_t, width)
        ############## FNO1d ##############

        ############## Temporal ##############
        y = X.reshape(batch_size * N_x, N_t, -1) # (batch * N_x, N_t, width)
        y = self.model_t(y) # (batch * N_x, N_t, width)
        y = y.reshape(batch_size, N_x, N_t, -1) # (batch, N_x, N_t, width)
        ############## Temporal ##############

        return x + y

class FNO_Jamba_2(nn.Module):
    def __init__(self, input_dim, output_dim, modes, width, num_layers, model_t_type):
        super(FNO_Jamba_2, self).__init__()
        self.in_proj = nn.Linear(input_dim, width)
        self.blocks = nn.ModuleList([
            FNO_Jamba_Layer_2(modes, width, model_t_type) 
            for _ in range(num_layers)
            ])
        self.fc1 = nn.Linear(width, width)
        self.fc2 = nn.Linear(width, width)
        self.fc3 = nn.Linear(width, output_dim)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.in_proj(x)
        for layer in self.blocks:
            x = layer(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.act(x)
        x = self.fc3(x)

        x = x.reshape(x.size(0), -1, x.size(-1))
        
        return x
    
if __name__ == "__main__":
    # x = torch.randn(2, 32, 64, 2)
    # model = FNO2d_Jamba(modes=4, width=16, num_layers_FNO=2, in_dim=2, out_dim=1, model_t_type="GRU", num_layers_t=1)
    # y = model(x)
    # print(y.shape)

    # x = torch.randn(2, 32, 64, 8)
    # model = FNO_Jamba_Layer(modes=4, width=8, model_t_type="GRU")
    # y = model(x)
    # print(y.shape)

    x = torch.randn(2, 32, 64, 2)
    model = FNO_Jamba_2(input_dim=2, output_dim=1, modes=8, width=32, num_layers=2, model_t_type="GRU")
    y = model(x)
    print(y.shape)
    pass