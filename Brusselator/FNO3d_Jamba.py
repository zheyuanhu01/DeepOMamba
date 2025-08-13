import torch.nn as nn
import numpy as np
import torch
from time_model import GRU, LSTM, Mamba_NO, MambaConfig


@torch.jit.script
def compl_mul2d(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # (batch, in_channel, x,y,t ), (in_channel, out_channel, x,y,t) -> (batch, out_channel, x,y,t)
    res =  torch.einsum("bixy,ioxy->boxy", a, b)
    return res


class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes1 = modes1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    def forward(self, x):
        batchsize = x.shape[0]
        size1 = x.shape[-2]
        size2 = x.shape[-1]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfftn(x, dim=[2, 3])

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-2), x.size(-1) // 2 + 1, device=x.device,
                                dtype=torch.cfloat)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        # Return to physical space
        x = torch.fft.irfftn(out_ft, s=(x.size(-2), x.size(-1)), dim=[2, 3])
        return x
  

class FNO_Jamba_Layer_1(nn.Module):
    def __init__(self, modes, width, model_t_type):
        super(FNO_Jamba_Layer_1, self).__init__()

        self.sp_conv = SpectralConv2d(width, width, modes, modes)
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

    def forward(self, x): # (batch, N_x, N_x, N_t, width)
        batch_size, N_x, N_t, width = x.size(0), x.size(1), x.size(-2), x.size(-1)

        x = x.permute(0, 3, 4, 1, 2) # (batch, N_t, width, N_x, N_x)
        x = x.reshape(batch_size * N_t, width, N_x, N_x) # (batch * N_t, width, N_x, N_x)

        x1 = self.sp_conv(x)
        x2 = self.w(x.view(batch_size * N_t, width, -1)).view(batch_size * N_t, width, N_x, N_x)
        x = x1 + x2
        x = self.act(x) # (batch * N_t, width, N_x, N_x)
        x = x.reshape(batch_size, N_t, width, N_x, N_x) # (batch, N_t, width, N_x, N_x)
        x = x.permute(0, 3, 4, 1, 2) # (batch, N_x, N_x, N_t, width)
        x = x.reshape(batch_size * N_x * N_x, N_t, -1) # (batch * N_x**2, N_t, width)
        x = self.model_t(x) # (batch * N_x**2, N_t, width)
        x = x.reshape(batch_size, N_x, N_x, N_t, -1) # (batch, N_x, N_t, width)
        
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

        self.sp_conv = SpectralConv2d(width, width, modes, modes)
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

    def forward(self, X): # (batch, N_x, N_x, N_t, width)
        batch_size, N_x, N_t, width = X.size(0), X.size(1), X.size(-2), X.size(-1)

        x = X.permute(0, 3, 4, 1, 2) # (batch, N_t, width, N_x, N_x)
        x = x.reshape(batch_size * N_t, width, N_x, N_x) # (batch * N_t, width, N_x, N_x)

        x1 = self.sp_conv(x)
        x2 = self.w(x.view(batch_size * N_t, width, -1)).view(batch_size * N_t, width, N_x, N_x)
        x = x1 + x2
        x = self.act(x) # (batch * N_t, width, N_x, N_x)
        x = x.reshape(batch_size, N_t, width, N_x, N_x) # (batch, N_t, width, N_x, N_x)
        x = x.permute(0, 3, 4, 1, 2) # (batch, N_x, N_x, N_t, width)

        y = X.reshape(batch_size * N_x * N_x, N_t, -1) # (batch * N_x**2, N_t, width)
        y = self.model_t(y) # (batch * N_x**2, N_t, width)
        y = y.reshape(batch_size, N_x, N_x, N_t, -1) # (batch, N_x, N_t, width)
        
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
    pass