import torch.nn as nn
import numpy as np
import torch
from time_model import GRU, LSTM, Mamba_NO, MambaConfig


class PR(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, grid_x):
        super(PR, self).__init__()
        
        self.grid_x = grid_x.reshape(-1)
        self.modes1 = modes1
        self.scale = (1 / (in_channels*out_channels))
        self.weights_pole = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.cfloat))
        self.weights_residue = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.cfloat))
       
    def output_PR(self, lambda1,alpha, weights_pole, weights_residue):   
        Hw=torch.zeros(weights_residue.shape[0],weights_residue.shape[0],weights_residue.shape[2],lambda1.shape[0], device=alpha.device, dtype=torch.cfloat)
        term1=torch.div(1,torch.sub(lambda1,weights_pole))
        Hw=weights_residue*term1
        output_residue1=torch.einsum("bix,xiok->box", alpha, Hw) 
        output_residue2=torch.einsum("bix,xiok->bok", alpha, -Hw) 
        return output_residue1,output_residue2    

    def forward(self, x):
        t=self.grid_x.cuda()
        #Compute input poles and resudes by FFT
        dt=(t[1]-t[0]).item()
        alpha = torch.fft.fft(x)
        lambda0=torch.fft.fftfreq(t.shape[0], dt)*2*np.pi*1j
        lambda1=lambda0.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        lambda1=lambda1.cuda()
    
        # Obtain output poles and residues for transient part and steady-state part
        output_residue1,output_residue2= self.output_PR(lambda1, alpha, self.weights_pole, self.weights_residue)
    
        # Obtain time histories of transient response and steady-state response
        x1 = torch.fft.ifft(output_residue1, n=x.size(-1))
        x1 = torch.real(x1)
        x2=torch.zeros(output_residue2.shape[0],output_residue2.shape[1],t.shape[0], device=alpha.device, dtype=torch.cfloat)    
        term1=torch.einsum("bix,kz->bixz", self.weights_pole, t.type(torch.complex64).reshape(1,-1))
        term2=torch.exp(term1) 
        x2=torch.einsum("bix,ioxz->boz", output_residue2,term2)
        x2=torch.real(x2)
        x2=x2/x.size(-1)
        return x1+x2
 

class LNO_Jamba_Layer_1(nn.Module):
    def __init__(self, modes, width, grid, model_t_type):
        super(LNO_Jamba_Layer_1, self).__init__()

        self.sp_conv = PR(width, width, modes, grid)
        self.w = nn.Conv1d(width, width, 1)

        self.norm = nn.InstanceNorm1d(width)
        
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
                d_intermediate=width,
                input_dim=width,
                output_dim=width,)

    def forward(self, x): # (batch, N_x, N_t, width)

        batch_size, N_x, N_t, width = x.size(0), x.size(1), x.size(2), x.size(3)

        x = x.permute(0, 2, 3, 1) # (batch, N_t, width, N_x)
        x = x.reshape(batch_size * N_t, width, N_x) # (batch * N_t, width, N_x)

        x1 = self.norm(self.sp_conv(self.norm(x)))
        x2 = self.w(x)
        x = x1 + x2
        x = torch.sin(x) # (batch * N_t, width, N_x)
        x = x.reshape(batch_size, N_t, width, N_x) # (batch, N_t, width, N_x)
        x = x.permute(0, 3, 1, 2) # (batch, N_x, N_t, width)
        x = x.reshape(batch_size * N_x, N_t, -1) # (batch * N_x, N_t, width)
        x = self.model_t(x) # (batch * N_x, N_t, width)
        x = x.reshape(batch_size, N_x, N_t, -1) # (batch, N_x, N_t, width)
        
        return x


class LNO_Jamba_1(nn.Module):
    def __init__(self, input_dim, output_dim, modes, width, num_layers, grid, time_model):
        super(LNO_Jamba_1, self).__init__()
        self.in_proj = nn.Linear(input_dim, width)
        self.width = width

        self.blocks = nn.ModuleList([
            LNO_Jamba_Layer_1(modes, width, grid, time_model) 
            for _ in range(num_layers)
            ])
        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = self.in_proj(x)
        for layer in self.blocks:
            x = layer(x)

        x = self.fc1(x)
        x = torch.sin(x)
        x = self.fc2(x)

        x = x.reshape(x.size(0), -1, x.size(-1))
        return x


class LNO_Jamba_Layer_2(nn.Module):
    def __init__(self, modes, width, grid, model_t_type):
        super(LNO_Jamba_Layer_2, self).__init__()

        self.sp_conv = PR(width, width, modes, grid)
        self.w = nn.Conv1d(width, width, 1)

        self.norm = nn.InstanceNorm1d(width)
        
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
                d_intermediate=width,
                input_dim=width,
                output_dim=width,)

    def forward(self, X): # (batch, N_x, N_t, width)

        batch_size, N_x, N_t, width = X.size(0), X.size(1), X.size(2), X.size(3)

        ############## LNO1d ##############
        x = X.permute(0, 2, 3, 1) # (batch, N_t, width, N_x)
        x = x.reshape(batch_size * N_t, width, N_x) # (batch * N_t, width, N_x)

        x1 = self.norm(self.sp_conv(self.norm(x)))
        x2 = self.w(x)
        x = x1 + x2
        x = torch.sin(x) # (batch * N_t, width, N_x)
        x = x.reshape(batch_size, N_t, width, N_x) # (batch, N_t, width, N_x)
        x = x.permute(0, 3, 1, 2) # (batch, N_x, N_t, width)
        ############## LNO1d ##############

        ############## Temporal ##############
        y = X.reshape(batch_size * N_x, N_t, -1) # (batch * N_x, N_t, width)
        y = self.model_t(y) # (batch * N_x, N_t, width)
        y = y.reshape(batch_size, N_x, N_t, -1) # (batch, N_x, N_t, width)
        ############## Temporal ##############
        return x + y

class LNO_Jamba_2(nn.Module):
    def __init__(self, input_dim, output_dim, modes, width, num_layers, grid, time_model):
        super(LNO_Jamba_2, self).__init__()
        self.in_proj = nn.Linear(input_dim, width)
        self.width = width

        self.blocks = nn.ModuleList([
            LNO_Jamba_Layer_2(modes, width, grid, time_model) 
            for _ in range(num_layers)
            ])
        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = self.in_proj(x)
        for layer in self.blocks:
            x = layer(x)

        x = self.fc1(x)
        x = torch.sin(x)
        x = self.fc2(x)

        x = x.reshape(x.size(0), -1, x.size(-1))
        return x

