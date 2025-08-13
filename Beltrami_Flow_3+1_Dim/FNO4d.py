import torch.nn as nn
import numpy as np
import torch

@torch.jit.script
def compl_mul4d(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    res = torch.einsum("bixyzw,ioxyzw->boxyzw", a, b)
    return res

class SpectralConv4d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, modes3, modes4):
        super(SpectralConv4d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2
        self.modes3 = modes3
        self.modes4 = modes4

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, self.modes4, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, self.modes4, dtype=torch.cfloat))
        self.weights3 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, self.modes4, dtype=torch.cfloat))
        self.weights4 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, self.modes4, dtype=torch.cfloat))
        self.weights5 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, self.modes4, dtype=torch.cfloat))
        self.weights6 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, self.modes4, dtype=torch.cfloat))
        self.weights7 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, self.modes4, dtype=torch.cfloat))
        self.weights8 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, self.modes4, dtype=torch.cfloat))

    def forward(self, x):
        batchsize = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfftn(x, dim=[2,3,4,5])
        
        z_dim = min(x_ft.shape[5], self.modes4)
        
        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x_ft.shape[2], x_ft.shape[3], x_ft.shape[4], self.modes4, device=x.device, dtype=torch.cfloat)
        
        # if x_ft.shape[4] > self.modes3, truncate; if x_ft.shape[4] < self.modes3, add zero padding 
        coeff = torch.zeros(batchsize, self.in_channels, self.modes1, self.modes2, self.modes3, self.modes4, device=x.device, dtype=torch.cfloat)        
        coeff[..., :z_dim] = x_ft[:, :, :self.modes1, :self.modes2, :self.modes3, :z_dim]
        out_ft[:, :, :self.modes1, :self.modes2, :self.modes3, :] = compl_mul4d(coeff, self.weights1)
        
        coeff = torch.zeros(batchsize, self.in_channels, self.modes1, self.modes2, self.modes3, self.modes4, device=x.device, dtype=torch.cfloat)        
        coeff[..., :z_dim] = x_ft[:, :, -self.modes1:, :self.modes2, :self.modes3, :z_dim]
        out_ft[:, :, -self.modes1:, :self.modes2, :self.modes3, :] = compl_mul4d(coeff, self.weights2)
        
        coeff = torch.zeros(batchsize, self.in_channels, self.modes1, self.modes2, self.modes3, self.modes4, device=x.device, dtype=torch.cfloat)        
        coeff[..., :z_dim] = x_ft[:, :, :self.modes1, -self.modes2:, :self.modes3, :z_dim]
        out_ft[:, :, :self.modes1, -self.modes2:, :self.modes3, :] = compl_mul4d(coeff, self.weights3)
        
        coeff = torch.zeros(batchsize, self.in_channels, self.modes1, self.modes2, self.modes3, self.modes4, device=x.device, dtype=torch.cfloat)        
        coeff[..., :z_dim] = x_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3, :z_dim]
        out_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3, :] = compl_mul4d(coeff, self.weights4)

        coeff = torch.zeros(batchsize, self.in_channels, self.modes1, self.modes2, self.modes3, self.modes4, device=x.device, dtype=torch.cfloat)        
        coeff[..., :z_dim] = x_ft[:, :, :self.modes1, :self.modes2, -self.modes3:, :z_dim]
        out_ft[:, :, :self.modes1, :self.modes2, -self.modes3:, :] = compl_mul4d(coeff, self.weights5)

        coeff = torch.zeros(batchsize, self.in_channels, self.modes1, self.modes2, self.modes3, self.modes4, device=x.device, dtype=torch.cfloat)        
        coeff[..., :z_dim] = x_ft[:, :, -self.modes1:, :self.modes2, -self.modes3:, :z_dim]
        out_ft[:, :, -self.modes1:, :self.modes2, -self.modes3:, :] = compl_mul4d(coeff, self.weights6)

        coeff = torch.zeros(batchsize, self.in_channels, self.modes1, self.modes2, self.modes3, self.modes4, device=x.device, dtype=torch.cfloat)        
        coeff[..., :z_dim] = x_ft[:, :, :self.modes1, -self.modes2:, -self.modes3:, :z_dim]
        out_ft[:, :, :self.modes1, -self.modes2:, -self.modes3:, :] = compl_mul4d(coeff, self.weights7)

        coeff = torch.zeros(batchsize, self.in_channels, self.modes1, self.modes2, self.modes3, self.modes4, device=x.device, dtype=torch.cfloat)        
        coeff[..., :z_dim] = x_ft[:, :, -self.modes1:, -self.modes2:, -self.modes3:, :z_dim]
        out_ft[:, :, -self.modes1:, -self.modes2:, -self.modes3:, :] = compl_mul4d(coeff, self.weights8)

        #Return to physical space
        x = torch.fft.irfftn(out_ft, s=(x.size(2), x.size(3), x.size(4), x.size(5)), dim=[2,3,4,5])
        return x

class FNO4d(nn.Module):
    def __init__(self, modes1, modes2, modes3, modes4, width, num_layers, in_dim, out_dim):
        super(FNO4d, self).__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        self.modes4 = modes4
        self.width = width

        # input channel is 3: (a(x, y), x, y)
        self.fc0 = nn.Linear(in_dim, width)

        self.sp_convs = nn.ModuleList([SpectralConv4d(
            width, width, modes1, modes2, modes3, modes4)
            for _ in range(num_layers)])

        self.ws = nn.ModuleList([nn.Conv1d(width, width, 1) for _ in range(num_layers)])

        self.fc1 = nn.Linear(width, width)
        self.fc2 = nn.Linear(width, width)
        self.fc3 = nn.Linear(width, out_dim)
        self.act = nn.GELU()

    def forward(self, x):
        length = len(self.ws)
        batchsize = x.shape[0]
        x = self.fc0(x)
        x = x.permute(0, 5, 1, 2, 3, 4)

        size_x, size_y, size_z, size_t = x.shape[-4], x.shape[-3], x.shape[-2], x.shape[-1]

        for i, (speconv, w) in enumerate(zip(self.sp_convs, self.ws)):
            x1 = speconv(x)
            x2 = w(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y, size_z, size_t)
            x = x1 + x2
            if i != length - 1:
                x = self.act(x)

        x = x.permute(0, 2, 3, 4, 5, 1)
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.act(x)
        x = self.fc3(x)

        x = x.reshape(x.size(0), -1, x.size(-1))
        return x

if __name__ == "__main__":
    x = torch.randn(16, 10, 10, 10, 10, 2)
    model = FNO4d(modes1=2, modes2=2, modes3=2, modes4=2, width=4, num_layers=2, in_dim=2, out_dim=3)
    y = model(x)
    print(y.shape)
    pass