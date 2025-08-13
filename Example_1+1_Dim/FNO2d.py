import torch.nn as nn
import numpy as np
import torch

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

class FNO2d(nn.Module):
    def __init__(self, modes1, modes2, width, num_layers, in_dim, out_dim):
        super(FNO2d, self).__init__()
        """
        Args:
            - modes1: list of int, number of modes in first dimension in each layer
            - modes2: list of int, number of modes in second dimension in each layer
            - width: int, optional, if layers is None, it will be initialized as [width] * [len(modes1) + 1] 
            - in_dim: number of input channels
            - out_dim: number of output channels
            - act: activation function, {tanh, gelu, relu, leaky_relu}, default: gelu 
        """
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width

        # input channel is 3: (a(x, y), x, y)
        self.fc0 = nn.Linear(in_dim, width)

        self.sp_convs = nn.ModuleList([SpectralConv2d(
            width, width, modes1, modes2)
            for _ in range(num_layers)])

        self.ws = nn.ModuleList([nn.Conv1d(width, width, 1) for _ in range(num_layers)])

        self.fc1 = nn.Linear(width, width)
        self.fc2 = nn.Linear(width, width)
        self.fc3 = nn.Linear(width, out_dim)
        self.act = nn.GELU()

    def forward(self, x):
        '''
        Args:
            - x : (batch size, x_grid, y_grid, 2)
        Returns:
            - x: (batch size, x_grid, y_grid, 1)
        '''
        size_1, size_2 = x.shape[1], x.shape[2]

        length = len(self.ws)
        batchsize = x.shape[0]
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)   # B, C, X, Y

        size_x, size_y = x.shape[-2], x.shape[-1]

        for i, (speconv, w) in enumerate(zip(self.sp_convs, self.ws)):
            x1 = speconv(x)
            x2 = w(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y)
            x = x1 + x2
            if i != length - 1:
                x = self.act(x)

        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.act(x)
        x = self.fc3(x)

        x = x.reshape(x.size(0), -1, x.size(-1))
        return x

if __name__ == "__main__":
    x = torch.randn(16, 10, 10, 2)
    model = FNO2d(4, 4, 8, 2, 2, 1)
    y = model(x)
    print(y.shape)
    pass