import torch.nn as nn
import numpy as np
import torch

class PR2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, T, X):
        super(PR2d, self).__init__()

        self.T, self.X = T.reshape(1, -1), X.reshape(1, -1)

        self.modes1 = modes1
        self.modes2 = modes2
        self.scale = (1 / (in_channels*out_channels))
        self.weights_pole1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1,  dtype=torch.cfloat))
        self.weights_pole2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes2, dtype=torch.cfloat))
        self.weights_residue = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1,  self.modes2, dtype=torch.cfloat))
    
    def output_PR(self, lambda1, lambda2, alpha, weights_pole1, weights_pole2, weights_residue):
        Hw=torch.zeros(weights_residue.shape[0],weights_residue.shape[0],weights_residue.shape[2],weights_residue.shape[3],lambda1.shape[0], lambda2.shape[0], device=alpha.device, dtype=torch.cfloat)
        term1=torch.div(1,torch.einsum("pbix,qbik->pqbixk",torch.sub(lambda1,weights_pole1),torch.sub(lambda2,weights_pole2)))
        Hw=torch.einsum("bixk,pqbixk->pqbixk",weights_residue,term1)
        Pk=Hw  # for ode, Pk=-Hw; for 2d pde, Pk=Hw; for 3d pde, Pk=-Hw; 
        output_residue1=torch.einsum("biox,oxikpq->bkox", alpha, Hw) 
        output_residue2=torch.einsum("biox,oxikpq->bkpq", alpha, Pk) 
        return output_residue1,output_residue2

    def forward(self, x):
        tx=self.T.cuda()
        ty=self.X.cuda()
        #Compute input poles and resudes by FFT
        dty=(ty[0,1]-ty[0,0]).item()  # location interval
        dtx=(tx[0,1]-tx[0,0]).item()  # time interval
        alpha = torch.fft.fft2(x, dim=[-2,-1])
        omega1=torch.fft.fftfreq(ty.shape[1], dty)*2*np.pi*1j   # location frequency
        omega2=torch.fft.fftfreq(tx.shape[1], dtx)*2*np.pi*1j   # time frequency
        omega1=omega1.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        omega2=omega2.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        lambda1=omega1.cuda()
        lambda2=omega2.cuda()
 
        # Obtain output poles and residues for transient part and steady-state part
        output_residue1,output_residue2 = self.output_PR(lambda1, lambda2, alpha, self.weights_pole1, self.weights_pole2, self.weights_residue)

        # Obtain time histories of transient response and steady-state response
        x1 = torch.fft.ifft2(output_residue1, s=(x.size(-2), x.size(-1)))
        x1 = torch.real(x1)    
        term1=torch.einsum("bip,kz->bipz", self.weights_pole1, ty.type(torch.complex64).reshape(1,-1))
        term2=torch.einsum("biq,kx->biqx", self.weights_pole2, tx.type(torch.complex64).reshape(1,-1))
        term3=torch.einsum("bipz,biqx->bipqzx", torch.exp(term1),torch.exp(term2))
        x2=torch.einsum("kbpq,bipqzx->kizx", output_residue2,term3)
        x2=torch.real(x2)
        x2=x2/x.size(-1)/x.size(-2)
        return x1+x2


class LNO2d(nn.Module):
    def __init__(self, input_dim, output_dim, width,modes1,modes2,T,X):
        super(LNO2d, self).__init__()
        
        self.width = width
        self.modes1 = modes1
        self.modes2 = modes2
        self.fc0 = nn.Linear(input_dim, self.width) 

        self.conv0 = PR2d(self.width, self.width, self.modes1, self.modes2, T, X)
        self.conv1 = PR2d(self.width, self.width, self.modes1, self.modes2, T, X)
        self.conv2 = PR2d(self.width, self.width, self.modes1, self.modes2, T, X)
        self.conv3 = PR2d(self.width, self.width, self.modes1, self.modes2, T, X)
        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        self.w3 = nn.Conv2d(self.width, self.width, 1)
        self.norm = nn.InstanceNorm2d(self.width)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, output_dim)

    def forward(self,x):
        # grid = self.get_grid(x.shape, x.device)
        # x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)

        x1 = self.norm(self.conv0(self.norm(x)))
        x2 = self.w0(x)
        x = x1 +x2
        x = torch.sin(x)

        x1 = self.norm(self.conv1(self.norm(x)))
        x2 = self.w1(x)
        x = x1 +x2
        x = torch.sin(x)

        x1 = self.norm(self.conv2(self.norm(x)))
        x2 = self.w2(x)
        x = x1 +x2
        x = torch.sin(x)

        x1 = self.norm(self.conv3(self.norm(x)))
        x2 = self.w3(x)
        x = x1 +x2

        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = torch.sin(x)
        x = self.fc2(x)

        x = x.reshape(x.size(0), -1, x.size(-1))
        return x

    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)

if __name__ == "__main__":
    x = torch.randn(16, 10, 10, 2)
    model = FNO2d(4, 4, 8, 2, 2, 1)
    y = model(x)
    print(y.shape)
    pass