import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import argparse
from torch.utils.data import TensorDataset, DataLoader

from time_model import GRU, LSTM, Mamba_NO

from time_model import GalerkinTransformer, Transformer, GNOT

def get_args():
    parser = argparse.ArgumentParser(description='DeepONet Training')
    parser.add_argument('--SEED', type=int, default=0)
    parser.add_argument('--grid_t', type=int, default=100, help="t-axis grid size")
    parser.add_argument('--K', type=int, default=4, help="")
    parser.add_argument('--N_train', type=int, default=10000)
    parser.add_argument('--N_test', type=int, default=1000)

    parser.add_argument('--num_epochs', type=int, default=1000, help="number of training epochs")
    parser.add_argument('--batch_size', type=int, default=16, help="minibatch size for SGD")
    parser.add_argument('--lr', type=float, default=1e-3, help="learning rate")

    parser.add_argument('--model', type=str, default="LSTM")

    parser.add_argument('--data2cuda', type=int, default=1)

    args = parser.parse_args()
    print(args)

    return args

args = get_args()

args.T = int(args.grid_t / 100)
args.grid_x = args.K * 4 + 1

SEED = args.SEED
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)

def generate_g_x(N, grid_x, K):
    k1 = (np.arange(1, K + 1) + 0.0).reshape(1, K, 1, 1, 1, 1)
    k2 = (np.arange(1, K + 1) + 0.0).reshape(1, 1, K, 1, 1, 1)

    A = np.random.uniform(low = 0.0, high = 1.0, size = (N, K, K, 1, 1, 1))
    B = np.random.uniform(low = 0.0, high = 1.0, size = (N, K, K, 1, 1, 1))
    C = np.random.rand()

    x = np.linspace(0, 2 * np.pi, grid_x + 1)[:-1]
    y = np.linspace(0, 2 * np.pi, grid_x + 1)[:-1]

    x, y = np.meshgrid(x, y, indexing='ij') # grid_x, grid_x
    x = x.reshape(1, 1, 1, grid_x, grid_x, 1) # 1, 1, 1, grid_x, grid_x, 1
    y = y.reshape(1, 1, 1, grid_x, grid_x, 1)

    g = A * np.sin(k1 * x + k2 * y) + \
        B * np.cos(k1 * x + k2 * y) # N, K, K, grid_x, grid_x, 1
    g = g.sum(1) # N, K, grid_x, grid_x, 1
    g = g.sum(1) # N, grid_x, grid_x, 1
    g = g + C

    d2u_dx2 = -(k1 ** 2) * A * np.sin(k1 * x + k2 * y) - \
        (k1 ** 2) * B * np.cos(k1 * x + k2 * y)

    d2u_dy2 = -(k2 ** 2) * A * np.sin(k1 * x + k2 * y) - \
        (k2 ** 2) * B * np.cos(k1 * x + k2 * y)

    Delta_g = d2u_dx2 + d2u_dy2
    Delta_g = Delta_g.sum(1)
    Delta_g = Delta_g.sum(1) # N, grid_x, grid_x, 1

    g = g.reshape(N, grid_x, grid_x, 1, 1)
    Delta_g = Delta_g.reshape(N, grid_x, grid_x, 1, 1)
    
    return g, Delta_g

def generate_h_x(N, grid_t, T):
    from data import AntideData
    s0 = [0]
    length_scale = 0.2
    data = AntideData(T, s0, grid_t, grid_t, length_scale, N, 1)

    h_t, h = data.X_train, data.y_train # N, grid_t, 1

    h_t = h_t.reshape(N, 1, 1, grid_t, 1)
    h = h.reshape(N, 1, 1, grid_t, 1)

    return h_t, h

def generate_data(N, grid_x, grid_t, T, K):
    g, Delta_g = generate_g_x(N, grid_x, K)
    h_t, h = generate_h_x(N, grid_t, T)

    u = g * h # N, grid_x, grid_x, grid_t, 1

    f = Delta_g * h + (g * h) ** 2 - g * h_t

    return u, f

y_train, f_train = generate_data(args.N_train, args.grid_x, args.grid_t, args.T, args.K)
y_test, f_test = generate_data(args.N_test, args.grid_x, args.grid_t, args.T, args.K)

# Setting device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

X_train, Y_train, X_test, Y_test = torch.from_numpy(f_train).float(), torch.from_numpy(y_train).float(), torch.from_numpy(f_test).float(), torch.from_numpy(y_test).float()

if args.data2cuda:
    X_train, Y_train, X_test, Y_test = X_train.to(device), Y_train.to(device), X_test.to(device), Y_test.to(device)

def Grid2Fourier(X, grid_x):
    # X shape: [N, grid_x, grid_x, grid_t, 1]
    X = X.permute(0, 3, 1, 2, 4).squeeze() # N, grid_t, grid_x, grid_x
    X = torch.fft.rfft2(X) / grid_x ** 2 # N, grid_t, grid_x, grid_x // 2 + 1
    X = X.reshape(X.size(0), X.size(1), -1) # N, grid_t, -1
    X_real = torch.real(X)
    X_imag = torch.imag(X)
    X = torch.cat([X_real, X_imag], -1) # N, grid_t, -1
    return X

def Fourier2Grid(X, grid_x):
    N, grid_t = X.size(0), X.size(1)
    X_real = X[:, :, :X.size(-1)//2]
    X_imag = X[:, :, X.size(-1)//2:]
    X = X_real + X_imag * 1j
    X[:, :, 0] = X_real[:, :, 0]
    X = X * grid_x ** 2
    X = X.reshape(N, grid_t, grid_x, grid_x // 2 + 1)
    X = torch.fft.irfft2(X, s=[grid_x] * 2)
    return X

X_train = Grid2Fourier(X_train, args.grid_x)
y_train = Grid2Fourier(Y_train, args.grid_x)
X_test = Grid2Fourier(X_test, args.grid_x)
y_test = Grid2Fourier(Y_test, args.grid_x)

Y_train = Y_train.squeeze().permute(0, 3, 1, 2)
Y_test = Y_test.squeeze().permute(0, 3, 1, 2)

print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
print(Y_train.shape, Y_test.shape)
io_dim = 2 * (args.grid_x // 2 + 1) * args.grid_x

if args.model == "LSTM":
    model = LSTM(input_dim=io_dim, output_dim=io_dim, hidden_dim=256, num_layers=1).to(device)
elif args.model == "GRU":
    model = GRU(input_dim=io_dim, output_dim=io_dim, hidden_dim=256, num_layers=1).to(device)
elif args.model == "Mamba":
    model = Mamba_NO(d_model=256,n_layer=1,d_intermediate=0,input_dim=io_dim,output_dim=io_dim).to(device)
elif args.model == "ST":
    model = GalerkinTransformer(dim_in=io_dim, dim_out=io_dim, dim_hid=128, depth=2, heads=4, dim_head=128, attn_type='standard', mlp_dim=128).to(device)
elif args.model == "GT":
    model = GalerkinTransformer(dim_in=io_dim, dim_out=io_dim, dim_hid=128, depth=2, heads=4, dim_head=128, attn_type='galerkin', mlp_dim=128).to(device)
elif args.model == "FT":
    model = GalerkinTransformer(dim_in=io_dim, dim_out=io_dim, dim_hid=128, depth=2, heads=4, dim_head=128, attn_type='fourier', mlp_dim=128).to(device)
elif args.model == "T":
    model = Transformer(ninput=io_dim, noutput=io_dim, nhidden=256, dim_feedforward=256, nhead=4, nlayers=1).to(device)
elif args.model == "GNOT":
    model = GNOT(dim_in=io_dim, dim_out=io_dim, dim_hid=128, depth=1, heads=4, dim_head=128, n_experts=2).to(device)

# Function to count parameters
def count_parameters(model): return sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total number of parameters: {count_parameters(model)}")

criterion = nn.MSELoss()

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=args.lr)

lr_lambda = lambda epoch: 1-epoch/args.num_epochs
if args.model in ["ST", "GT", "FT", "T", "GNOT"]:
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=args.lr,
        total_steps=args.num_epochs,
        div_factor=1e4,
        pct_start=0.2,
        final_div_factor=1e4,
    )
else:
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

train_dataset = TensorDataset(X_train, y_train, Y_train)
train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)

test_dataset = TensorDataset(X_test, y_test, Y_test)
test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False)

loss_traj = []
# Training loop
for epoch in tqdm(range(args.num_epochs)):
    
    # Forward pass
    for batch_idx, (data, targets, _) in enumerate(train_loader):
        model.train()  # Set the model to training mode
        optimizer.zero_grad()  # Clear gradients
        data, targets = data.to(device), targets.to(device)
        outputs = model(data)
        # loss = criterion(outputs, targets)  # Target is the same as input
        loss = torch.linalg.vector_norm(outputs - targets) / torch.linalg.vector_norm(targets)
        loss.backward()
        optimizer.step()

        outputs_grid = Fourier2Grid(outputs, args.grid_x)
        targets_grid = Fourier2Grid(targets, args.grid_x)
        loss_saved = criterion(outputs_grid, targets_grid)
        loss_traj.append(loss_saved.item())


    scheduler.step()

    # Print loss every epoch
    if (epoch+1)%int(100)==0:
        output_test, label_test = [], []
        output_grid, label_grid = [], []

        for batch_idx, (data, targets, Targets) in enumerate(train_loader):
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)

            output_test.append(outputs.detach().cpu())
            label_test.append(targets.detach().cpu())

            outputs_grid = Fourier2Grid(outputs, args.grid_x)
            targets_grid = Fourier2Grid(targets, args.grid_x)

            # outputs_grid = Fourier2Grid(targets, args.grid_x)
            # targets_grid = Targets
            # print(outputs_grid.shape, targets_grid.shape)

            output_grid.append(outputs_grid.detach().cpu())
            label_grid.append(targets_grid.detach().cpu())

        output_test = torch.cat(output_test, 0)
        label_test = torch.cat(label_test, 0)

        error_train = torch.norm((output_test - label_test).reshape(-1)) / torch.norm((label_test).reshape(-1))

        output_grid = torch.cat(output_grid, 0)
        label_grid = torch.cat(label_grid, 0)

        error_train_grid = torch.norm((output_grid - label_grid).reshape(-1)) / torch.norm((label_grid).reshape(-1))

        output_test, label_test = [], []
        output_grid, label_grid = [], []

        for batch_idx, (data, targets, Targets) in enumerate(test_loader):
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)

            output_test.append(outputs.detach().cpu())
            label_test.append(targets.detach().cpu())

            outputs_grid = Fourier2Grid(outputs, args.grid_x)
            targets_grid = Fourier2Grid(targets, args.grid_x)

            # outputs_grid = Fourier2Grid(targets, args.grid_x)
            # targets_grid = Targets
            # print(outputs_grid.shape, targets_grid.shape)

            output_grid.append(outputs_grid.detach().cpu())
            label_grid.append(targets_grid.detach().cpu())

        output_test = torch.cat(output_test, 0)
        label_test = torch.cat(label_test, 0)

        error_test = torch.norm((output_test - label_test).reshape(-1)) / torch.norm((label_test).reshape(-1))

        output_grid = torch.cat(output_grid, 0)
        label_grid = torch.cat(label_grid, 0)

        error_test_grid = torch.norm((output_grid - label_grid).reshape(-1)) / torch.norm((label_grid).reshape(-1))

        print(f"Epoch {epoch+1}, Train: {error_train.item():.3e} {error_train_grid.item():.3e}, Test: {error_test.item():.3e} {error_test_grid.item():.3e}")

loss_traj = np.asarray(loss_traj)
filename = "RD_Fourier_3D_" + \
    "Model=" + str(args.model) + "_" + \
    "Seed=" + str(args.SEED) + \
    ".txt"
np.savetxt(filename, loss_traj)