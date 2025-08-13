import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import argparse
from torch.utils.data import TensorDataset, DataLoader

from FNO4d import FNO4d

from FFNO4d import FFNO4d

from FNO4d_Jamba import FNO_Jamba_1, FNO_Jamba_2


def get_args():
    parser = argparse.ArgumentParser(description='DeepONet Training')
    parser.add_argument('--SEED', type=int, default=0)

    parser.add_argument('--grid_x', type=int, default=17, help="x-axis grid size")
    parser.add_argument('--grid_t', type=int, default=100, help="t-axis grid size")

    parser.add_argument('--T', type=int, default=1, help="terminal time")

    parser.add_argument('--K', type=int, default=4, help="")

    parser.add_argument('--N_train', type=int, default=10000)
    parser.add_argument('--N_test', type=int, default=10000)

    parser.add_argument('--num_epochs', type=int, default=1000, help="number of training epochs")
    parser.add_argument('--batch_size', type=int, default=16, help="minibatch size for SGD")
    parser.add_argument('--lr', type=float, default=1e-3, help="learning rate")

    parser.add_argument('--model', type=str, default="FNO")

    args = parser.parse_args()
    print(args)

    return args

args = get_args()

SEED = args.SEED
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)

def generate_g_x(N, grid_x, K):
    k1 = (np.arange(1, K + 1) + 0.0).reshape(1, K, 1, 1, 1, 1, 1, 1)
    k2 = (np.arange(1, K + 1) + 0.0).reshape(1, 1, K, 1, 1, 1, 1, 1)
    k3 = (np.arange(1, K + 1) + 0.0).reshape(1, 1, 1, K, 1, 1, 1, 1)

    A = np.random.uniform(low = 0.0, high = 1.0, size = (N, K, K, K, 1, 1, 1, 1))
    B = np.random.uniform(low = 0.0, high = 1.0, size = (N, K, K, K, 1, 1, 1, 1))
    C = np.random.rand()

    x = np.linspace(0, 2 * np.pi, grid_x + 1)[:-1]
    y = np.linspace(0, 2 * np.pi, grid_x + 1)[:-1]
    z = np.linspace(0, 2 * np.pi, grid_x + 1)[:-1]
    x, y, z = np.meshgrid(x, y, z, indexing='ij') # grid_x, grid_x, grid_x
    x = x.reshape(1, 1, 1, 1, grid_x, grid_x, grid_x, 1)
    y = y.reshape(1, 1, 1, 1, grid_x, grid_x, grid_x, 1)
    z = z.reshape(1, 1, 1, 1, grid_x, grid_x, grid_x, 1)

    g = A * np.sin(k1 * x + k2 * y + k3 * z) + \
        B * np.cos(k1 * x + k2 * y + k3 * z) # N, K, K, K, grid_x, grid_x, grid_x, 1
    g = g.sum(1) # N, K, K, grid_x, grid_x, grid_x, 1
    g = g.sum(1) # N, K, grid_x, grid_x, grid_x, 1
    g = g.sum(1) # N, grid_x, grid_x, grid_x, 1
    g = g + C

    d2u_dx2 = -(k1 ** 2) * A * np.sin(k1 * x + k2 * y + k3 * z) - \
        (k1 ** 2) * B * np.cos(k1 * x + k2 * y + k3 * z)

    d2u_dy2 = -(k2 ** 2) * A * np.sin(k1 * x + k2 * y + k3 * z) - \
        (k2 ** 2) * B * np.cos(k1 * x + k2 * y + k3 * z)
    
    d2u_dz2 = -(k3 ** 2) * A * np.sin(k1 * x + k2 * y + k3 * z) - \
        (k3 ** 2) * B * np.cos(k1 * x + k2 * y + k3 * z)

    Delta_g = d2u_dx2 + d2u_dy2 + d2u_dz2
    Delta_g = Delta_g.sum(1)
    Delta_g = Delta_g.sum(1)
    Delta_g = Delta_g.sum(1)

    g = g.reshape(N, grid_x, grid_x, grid_x, 1, 1)
    Delta_g = Delta_g.reshape(N, grid_x, grid_x, grid_x, 1, 1)
    
    return g, Delta_g

def generate_h_x(N, grid_t, T):
    from data import AntideData
    s0 = [0]
    length_scale = 0.2
    data = AntideData(T, s0, grid_t, grid_t, length_scale, N, 1)

    h_t, h = data.X_train, data.y_train # N, grid_t, 1

    h_t = h_t.reshape(N, 1, 1, 1, grid_t, 1)
    h = h.reshape(N, 1, 1, 1, grid_t, 1)

    return h_t, h

def generate_data(N, grid_x, grid_t, T, K):
    g, Delta_g = [], []
    for _ in tqdm(range(N // 100)):
        tg, tDelta_g = generate_g_x(100, grid_x, K)
        g.append(tg)
        Delta_g.append(tDelta_g)
    g, Delta_g = np.concatenate(g, 0), np.concatenate(Delta_g, 0)
    h_t, h = generate_h_x(N, grid_t, T)

    u = g * h # N, grid_x, grid_x, grid_x, grid_t, 1

    f = Delta_g * h + (g * h) ** 2 - g * h_t

    x = np.linspace(0, 2 * np.pi, grid_x + 1)[:-1]
    y = np.linspace(0, 2 * np.pi, grid_x + 1)[:-1]
    z = np.linspace(0, 2 * np.pi, grid_x + 1)[:-1]
    t = np.linspace(0, T, grid_t)
    x, y, z, t = np.meshgrid(x, y, z, t, indexing='ij') # [grid_x, grid_x, grid_x, grid_t]

    x = x.reshape(1, grid_x, grid_x, grid_x, grid_t, 1)
    y = y.reshape(1, grid_x, grid_x, grid_x, grid_t, 1)
    z = z.reshape(1, grid_x, grid_x, grid_x, grid_t, 1)
    t = t.reshape(1, grid_x, grid_x, grid_x, grid_t, 1) # [1, grid_x, grid_x, grid_x, grid_t, 1]

    x = np.tile(x, (N, 1, 1, 1, 1, 1))
    y = np.tile(y, (N, 1, 1, 1, 1, 1))
    z = np.tile(z, (N, 1, 1, 1, 1, 1))
    t = np.tile(t, (N, 1, 1, 1, 1, 1))

    f = np.concatenate([f, x, y, z, t], -1) # N, grid_x, grid_x, grid_x, grid_t, 5
    return u, f

u_train, f_train = generate_data(args.N_train, args.grid_x, args.grid_t, args.T, args.K)
u_test, f_test = generate_data(args.N_test, args.grid_x, args.grid_t, args.T, args.K)

# Setting device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# X_train, y_train, X_test, y_test = torch.from_numpy(f_train).float().to(device), torch.from_numpy(u_train).float().to(device), torch.from_numpy(f_test).float().to(device), torch.from_numpy(u_test).float().to(device)
X_train, y_train, X_test, y_test = torch.from_numpy(f_train).float(), torch.from_numpy(u_train).float(), torch.from_numpy(f_test).float(), torch.from_numpy(u_test).float()

y_train = y_train.reshape(args.N_train, args.grid_x ** 3 * args.grid_t, 1)
y_test = y_test.reshape(args.N_test, args.grid_x ** 3 * args.grid_t, 1)

print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

in_dim = 5

def get_model(model):
    if args.model == "FNO":
        model = FNO4d(modes1=9, modes2=9, modes3=9, modes4=16, width=8, num_layers=2, in_dim=in_dim, out_dim=1).to(device)

    elif args.model == "FNO_GRU_1":
        model = FNO_Jamba_1(input_dim=in_dim, output_dim=1, modes=9, width=32, num_layers=1, model_t_type="GRU").to(device)
    elif args.model == "FNO_LSTM_1":
        model = FNO_Jamba_1(input_dim=in_dim, output_dim=1, modes=9, width=32, num_layers=1, model_t_type="LSTM").to(device)
    elif args.model == "FNO_Mamba_1":
        model = FNO_Jamba_1(input_dim=in_dim, output_dim=1, modes=9, width=32, num_layers=1, model_t_type="Mamba").to(device)

    elif args.model == "FNO_GRU_2":
        model = FNO_Jamba_2(input_dim=in_dim, output_dim=1, modes=9, width=32, num_layers=1, model_t_type="GRU").to(device)
    elif args.model == "FNO_LSTM_2":
        model = FNO_Jamba_2(input_dim=in_dim, output_dim=1, modes=9, width=32, num_layers=1, model_t_type="LSTM").to(device)
    elif args.model == "FNO_Mamba_2":
        model = FNO_Jamba_2(input_dim=in_dim, output_dim=1, modes=9, width=32, num_layers=1, model_t_type="Mamba").to(device)

    elif args.model == "FFNO":
        model = FFNO4d(modes_x=9, modes_y=9, modes_z=9, modes_t=16, width=128, input_dim=in_dim, output_dim=1, n_layers=2).to(device)

    return model

model = get_model(args.model)

# Function to count parameters
def count_parameters(model): return sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total number of parameters: {count_parameters(model)}")

# Loss function
criterion = nn.MSELoss()

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=args.lr)

lr_lambda = lambda epoch: 1-epoch/args.num_epochs
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)

test_dataset = TensorDataset(X_test, y_test)
test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False)

# Training loop
for epoch in tqdm(range(args.num_epochs)):
    
    # Forward pass
    for batch_idx, (data, targets) in enumerate(train_loader):
        model.train()  # Set the model to training mode
        optimizer.zero_grad()  # Clear gradients
        data, targets = data.to(device), targets.to(device)
        outputs = model(data)
        loss = criterion(outputs, targets)  # Target is the same as input
        loss.backward()
        optimizer.step()
    scheduler.step()

    # Print loss every epoch
    if (epoch+1)%int(50)==0 or epoch == args.num_epochs - 1:
        output_train, label_train = [], []

        for batch_idx, (data, targets) in enumerate(train_loader):
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)
            output_train.append(outputs.detach().cpu())
            label_train.append(targets.detach().cpu())
        
        output_train = torch.cat(output_train, 0)
        label_train = torch.cat(label_train, 0)

        error_train = torch.norm((output_train - label_train).reshape(-1)) / torch.norm((label_train).reshape(-1))
        

        output_test, label_test = [], []

        for batch_idx, (data, targets) in enumerate(test_loader):
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)
            output_test.append(outputs.detach().cpu())
            label_test.append(targets.detach().cpu())

        output_test = torch.cat(output_test, 0)
        label_test = torch.cat(label_test, 0)

        error_test = torch.norm((output_test - label_test).reshape(-1)) / torch.norm((label_test).reshape(-1))
        
        print(f"Epoch {epoch+1}, Train Loss: {loss.item():.3e}, Train Rel L2: {error_train.item():.3e}, Test Rel L2: {error_test.item():.3e}")

