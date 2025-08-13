import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import argparse
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import os

from FNO3d import FNO3d

from FFNO3d import FFNO3d

from FNO3d_Jamba import FNO_Jamba_1, FNO_Jamba_2


def get_args():
    parser = argparse.ArgumentParser(description='DeepONet Training')
    parser.add_argument('--SEED', type=int, default=0)

    parser.add_argument('--grid_x', type=int, default=28, help="x-axis grid size")
    parser.add_argument('--grid_t', type=int, default=1000, help="t-axis grid size")

    parser.add_argument('--T', type=int, default=1, help="terminal time")

    parser.add_argument('--num_epochs', type=int, default=1000, help="number of training epochs")
    parser.add_argument('--batch_size', type=int, default=16, help="minibatch size for SGD")
    parser.add_argument('--lr', type=float, default=1e-3, help="learning rate")

    parser.add_argument('--model', type=str, default="FNO")

    parser.add_argument('--aug', type=int, default=1)

    args = parser.parse_args()
    print(args)

    return args

args = get_args()

SEED = args.SEED
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)

f, u = [], []
for path in os.listdir("Data"):
    data = np.load("Data/" + path)
    f_ = data['F'][:, :args.grid_t]
    u_ = data['U'][:, :args.grid_t]
    f.append(f_)
    u.append(u_)
f = np.concatenate(f, 0)
u = np.concatenate(u, 0)

f_train, f_test, y_train, y_test = train_test_split(
    f, u, test_size=0.1, random_state=42
)

args.N_train = f_train.shape[0]
args.N_test = f_test.shape[0]

print(f_train.shape, y_train.shape, f_test.shape, y_test.shape)
def process_f(f, grid_x):
    N = f.shape[0]
    grid_t = f.shape[1]
    f = f.reshape(N, 1, 1, grid_t, 1)
    f = np.tile(f, (1, grid_x, grid_x, 1, 1))
    return f

def process_u(u):
    N, grid_t, grid_x = u.shape[0], u.shape[1], u.shape[2]
    u = np.transpose(u, (0, 2, 3, 1))
    u = u.reshape(N, grid_x, grid_x, grid_t, 1)
    return u

def aug_f(f, N, grid_x, grid_t):
    x = np.linspace(0, 1, grid_x)
    y = np.linspace(0, 1, grid_x)
    t = np.linspace(0, 1, grid_t)
    x, y, t = np.meshgrid(x, y, t, indexing='ij') # [grid_x, grid_x, grid_t]
    x = x.reshape(1, grid_x, grid_x, grid_t, 1)
    y = y.reshape(1, grid_x, grid_x, grid_t, 1)
    t = t.reshape(1, grid_x, grid_x, grid_t, 1) # [1, grid_x, grid_x, grid_t, 1]
    x = np.tile(x, (N, 1, 1, 1, 1))
    y = np.tile(y, (N, 1, 1, 1, 1))
    t = np.tile(t, (N, 1, 1, 1, 1)) # [N, grid_x, grid_x, grid_t, 1]
    f = np.concatenate([f, x, y, t], -1) # N, grid_x, grid_t, 4
    return f

if args.model not in ['GRU', 'LSTM', 'Mamba', 'GNOT', 'GT', 'FT', 'ST', 'T']:
    f_train = process_f(f_train, args.grid_x)
    f_test = process_f(f_test, args.grid_x)
    if args.aug == 1:
        f_train = aug_f(f_train, args.N_train, args.grid_x, args.grid_t)
        f_test = aug_f(f_test, args.N_test, args.grid_x, args.grid_t)
else:
    f_train = f_train.reshape(f_train.shape[0], 1, 1, -1, 1)
    f_test = f_test.reshape(f_test.shape[0], 1, 1, -1, 1)
y_train = process_u(y_train)
y_test = process_u(y_test)

# Setting device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

X_train, y_train, X_test, y_test = torch.from_numpy(f_train).float().to(device), torch.from_numpy(y_train).float().to(device), torch.from_numpy(f_test).float().to(device), torch.from_numpy(y_test).float().to(device)

y_train = y_train.reshape(y_train.size(0), args.grid_x ** 2 * args.grid_t, 1)
y_test = y_test.reshape(y_test.size(0), args.grid_x ** 2 * args.grid_t, 1)

print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

in_dim = 1 if args.aug == 0 else 4

def get_model(model):
    if args.model == "FNO":
        model = FNO3d(modes1=12, modes2=12, modes3=16, width=8, num_layers=2, in_dim=in_dim, out_dim=1).to(device)

    elif args.model == "FNO_GRU_1":
        model = FNO_Jamba_1(input_dim=in_dim, output_dim=1, modes=12, width=8, num_layers=1, model_t_type="GRU").to(device)
    elif args.model == "FNO_LSTM_1":
        model = FNO_Jamba_1(input_dim=in_dim, output_dim=1, modes=12, width=8, num_layers=1, model_t_type="LSTM").to(device)
    elif args.model == "FNO_Mamba_1":
        model = FNO_Jamba_1(input_dim=in_dim, output_dim=1, modes=12, width=8, num_layers=1, model_t_type="Mamba").to(device)

    elif args.model == "FNO_GRU_2":
        model = FNO_Jamba_2(input_dim=in_dim, output_dim=1, modes=12, width=8, num_layers=1, model_t_type="GRU").to(device)
    elif args.model == "FNO_LSTM_2":
        model = FNO_Jamba_2(input_dim=in_dim, output_dim=1, modes=12, width=8, num_layers=1, model_t_type="LSTM").to(device)
    elif args.model == "FNO_Mamba_2":
        model = FNO_Jamba_2(input_dim=in_dim, output_dim=1, modes=12, width=8, num_layers=1, model_t_type="Mamba").to(device)

    elif args.model == "FFNO":
        model = FFNO3d(modes_x=12, modes_y=12, modes_z=16, width=8, input_dim=in_dim, output_dim=1, n_layers=2).to(device)

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
if args.model in ['DeepONet_GT', 'DeepONet_FT', 'DeepONet_ST', 'DeepONet_T', 'DeepONet_GNOT', \
                  "GT", "FT", "ST", "T", "GNOT"]:
    print("Using Transformer scheduler")
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
        outputs = model(data)
        loss = criterion(outputs, targets)  # Target is the same as input
        loss.backward()
        optimizer.step()
    scheduler.step()

    # Print loss every epoch
    if (epoch+1)%int(100)==0: 
        output_train, label_train = [], []

        for batch_idx, (data, targets) in enumerate(train_loader):
            outputs = model(data)
            output_train.append(outputs.detach().cpu())
            label_train.append(targets.detach().cpu())
        
        output_train = torch.cat(output_train, 0)
        label_train = torch.cat(label_train, 0)

        error_train = torch.norm((output_train - label_train).reshape(-1)) / torch.norm((label_train).reshape(-1))
        

        output_test, label_test = [], []

        for batch_idx, (data, targets) in enumerate(test_loader):
            outputs = model(data)
            output_test.append(outputs.detach().cpu())
            label_test.append(targets.detach().cpu())

        output_test = torch.cat(output_test, 0)
        label_test = torch.cat(label_test, 0)

        error_test = torch.norm((output_test - label_test).reshape(-1)) / torch.norm((label_test).reshape(-1))
        
        print(f"Epoch {epoch+1}, Train Loss: {loss.item():.3e}, Train Rel L2: {error_train.item():.3e}, Test Rel L2: {error_test.item():.3e}")

