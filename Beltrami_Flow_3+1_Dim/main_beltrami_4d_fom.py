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

    parser.add_argument('--N_train', type=int, default=900)
    parser.add_argument('--N_test', type=int, default=100)

    parser.add_argument('--num_epochs', type=int, default=1000, help="number of training epochs")
    parser.add_argument('--batch_size', type=int, default=8, help="minibatch size for SGD")
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

def generate_data(N, grid_x, grid_t, T):
    x = np.linspace(0, 2 * np.pi, grid_x)
    y = np.linspace(0, 2 * np.pi, grid_x)
    z = np.linspace(0, 2 * np.pi, grid_x)
    t = np.linspace(0, T, grid_t)

    x, y, z, t = np.meshgrid(x, y, z, t, indexing='ij') # grid_x, grid_x, grid_x, grid_t

    x = np.expand_dims(x, axis=0) # 1, grid_x, grid_x, grid_x, grid_t
    y = np.expand_dims(y, axis=0) # 1, grid_x, grid_x, grid_x, grid_t
    z = np.expand_dims(z, axis=0) # 1, grid_x, grid_x, grid_x, grid_t
    t = np.expand_dims(t, axis=0) # 1, grid_x, grid_x, grid_x, grid_t

    a = np.random.rand(N, 1, 1, 1, 1)
    pm = (np.random.rand(N, 1, 1, 1, 1) > 0.5) + 0.0
    d = np.random.rand(N, 1, 1, 1, 1) + 0.5

    u = -a * (np.exp(a * x) * np.sin(a * y + pm * d * z) + np.exp(a * z) * np.cos(a * x + pm * d * y)) * np.exp(-d*d*t)
    v = -a * (np.exp(a * y) * np.sin(a * z + pm * d * x) + np.exp(a * x) * np.cos(a * y + pm * d * z)) * np.exp(-d*d*t)
    w = -a * (np.exp(a * z) * np.sin(a * x + pm * d * y) + np.exp(a * y) * np.cos(a * z + pm * d * x)) * np.exp(-d*d*t)

    U = np.stack([u, v, w], -1) # N, grid_x, grid_x, grid_x, grid_t, 3

    U_init = U[:, :, :, :, 0:1, :] # N, grid_x, grid_x, grid_x, 1, 3
    U_init = np.tile(U_init, (1, 1, 1, 1, grid_t, 1))

    U_x0, U_x1 = U[:, 0:1, :, :, :, :], U[:, -1:, :, :, :, :] # N, 1, grid_x, grid_x, grid_t, 3
    U_x0 = np.tile(U_x0, (1, grid_x, 1, 1, 1, 1))
    U_x1 = np.tile(U_x1, (1, grid_x, 1, 1, 1, 1))

    U_y0, U_y1 = U[:, :, 0:1, :, :, :], U[:, :, -1:, :, :, :] # N, grid_x, 1, grid_x, grid_t, 3
    U_y0 = np.tile(U_y0, (1, 1, grid_x, 1, 1, 1))
    U_y1 = np.tile(U_y1, (1, 1, grid_x, 1, 1, 1))

    U_z0, U_z1 = U[:, :, :, 0:1, :, :], U[:, :, :, -1:, :, :] # N, grid_x, grid_x, 1, grid_t, 3
    U_z0 = np.tile(U_z0, (1, 1, 1, grid_x, 1, 1))
    U_z1 = np.tile(U_z1, (1, 1, 1, grid_x, 1, 1))

    F = np.concatenate([U_init, U_x0, U_x1, U_y0, U_y1, U_z0, U_z1], -1) # N, grid_x, grid_x, grid_x, grid_t, 27

    return U, F

u_train, f_train = generate_data(args.N_train, args.grid_x, args.grid_t, args.T)
u_test, f_test = generate_data(args.N_test, args.grid_x, args.grid_t, args.T)

# Setting device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

X_train, y_train, X_test, y_test = torch.from_numpy(f_train).float().to(device), torch.from_numpy(u_train).float().to(device), torch.from_numpy(f_test).float().to(device), torch.from_numpy(u_test).float().to(device)
# X_train, y_train, X_test, y_test = torch.from_numpy(f_train).float(), torch.from_numpy(u_train).float(), torch.from_numpy(f_test).float(), torch.from_numpy(u_test).float()

y_train = y_train.reshape(args.N_train, args.grid_x ** 3 * args.grid_t, 3)
y_test = y_test.reshape(args.N_test, args.grid_x ** 3 * args.grid_t, 3)

print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

in_dim = 21

def get_model(model):
    if args.model == "FNO":
        model = FNO4d(modes1=9, modes2=9, modes3=9, modes4=16, width=8, num_layers=2, in_dim=in_dim, out_dim=3).to(device)

    elif args.model == "FNO_GRU_1":
        model = FNO_Jamba_1(input_dim=in_dim, output_dim=3, modes=9, width=16, num_layers=1, model_t_type="GRU").to(device)
    elif args.model == "FNO_LSTM_1":
        model = FNO_Jamba_1(input_dim=in_dim, output_dim=3, modes=9, width=16, num_layers=1, model_t_type="LSTM").to(device)
    elif args.model == "FNO_Mamba_1":
        model = FNO_Jamba_1(input_dim=in_dim, output_dim=3, modes=9, width=16, num_layers=1, model_t_type="Mamba").to(device)

    elif args.model == "FNO_GRU_2":
        model = FNO_Jamba_2(input_dim=in_dim, output_dim=3, modes=9, width=16, num_layers=1, model_t_type="GRU").to(device)
    elif args.model == "FNO_LSTM_2":
        model = FNO_Jamba_2(input_dim=in_dim, output_dim=3, modes=9, width=16, num_layers=1, model_t_type="LSTM").to(device)
    elif args.model == "FNO_Mamba_2":
        model = FNO_Jamba_2(input_dim=in_dim, output_dim=3, modes=9, width=16, num_layers=1, model_t_type="Mamba").to(device)

    elif args.model == "FFNO":
        model = FFNO4d(modes_x=9, modes_y=9, modes_z=9, modes_t=16, width=32, input_dim=in_dim, output_dim=3, n_layers=2).to(device)
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
        # loss = torch.linalg.vector_norm(outputs - targets) / torch.linalg.vector_norm(targets)
        loss.backward()
        optimizer.step()
    scheduler.step()

    # Print loss every epoch
    if (epoch+1)%int(100)==0: 
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

