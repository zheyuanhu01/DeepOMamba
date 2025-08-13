import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import argparse
from torch.utils.data import TensorDataset, DataLoader
import time
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import os

from PODDON_TGV import PODDON_GRU, PODDON_LSTM, PODDON_Mamba, MambaConfig

from PODDON_TGV import GalerkinTransformer, Transformer, GNOT

def get_args():
    parser = argparse.ArgumentParser(description='DeepONet Training')
    parser.add_argument('--SEED', type=int, default=0)

    parser.add_argument('--grid_x', type=int, default=28, help="x-axis grid size")
    parser.add_argument('--grid_t', type=int, default=201, help="t-axis grid size")

    parser.add_argument('--T', type=int, default=1, help="terminal time")

    parser.add_argument('--num_epochs', type=int, default=1000, help="number of training epochs")
    parser.add_argument('--batch_size', type=int, default=16, help="minibatch size for SGD")
    parser.add_argument('--lr', type=float, default=1e-3, help="learning rate")

    parser.add_argument('--model', type=str, default="FNO")

    parser.add_argument('--n_components_decode', type=int, default=256)

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


f_train, f_test, u_train, u_test = train_test_split(
    f, u, test_size=0.1, random_state=42
)

args.N_train = f_train.shape[0]
args.N_test = f_test.shape[0]

def process_u(u):
    N, grid_t = u.shape[0], u.shape[1]
    u = u.reshape(N, grid_t, -1)
    return u

u_train = process_u(u_train)
u_test = process_u(u_test)
f_train = process_u(f_train)
f_test = process_u(f_test)


print("pca for label...")
u_train_pca = u_train.reshape(args.N_train * args.grid_t, -1)
u_test_pca = u_test.reshape(args.N_test * args.grid_t, -1)
start = time.time()
pca = PCA(n_components=args.n_components_decode).fit(u_train_pca)
end = time.time()
print("PCA Time: ", end - start)
print("# Components:", pca.n_components_)
const = np.sqrt(u_train_pca.shape[-1])
POD_Basis = pca.components_.T * const
POD_Mean = pca.mean_
print("POD shapes: ", POD_Basis.shape, POD_Mean.shape)
f = pca.fit_transform(u_test_pca)
u_test_pred = pca.inverse_transform(f)
print("PCA Error: ", np.linalg.norm(u_test_pca - u_test_pred) / np.linalg.norm(u_test), np.linalg.norm(u_test_pca))

In_Basis, In_Mean = np.ones((1, 1)), np.zeros((1,))

# Setting device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

X_train, Y_train, X_test, Y_test = torch.from_numpy(f_train).float().to(device), torch.from_numpy(u_train).float().to(device), torch.from_numpy(f_test).float().to(device), torch.from_numpy(u_test).float().to(device)
print("Data Tensor Shapes: ", X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)

POD_Basis = torch.from_numpy(POD_Basis).float().to(device)
POD_Mean = torch.from_numpy(POD_Mean).float().to(device)
In_Basis = torch.from_numpy(In_Basis).float().to(device)
In_Mean = torch.from_numpy(In_Mean).float().to(device)

def get_model(model):
    if args.model == "GRU":
        model = PODDON_GRU(input_dim=1, output_dim=args.n_components_decode, hidden_dim=256, num_layers=2, POD_Basis=POD_Basis, POD_Mean=POD_Mean, In_Basis=In_Basis, In_Mean=In_Mean).to(device)
    elif args.model == "LSTM":
        model = PODDON_LSTM(input_dim=1, output_dim=args.n_components_decode, hidden_dim=256, num_layers=2, POD_Basis=POD_Basis, POD_Mean=POD_Mean, In_Basis=In_Basis, In_Mean=In_Mean).to(device)
    elif args.model == "Mamba":
        config = MambaConfig(d_model=256,n_layer=2,vocab_size=0,ssm_cfg=dict(layer="Mamba1"),rms_norm=True,residual_in_fp32=True,fused_add_norm=True)
        model = PODDON_Mamba(256, 2, 0, 1, args.n_components_decode, POD_Basis, POD_Mean, In_Basis, In_Mean, config.ssm_cfg).to(device)
    elif args.model == "ST":
        model = GalerkinTransformer(dim_in=1, dim_out=args.n_components_decode, dim_hid=256, depth=1, heads=4, dim_head=256, attn_type='standard', mlp_dim=256, POD_Basis=POD_Basis, POD_Mean=POD_Mean, In_Basis=In_Basis, In_Mean=In_Mean).to(device)
    elif args.model == "GT":
        model = GalerkinTransformer(dim_in=1, dim_out=args.n_components_decode, dim_hid=256, depth=1, heads=4, dim_head=256, attn_type='galerkin', mlp_dim=256, POD_Basis=POD_Basis, POD_Mean=POD_Mean, In_Basis=In_Basis, In_Mean=In_Mean).to(device)
    elif args.model == "FT":
        model = GalerkinTransformer(dim_in=1, dim_out=args.n_components_decode, dim_hid=256, depth=1, heads=4, dim_head=256, attn_type='fourier', mlp_dim=256, POD_Basis=POD_Basis, POD_Mean=POD_Mean, In_Basis=In_Basis, In_Mean=In_Mean).to(device)
    elif args.model == "T":
        model = Transformer(ninput=1, noutput=args.n_components_decode, nhidden=256, dim_feedforward=256, nhead=4, nlayers=1, POD_Basis=POD_Basis, POD_Mean=POD_Mean, In_Basis=In_Basis, In_Mean=In_Mean).to(device)
    elif args.model == "GNOT":
        model = GNOT(dim_in=1, dim_out=args.n_components_decode, dim_hid=256, depth=1, heads=4, dim_head=256, n_experts=2, POD_Basis=POD_Basis, POD_Mean=POD_Mean, In_Basis=In_Basis, In_Mean=In_Mean).to(device)

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

train_dataset = TensorDataset(X_train, Y_train)
train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)

test_dataset = TensorDataset(X_test, Y_test)
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

