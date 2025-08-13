import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import argparse
from torch.utils.data import TensorDataset, DataLoader
import time
from sklearn.decomposition import PCA

from PODDON_TGV import PODDON_GRU, PODDON_LSTM, PODDON_Mamba, MambaConfig

from PODDON_TGV import GalerkinTransformer, Transformer, GNOT

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

    parser.add_argument('--model', type=str, default="GRU")

    parser.add_argument('--n_components_decode', type=int, default=128)
    parser.add_argument('--n_components_encode', type=int, default=128)

    args = parser.parse_args()
    print(args)

    return args

args = get_args()

SEED = args.SEED
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)

print("Generating Data in FOM...")
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

    U_init = U[:, :, :, :, 0, :] # N, grid_x, grid_x, grid_x, 3
    U_init = np.expand_dims(U_init, 1) # N, 1, grid_x, grid_x, grid_x, 3
    U_init = np.tile(U_init, (1, grid_t, 1, 1, 1, 1)) # N, grid_t, grid_x, grid_x, grid_x, 3
    U_init = U_init.reshape(N, grid_t, -1)

    U_x0, U_x1 = U[:, 0, :, :, :, :], U[:, -1, :, :, :, :] # N, grid_x, grid_x, grid_t, 3
    U_x0, U_x1 = np.transpose(U_x0, (0, 3, 1, 2, 4)), np.transpose(U_x1, (0, 3, 1, 2, 4))
    U_x0, U_x1 = U_x0.reshape(N, grid_t, -1), U_x1.reshape(N, grid_t, -1)

    U_y0, U_y1 = U[:, :, 0, :, :, :], U[:, :, -1, :, :, :] # N, grid_x, grid_x, grid_t, 3
    U_y0, U_y1 = np.transpose(U_y0, (0, 3, 1, 2, 4)), np.transpose(U_y1, (0, 3, 1, 2, 4))
    U_y0, U_y1 = U_y0.reshape(N, grid_t, -1), U_y1.reshape(N, grid_t, -1)

    U_z0, U_z1 = U[:, :, :, 0, :, :], U[:, :, :, -1, :, :] # N, grid_x, grid_x, 1, grid_t, 3
    U_z0, U_z1 = np.transpose(U_z0, (0, 3, 1, 2, 4)), np.transpose(U_z1, (0, 3, 1, 2, 4))
    U_z0, U_z1 = U_z0.reshape(N, grid_t, -1), U_z1.reshape(N, grid_t, -1)

    F = np.concatenate([U_init, U_x0, U_x1, U_y0, U_y1, U_z0, U_z1], -1) # N, grid_t, -1


    U = np.transpose(U, (0, 4, 1, 2, 3, 5)) # N, grid_t, grid_x, grid_x, grid_x, 3
    U = U.reshape(N, grid_t, -1)

    return U, F

# u_train, f_train = generate_data(args.N_train, args.grid_x, args.grid_t, args.T)
# u_test, f_test = generate_data(args.N_test, args.grid_x, args.grid_t, args.T)

u_train, f_train = [], []
for i in tqdm(range(args.N_train // 10)):
    u_, f_ = generate_data(10, args.grid_x, args.grid_t, args.T)
    u_train.append(u_)
    f_train.append(f_)
u_train = np.concatenate(u_train, 0)
f_train = np.concatenate(f_train, 0)

u_test, f_test = [], []
for i in tqdm(range(args.N_test // 10)):
    u_, f_ = generate_data(10, args.grid_x, args.grid_t, args.T)
    u_test.append(u_)
    f_test.append(f_)
u_test = np.concatenate(u_test, 0)
f_test = np.concatenate(f_test, 0)

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


print("pca for input...")
f_train_pca = f_train.reshape(args.N_train * args.grid_t, -1)
f_test_pca = f_test.reshape(args.N_test * args.grid_t, -1)
start = time.time()
pca = PCA(n_components=args.n_components_encode).fit(f_train_pca)
end = time.time()
print("PCA Time: ", end - start)
print("# Components:", pca.n_components_)
const = np.sqrt(f_train_pca.shape[-1])
In_Basis = pca.components_.T * const
In_Mean = pca.mean_
print("InPOD shapes: ", In_Basis.shape, In_Mean.shape)
f = pca.fit_transform(f_test_pca)
f_test_pred = pca.inverse_transform(f)
print("PCA Error: ", np.linalg.norm(f_test_pca - f_test_pred) / np.linalg.norm(f_test), np.linalg.norm(f_test_pca))


# Setting device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

X_train, Y_train, X_test, Y_test = torch.from_numpy(f_train).float().to(device), torch.from_numpy(u_train).float().to(device), torch.from_numpy(f_test).float().to(device), torch.from_numpy(u_test).float().to(device)
print("Data Tensor Shapes: ", X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)

POD_Basis = torch.from_numpy(POD_Basis).float().to(device)
POD_Mean = torch.from_numpy(POD_Mean).float().to(device)
In_Basis = torch.from_numpy(In_Basis).float().to(device)
In_Mean = torch.from_numpy(In_Mean).float().to(device)

in_dim = X_train.shape[-1]

def get_model(model):
    if args.model == "GRU":
        model = PODDON_GRU(input_dim=args.n_components_encode, output_dim=args.n_components_decode, hidden_dim=256, num_layers=1, POD_Basis=POD_Basis, POD_Mean=POD_Mean, In_Basis=In_Basis, In_Mean=In_Mean).to(device)
    elif args.model == "LSTM":
        model = PODDON_LSTM(input_dim=args.n_components_encode, output_dim=args.n_components_decode, hidden_dim=256, num_layers=1, POD_Basis=POD_Basis, POD_Mean=POD_Mean, In_Basis=In_Basis, In_Mean=In_Mean).to(device)
    elif args.model == "Mamba":
        config = MambaConfig(d_model=256,n_layer=1,vocab_size=0,ssm_cfg=dict(layer="Mamba1"),rms_norm=True,residual_in_fp32=True,fused_add_norm=True)
        model = PODDON_Mamba(256, 1, 0, args.n_components_encode, args.n_components_decode, POD_Basis, POD_Mean, In_Basis, In_Mean, config.ssm_cfg).to(device)
    elif args.model == "ST":
        model = GalerkinTransformer(dim_in=args.n_components_encode, dim_out=args.n_components_decode, dim_hid=256, depth=1, heads=4, dim_head=256, attn_type='standard', mlp_dim=256, POD_Basis=POD_Basis, POD_Mean=POD_Mean, In_Basis=In_Basis, In_Mean=In_Mean).to(device)
    elif args.model == "GT":
        model = GalerkinTransformer(dim_in=args.n_components_encode, dim_out=args.n_components_decode, dim_hid=256, depth=1, heads=4, dim_head=256, attn_type='galerkin', mlp_dim=256, POD_Basis=POD_Basis, POD_Mean=POD_Mean, In_Basis=In_Basis, In_Mean=In_Mean).to(device)
    elif args.model == "FT":
        model = GalerkinTransformer(dim_in=args.n_components_encode, dim_out=args.n_components_decode, dim_hid=256, depth=1, heads=4, dim_head=256, attn_type='fourier', mlp_dim=256, POD_Basis=POD_Basis, POD_Mean=POD_Mean, In_Basis=In_Basis, In_Mean=In_Mean).to(device)
    elif args.model == "T":
        model = Transformer(ninput=args.n_components_encode, noutput=args.n_components_decode, nhidden=256, dim_feedforward=256, nhead=4, nlayers=1, POD_Basis=POD_Basis, POD_Mean=POD_Mean, In_Basis=In_Basis, In_Mean=In_Mean).to(device)
    elif args.model == "GNOT":
        model = GNOT(dim_in=args.n_components_encode, dim_out=args.n_components_decode, dim_hid=256, depth=1, heads=4, dim_head=256, n_experts=2, POD_Basis=POD_Basis, POD_Mean=POD_Mean, In_Basis=In_Basis, In_Mean=In_Mean).to(device)

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

