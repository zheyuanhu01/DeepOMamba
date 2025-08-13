import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import argparse
from torch.utils.data import TensorDataset, DataLoader
from FNO2d import FNO2d
from FNO2d_Jamba import FNO_Jamba_1, FNO_Jamba_2
from FFNO2d import FFNO
from DON_2d import POD_GRU, POD_LSTM, POD_Mamba, MambaConfig
from DON_2d import POD_GalerkinTransformer, POD_Transformer, POD_GNOT

parser = argparse.ArgumentParser(description='DeepONet Training')
parser.add_argument('--SEED', type=int, default=0)

parser.add_argument('--grid_x', type=int, default=100, help="x-axis grid size")
parser.add_argument('--grid_t', type=int, default=100, help="t-axis grid size")

parser.add_argument('--L', type=float, default=5, help="domain size")
parser.add_argument('--T', type=float, default=5, help="terminal time")

parser.add_argument('--N_train', type=int, default=27000)
parser.add_argument('--N_test', type=int, default=3000)

parser.add_argument('--num_epochs', type=int, default=100, help="number of training epochs")
parser.add_argument('--batch_size', type=int, default=16, help="minibatch size for SGD")
parser.add_argument('--lr', type=float, default=1e-3, help="learning rate")

parser.add_argument('--model', type=str, default="GRU")

parser.add_argument('--FNO_hidden_dim', type=int, default=64)
parser.add_argument('--FNO_num_layers', type=int, default=1)
parser.add_argument('--FNO_modes_x', type=int, default=32)
parser.add_argument('--FNO_modes_t', type=int, default=16)

parser.add_argument('--LSTM_hidden_dim', type=int, default=256)
parser.add_argument('--LSTM_num_layers', type=int, default=1)

parser.add_argument('--GRU_hidden_dim', type=int, default=256)
parser.add_argument('--GRU_num_layers', type=int, default=1)

parser.add_argument('--MambaLLM_d_model', type=int, default=256)
parser.add_argument('--MambaLLM_n_layer', type=int, default=1)

parser.add_argument('--data2cuda', type=int, default=1)

parser.add_argument('--save_loss', type=int, default=0)
parser.add_argument('--save_model', type=int, default=0)

args = parser.parse_args()
print(args)

SEED = args.SEED
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)

def generate_data_1(grid_x=1000, grid_t=100, L=5, T=5):

    def u(k, A, alpha, beta, delta, b1, b0, b, x, t):
        inside_sec = k * (x - delta * k * k * t) - (b * torch.arctan(A * t) + b1 * t + b0)
        return 12 * beta * (k ** 2) / (torch.cosh(inside_sec) ** 2)

    def f(k, A, alpha, beta, delta, b1, b0, b, x, t):
        temp1 = 12 * k * beta / alpha
        temp2 = (k**3) * (4 * beta - delta) - b * A / (1 + A**2 * t**2) - b1
        inside_sec = k * (x - delta * k * k * t) - (b * torch.arctan(A * t) + b1 * t + b0)
        return temp1 * temp2 / (torch.cosh(inside_sec) ** 2)


    # Prepare grid points

    x = np.linspace(-L, L, grid_x)
    t = np.linspace(0, T, grid_t)
    x, t = np.meshgrid(x, t, indexing='ij') # [grid_x, grid_t]
    x, t = x.reshape(-1), t.reshape(-1)
    x, t = torch.from_numpy(x).float(), torch.from_numpy(t).float()
    t0 = torch.zeros(grid_x * grid_t)
    x1 = torch.zeros(grid_x * grid_t) + L
    x2 = torch.zeros(grid_x * grid_t) - L

    # PDE parameters

    # k = np.random.rand() * 0.5 + 0.001 # [0.5, 1.5]
    # A = np.random.rand() + 2.5 # [2.5, 3.5]
    # alpha = np.random.rand() + 1.5 # [1.5, 2.5]
    # beta = (np.random.rand() * 0.25 + 0.25) / k ** 2 # [0.25, 0.5]
    # b1 = (np.random.rand() - 0.5) * 10 * k # [-3, 3]
    # b0 = (np.random.rand() - 0.5) * 50 * k # [-25, 25]
    # b = (np.random.rand() - 0.5) * 10 * k # [-2.5, 2.5]
    # delta = (np.random.rand() - 0.5) * 8 / k ** 3 # [-4, 4]

    k = np.random.rand() + 0.5 # [0.5, 1.5]
    A = np.random.rand() + 2.5 # [2.5, 3.5]
    alpha = np.random.rand() + 1.5 # [1.5, 2.5]
    beta = np.random.rand() * 0.25 + 0.125 # [0.125, 0.375]
    b1 = (np.random.rand() - 0.5) * 6 # [-3, 3]
    b0 = (np.random.rand() - 0.5) * 2 # [-1, 1]
    b = (np.random.rand() - 0.5) * 2 # [-1, 1]
    delta = (np.random.rand() - 0.5) * 8 # [-4, 4]

    x.requires_grad_()
    ff = lambda x, t: f(k, A, alpha, beta, delta, b1, b0, b, x, t)
    out_f = ff(x, t)
    # f_x = torch.autograd.grad(out_f.sum(), x, create_graph=True)[0] # grid_x \times grid_t

    uu = lambda x, t: u(k, A, alpha, beta, delta, b1, b0, b, x, t)
    u_sol = uu(x, t)
    initial = uu(x, t0)

    boundary_1 = uu(x1, t)

    x2.requires_grad_()
    boundary_2 = uu(x2, t)
    boundary_3 = torch.autograd.grad(boundary_2.sum(), x2, create_graph=True)[0]

    f_x = torch.stack([out_f, initial, boundary_1, boundary_2, boundary_3], -1) # grid_x \times grid_t
    f_x = f_x.reshape(grid_x, grid_t, 5)
    u_sol = u_sol.reshape(grid_x * grid_t, 1)

    alpha = torch.zeros(grid_x, grid_t, 1) + alpha
    beta = torch.zeros(grid_x, grid_t, 1) + beta

    f_x = torch.cat([f_x, alpha, beta], -1) # grid_x, grid_t, 7

    f_x = f_x.detach().to("cpu")
    u_sol = u_sol.detach().to("cpu")

    return f_x, u_sol

def generate_data_2(grid_x=1000, grid_t=100, L=5, T=5):

    def u(k, A, alpha, beta, delta, b2, b1, b0, x, t):
        inside_sec = k * (x - delta * k * k * t) - torch.exp(b2 * t**2 + b1 * t + b0)
        return 12 * beta * (k ** 2) / (torch.cosh(inside_sec) ** 2)

    def f(k, A, alpha, beta, delta, b2, b1, b0, x, t):
        temp1 = 12 * k * beta / alpha
        temp2 = (k**3) * (4 * beta - delta) - (2 * b2 * t + b1) * torch.exp(b2 * t**2 + b1 * t + b0)
        inside_sec = k * (x - delta * k * k * t) - torch.exp(b2 * t**2 + b1 * t + b0)
        return temp1 * temp2 / (torch.cosh(inside_sec) ** 2)

    # Prepare grid points

    x = np.linspace(-L, L, grid_x)
    t = np.linspace(0, T, grid_t)
    x, t = np.meshgrid(x, t, indexing='ij') # [grid_x, grid_t]
    x, t = x.reshape(-1), t.reshape(-1)
    x, t = torch.from_numpy(x).float(), torch.from_numpy(t).float()
    t0 = torch.zeros(grid_x * grid_t)
    x1 = torch.zeros(grid_x * grid_t) + L
    x2 = torch.zeros(grid_x * grid_t) - L

    # PDE parameters

    k = np.random.rand() + 0.5 # [0.5, 1.5]
    A = np.random.rand() + 2.5 # [2.5, 3.5]
    alpha = np.random.rand() + 1.5 # [1.5, 2.5]
    beta = np.random.rand() * 0.25 + 0.125 # [0.125, 0.375]
    b2 = (np.random.rand() - 2) # [-2, -1]
    b1 = (np.random.rand() - 0.5) * 2 # [-1, 1]
    b0 = (np.random.rand() - 0.5) * 2 # [-1, 1]
    delta = (np.random.rand() - 0.5) * 8 # [-4, 4]

    # k = np.random.rand() * 0.5 + 0.001 # [0.5, 1.5]
    # A = np.random.rand() + 2.5 # [2.5, 3.5]
    # alpha = (np.random.rand() + 1.5) / k * 24 # [1.5, 2.5]
    # beta = (np.random.rand() * 0.25 + 0.25) / k ** 2 # [0.25, 0.5]
    # b2 = (np.random.rand() - 1) * 2 # [-2, 0]
    # b1 = (np.random.rand() - 0.5) * 6 # [-3, 3]
    # b0 = (np.random.rand() - 0.5) * 2 # [-1, 1]
    # delta = (np.random.rand() - 0.5) * 8 / k ** 3 # [-4, 4]

    x.requires_grad_()
    ff = lambda x, t: f(k, A, alpha, beta, delta, b2, b1, b0, x, t)
    out_f = ff(x, t)
    # f_x = torch.autograd.grad(out_f.sum(), x, create_graph=True)[0] # grid_x \times grid_t

    uu = lambda x, t: u(k, A, alpha, beta, delta, b2, b1, b0, x, t)
    u_sol = uu(x, t)
    initial = uu(x, t0)

    boundary_1 = uu(x1, t)

    x2.requires_grad_()
    boundary_2 = uu(x2, t)
    boundary_3 = torch.autograd.grad(boundary_2.sum(), x2, create_graph=True)[0]

    f_x = torch.stack([out_f, initial, boundary_1, boundary_2, boundary_3], -1) # grid_x \times grid_t
    f_x = f_x.reshape(grid_x, grid_t, 5)
    u_sol = u_sol.reshape(grid_x * grid_t, 1)

    alpha = torch.zeros(grid_x, grid_t, 1) + alpha
    beta = torch.zeros(grid_x, grid_t, 1) + beta

    f_x = torch.cat([f_x, alpha, beta], -1) # grid_x, grid_t, 7

    f_x = f_x.detach().to('cpu')
    u_sol = u_sol.detach().to('cpu')

    return f_x, u_sol

def generate_data_3(grid_x=1000, grid_t=100, L=5, T=5):

    def F(a, b, d, x, t):
        return (x + a) ** 2 + (t + b) ** 2 + d
    
    def H(a, b, d, x, t):
        return (x + a) / torch.sqrt((t + b) ** 2 + d)
    
    def u(a, b, d, beta, gamma, x, t):
        return 12 * beta * gamma / F(a, b, d, x, t)
    
    def f(a, b, d, alpha, beta, gamma, x, t):
        const = 12 * beta * gamma / alpha
        t1 = 6 * beta * (gamma + 1) / (F(a, b, d, x, t) ** 2)
        t2 = 8 * beta * ((t + b) ** 2 + d) / (F(a, b, d, x, t) ** 3)
        t3 = (t + b) * (x + a) / ((t + b) ** 2 + d) / F(a, b, d, x, t)
        t4 = (t + b) / (torch.sqrt((t + b) ** 2 + d) ** 3) * torch.arctan(H(a, b, d, x, t))
        return const * (t1 - t2 - t3 - t4)

    # Prepare grid points

    x = np.linspace(-L, L, grid_x)
    t = np.linspace(0, T, grid_t)
    x, t = np.meshgrid(x, t, indexing='ij') # [grid_x, grid_t]
    x, t = x.reshape(-1), t.reshape(-1)
    x, t = torch.from_numpy(x).float(), torch.from_numpy(t).float()
    t0 = torch.zeros(grid_x * grid_t)
    x1 = torch.zeros(grid_x * grid_t) + L
    x2 = torch.zeros(grid_x * grid_t) - L

    # PDE parameters

    # a = (np.random.rand() - 0.5) * 100 # [-50, 50]
    # alpha = np.random.rand() + 1.5 # [1.5, 2.5]
    # beta = np.random.rand() * 0.25 + 0.125 # [0.125, 0.375]
    # b = (np.random.rand() - 0.5) * 8  # [-4, 4]
    # d = np.random.rand() * 3 + 1 # [1, 4]
    # gamma = (np.random.rand() + 0.5) * 1 # [0.5, 1.5]
    k = np.random.rand() + 0.5 # [0.5, 1.5]
    a = (np.random.rand() - 0.5) * 6 # [-3, 3]
    alpha = np.random.rand() + 1.5 # [1.5, 2.5]
    beta = np.random.rand() * 0.25 + 0.125 # [0.125, 0.375]
    b = (np.random.rand() - 0.5) * 6 - 1 # [-4, 2]
    d = np.random.rand() * 0.5 + 0.3 # [1, 4]
    gamma = (np.random.rand() + 0.5) * 1 # [0.5, 1.5]

    x.requires_grad_()
    ff = lambda x, t: f(a, b, d, alpha, beta, gamma, x, t)
    out_f = ff(x, t)
    # f_x = torch.autograd.grad(out_f.sum(), x, create_graph=True)[0] # grid_x \times grid_t

    uu = lambda x, t: u(a, b, d, beta, gamma, x, t)
    u_sol = uu(x, t)
    initial = uu(x, t0)

    boundary_1 = uu(x1, t)

    x2.requires_grad_()
    boundary_2 = uu(x2, t)
    boundary_3 = torch.autograd.grad(boundary_2.sum(), x2, create_graph=True)[0]

    f_x = torch.stack([out_f, initial, boundary_1, boundary_2, boundary_3], -1) # grid_x \times grid_t
    f_x = f_x.reshape(grid_x, grid_t, 5)
    u_sol = u_sol.reshape(grid_x * grid_t, 1)

    alpha = torch.zeros(grid_x, grid_t, 1) + alpha
    beta = torch.zeros(grid_x, grid_t, 1) + beta

    f_x = torch.cat([f_x, alpha, beta], -1) # grid_x, grid_t, 7

    f_x = f_x.detach().to('cpu')
    u_sol = u_sol.detach().to('cpu')
    return f_x, u_sol

X_train, y_train = [], []
for i in tqdm(range(args.N_train // 3)):
    x, y = generate_data_1(args.grid_x, args.grid_t, args.L, args.T)
    X_train.append(x)
    y_train.append(y)
    x, y = generate_data_2(args.grid_x, args.grid_t, args.L, args.T)
    X_train.append(x)
    y_train.append(y)
    x, y = generate_data_3(args.grid_x, args.grid_t, args.L, args.T)
    X_train.append(x)
    y_train.append(y)
X_train = torch.stack(X_train, 0)
y_train = torch.stack(y_train, 0)
print(X_train.shape, y_train.shape)

print(X_train.isnan().any(), y_train.isnan().any())

X_test, y_test = [], []
for i in tqdm(range(args.N_test // 3)):
    x, y = generate_data_1(args.grid_x, args.grid_t, args.T)
    X_test.append(x)
    y_test.append(y)
    x, y = generate_data_2(args.grid_x, args.grid_t, args.T)
    X_test.append(x)
    y_test.append(y)
    x, y = generate_data_3(args.grid_x, args.grid_t, args.T)
    X_test.append(x)
    y_test.append(y)
X_test = torch.stack(X_test, 0)
y_test = torch.stack(y_test, 0)
print(X_test.shape, y_test.shape)

# Setting device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if args.data2cuda:
    X_train, y_train, X_test, y_test = X_train.to(device), y_train.to(device), X_test.to(device), y_test.to(device)

x = np.linspace(-args.L, args.L, args.grid_x)
t = np.linspace(0, args.T, args.grid_t)
x, t = np.meshgrid(x, t, indexing='ij') # [grid_x, grid_t]
x, t = torch.from_numpy(x.reshape(-1)).float().to(device), torch.from_numpy(t.reshape(-1)).float().to(device)

in_dim = 7

if args.model == "FNO":
    model = FNO2d(modes1=32, modes2=32, width=24, num_layers=2, in_dim=in_dim, out_dim=1).to(device)

elif args.model == "FFNO":
    model = FFNO(modes=32, width=96, input_dim=in_dim, output_dim=1, n_layers=2).to(device)

elif args.model == "FNO_GRU_1":
    model = FNO_Jamba_1(input_dim=in_dim, output_dim=1, modes=32, width=128, num_layers=1, model_t_type="GRU").to(device)
elif args.model == "FNO_GRU_2":
    model = FNO_Jamba_2(input_dim=in_dim, output_dim=1, modes=32, width=128, num_layers=1, model_t_type="GRU").to(device)

elif args.model == "FNO_LSTM_1":
    model = FNO_Jamba_1(input_dim=in_dim, output_dim=1, modes=32, width=128, num_layers=1, model_t_type="LSTM").to(device)
elif args.model == "FNO_LSTM_2":
    model = FNO_Jamba_2(input_dim=in_dim, output_dim=1, modes=32, width=128, num_layers=1, model_t_type="LSTM").to(device)

elif args.model == "FNO_Mamba_1":
    model = FNO_Jamba_1(input_dim=in_dim, output_dim=1, modes=32, width=128, num_layers=1, model_t_type="Mamba").to(device)
elif args.model == "FNO_Mamba_2":
    model = FNO_Jamba_2(input_dim=in_dim, output_dim=1, modes=32, width=128, num_layers=1, model_t_type="Mamba").to(device)

elif args.model == "GRU":
    model = POD_GRU(input_dim=in_dim * args.grid_x, output_dim=args.grid_x, hidden_dim=256, num_layers=1).to(device)
elif args.model == "LSTM":
    model = POD_LSTM(input_dim=in_dim * args.grid_x, output_dim=args.grid_x, hidden_dim=256, num_layers=1).to(device)
elif args.model == "Mamba":
    config = MambaConfig(
            d_model=256,
            n_layer=1,
            vocab_size=0,
            ssm_cfg=dict(layer="Mamba1"),
            rms_norm=True,
            residual_in_fp32=True,
            fused_add_norm=True
        )
    model = POD_Mamba(256, 1, 0, in_dim * args.grid_x, args.grid_x, config.ssm_cfg).to(device)
elif args.model == "GT":
    model = POD_GalerkinTransformer(dim_in=in_dim * args.grid_x, dim_out=args.grid_x, dim_hid=256, depth=1, heads=4, dim_head=256, attn_type='galerkin', mlp_dim=256).to(device)
elif args.model == "ST":
    model = POD_GalerkinTransformer(dim_in=in_dim * args.grid_x, dim_out=args.grid_x, dim_hid=256, depth=1, heads=4, dim_head=256, attn_type='standard', mlp_dim=256).to(device)
elif args.model == "T":
    model = POD_Transformer(ninput=in_dim * args.grid_x, noutput=args.grid_x, nhidden=256, dim_feedforward=256, nhead=4, nlayers=1).to(device)
elif args.model == "GNOT":
    model = POD_GNOT(dim_in=in_dim * args.grid_x, dim_out=args.grid_x, dim_hid=256, depth=1, heads=4, dim_head=256, n_experts=2).to(device)

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

loss_traj = []
# Training loop
for epoch in tqdm(range(args.num_epochs)):
    current_loss = []
    # Forward pass
    for batch_idx, (data, targets) in enumerate(train_loader):
        model.train()  # Set the model to training mode
        optimizer.zero_grad()  # Clear gradients
        data, targets = data.to(device), targets.to(device)
        outputs = model(data)
        loss = torch.linalg.vector_norm(outputs - targets) / torch.linalg.vector_norm(targets)  # Target is the same as input
        loss.backward()
        optimizer.step()
        current_loss.append(loss.item())
    
    current_loss = sum(current_loss) / len(current_loss)
    loss_traj.append(current_loss)
    scheduler.step()

    # Print loss every epoch
    if (epoch+1)%int(10)==0 or epoch == args.num_epochs - 1:
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
        # print(f"Epoch {epoch+1}, Train Loss: {loss.item()}, Train Rel L2: {error_train.item()}, Test Rel L2: {error_test.item()}")

if args.save_loss:
    loss_traj = np.asarray(loss_traj)
    filename = "Data/fKdV_" + \
        "Model=" + str(args.model) + "_" + \
        "Seed=" + str(args.SEED) + \
        ".txt"
    np.savetxt(filename, loss_traj)

if args.save_model:
    import os
    print("Saving model...")
    PATH = "fKdV"
    if not os.path.isdir(PATH): os.mkdir(PATH)  
    torch.save(model.state_dict(), PATH + "/" + args.model + "_Model.pth")