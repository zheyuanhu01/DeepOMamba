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
from burgers_data import AntideData, AntideAntideData

parser = argparse.ArgumentParser(description='DeepONet Training')
parser.add_argument('--SEED', type=int, default=0)

parser.add_argument('--T', type=int, default=1, help="")

parser.add_argument('--grid_x', type=int, default=100, help="x-axis grid size")
parser.add_argument('--grid_t', type=int, default=100, help="t-axis grid size")

parser.add_argument('--N_train', type=int, default=27000)
parser.add_argument('--N_test', type=int, default=3000)

parser.add_argument('--num_epochs', type=int, default=100, help="number of training epochs")
parser.add_argument('--batch_size', type=int, default=16, help="minibatch size for SGD")
parser.add_argument('--lr', type=float, default=1e-3, help="learning rate")

parser.add_argument('--model', type=str, default="FNO")

parser.add_argument('--save_loss', type=int, default=0)
parser.add_argument('--save_model', type=int, default=0)

args = parser.parse_args()
print(args)

SEED = args.SEED
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)

def generate_t(grid_t, T):
    # assert grid_t // 100 == T
    # data
    s0 = [0]
    sensor_in = grid_t
    sensor_out = grid_t
    length_scale = 0.2
    train_num = args.N_train // 4
    test_num = args.N_test // 4

    np.random.seed(args.SEED)
    data = AntideData(T, s0, sensor_in, sensor_out, length_scale, train_num, test_num)
    g_3_train = data.X_train
    g_2_train = data.y_train
    g_3_test = data.X_test
    g_2_test = data.y_test

    np.random.seed(args.SEED)
    s0 = [0, 0]
    data = AntideAntideData(T, s0, sensor_in, sensor_out, length_scale, train_num, test_num)
    g_1_train = data.y_train
    g_1_test = data.y_test

    return g_1_train, g_1_test, g_2_train, g_2_test, g_3_train, g_3_test

g_1_train, g_1_test, g_2_train, g_2_test, g_3_train, g_3_test = generate_t(args.grid_t, args.T)

def generate_data(grid_x, grid_t, g_1, g_2, g_3):
    g_1 = g_1.T # [1, grid_t]
    g_2 = g_2.T # [1, grid_t]
    g_3 = g_3.T # [1, grid_t]

    x = np.linspace(-5, 5, grid_x)
    t = np.linspace(0.5, 0.5 + args.T, grid_t)
    x, t = np.meshgrid(x, t, indexing='ij') # [grid_x, grid_t]

    c1 = np.random.rand() * 3
    c2 = np.random.rand() * 6 - 3

    def u1_sol(x, t):
        return x / t - c1 / t - (g_1 - t * g_2) / t
    
    u0 = np.tile(g_2, (grid_x, 1))
    u0_init = u0[:, 0:1] # [grid_x, 1]
    u0_boundary_1 = u0[0:1, :] # [1, grid_t]
    u0_boundary_2 = u0[-1:, :] # [1, grid_t]

    u0_init = np.tile(u0_init, (1, grid_t)) # [grid_x, grid_t]
    u0_boundary_1 = np.tile(u0_boundary_1, (grid_x, 1))
    u0_boundary_2 = np.tile(u0_boundary_2, (grid_x, 1))

    u1 = u1_sol(x, t)
    u1_init = u1[:, 0:1] # [grid_x, 1]
    u1_boundary_1 = u1[0:1, :] # [1, grid_t]
    u1_boundary_2 = u1[-1:, :] # [1, grid_t]

    u1_init = np.tile(u1_init, (1, grid_t)) # [grid_x, grid_t]
    u1_boundary_1 = np.tile(u1_boundary_1, (grid_x, 1))
    u1_boundary_2 = np.tile(u1_boundary_2, (grid_x, 1))
    
    const =  np.sqrt(2 * c1)
    def u2_sol(x, t):
        return - const * np.tanh(0.5 * const * (g_1 - x + c2)) + g_2
    u2 = u2_sol(x, t)
    u2_init = u2[:, 0:1] # [grid_x, 1]
    u2_boundary_1 = u2[0:1, :] # [1, grid_t]
    u2_boundary_2 = u2[-1:, :] # [1, grid_t]

    u2_init = np.tile(u2_init, (1, grid_t)) # [grid_x, grid_t]
    u2_boundary_1 = np.tile(u2_boundary_1, (grid_x, 1))
    u2_boundary_2 = np.tile(u2_boundary_2, (grid_x, 1))

    const =  np.sqrt(2 * c1)
    def u3_sol(x, t):
        return const / t * np.tanh(0.5 * const * ((x - g_1) / t + c2)) + (x - g_1) / t + g_2
    u3 = u3_sol(x, t)
    u3_init = u3[:, 0:1] # [grid_x, 1]
    u3_boundary_1 = u3[0:1, :] # [1, grid_t]
    u3_boundary_2 = u3[-1:, :] # [1, grid_t]

    u3_init = np.tile(u3_init, (1, grid_t)) # [grid_x, grid_t]
    u3_boundary_1 = np.tile(u3_boundary_1, (grid_x, 1))
    u3_boundary_2 = np.tile(u3_boundary_2, (grid_x, 1))

    f = np.tile(g_3, (grid_x, 1))

    f0 = np.stack([f, u0_init, u0_boundary_1, u0_boundary_2], -1) # N_xN_t, 4
    f0 = f0.reshape(grid_x, grid_t, 4) # N_x, N_t, 4

    f1 = np.stack([f, u1_init, u1_boundary_1, u1_boundary_2], -1) # N_xN_t, 4
    f1 = f1.reshape(grid_x, grid_t, 4) # N_x, N_t, 4

    f2 = np.stack([f, u2_init, u2_boundary_1, u2_boundary_2], -1) # N_xN_t, 4
    f2 = f2.reshape(grid_x, grid_t, 4) # N_x, N_t, 4

    f3 = np.stack([f, u3_init, u3_boundary_1, u3_boundary_2], -1) # N_xN_t, 4
    f3 = f3.reshape(grid_x, grid_t, 4) # N_x, N_t, 4

    f = np.stack([f0, f1, f2, f3], 0)
    u = np.stack([u0, u1, u2, u3], 0)
    u = u.reshape(4, -1, 1)

    return f, u

X_train, y_train = [], []
for i in tqdm(range(args.N_train // 4)):
    x, y = generate_data(args.grid_x, args.grid_t, g_1_train[i], g_2_train[i], g_3_train[i])
    X_train.append(x)
    y_train.append(y)
X_train = np.concatenate(X_train, 0)
y_train = np.concatenate(y_train, 0)
print(X_train.shape, y_train.shape)

X_test, y_test = [], []
for i in tqdm(range(args.N_test // 4)):
    x, y = generate_data(args.grid_x, args.grid_t, g_1_test[i], g_2_test[i], g_3_test[i])
    X_test.append(x)
    y_test.append(y)
X_test = np.concatenate(X_test, 0)
y_test = np.concatenate(y_test, 0)
print(X_test.shape, y_test.shape)

# Setting device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

X_train, y_train, X_test, y_test = torch.from_numpy(X_train).float().to(device), torch.from_numpy(y_train).float().to(device), torch.from_numpy(X_test).float().to(device), torch.from_numpy(y_test).float().to(device)

in_dim = 4

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
    model = POD_Mamba(256, 1, 0, \
                     in_dim * args.grid_x, args.grid_x, config.ssm_cfg).to(device)
elif args.model == "GT":
    model = POD_GalerkinTransformer(dim_in=in_dim * args.grid_x, dim_out=args.grid_x, dim_hid=256, depth=1, heads=4, dim_head=256, attn_type='galerkin', mlp_dim=256).to(device)
elif args.model == "ST":
    model = POD_GalerkinTransformer(dim_in=in_dim * args.grid_x, dim_out=args.grid_x, dim_hid=256, depth=1, heads=4, dim_head=256, attn_type='standard', mlp_dim=256).to(device)
elif args.model == "FT":
    model = POD_GalerkinTransformer(dim_in=in_dim * args.grid_x, dim_out=args.grid_x, dim_hid=256, depth=1, heads=4, dim_head=256, attn_type='fourier', mlp_dim=256).to(device)
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

# Training loop
for epoch in tqdm(range(args.num_epochs)):
    
    # Forward pass
    for batch_idx, (data, targets) in enumerate(train_loader):
        model.train()  # Set the model to training mode
        optimizer.zero_grad()  # Clear gradients
        outputs = model(data)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
    scheduler.step()

    # Print loss every epoch
    if (epoch+1)%int(25)==0 or epoch == args.num_epochs - 1: 
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
        # print(f"Epoch {epoch+1}, Train Loss: {loss.item()}, Train Rel L2: {error_train.item()}, Test Rel L2: {error_test.item()}")

if args.save_loss:
    loss_traj = np.asarray(loss_traj)
    filename = "Data/Burgers_" + \
        "Model=" + str(args.model) + "_" + \
        "Seed=" + str(args.SEED) + \
        ".txt"
    np.savetxt(filename, loss_traj)

if args.save_model:
    import os
    print("Saving model...")
    PATH = "Burgers"
    if not os.path.isdir(PATH): os.mkdir(PATH)  
    torch.save(model.state_dict(), PATH + "/" + args.model + "_Model.pth")