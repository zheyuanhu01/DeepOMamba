import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import argparse
from torch.utils.data import TensorDataset, DataLoader
from FNO2d import FNO2d
from FNO2d_Jamba import FNO_Jamba_1, FNO_Jamba_2
from FFNO2d_Flexible import FFNO
from DON_2d import POD_GRU, POD_LSTM, POD_Mamba, MambaConfig
from DON_2d import POD_GalerkinTransformer, POD_Transformer, POD_GNOT
from LNO2d import LNO2d
from LNO2d_Jamba import LNO_Jamba_1, LNO_Jamba_2
import scipy.io

class MatReader(object):
    def __init__(self, file_path, to_torch=True, to_cuda=False, to_float=True):
        super(MatReader, self).__init__()

        self.to_torch = to_torch
        self.to_cuda = to_cuda
        self.to_float = to_float

        self.file_path = file_path

        self.data = None
        self.old_mat = None
        self._load_file()

    def _load_file(self):
        try:
            self.data = scipy.io.loadmat(self.file_path)
            self.old_mat = True
        except:
            #self.data = h5py.File(self.file_path)
            self.old_mat = False

    def load_file(self, file_path):
        self.file_path = file_path
        self._load_file()

    def read_field(self, field):
        x = self.data[field]

        if not self.old_mat:
            x = x[()]
            x = np.transpose(x, axes=range(len(x.shape) - 1, -1, -1))

        if self.to_float:
            x = x.astype(np.float32)

        if self.to_torch:
            x = torch.from_numpy(x)

            if self.to_cuda:
                x = x.cuda()

        return x

    def set_cuda(self, to_cuda):
        self.to_cuda = to_cuda

    def set_torch(self, to_torch):
        self.to_torch = to_torch

    def set_float(self, to_float):
        self.to_float = to_float

def get_args():
    parser = argparse.ArgumentParser(description='DeepONet Training')
    parser.add_argument('--SEED', type=int, default=0)

    parser.add_argument('--N_train', type=int, default=450, help="")
    parser.add_argument('--N_test', type=int, default=50, help="")

    parser.add_argument('--grid_x', type=int, default=20, help="x-axis grid size")
    parser.add_argument('--grid_t', type=int, default=200, help="t-axis grid size")

    parser.add_argument('--T', type=int, default=10, help="terminal time")

    parser.add_argument('--num_epochs', type=int, default=1000, help="number of training epochs")
    parser.add_argument('--batch_size', type=int, default=16, help="minibatch size for SGD")
    parser.add_argument('--lr', type=float, default=1e-3, help="learning rate")

    parser.add_argument('--model', type=str, default="FNO")


    parser.add_argument('--LSTM_hidden_dim', type=int, default=256)
    parser.add_argument('--LSTM_num_layers', type=int, default=1)

    parser.add_argument('--GRU_hidden_dim', type=int, default=256)
    parser.add_argument('--GRU_num_layers', type=int, default=1)

    parser.add_argument('--MambaLLM_d_model', type=int, default=256)
    parser.add_argument('--MambaLLM_n_layer', type=int, default=1)

    parser.add_argument('--save_loss', type=int, default=0)
    parser.add_argument('--save_model', type=int, default=0)

    args = parser.parse_args()
    print(args)

    return args

args = get_args()

SEED = args.SEED
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)

reader = MatReader('RD_ftx_k0.01_D0.01.mat')
f_train = reader.read_field('F_train').numpy()
y_train = reader.read_field('U_train').numpy()
f_test = reader.read_field('F_test').numpy()
y_test = reader.read_field('U_test').numpy()
f_train = f_train.reshape(-1, args.grid_x, args.grid_t, 1)
f_test = f_test.reshape(-1, args.grid_x, args.grid_t, 1)
y_train = y_train.reshape(-1, args.grid_x * args.grid_t, 1)
y_test = y_test.reshape(-1, args.grid_x * args.grid_t, 1)

def aug_f(f, N, grid_x, grid_t):
    x = np.linspace(0, 1, grid_x)
    t = np.linspace(0, 1, grid_t)
    x, t = np.meshgrid(x, t, indexing='ij') # [grid_x, grid_t]
    x = x.reshape(1, grid_x, grid_t, 1)
    t = t.reshape(1, grid_x, grid_t, 1)
    x = np.tile(x, (N, 1, 1, 1))
    t = np.tile(t, (N, 1, 1, 1))
    f = np.concatenate([f, x, t], -1) # N, grid_x, grid_t, 3
    return f

f_train = aug_f(f_train, args.N_train, args.grid_x, args.grid_t)
f_test = aug_f(f_test, args.N_test, args.grid_x, args.grid_t)

# Setting device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

X_train, y_train, X_test, y_test = torch.from_numpy(f_train).float().to(device), torch.from_numpy(y_train).float().to(device), torch.from_numpy(f_test).float().to(device), torch.from_numpy(y_test).float().to(device)

print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

in_dim = 3

if args.model == "FNO":
    model = FNO2d(modes1=6, modes2=32, width=32, num_layers=2, in_dim=in_dim, out_dim=1).to(device)

elif args.model == "FFNO":
    model = FFNO(modes_x=6, modes_y=32, width=64, input_dim=in_dim, output_dim=1, n_layers=2).to(device)

elif args.model == "FNO_GRU_1":
    model = FNO_Jamba_1(input_dim=in_dim, output_dim=1, modes=6, width=128, num_layers=2, model_t_type="GRU").to(device)
elif args.model == "FNO_LSTM_1":
    model = FNO_Jamba_1(input_dim=in_dim, output_dim=1, modes=6, width=128, num_layers=2, model_t_type="LSTM").to(device)
elif args.model == "FNO_Mamba_1":
    model = FNO_Jamba_1(input_dim=in_dim, output_dim=1, modes=6, width=128, num_layers=2, model_t_type="Mamba").to(device)

elif args.model == "FNO_GRU_2":
    model = FNO_Jamba_2(input_dim=in_dim, output_dim=1, modes=6, width=128, num_layers=2, model_t_type="GRU").to(device)
elif args.model == "FNO_LSTM_2":
    model = FNO_Jamba_2(input_dim=in_dim, output_dim=1, modes=6, width=128, num_layers=2, model_t_type="LSTM").to(device)
elif args.model == "FNO_Mamba_2":
    model = FNO_Jamba_2(input_dim=in_dim, output_dim=1, modes=6, width=128, num_layers=2, model_t_type="Mamba").to(device)

elif args.model == "GRU":
    model = POD_GRU(input_dim=in_dim * args.grid_x, output_dim=args.grid_x, hidden_dim=args.GRU_hidden_dim, num_layers=args.GRU_num_layers).to(device)
elif args.model == "LSTM":
    model = POD_LSTM(input_dim=in_dim * args.grid_x, output_dim=args.grid_x, hidden_dim=args.LSTM_hidden_dim, num_layers=args.LSTM_num_layers).to(device)
elif args.model == "Mamba":
    config = MambaConfig(
            d_model=args.MambaLLM_d_model,
            n_layer=args.MambaLLM_n_layer,
            vocab_size=0,
            ssm_cfg=dict(layer="Mamba1"),
            rms_norm=True,
            residual_in_fp32=True,
            fused_add_norm=True
        )
    model = POD_Mamba(args.MambaLLM_d_model, args.MambaLLM_n_layer, 0, \
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


elif args.model =="LNO":
    T = torch.linspace(0, 1, args.grid_t)
    X = torch.linspace(0, 1, args.grid_x)
    model = LNO2d(input_dim=in_dim, output_dim=1, width=64, modes1=4, modes2=4, T=T, X=X)
    model = model.to(device)

elif args.model == "LNO_LSTM_1":
    X = torch.linspace(0, 1, args.grid_x)
    model = LNO_Jamba_1(input_dim=in_dim, output_dim=1, modes=4, width=32, num_layers=4, grid=X, time_model="LSTM")
    model = model.to(device)
elif args.model == "LNO_LSTM_2":
    X = torch.linspace(0, 1, args.grid_x)
    model = LNO_Jamba_2(input_dim=in_dim, output_dim=1, modes=4, width=32, num_layers=4, grid=X, time_model="LSTM")
    model = model.to(device)
elif args.model == "LNO_GRU_1":
    X = torch.linspace(0, 1, args.grid_x)
    model = LNO_Jamba_1(input_dim=in_dim, output_dim=1, modes=4, width=32, num_layers=4, grid=X, time_model="GRU")
    model = model.to(device)
elif args.model == "LNO_GRU_2":
    X = torch.linspace(0, 1, args.grid_x)
    model = LNO_Jamba_2(input_dim=in_dim, output_dim=1, modes=4, width=32, num_layers=4, grid=X, time_model="GRU")
    model = model.to(device)
elif args.model == "LNO_Mamba_1":
    X = torch.linspace(0, 1, args.grid_x)
    model = LNO_Jamba_1(input_dim=in_dim, output_dim=1, modes=4, width=32, num_layers=4, grid=X, time_model="Mamba")
    model = model.to(device)
elif args.model == "LNO_Mamba_2":
    X = torch.linspace(0, 1, args.grid_x)
    model = LNO_Jamba_2(input_dim=in_dim, output_dim=1, modes=4, width=32, num_layers=4, grid=X, time_model="Mamba")
    model = model.to(device)

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

# scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

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
    if (epoch+1)%int(250)==0: 
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

if args.save_model:
    import os
    print("Saving model...")
    PATH = "RD_Transient_ftx"
    if not os.path.isdir(PATH): os.mkdir(PATH)  
    torch.save(model.state_dict(), PATH + "/" + args.model + "_Model.pth")