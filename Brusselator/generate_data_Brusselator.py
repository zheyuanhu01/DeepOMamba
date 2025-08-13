import argparse
import numpy as np
from matplotlib import pyplot as plt
from pde import PDE, PDEBase, FieldCollection, PlotTracker, ScalarField, UnitGrid, MemoryStorage, movie_scalar, CartesianGrid, DiffusionPDE
import os
import sys
import h5py
from generate_samples import generate_GPsamples


t_range, dt = 3, 1e-3
interval0 = 0.001
NT=int(t_range/interval0+1)
tt=np.linspace(0,t_range,NT)
N = 100
l = 0.2
sig = 1

S = generate_GPsamples(N, sig, l, tt)

sam_start1 = 0
sam_end1 = sam_start1 + N

        
print('Number of total samples for training:',N)  

a, b = 1, 3.0  
d0, d1 = 1, 0.5
nx, ny = 28, 28
grid = UnitGrid([nx,ny])
u = ScalarField(grid, a, label="Field $u$")
v = b / a + 0.1 * ScalarField.random_normal(grid, label="Field $v$")
state = FieldCollection([u, v])

import time
start = time.time() 
storage_outer1 = []
## training data
for i in range(N):

    print('Iteration: {}'.format(i+1))
    get = []
    storage = MemoryStorage()
    S1 = S[sam_start1+i,:]
    eq = PDE(
        {
            "u": f"{d0} * laplace(u) + {a} - ({b} + 1) * u + u**2 * v + forcing(t)",
            "v": f"{d1} * laplace(v) + {b} * u - u**2 * v",
        },
        user_funcs = {"forcing": lambda t: np.interp(t, tt, S1)}
    )

    tracker = [storage.tracker(interval=interval0)]
    sol = eq.solve(state, t_range=t_range, dt=dt, tracker=tracker)
    output1 = np.array((storage.data))[:,0,:,:]   # v
    storage_outer1.append(output1)
finish = time.time() - start
print('Total time: ',finish)  
outputs_all = np.array(storage_outer1) 
# U_train = outputs_all[:int(N*0.8),:,:]
# U_test = outputs_all[int(N*0.8):,:,:]
# F_train = S[:int(N*0.8),:]
# F_test = S[int(N*0.8):,:]

datafile1 = 'Brusselator_force_data.npz'.format(a,b)
datadir = 'Data/'
os.makedirs(datadir, exist_ok=True)
np.savez(os.path.join(datadir, datafile1), F=S, U=outputs_all)



