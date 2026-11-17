# -*- coding: utf-8 -*-
"""Cerebellum pre-training.

Trains the feed-forward cerebellar network (:class:`~ccdl.models.FFNet`) to
approximate the inverse-model correction over a set of perturbations, producing
the pretrained cerebellum weights later loaded by the full CCNet training.
"""
import os, shutil
import sys
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

import torch
from torch import nn
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from ccdl.paths import DATA_DIR
from ccdl.utils import save_var, pr_print
from ccdl.reaching import ReachingDataset
from ccdl.models import FFNet


#from IPython.display import display, HTML
#display(HTML("<style>.container { width:70% !important; }</style>"))

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f'using {device.__str__()}')
dtype = 'float32'

npdtype = getattr(np, dtype)
torchdtype = getattr(torch, dtype)

EPOCHS = 20000
ITERATIONS=5
SEED = 127
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)


DATADIR = str(DATA_DIR)

# # Cerebellar module pretrain
# In this notebook, we explore the different ways in which we can pretrain the cerebellar, feedforward module for usage in the CCNet.
# The main objective is to get a model that ___CORRECTLY___ learn the output 
# $$\delta\theta = \theta^* - \theta^0 = P^{-1}(y^*) - \theta^0$$
# where is the motor plant through which the network generate the readout on which error is computed.
# In this first case, $P(\theta)$ is defined as
# $$
# y_1 = l_1\,cos(\theta_1) + l_2\, cos(\theta_1 + \theta_2)\\
# y_2 =  l_1\, sin(\theta_1) + l_2\, sin(\theta_1 + \theta_2)
# $$
# In order to properly pretrain this module without "cheating", we train the model on a series of perturbation dataset obtained by translation and radius change of the center-out reaching task ring of target points, while the perturbation on which the pretrained model will be used will be a rotational task. I also include a validation/test set of data (still no rotation), for generalization evaluation (indispensable here).
# 
# <br>
# 
# 
# > :memo: __NOTE: for increasing the number of data points, I treat perturbation in two different ways, equivalent once the model is correct (but not otherwise): give the initial dataset and the perturbed one, just by inverting the labels, I can generate a new datapoint where the network has to generalize to the opposite case, i.e. starting from the _perturbed output_, it has to provide the correction for getting the initial output.__ ___(99% of confidence this thing make sense :) )___.<br><br>
# > Notation:<br>
# >-   __unperturbed dataset__: &nbsp;&nbsp; $ \square^{0}$, <br>
# >   __perturbed dataset__:  &nbsp;&nbsp; $\square^{p}$, <br>
# >   __actual initial dataset__:  &nbsp;&nbsp; $\square^{base}$, <br>
# >   __actual target dataset__:  &nbsp;&nbsp;  $\square^{*}$
# >- inputs: $x$
# >- network outputs: $\theta$ 
# >- readout: $y$ 
# >- correction: $\delta \theta = \theta^* - \theta^{base} = P^{-1}(y^*) - \theta^{base} $
# >- the input x is always $y^*$
# >
# > In order to let the plant $P$ remain the same, we need to properly assign all the variables. As explained above, we have two different cases:
# > -  Standard case:
# >    - $\theta^* = \theta^p$, &nbsp;&nbsp;   $\theta^{base} = \theta^0$, &nbsp;&nbsp;   $y^* = y^p$
# >    - $\delta \theta = \theta^p - \theta^0 = P^{-1}(y^p) - \theta^{0} $
# >    - then, input $x = y^p$
# >
# > -  Inverted case:
# >    - $\theta^* = \theta^0$, &nbsp;&nbsp;  $\theta^{base} = \theta^p$, &nbsp;&nbsp;  $y^* = y^0$
# >    - $\delta \theta = \theta^0 - \theta^p = P^{-1}(y^0) - \theta^{p} $
# >    - then, input $x = y^0$


## Functions

def prepare_datasets(dataset_params, batch_size=2000,cpu = False, return_std=False):
    # Generate combinations of radii and shifts for perturbations
    radii = np.arange(0.85, 1.151, 0.05)
    shifts = np.arange(-0.02, 0.021, 0.01)
    shifts = [(a, b) for a in shifts for b in shifts]
    combs = [(rad, pos) for rad in radii for pos in shifts]
    # Initialize dataset and generate unperturbed dataset
    N = len(combs)
    dataset = []
    reaching = ReachingDataset(dataset_params)
    inputs_zero, targets_zero = reaching.generate_unperturbed_dataset('angles')

    dev = device if not cpu else 'cpu'
    # Loop through combinations to create perturbed datasets
    with torch.no_grad():
        for i, (radius, transl) in enumerate(combs):
            perturbation_params = {
                'angle': 0.,
                'amplitude_factor': radius,
                'translation': transl
            }
            
            reaching.apply_perturbation(perturbation_params)
            inputs_p, targets_p = reaching.generate_perturbed_dataset('angles', update_inputs=True, norm_inputs=True)

            # Compute correction for standard case and add to dataset
            correction = (targets_p - targets_zero).detach().clone().to(dev)
            cereb_inputs = torch.cat([inputs_p, targets_zero], dim=2).detach().clone().to(dev)
            dataset.append([cereb_inputs, correction])

            # Compute correction for inverted case and add to dataset
            correction = (targets_zero - targets_p).detach().clone().to(dev)
            cereb_inputs = torch.cat([inputs_zero, targets_p], dim=2).detach().clone().to(dev)
            dataset.append([cereb_inputs, correction])

    # Separate inputs and targets from the dataset
    inputs, targets = list(zip(*dataset))
    inputs = list(inputs)
    targets = list(targets)

    inputs2 = [inputs[i][j] for i in range(len(inputs)) for j in range(len(inputs[i]))]
    targets2 = [targets[i][j] for i in range(len(targets)) for j in range(len(targets[i]))]
    
    dataloader = DataLoader(list(zip(inputs2, targets2)), batch_size=batch_size, shuffle=True, pin_memory=False)

    # Split off a validation set from the main dataset
    val_inputs = []
    val_targets = []
    for i in range(len(inputs) // 20):
        val = np.random.randint(0, len(inputs))
        val_inputs.append(inputs.pop(val))
        val_targets.append(targets.pop(val))
    
    if return_std:
        return dataloader, (val_inputs, val_targets), targets_zero.std().item()
    return dataloader, (val_inputs, val_targets)

def train(model, dataloader, validation_dataset, epochs, criterion, opt, scheduler):
    train_losses = []
    val_losses = []
    start = datetime.now()

    with torch.no_grad():
        init_losses = []
        for batch_inputs, batch_targets in dataloader:
            batch_inputs = torch.cat([batch_inputs[i] for i in range(batch_inputs.shape[0])], dim=0)
            batch_targets = torch.cat([batch_targets[i] for i in range(batch_targets.shape[0])], dim=0)
            out = model(batch_inputs)
            loss = criterion(out, batch_targets)
            init_losses.append(loss.item())

        val_losses = []
        for inp, targ in zip(*validation_dataset):
            out = model(inp)
            val_losses.append(criterion(out, targ).item())
        print(f"initialization: loss= {np.mean(init_losses).item():.5e}\t,  val_loss = {np.mean(val_losses):.5e}")


    for i in range(1, epochs+1):
        iter_losses = []
        for batch_inputs, batch_targets in dataloader:
            opt.zero_grad()
            batch_inputs = torch.cat([batch_inputs[i] for i in range(batch_inputs.shape[0])], dim=0)
            batch_targets = torch.cat([batch_targets[i] for i in range(batch_targets.shape[0])], dim=0)
            out = model(batch_inputs)
            loss = criterion(out, batch_targets)
            if loss >0. and not np.isnan(loss.item()):
                loss.backward()
                opt.step()
            iter_losses.append(loss.item())
            del batch_inputs, batch_targets, loss, out
            if device.type == 'cuda':
                torch.cuda.empty_cache()
        loss = np.mean(iter_losses)
        train_losses.append(loss)
        if scheduler:
            scheduler.step(loss)

        if i%200==0:
            # computing test dataset error
            iter_losses = []
            with torch.no_grad():
                for inp, targ in zip(*validation_dataset):
                    out = model(inp)
                    iter_losses.append(criterion(out, targ).item())
            iter_losses = np.mean(iter_losses)
            val_losses.append(iter_losses)
            print(f"epoch {i} : loss= {loss.item():.5e}\t,  val_loss = {iter_losses:.5e},\t time = {datetime.now() - start}")
        else:
            print(f"epoch {i} : loss= {loss.item():.5e}\t", end='\r')

    return model, train_losses, val_losses


def test(model, dataloader, test_dataset, criterion):
    train_loss = []
    for batch_inputs, batch_targets in dataloader:
        batch_inputs = torch.cat([batch_inputs[i] for i in range(batch_inputs.shape[0])], dim=0)
        batch_targets = torch.cat([batch_targets[i] for i in range(batch_targets.shape[0])], dim=0)
        out = model(batch_inputs)
        train_loss.append(criterion(out, batch_targets).item())
    
    test_loss = []
    for inp, targ in zip(*test_dataset):
        out = model(inp)
        test_loss.append(criterion(out, targ).item())
    return np.mean(train_loss), np.mean(test_loss)

dataset_params = {
    "tasks": 8,
    "small_ring": False,
    "target": "moving",
    "interpolation": "bell",
    "timesteps": 130,
    "hold_cue_time": 30,
    "t_in_duration": 30,
    "small_ring_rot": True,
    "rings_size": [0.1, 0.05],
    "joints_length": [0.22, 0.27],
    "input_type": 'autoenc',
    "input_mode": "full-go",
    'device': device.__str__(),
    'dtype': 'float32'
}

params = dataset_params

saving_folder = os.path.join(DATADIR, 'cereb_pre_oldcode')
if os.path.exists(saving_folder):
    shutil.rmtree(saving_folder)
    

# Prepare datasets
train_loader, validation = prepare_datasets(params, batch_size=3500)
input_size = train_loader.dataset[0][0].shape[-1]

# Define model
net_name = '2x1024_noseq'
lr = 0.0001
lr_scheduler = True
print(datetime.now())

print(f'training for {EPOCHS} epochs.')
criterion = nn.MSELoss()

losses = {}
name = net_name+f'lr_{lr:.1e}'
name = name+f'_lr_scheduler' if lr_scheduler else name
folder = os.path.join(saving_folder, name)
os.makedirs(folder)
for iteration in range(ITERATIONS):
    i_name = f"{iteration}.sd"
    pr_print(['bold', 'red', 'underline'],name+'_'+i_name)
    filename = os.path.join(folder, i_name)
    if os.path.exists(filename):
        continue
    
    cereb = FFNet(input_size, 2, [1024]*2, True, 'xavier_normal', init_gain=.1, h_nlin='relu', out_nlin='linear')
    cereb.to(device)

    print(cereb)

    opt = torch.optim.Adam(cereb.parameters(), lr=lr)
    if lr_scheduler:
        scheduler = ReduceLROnPlateau(opt, patience=150, factor=0.2, verbose=True,min_lr=1e-6)
    else:
        scheduler = None
    cereb, train_losses, val_losses = train(cereb, train_loader, validation, EPOCHS, criterion, opt, scheduler)

    losses[name] = (train_losses, val_losses)
    
    torch.save(cereb.state_dict(), filename)

    if device.type == 'cuda':
        torch.cuda.empty_cache()
    del cereb

save_var(losses, os.path.join(saving_folder, 'losses.pkl'))

