# -*- coding: utf-8 -*-
"""
Created on 06/03/2023
@author: Leonardo Agueci - leoagueci@gmail.com

Cortex pre-training loop (:class:`CCNetPreTrainer`): trains the cortical RNN on
the reaching task before the cerebellar/distributed-learning stage.
"""
import sys
import copy

from datetime import datetime

import matplotlib.pyplot as plt

# plt.switch_backend('Qt5Agg')

from matplotlib.cm import rainbow

import torch
import torch.nn as nn
from torch import Tensor
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
import numpy as np
import random

from torch.utils.tensorboard.writer import SummaryWriter

from ccdl.utils import *
from ccdl.reaching import *
from ccdl.models import *
from ccdl.ccnet import CerebelloCorticalNet

class CCNetPreTrainer:
    """Trainer for the cortex pre-training stage.

    Optimises the cortical RNN on the reaching task (cerebellum inactive) to
    produce the baseline controller later used by :class:`~ccdl.trainer.CCNetTrainer`.
    """

    def __init__(self, net_params, train_params=None, model_class='CerebelloCorticalNet', opt=torch.optim.Adam,
                 criterion=nn.MSELoss):

        self.net_params = net_params
        self.device = self.net_params['device']

        self.opt = opt
        self.criterion = criterion()

        assert type(model_class) == str, "model_class must be a string"
        assert model_class in globals(), f"model_class {model_class} not found"

        model_class = globals()[model_class]
        self.model = self.init_model(model_class)
        self.model_state = self.init_state()

        self.dataset = (None, None)
        # initialize training parameters
        self.training_params = self.store_train_params(train_params) if train_params else {}

        self.pretrain = False

        # variables initialization
        self.mse_losses = []  # the error loss
        self.tot_losses = []  # mse_loss + reg
        self.outputs = None
        self.net_buffer = None
        self.nets = []
        self.gradients = {}
        self.cortex_error = None

        self.logfile = None
        self.dataset_generator = None

        # Variable used for trainings without interruptions (when True)
        self.force = False
        # Init. correction dataset as none, used for pretraining usually
        self.correction_dataset = None
        # Default feedback type
        self.input_2_type = None
        # init cortex testing model_state
        self.test_state = {'cereb_feedback': False, 'cereb_readout': False, 'modules_plasticity': None,
                           'plant': self.model_state['plant']}

    def init_model(self, model_class):
        args = self.net_params['args']
        kwargs = copy.deepcopy(self.net_params)
        kwargs.pop('args')
        print(f'using model_class {model_class.__name__}')
        return model_class(*args, **kwargs)

    def _model_call(self, x, **kwargs):
        # Internal method to handle conditional 'init_state' argument
        if 'initial_theta' in kwargs:
            var = kwargs.pop('initial_theta')
            return self.model(x, **kwargs, var=var)
        else:
            return self.model(x, return_hidden=True, return_thetas=True, **kwargs)

    @staticmethod
    def default_training_params():
        return {
            "lr": 0.01,
            "epochs": 100,
            "l1": 0.,
            "l2": 0.,
            "activity_penalty": 0.,
            "reg_drop": None,
            "lr_scheduler": None,
            "early_stop": False,
            "local_lr": 1e-4,
            "local_method": 'mean',
            "es_params": [300, 1e-9],
            "es_patience": 300,
            "es_eps": 1e-9
        }

    def init_state(self):
        return self.model.default_model_state()

    def store_train_params(self, train_params):
        """method storing training parameters"""
        final_train_params = self.default_training_params()
        if train_params:
            final_train_params.update(train_params)
            final_train_params['es_patience'] = train_params['es_params'][0]
            final_train_params['es_eps'] = float(train_params['es_params'][1])
            if final_train_params['reg_drop'] is not None:
                final_train_params['reg_drop_factor'] = final_train_params['reg_drop'][0]
                final_train_params['reg_drop_time'] = final_train_params['reg_drop'][1]
            else:
                final_train_params['reg_drop_factor'] = 0.
                final_train_params['reg_drop_time'] = -1
            if final_train_params['local_method'] is None:
                final_train_params['local_method'] = 'mean'
        return final_train_params

    def load_cortex(self, state_dict):
        """method loading pretrained cortex"""
        try:
            self.model.cortex.load_state_dict(state_dict)
            self.printfunc("Pretrained CXmodel correctly loaded.")
        except:
            current_state = self.model.cortex.state_dict()
            self.printfunc("The state_dict provided does not match the current model structure.\n")
            self.printfunc("Saved state_dict keys:")
            self.printfunc(str([(name, value.shape, value.device) for name, value in state_dict.items()]))
            self.printfunc("\n\nCurrent state_dict keys:")
            self.printfunc(str([(name, value.shape, value.device) for name, value in current_state.items()]))
            raise IOError

    def load_cereb(self, state_dict):
        """method loading pretrained cerebellum"""
        try:
            self.model.cerebellum.load_state_dict(state_dict)
            self.printfunc("Pretrained model correctly loaded.")
        except:
            current_state = self.model.cerebellum.state_dict()
            self.printfunc("The state_dict provided does not match the current model structure.\n")
            self.printfunc("Saved state_dict keys:")
            self.printfunc(str([(name, value.shape) for name, value in state_dict.items()]))
            self.printfunc("\n\nCurrent state_dict keys:")
            self.printfunc(str([(name, value.shape) for name, value in current_state.items()]))
            raise IOError

    def load_model(self, state_dict):
        """method loading pretrained models"""
        try:
            self.model.load_state_dict(state_dict)
            self.printfunc("Pretrained model correctly loaded.")
        except:
            if self.force:
                raise IOError("Error occurred while loading the state_dict")
            else:
                current_state = self.model.state_dict()
                self.printfunc("The state_dict provided does not match the current model structure.\n")
                self.printfunc("Saved state_dict:\n" + str(list(state_dict.keys())))
                self.printfunc("\n\nCurrent state_dict:\n" + str(list(current_state.keys())))

                choice = input(" Do you want to try loading only the matching parameters? (Not safe)")
                if choice.lower() in ('y', 'yes'):
                    self.model.load_state_dict(state_dict, strict=False)
                    self.printfunc("Pretrained model completed")
                else:
                    raise IOError

    def plastic_params(self, module=None, flatten=False):
        """Helper function, returning all plastic parameters in a given module. if none, module is the entire net"""
        if not module:
            module = self.model
        names = []
        params = []
        for name, param in module.named_parameters():
            if param.requires_grad:
                if flatten:
                    params.append(param.reshape(-1))
                else:
                    params.append(param)
                names.append(name)
        return params, names

    def all_params(self, module=None):
        """Helper function, returning all parameters in a given module. if none, module is the entire net"""
        if not module:
            module = self.model
        names = []
        params = []
        for name, param in module.named_parameters():
            params.append(param)
            names.append(name)
        return params, names

    @staticmethod
    def sd_to_cpu(sd, device):
        """Move state_dict to the new device"""
        for k, v in sd.items():
            sd[k] = v.to(device)
        return sd

    @staticmethod
    def plot_distro(param, bins=50):
        assert hasattr(param, 'shape'), "Please pass a valid parameter (weight, bias, etc)"
        with torch.no_grad():
            if len(param.shape) > 1:
                param = param.detach().view(-1).numpy()
                n, bias, patches = plt.hist(param, bins)
                plt.show()

    def printfunc(self, output, formatting=None, end=None):
        printsave(output, self.logfile, True, formatting, end)

    def L1_Loss(self):
        params, _ = self.plastic_params(flatten=True)
        params = torch.cat(params)
        return torch.norm(params, 1) / params.shape[0]

    def L2_Loss(self):
        params, _ = self.plastic_params(flatten=True)
        params = torch.cat(params)
        return torch.norm(params, 2) / params.shape[0]

    def activity_loss_function(self, hidden):
        return torch.mean(hidden ** 2)

    def reset_loss(self):
        self.mse_loss = 0.
        self.activity_loss = torch.tensor(0.).to(self.device)
        self.l1_loss = torch.tensor(0.).to(self.device)
        self.l2_loss = torch.tensor(0.).to(self.device)

    def compute_cortex_loss(self, thetas=None, input_2=None, input_2_type=None):
        if thetas is None:
            self.model.train(**self.test_state)
            inputs, targets = self.dataset
            outputs, _, _ = self._model_call(inputs, input_2=input_2, input_2_type=input_2_type)
            error = self.criterion(outputs, targets)
            self.model.train(**self.model_state)
        else:
            self.model.eval(**self.model_state)
            inputs, targets = self.dataset
            outputs = inputs.new_zeros((inputs.shape[0], inputs.shape[1], self.model.output_size))
            timesteps = inputs.shape[0]
            for t in range(timesteps):
                outputs[t] = self.model.plant(thetas[0, t])
            error = self.criterion(outputs, targets)
            self.model.train(**self.model_state)
        return error.item()

    def compute_loss(self, outputs, target, activity):
        n = target.shape[1]
        assert any([name in self.training_params.keys() for name in ['l1', 'l2', 'activity_penalty']]), \
            "Please define l1, l2 and activity_penalty, necessary for computing loss"

        loss = self.criterion(outputs, target)
        self.mse_loss = loss.detach().clone().item()

        if self.training_params['l1'] > 0.:
            l1_loss = self.training_params['l1'] * self.L1_Loss()
            self.l1_loss = l1_loss
            loss += l1_loss
        if self.training_params['l2'] > 0.:
            l2_loss = self.training_params['l2'] * self.L2_Loss()
            self.l2_loss = l2_loss
            loss += l2_loss
        if self.training_params['activity_penalty'] > 0.:
            activity_loss = self.training_params['activity_penalty'] * self.activity_loss_function(activity)
            self.activity_loss = activity_loss
            loss += activity_loss
        return loss

    def early_stopping(self):
        """plateau"""
        stop_flag = False
        stop_pos = None
        epoch = len(self.mse_losses)
        last_pos = epoch - 1
        best = np.min(np.array(self.mse_losses))
        best_pos = np.argmin(np.array(self.mse_losses))

        # threshold stop
        if np.all(np.array(self.mse_losses[-self.training_params['es_patience']:]) < self.training_params['es_eps']):
            stop_flag = True
            buffer_pos = -(epoch - int(best_pos))
            stop_pos = best_pos
            self.printfunc("threshold stopping triggered.")

        # plateau stop (no decrease & slow decrease)
        else:
            if epoch > self.training_params['es_patience']:
                no_decrease = (best_pos < epoch - self.training_params['es_patience'])
                slow_decrease = (np.max(self.mse_losses[-self.training_params['es_patience']:]) - best < self.training_params['es_eps'])

                if no_decrease or slow_decrease:
                    stop_flag = True
                    buffer_pos = -(epoch - int(best_pos))
                    stop_pos = best_pos
                    if no_decrease:
                        self.printfunc("no_decrease stopping triggered.")
                    else:
                        self.printfunc("slow_decrease stopping triggered.")
        if stop_flag:
            try:
                self.model.load_state_dict(self.net_buffer.buf[buffer_pos])
            except:
                self.printfunc('Something occurred in the early stopping module (plateau), check net buffer vals'
                               '(len=%d, buffer_pos=%d).' % (len(self.net_buffer.buf), buffer_pos),
                               formatting=['red', 'bold'])
        return stop_flag, stop_pos

    """
    if best_pos < epoch - self.training_params['es_patience']
    if 0 < self.training_params['es_patience'] < epoch and best <= self.training_params['es_eps'] :
        bad_epochs = last_pos - best_pos
        no_decrease = (bad_epochs >= self.training_params['es_patience']  - 1)
        slow_decrease = np.all(np.abs(self.mse_losses[-self.training_params['es_patience']:] - best) < self.training_params['es_eps'] )
        """

    def _training_loop(self, epoch, optimizers, scheduler, record_gradients, debug):
        # make sure the model status is correct
        self.model.train(**self.model_state, pretraining=self.pretrain)

        inputs, targets = self.dataset
        assert inputs is not None, "dataset was not properly initialized (inputs)."
        assert targets is not None, "dataset was not properly initialized (targets)."
        self.reset_loss()
        batch_size = inputs.shape[1]

        # input_2 used for feeding back the correction signal
        if 'correction_fb' in self.training_params.keys() and self.training_params['correction_fb']:
            input_2 = self.correction_dataset
        else:
            input_2 = None

        # zero the gradient buffers
        optimizers[0].zero_grad()
        if len(optimizers) > 1 and epoch >= self.training_params['fb_lr_start']:
            if epoch == self.training_params['fb_lr_start']:
                self.printfunc("Starting feedback learning...", ['bold', 'underline', 'blue'])
            optimizers[1].zero_grad()
        outputs, activity, thetas = self._model_call(inputs, input_2=input_2, input_2_type=self.input_2_type)
        thetas_mem = {
            'cortex': thetas[0].abs().mean().item(),
            'cereb':  thetas[1].abs().mean().item()
        }

        if debug:
            mean_hidden_activity = activity.abs().mean().item()

        loss = self.compute_loss(outputs, targets, activity)

        # backprop only if different from zero and nan
        if np.abs(loss.item()) > 0.:
            loss.backward()
            optimizers[0].step()
            if len(optimizers) > 1 and epoch >= self.training_params['fb_lr_start']:
                optimizers[1].step()

        # saving gradients
        if record_gradients or debug:
            cx_params, cx_names = self.plastic_params(self.model.cortex)
            cx_grads = {name: param.grad.view(-1).abs().mean().item() for name, param in zip(cx_names, cx_params)}
            if debug and epoch % debug == 0:
                self.printfunc("GRADS:")
                self.printfunc(f"\tcortex grads(mean abs)")
                if len(cx_grads) == 0:
                    self.printfunc("\t\t[]")
            for name, val in cx_grads.items():
                if name in ['rnn.input2h_2.weight', 'rnn.input2h_2.bias']:
                    if len(optimizers) > 1 and epoch < self.training_params['fb_lr_start']:
                        continue
                if record_gradients:
                    self.gradients['cortex'][name].append(val)
                if debug and epoch % debug == 0:
                    self.printfunc(f"\t\t {name}: {val:.3e}")

            cb_params, cb_names = self.plastic_params(self.model.cerebellum)
            cb_grads = {name: param.grad.view(-1).abs().mean().item() for name, param in zip(cb_names, cb_params)}
            if debug and epoch % debug == 0:
                self.printfunc("\tcereb grads(mean abs)")
                if len(cb_grads) == 0:
                    self.printfunc("\t\t[]")
            for name, val in cb_grads.items():
                if record_gradients:
                    self.gradients['cerebellum'][name].append(val)
                if debug and epoch % debug == 0:
                    self.printfunc(f"\t\t {name}: {val:.3e}")

        if debug and epoch % debug == 0:
            cortex_params, cortex_names = self.plastic_params(self.model.cortex)
            cortex_grads = [cortex_params[i].grad.view(-1).abs().mean().item() for i in range(len(cortex_params))]
            cortex_params, cortex_names2 = self.all_params(self.model.cortex)
            cortex_mean_param = [cortex_params[i].view(-1).abs().mean().item() for i in range(len(cortex_params))]

            cereb_params, cereb_names = self.plastic_params(self.model.cerebellum)
            cereb_grads = [cereb_params[i].grad.view(-1).abs().mean().item() for i in range(len(cereb_params))]
            cereb_params, cereb_names2 = self.all_params(self.model.cerebellum)
            cereb_mean_param = [cereb_params[i].view(-1).abs().mean().item() for i in range(len(cereb_params))]

        if scheduler is not None:
            if self.training_params['lr_scheduler'] == 'cosine':
                scheduler.step()
            else:
                scheduler.step(loss)

        if record_gradients:
            if self.tensorboard:
                self.writer.add_scalars('cortex_grads', cx_grads, epoch)
                self.writer.add_scalars('cerebellum_grads', cb_grads, epoch)

        if self.tensorboard:
            self.writer.add_scalars('thetas', thetas_mem, epoch)
        if debug and epoch % debug == 0:
            mean_hidden_activity = mean_hidden_activity / batch_size
            self.printfunc(f"mean hidden activity: {mean_hidden_activity}")

            self.printfunc("PARAMS MEAN-ABS VALUE:")
            self.printfunc("\tcortex")
            self.printfunc("\t\t" + "".join(
                [f"{cortex_names2[j]}: {cortex_mean_param[j]:.3e}\t" for j in range(len(cortex_mean_param))]))
            self.printfunc("\tcerebellum:")
            self.printfunc("\t\t" + "".join(
                [f"{cereb_names2[j]}: {cereb_mean_param[j]:.3e}\t" for j in range(len(cereb_mean_param))]),
                end='\n\n')

            self.printfunc("THETAS ABS-MEAN:")
            self.printfunc(thetas_mem)
            self.printfunc("PLANT:")
            self.printfunc(self.model.plant.__class__)

        self.mse_losses.append(self.mse_loss)
        self.tot_losses.append(loss.item())
        if not self.pretrain:
            self.cortex_error = self.compute_cortex_loss(thetas, input_2=input_2,
                                                          input_2_type=self.input_2_type)

        return loss, outputs

    def train(self, dataset_generator, model_state=None, pretrain=False, washout=False, datatype=None,
              train_params=None, opt=None, outsave_step=0, print_steps=1, save_w_steps=0, record_gradients=False,
              save_cortex=True, verbose=True, debug=False, plot=False, tensorboard_params={}, text="", logfile=None,
              force=False, skip_init_loss=False, record_angle=False):
        
        self.force = force
        self.tensorboard = True if tensorboard_params != {} else False
        self.logfile = logfile

        if debug:
            if pretrain:
                self.printfunc("Starting training in 'pretrain' mode")
            else:
                self.printfunc("Starting training in 'train' mode")

        if model_state is not None:
            assert type(model_state) == dict and model_state.keys() == self.model_state.keys(), \
                "Please provide as model_state a dict with keys: %s" % str(list(self.model_state.keys()))
            self.model_state = model_state

        # updating training_params
        if not train_params and not self.training_params:
            self.printfunc('No training parameters provided. Using default settings')
        self.training_params = self.store_train_params(train_params)

        # Set training parameters
        self.pretrain = pretrain
        self.training_params['pretrain'] = pretrain

        self.dataset_generator = dataset_generator

        if verbose:
            self.printfunc('Training initialization...\n')

        assert hasattr(self, 'training_params'), "training parameters not assigned yet. Please provide it as kwarg ('train_params')\
                                         or use store_train_params(train_params) method."

        if opt:
            if type(opt) == str:
                try:
                    optimizer = getattr(torch.optim, opt)
                except:
                    self.printfunc('Optimizer not found, using adam.')
                    optimizer = torch.optim.Adam
            else:
                optimizer = opt
        else:
            optimizer = self.opt
        assert hasattr(self, 'opt'), "Optimizer not assigned yet. Please provide it as kwarg ('opt')"

        self.model.train(**self.model_state, pretraining=pretrain)

        # Generating the dataset
        dt_text = datatype if datatype is not None else 'Default'
        self.printfunc(f"Using {dt_text} dataset.")

        correction_datatype = None
        input_2_type = None
        if self.model.plant.name() == 'Identity':
            if 'correction_fb' in self.training_params.keys() and self.training_params['correction_fb']:
                correction_datatype = datatype if 'correction_datatype' not in self.training_params.keys() \
                    else self.training_params['correction_datatype']
                input_2_type = correction_datatype if (correction_datatype == 'angles') else 'output'
        else:
            if 'correction_fb' in self.training_params.keys() and self.training_params['correction_fb']:
                correction_datatype = 'angles' if 'correction_datatype' not in self.training_params.keys() \
                    else self.training_params['correction_datatype']
                input_2_type = correction_datatype if (correction_datatype == 'angles') else 'output'

        if pretrain:
            self.dataset = dataset_generator.generate_unperturbed_dataset(datatype)
            if 'correction_fb' in self.training_params.keys() and self.training_params['correction_fb']:
                self.correction_dataset = dataset_generator.generate_unperturbed_dataset(correction_datatype)[1]
        else:
            self.dataset = dataset_generator.generate_perturbed_dataset(datatype)
            self.unperturbed_dataset = dataset_generator.generate_unperturbed_dataset(datatype)
            if 'correction_fb' in self.training_params.keys() and self.training_params['correction_fb']:
                self.correction_dataset = dataset_generator.generate_perturbed_dataset(correction_datatype)[1]

        self.input_2_type = input_2_type

        if debug:
            self.printfunc("\nModel summary:", ['green', 'underline'])
            self.printfunc(self.model)
            self.printfunc("\nModel state:", ['green', 'underline'])
            self.printfunc(self.model.get_model_state())

            if plot:
                self.printfunc("Plotting dataset...")
                dataset_generator.plot_dataset2(dataset=self.dataset, pert=True, tasks=[1, 2, 5])

        ## SETTING UP OPTIMIZERs
        betas = tuple(self.training_params.get('betas', (0.9, 0.999)))
        params_dict = []
        lr = self.training_params['lr']

        # cerebellum params
        cb_params = self.plastic_params(self.model.cerebellum)[0]
        if len(cb_params) > 0:
            params_dict.append({'params': cb_params, 'lr': lr, 'betas': betas})

        # cortex params
        cx_params, cx_names = self.plastic_params(self.model.cortex)

        fb_params = []
        # extracting feedback params
        for p in ['weight', 'bias']:
            fb_name = 'rnn.input2h_2.' + p
            if fb_name in cx_names:
                index = cx_names.index(fb_name)
                fb_params.append(cx_params.pop(index))
                cx_names.pop(index)

        cx_lr = self.training_params.get('cx_lr')
        if len(cx_params) > 0:
            if cx_lr:
                self.printfunc(f"Using custom learning ({cx_lr:.2E}) rate for cx-weights")
            else:
                cx_lr = lr
            params_dict.append({'params': cx_params, 'lr': cx_lr, 'betas': betas})

        optimizers = [optimizer(params_dict)]

        if len(fb_params) > 0:
            fb_lr = self.training_params.get('fb_lr')
            if fb_lr:
                self.printfunc(f"Using custom learning ({fb_lr:.2E}) rate for fb-weights")
                fb_lr_start = self.training_params.get('fb_lr_start')
                if not fb_lr_start:
                    self.training_params['fb_lr_start'] = 0
            else:
                fb_lr = lr
            optimizers.append(optimizer(fb_params, lr=fb_lr))

        # initializing TensorBoard writer
        if self.tensorboard:
            self.writer = SummaryWriter(**tensorboard_params)

        # support variable initialization
        scheduler = None
        lr_mem = []
        outputs_memory = []
        epoch = 0

        # initializing gradients storing variable
        if record_gradients:
            self.printfunc("\nGradients will be recorded.")
            self.gradients['cortex'] = {name: [] for (_, name) in zip(*self.plastic_params(self.model.cortex))}
            self.gradients['cerebellum'] = {name: [] for _, name in zip(*self.plastic_params(self.model.cerebellum))}
            self.gradients['cortex_local'] = []

        # initializing LR scheduler
        assert self.training_params['lr_scheduler'] in [None, 'cosine', 'ReduceLROnPlateau'], \
            "lr_scheduler must be 'cosine' or 'ReduceLROnPlateau' or None"
        if isinstance(self.training_params['lr_scheduler'], int):
            self.printfunc("Using ReduceLROnPlateau LRscheduler.")
            scheduler = ReduceLROnPlateau(optimizers[0], patience=self.training_params['lr_scheduler'],
                                          factor=0.2, verbose=bool(debug), min_lr=1e-6)
        elif self.training_params['lr_scheduler'] == 'cosine':
            self.printfunc("Using CosineAnnealing LRscheduler.")
            scheduler = CosineAnnealingLR(optimizers[0], self.training_params['epochs'],
                                          eta_min=self.training_params['lr'] / 100)

        with torch.no_grad():
            self.model.eval(**self.model_state)
            output, activity, thetas = self._model_call(self.dataset[0], input_2=self.correction_dataset,
                                                         input_2_type=self.input_2_type)
            if 'correction_fb' in self.training_params.keys() and self.training_params['correction_fb']:
                self.printfunc("\nUsing correction feedback.", ["bold", "underline"])
                self.prev_output = output.detach().clone()
            if not pretrain:
                pretrain_model_state = {'cereb_feedback': False, 'cereb_readout': False,
                                        'modules_plasticity': None, 'local_plasticity': False,
                                        'plant': self.model_state['plant']}
                self.model.eval(**pretrain_model_state)
                if verbose:
                    self.printfunc("Pretrain model_state: %s" % str(pretrain_model_state))
                    self.printfunc("Pretrain model: %s" % str(self.model))
                unp_output, _, _ = self._model_call(self.unperturbed_dataset[0],
                                                     input_2=self.unperturbed_dataset[1],
                                                     input_2_type=self.input_2_type)
                self.model.train(**self.model_state)

            if debug:
                if len(self.dataset[1].shape) == 3:
                    if plot:
                        c = rainbow(np.linspace(0, 1, self.dataset[1].shape[1]))
                        plt.figure(2)
                        plt.title("Starting trajectories")
                        for i in range(self.dataset[1].shape[1]):
                            plt.plot(self.dataset[1][:, i, 0], self.dataset[1][:, i, 1], color=c[i], alpha=0.1)
                            plt.plot(output[:, i, 0], output[:, i, 1], color=c[i], label=f'task {i}')
                        plt.legend()
                        plt.show()

                        plt.figure(3)
                        plt.title("Pretrain trajectories")
                        for i in range(self.dataset[1].shape[1]):
                            plt.plot(self.dataset[1][:, i, 0], self.dataset[1][:, i, 1], color=c[i], alpha=0.1)
                            plt.plot(unp_output[:, i, 0], unp_output[:, i, 1], color=c[i], label=f'task {i}')
                        plt.legend()
                        plt.show()

                        plt.figure(4)
                        plt.title("plotting thetas cortex")
                        for i in range(self.dataset[1].shape[1]):
                            plt.plot(thetas[0, :, i, 0], thetas[0, :, i, 1], color=c[i], label=f'task {i}')
                        plt.legend()
                        plt.show()

                        plt.figure(5)
                        plt.title("plotting thetas cereb")
                        for i in range(self.dataset[1].shape[1]):
                            plt.plot(thetas[1, :, i, 0], thetas[1, :, i, 1], color=c[i], label=f'task {i}')
                        plt.legend()
                        plt.show()

                    self.printfunc("params test:")
                    self.printfunc("rnn.h2h.weight: " + str(self.model.cortex.rnn.h2h.weight.cpu().numpy()[0, :3]))
                    self.printfunc("rnn.h2h.bias: " + str(self.model.cortex.rnn.h2h.bias.cpu().numpy()[:3]))
                    self.printfunc("rnn.input2h.weight: " + str(self.model.cortex.rnn.input2h.weight.cpu().numpy()[0, :3]))
                    self.printfunc("rnn.input2h.bias: " + str(self.model.cortex.rnn.input2h.bias.cpu().numpy()[:3]))
                    self.printfunc("rnn.input2h_2.weight: " + str(self.model.cortex.rnn.input2h_2.weight.cpu().numpy()[0, :3]))
                    self.printfunc("fc.weight: " + str(self.model.cortex.fc.weight.cpu().numpy()[0, :3]))
                    try:
                        self.printfunc("rnn.input2h_2.bias: " + str(self.model.cortex.rnn.input2h_2.bias.cpu().numpy()[:3]))
                    except:
                        pass
                    try:
                        self.printfunc("fc.bias: " + str(self.model.cortex.fc.bias.cpu().numpy()[0, :3]))
                    except:
                        pass
                else:
                    self.printfunc(
                        "starting trajectories will not be printed, since the dataset is not (time, task, dim)-shaped")

        # loss arrays initialization
        self.reset_loss()
        loss = self.compute_loss(output, self.dataset[1], activity)
        best_loss = loss.item()
        self.tot_losses = [loss.item()]
        self.mse_losses = [self.mse_loss] if not skip_init_loss else []
        self.metrics = {
            'loss': [],
            'mse': [],
            'l1_loss': [],
            'l2_loss': [],
            'activity_loss': []
        }
        if record_angle:
            self.metrics['angular_error_rad'] = []
            self.metrics['takeoff_angular_error_rad'] = []

        if not pretrain:
            self.metrics['cortex_error'] = []
            self.metrics['nocb_error'] = []
        self.hyperparams = {item: [] for item in ['lr', 'l1', 'l2', 'activity_penalty']
                            if self.training_params[item] != 0.}

        self.printfunc(f"Starting loss: {self.tot_losses[0]} (mse_loss: {self.mse_loss}, l1_loss: {self.l1_loss.item()}, "
                       f"l2_loss: {self.l2_loss.item()}, activity_loss: {self.activity_loss.item()})\n")
        if not pretrain:
            self.printfunc("Unperturbed loss: {}".format(self.criterion(unp_output, self.unperturbed_dataset[1])))

        metrics = {
            'loss': self.tot_losses[0],
            'mse': self.mse_loss,
            'l1_loss': self.l1_loss.item(),
            'l2_loss': self.l2_loss.item(),
            'activity_loss': self.activity_loss.item()
        }

        if not pretrain:
            metrics['cortex_error'] = self.compute_cortex_loss(thetas,
                                                                input_2=self.correction_dataset,
                                                                input_2_type=self.input_2_type)
            metrics['nocb_error'] = self.compute_cortex_loss()

        for k, v in metrics.items():
            self.metrics[k].append(v)
            if self.tensorboard:
                self.writer.add_scalars(k, v, 0)

        if outsave_step:
            self.outputs = [output]

        # parameters buffer initialization
        self.net_buffer = Buffer(max(self.training_params['es_patience'], 100))

        if save_w_steps > 0:
            self.nets = [copy.deepcopy(self.model.state_dict())]

        self.net_buffer.add(copy.deepcopy(self.model.state_dict()))

        start_time = datetime.now()

        if outsave_step > 0:
            outputs_memory = []

        early_stop_message = ""
        sys.stdout.write("\033[K")
        self.printfunc(f'Starting training: {start_time}', ['bold'])

        for epoch in range(1, self.training_params['epochs'] + 1):

            hyperparams = {item: self.training_params[item] for item in self.hyperparams.keys()}
            for k, v in hyperparams.items():
                self.hyperparams[k] = hyperparams[k]
                if self.tensorboard:
                    self.writer.add_scalar(f'{k}', v, epoch)

            if epoch == self.training_params['reg_drop_time']:
                self.training_params['l1'] *= self.training_params['reg_drop_factor']
                self.training_params['l2'] *= self.training_params['reg_drop_factor']
                self.training_params['activity_penalty'] *= self.training_params['reg_drop_factor']

                self.printfunc(f"Decreasing regularizers. new values: l1 = {self.training_params['l2']}, "
                               f"l2 = {self.training_params['l2']}, "
                               f"activity penalty = {self.training_params['activity_penalty']}\n",
                               formatting=['red', 'bold'])

            loss, output = self._training_loop(epoch, optimizers, scheduler, record_gradients, debug)
            self.prev_output = output.detach().clone()
            if scheduler is not None:
                lr_mem.append(optimizers[0].param_groups[0]['lr'])
            if outsave_step > 0 and epoch % outsave_step == 0:
                outputs_memory.append(output.detach().clone().cpu())

            metrics = {
                'loss': self.tot_losses[-1],
                'mse': self.mse_loss,
                'l1_loss': self.l1_loss.item(),
                'l2_loss': self.l2_loss.item(),
                'activity_loss': self.activity_loss.item()
            }
            
            if record_angle:
                t_start = dataset_generator.hold_cue_time  # skip the hold period
                t_stop = (self.dataset[1].shape[0] - t_start)//2 # skip the end period
                with torch.no_grad():
                    metrics['angular_error_rad'] = dataset_generator.compute_signed_angular_error(
                        output, self.dataset[1], t_start=t_start, mean=True
                    )
                    metrics['takeoff_angular_error_rad'] = dataset_generator.compute_signed_angular_error(
                        output, self.dataset[1], t_start=t_start, t_stop=t_stop, mean=True
                    )

            if not pretrain:
                if self.cortex_error is not None:
                    metrics['cortex_error'] = self.cortex_error
                    metrics['nocb_error'] = self.compute_cortex_loss()
                else:
                    metrics['nocb_error'] = self.compute_cortex_loss()
            
            for k, v in metrics.items():
                self.metrics[k].append(v)
                if self.tensorboard:
                    self.writer.add_scalar(k, v, epoch)

            # debug utility
            if debug:
                if epoch % debug == 0:
                    self.printfunc(self.model.get_model_state())
                    additional_text = f", l1={self.l1_loss.item():.3E}, l2={self.l2_loss.item():.3E}, " \
                                      f"act_loss={self.activity_loss.item()})"
                    if self.training_params['lr_scheduler']:
                        additional_text += f", lr={lr_mem[-1]}"
                    self.printfunc(f"Epoch {epoch} ({text}): loss = {loss.item():.6E}"
                                   f"{additional_text} (mse={self.mse_loss:.6E}), runtime:"
                                   f"{str(datetime.now() - start_time)}\n\n\n\n",
                                   formatting=['bold', 'purple'])
            else:
                if epoch % print_steps == 0:
                    additional_text = ""
                    if verbose:
                        if self.training_params['l1'] > 0. or self.training_params['l2'] > 0. or self.training_params['activity_penalty']:
                            additional_text = f" , l1={self.l1_loss.item()}, l2={self.l2_loss.item()}," \
                                              f"act_loss={self.activity_loss.item()}"
                        if self.training_params['lr_scheduler']:
                            additional_text += f", lr={lr_mem[-1]}"
                        additional_text += f" | Runtime: {str(datetime.now() - start_time)}"
                    self.printfunc(
                        f"Epoch {epoch} ({text}): loss = {loss.item()} (mse={self.mse_loss}{additional_text})")

            self.net_buffer.add(copy.deepcopy(self.model.state_dict()))
            if save_w_steps > 0 and epoch % save_w_steps == 0:
                state_dict = self.sd_to_cpu(copy.deepcopy(self.model.state_dict()), 'cpu')
                self.nets.append(state_dict)

            if self.training_params['early_stop']:
                stop, stop_pos = self.early_stopping()
                if stop:
                    best_loss = self.mse_losses[stop_pos]
                    early_stop_message = " (stopping, best model at epoch %d, loss = %f)" % (stop_pos, best_loss)
                    break
                else:
                    best_loss = loss.item()
            else:
                best_loss = loss.item()

        if self.training_params['l1'] > 0. or self.training_params['l2'] > 0. or self.training_params['activity_penalty']:
            reg_text = f" (mse={self.mse_loss:.6E}, l1={self.l1_loss.item():.3E}, l2={self.l2_loss.item():.3E}," \
                       f"act_loss={self.activity_loss.item():.3E})"
        else:
            reg_text = f" (mse={self.mse_loss:.6E})"
        self.printfunc(text + " completed at epoch %d. Tot-time: %s\n\tStopping loss = %.9f %s%s." % (
            epoch, str(datetime.now() - start_time), loss.item(), reg_text, early_stop_message),
                       formatting=['bold', 'green'])
        self.printfunc("Time: " + str(datetime.now()), formatting=['bold'], end='\n\n')

        results = {
            'tot_losses': self.tot_losses,
            'mse_losses': self.mse_losses,
            'metrics': self.metrics,
            'model_sd': self.model.state_dict()
        }

        if save_cortex:
            results['cortex_sd'] = self.model.cortex.state_dict()
        if save_w_steps > 0:
            results['parameters'] = self.nets
        if outsave_step > 0:
            results['output'] = outputs_memory
        if scheduler is not None:
            results['lr_mem'] = lr_mem
        if record_gradients:
            results['gradients'] = self.gradients
        if record_angle:
            results['angular_error_rad'] = self.metrics['angular_error_rad']
            results['takeoff_angular_error_rad'] = self.metrics['takeoff_angular_error_rad']
        performance = {
            'epochs': stop_pos if (self.training_params['early_stop'] and stop_pos is not None) else self.training_params['epochs'],
            'best_loss': best_loss,
            'tot_epochs': len(self.mse_losses)
        }
        return results, performance
