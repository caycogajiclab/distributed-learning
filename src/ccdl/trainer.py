# -*- coding: utf-8 -*-
# Created on 06/03/2023
# Author: Leonardo Agueci - leonardo.agueci@ens.psl.eu
"""Training loop for the cerebello-cortical network (:class:`CCNetTrainer`).

Handles standard (backprop + optional local plasticity) training of a
CerebelloCorticalNet on the reaching task: loss assembly (task and
regularisation terms), optimizer/scheduler setup, logging, and evaluation.
See the notes below on ``model_state`` and the scheduler convention.
"""
#
# (SOME) INFO on the code:
# model_state: dict controlling the active configuration of the network at
# each point in the computation graph. Keys:
#   'cereb_feedback'    : bool — whether the cerebellum's output is fed back
#                         to the cortex as input (correction signal).
#   'cereb_readout'     : bool — whether the cerebellum contributes to the
#                         final motor output.
#   'modules_plasticity': dict or None — per-module gradient enable flags
#                         (e.g. {'cortex_hw': True, 'cereb': False}).
#                         None means use the model's current requires_grad state.
#   'local_plasticity'  : bool — whether the cortex hidden layer is updated
#                         via a local Hebbian/delta rule in addition to backprop.
#   'plant'             : bool — whether the biomechanical plant (arm model)
#                         is active in the forward pass.
#
# self.model_state  : active state during training, set via model.train(**self.model_state) 
#                     or model.eval(**self.model_state)
# self.cx_only_state: cerebellum fully disabled — used for cx-only loss eval
#                     and for testing cortex performance in isolation.
# self.cbfb_state   : cereb_feedback enabled, readout disabled (cx+fb loss).
#
# Scheduler convention for cx_scheduler / cb_scheduler:
#   - None / False : no scheduler
#   - True         : CosineAnnealing, T_max = epochs
#   - integer N    : CosineAnnealing, T_max = N

import sys
from copy import deepcopy
from datetime import datetime
from collections import OrderedDict
import warnings

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard.writer import SummaryWriter

from ccdl.utils import *
from ccdl.reaching import *
from ccdl.models import *
from ccdl.ccnet import CerebelloCorticalNet


class CCNetTrainer:
    
    def __init__(self, net_params, train_params=None, 
                 model_class='CerebelloCorticalNet', opt=torch.optim.Adam,
                 criterion=nn.MSELoss, logfile=None, verbose=False):

        self.logfile    = logfile
        self.net_params = net_params
        self.device     = self.net_params['device']
        self.opt        = opt
        self.criterion  = criterion()
        self.verbose    = verbose

        
        assert type(model_class) == str, "model_class must be a string"
        assert model_class in globals(), f"model_class {model_class} not found"
        model_class      = globals()[model_class]
        self.model       = self.init_model(model_class)
        self.model_state = self.model.default_model_state()
        self.dataset     = (None, None)
        
        # initialize training parameters
        self.training_params = self.store_train_params(train_params) if train_params else {}
        self.pretrain = False

        # variables initialization
        self.mse_losses   = []   # MSE only
        self.tot_losses   = []   # mse_loss + reg
        self.outputs      = None
        self.nets         = []
        self.gradients    = {}
        self.gradients_max         = {}
        self.clipped_gradients_max = {}
        self.cortex_error = None
        self.dataset_generator = None

        self.force              = False  # variable used for trainings without interruptions
        self.correction_dataset = None   # init correction dataset as None
        self.input_2_type       = None   # default feedback type
        self.prev_output        = None
        self.changes            = {}

        # cx_only_state: evaluation state where the cerebellum is fully disabled
        # (no feedback, no readout, no plasticity). Used to compute the cortex-only
        # loss and to evaluate cortex performance in isolation.
        self.cx_only_state = {
            'cereb_feedback': False, 'cereb_readout': False,
            'modules_plasticity': None, 'local_plasticity': False,
            'plant': self.model_state['plant']
        }
        self.cb_fb_state = {'cereb_feedback':True,'cereb_readout':False,
                            'modules_plasticity':None,'local_plasticity':False,
                            'plant':self.model_state['plant']}
    
    @staticmethod
    def default_training_params():
        return {
            'cx_lr':            1e-6,
            'cb_lr':            1e-4,
            'cx_scheduler':     False,
            'cb_scheduler':     False,
            'epochs':           100,
            'l1':               0.,
            'l2':               0.,
            'activity_penalty': 0.,
            'reg_drop':         None,   # (multiplicative factor, application epoch); None if not used
            'early_stop':       False,
            'local_lr':         1e-4,
            'local_method':     'mean',
            'fb_switch': False,   # whether to toggle cereb_feedback on/off in sync with local rule steps
            'es_params':        [300, 1e-9],
            'es_patience':      300,
            'es_eps':           1e-9,
            'grad_clip': None,   # (float) None = disabled,
            'cx_error':         'cx-only',
        }
    
    def compute_loss(self, outputs, target, activity, batch_train=True):
        n = target.shape[1]
        assert any([name in self.training_params.keys() for name in ['l1', 'l2', 'activity_penalty']]), "Please define l1, l2 and activity_penalty, necessary for computing loss"

        if batch_train:
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
            if self.training_params['activity_penalty']  > 0.:
                activity_loss = self.training_params['activity_penalty']  * self.activity_loss_function(activity)
                self.activity_loss = activity_loss
                loss += activity_loss
        else:
            loss = 1 / n * self.criterion(outputs, target)
            self.mse_loss += loss.detach().clone().item()

            if self.training_params['l1'] > 0.:
                l1_loss = 1 / n * (self.training_params['l1'] * self.L1_Loss())
                self.l1_loss += l1_loss
                loss += l1_loss
            if self.training_params['l2'] > 0.:
                l2_loss = 1 / n * (self.training_params['l2'] * self.L2_Loss())
                self.l2_loss += l2_loss
                loss += l2_loss
            if self.training_params['activity_penalty']  > 0.:
                activity_loss = 1 / n * (self.training_params['activity_penalty']  * self.activity_loss_function(activity))
                self.activity_loss += activity_loss
                loss += activity_loss
        return loss

    def compute_cortex_loss(self, thetas = None, input_2=None, input_2_type=None,
                            temp_state=None, debug=False):
        if thetas is None:
            temp_state = self.cx_only_state if temp_state is None else temp_state
            self.model.train(**temp_state)
            if debug:
                self.printfunc("cortex_loss model state:",formatting=['purple'])
                self.printfunc(self.model.get_model_state(),formatting=['purple'])
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
                outputs[t] = self.model.plant(thetas[0,t])
            error = self.criterion(outputs, targets)
            self.model.train(**self.model_state)
        return error

    def _training_loop(self, epoch, optimizers, record_gradients, debug_step):
        if debug_step:
            torch.autograd.set_detect_anomaly(True)

        self.model.train(**self.model_state, pretraining=self.pretrain)

        inputs, targets = self.dataset
        assert inputs is not None, "dataset was not properly initialized (inputs)."
        assert targets is not None, "dataset was not properly initialized (targets)."
        self.reset_loss()
        batch_size = inputs.shape[1]

        input_2 = None
        if 'correction_fb' in self.training_params and self.training_params['correction_fb']:
            input_2 = self.correction_dataset

        if 'fb' in optimizers.keys() and epoch == self.training_params['fb_lr_start']:
            self.printfunc(f"Starting feedback learning (epoch {epoch}).", ['bold', 'underline', 'blue'])

        outputs, activity, thetas = self._model_call(inputs, input_2=input_2, input_2_type=self.input_2_type)
        thetas_mem = {'cortex': thetas[0].abs().mean().item(),
                    'cereb':  thetas[1].abs().mean().item()}
        if debug_step:
            mean_hidden_activity = activity.abs().mean().item()

        loss = self.compute_loss(outputs, targets, activity)

        # ==================
        # backprop part
        # ==================
        grad_clip = self.training_params.get('grad_clip')
        grads, grads_max, premax = {}, {}, {}

        cx_error = self.training_params.get('cx_error', 'cx-only') or 'cx-only'

        for key, optimizer in optimizers.items():
            if key == 'fb' and epoch < self.training_params.get('fb_lr_start', 0):
                continue
            if key == 'cx' and epoch < self.training_params.get('cx_lr_start', 0):
                continue

            optimizer.zero_grad()

            if key == 'cx':
                if debug_step and self.model_state['local_plasticity']:
                    warnings.warn("Using Cortex optimizer within local trainer")
                if epoch == 1:
                    self.printfunc(f'cx optimizer, using error {cx_error}',
                                formatting=['purple'], verbose=True)
                print_state_flag = epoch % debug_step == 0 if debug_step else False
                if cx_error == 'cx-only':
                    cx_loss = self.compute_cortex_loss(temp_state=self.cx_only_state,
                                                    debug=print_state_flag)
                elif cx_error == 'cx+fb':
                    cx_loss = self.compute_cortex_loss(temp_state=self.cb_fb_state,
                                                    debug=print_state_flag)
                else:
                    self.cortex_error = loss.item()
                    if grad_clip:
                        if record_gradients:
                            premax.update(self._snapshot_grad_maxima({key: optimizer}))
                        params_list = [p for pg in optimizer.param_groups for p in pg['params']]
                        torch.nn.utils.clip_grad_norm_(params_list, max_norm=grad_clip)
                    if record_gradients or debug_step:
                        g, gmax = self._record_gradients({key: optimizer})
                        grads.update(g); grads_max.update(gmax)
                    optimizer.step()
                    continue

                self.cortex_error = cx_loss.clone().item()
                if np.abs(cx_loss.item()) > 0.:
                    cx_loss.backward()
                else:
                    continue

            else:
                if np.abs(loss.item()) > 0.:
                    _cx_active = ('cx' in optimizers
                                and epoch >= self.training_params['cx_lr_start']
                                and cx_error not in ('cx-only', 'cx+fb'))
                    loss.backward(retain_graph=_cx_active)
                else:
                    continue

            # grad clip
            if grad_clip:
                if record_gradients:
                    premax.update(self._snapshot_grad_maxima({key: optimizer}))
                params_list = [p for pg in optimizer.param_groups for p in pg['params']]
                torch.nn.utils.clip_grad_norm_(params_list, max_norm=grad_clip)

            if record_gradients or debug_step:
                g, gmax = self._record_gradients({key: optimizer})
                grads.update(g); grads_max.update(gmax)

            optimizer.step()

        # ==================
        # local rule
        # ==================
        delta_w = None
        if (self.model_state['local_plasticity']
                and epoch > self.training_params['local_rule_start']
                and epoch % self.training_params['local_rule_step'] == 0):
            delta_w = self.model.delta_rule_on_cortex_(
                self.training_params['local_lr'],
                self.training_params['local_method'],
                return_deltaw=True
            ).abs().mean().item()

        if record_gradients:
            self._store_gradients(grads, grads_max, delta_w, epoch, premax=premax)

        if debug_step and epoch % debug_step == 0:
            self._print_grad_summary(grads, grads_max, premax, delta_w,
                                    thetas_mem, mean_hidden_activity, epoch)

        if self.tensorboard:
            self.writer.add_scalars('thetas', thetas_mem, epoch)
        if debug_step and epoch % debug_step == 0:
            mean_hidden_activity = mean_hidden_activity / batch_size
            self.printfunc(f"\nmean hidden activity: {mean_hidden_activity}")
            self.printfunc("THETAS ABS-MEAN:")
            self.printfunc(thetas_mem)
            self.printfunc("PLANT:")
            self.printfunc(self.model.plant.__class__)

        self.mse_losses.append(self.mse_loss)
        self.tot_losses.append(loss.item())
        if 'cx' not in optimizers:
            self.cortex_error = self.compute_cortex_loss(thetas, input_2=input_2,
                                                        input_2_type=self.input_2_type).detach().item()
        return loss, outputs

    def train(self, dataset_generator, model_state=None, pretrain=False, washout=False, datatype=None,
              train_params=None, opt=None, outsave_step=0, print_steps=1, save_w_steps=0, record_gradients=False,
              save_cortex=True, verbose=True, debug_step=False, tensorboard_params={}, text="", logfile=None, 
              record_changes=False, force=False, **kwargs):
        
        assert not pretrain or type(self) is not CCNetTrainer, (
            "pretrain=True passed to CCNetTrainer. Use CCNetPreTrainer for pretraining."
        )
        
        optimizer = self._setup_training(
            model_state, train_params, opt, logfile,
            verbose, force, dataset_generator,
            tensorboard_params, debug_step, kwargs)
        
        self._setup_dataset(dataset_generator, datatype, washout, debug_step, verbose)
        
        optimizers, schedulers, T_maxes = self._setup_optimizers(optimizer)
        
        
        # initializing TensorBoard writer
        # kwargs: (log_dir=None,comment='',purge_step=None,max_queue=10,flush_secs=120,filename_suffix='')
        if self.tensorboard:
            self.writer = SummaryWriter(**tensorboard_params)
        
        (output, best_loss,
         outputs_memory, save_idx,
         early_stop_message, stop_pos) = self._pre_loop_init(
            optimizers, outsave_step, save_w_steps,
            record_gradients, record_changes, debug_step)

        start_time = datetime.now()
        early_stop_message = ""
        sys.stdout.write("\033[K")  # Clear to the end of line
        self.printfunc(f'Starting training: {start_time}', 
                       ['bold','green'], verbose=True)

        if debug_step:
            self.printfunc(f'Using learning rates: '+ "".join([f" {k}_lr={optimizers[k].param_groups[0]['lr']}" for k in optimizers.keys()]))
            if self.model_state['local_plasticity']:
                self.printfunc(f"local plasticity on. input2h_2.requires_grad = {self.model.cortex.rnn.input2h_2.weight.requires_grad}", formatting=['red'])

        if self.model_state['local_plasticity'] and self.model.cortex.rnn.h2h.weight.requires_grad:
            self.printfunc(f"WARNING: local rule on while hidden layer requires_grad=True", formatting=['red'])
        
        for epoch in range(1, self.training_params['epochs'] + 1):

            for item in ['l1', 'l2', 'activity_penalty']:
                if item in self.hyperparams:
                    self.hyperparams[item].append(self.training_params[item])
            for key in optimizers:
                self.hyperparams[f'{key}_lr'].append(
                    optimizers[key].param_groups[0]['lr'])
            if self.tensorboard:
                for k, vals in self.hyperparams.items():
                    if vals:
                        self.writer.add_scalar(k, vals[-1], epoch)

            if epoch == self.training_params['reg_drop_time']:
                self.training_params['l1'] *= self.training_params['reg_drop_factor']
                self.training_params['l2'] *= self.training_params['reg_drop_factor']
                self.training_params['activity_penalty']  *= self.training_params['reg_drop_factor']
                
                self.printfunc(
                    f"Decreasing regularizers. new values: "
                    f"l1 = {self.training_params['l1'] }, "
                    f"l2 = {self.training_params['l2'] }, " 
                    f"activity penalty = {self.training_params['activity_penalty'] }\n", 
                    formatting=['red', 'bold'])

            # local rule handler
            if self.model_state['local_plasticity']:
                if self.training_params['fb_switch']:
                    if epoch> self.training_params['local_rule_start'] and epoch % self.training_params['local_rule_step'] == 0:
                        self.model_state['cereb_feedback'] = True
                    else:
                        self.model_state['cereb_feedback'] = False                    
            
            loss, output = self._training_loop(epoch, optimizers, record_gradients, debug_step)
            self.prev_output = output.detach().clone()

            
            # --- scheduler step ---
            for key, sched in schedulers.items():
                if epoch <= T_maxes[key]:
                    sched.step()
        
            # --- net buffer + changes ---
            self.net_buffer.add(deepcopy(self.model.state_dict()))
            if record_changes:
                parameter_changes = self.compute_parameter_changes(
                    self.net_buffer.buf[-1], self.net_buffer.buf[-2])
                for key in self.changes.keys():
                    self.changes[key].append(parameter_changes[key])
                    
            # --- weight checkpoint ---
            if save_w_steps > 0 and epoch % save_w_steps == 0:
                self.nets.append(
                    self.sd_to_cpu(deepcopy(self.model.state_dict()), 'cpu'))
            
            # --- early stopping ---
            if self.training_params['early_stop']:
                stop, stop_pos = self.early_stopping()
                if stop:
                    best_loss = self.mse_losses[stop_pos]
                    early_stop_message = " (stopping, best model at epoch %d, loss = %f)" % (stop_pos,
                                                                                             best_loss)
                    break
                else:
                    best_loss = loss.item()
            else:
                best_loss = loss.item()
                
            ###########################
            # Saving data
            ###########################
            
            # --- output snapshot ---
            if outsave_step > 0 and epoch % outsave_step == 0:
                outputs_memory[save_idx] = output.detach().cpu()
                save_idx += 1

            metrics = {
                'loss': self.tot_losses[-1],
                'mse': self.mse_loss,
                'l1_loss': self.l1_loss.item(),
                'l2_loss': self.l2_loss.item(),
                'activity_loss': self.activity_loss.item(),
                #saving learning rates
                'cx_lr': optimizers['cx'].param_groups[0]['lr'] if 'cx' in optimizers else None,
                'cb_lr': optimizers['cb'].param_groups[0]['lr'] if 'cb' in optimizers else None,
            }
            
            metrics['cortex_error'] = self.compute_cortex_loss(
                temp_state=self.cb_fb_state).detach().item()
            metrics['nocb_error'] = self.compute_cortex_loss(
                temp_state=self.cx_only_state).detach().item()

            for k, v in metrics.items():
                self.metrics[k].append(v)
                if self.tensorboard:
                    self.writer.add_scalar(k, v, epoch)
                
            # --- logging ---
            if debug_step and epoch % debug_step == 0:
                additional_text = (
                    f", l1={self.l1_loss.item():.3E}, "
                    f"l2={self.l2_loss.item():.3E}, "
                    f"act_loss={self.activity_loss.item():.3E}")
                for key in optimizers:
                    vals = self.hyperparams[f'{key}_lr']
                    if vals:
                        additional_text += f", {key}_lr={vals[-1]:.3E}"
                self.printfunc(
                    f"Epoch {epoch} ({text}):\n loss={loss.item():.6E} "
                    f"mse={self.mse_loss:.6E}, "
                    f"{additional_text}"
                    f"runtime={str(datetime.now() - start_time)}\n\n\n\n",
                    formatting=['bold', 'purple'])
            
            elif epoch % print_steps == 0:
                additional_text = ""
                if verbose:
                    if self.training_params['l1'] > 0. \
                            or self.training_params['l2'] > 0. \
                            or self.training_params['activity_penalty'] :
                        additional_text += (
                            f", l1={self.l1_loss.item()}, "
                            f"l2={self.l2_loss.item()},"\
                            f"act_loss={self.activity_loss.item()}")
                    # add if condition for plotting learning rates if schedulers are used
                    if self.training_params['cx_scheduler'] or self.training_params['cb_scheduler']:
                        additional_text += f" | cx_lr: {optimizers['cx'].param_groups[0]['lr']}" if 'cx' in optimizers else ""
                        additional_text += f", cb_lr: {optimizers['cb'].param_groups[0]['lr']}" if 'cb' in optimizers else ""
                    for key in optimizers:
                        vals = self.hyperparams[f'{key}_lr']
                        if vals:
                            additional_text += f", {key}_lr={vals[-1]:.3E}"
                    additional_text += f" | Runtime: {str(datetime.now() - start_time)}"
                elapsed = datetime.now() - start_time
                _cx_err_str = f"{self.cortex_error:.6}" if self.cortex_error is not None else "n/a"
                self.printfunc(
                    f"Epoch {epoch:5d}   ({text+'): ':<{35}s} "
                    f"loss = {loss.item():.6E} "\
                    f"\t Elapsed: {str(elapsed).split('.')[0]} "\
                    f"(mse={self.mse_loss:.6}, cx_error={_cx_err_str}{additional_text})."
                    )


        if self.training_params['l1'] > 0. or self.training_params['l2'] > 0. or self.training_params['activity_penalty'] :
            # TODO: error, reg_loss was changed. Solve it!!!
            reg_text = f" (mse={self.mse_loss:.6E}, l1={self.l1_loss.item():.3E}, l2={self.l2_loss.item():.3E}," \
                       f"act_loss={self.activity_loss.item():.3E})"
        else:
            reg_text = f" (mse={self.mse_loss:.6E})"
        self.printfunc(text + " completed at epoch %d. Tot-time: %s\n\tStopping loss = %.9f %s%s." % (
            epoch, str(datetime.now() - start_time), loss.item(), reg_text, early_stop_message),
                       formatting=['bold', 'green'])
        self.printfunc("Time: "+str(datetime.now()), formatting=['bold'], end='\n\n', verbose=True)
        self.printfunc("Print model\n"+self.model.__repr__(), verbose=True)

        results = {'tot_losses': self.tot_losses,
                   'mse_losses': self.mse_losses,
                   'metrics': self.metrics,
                   'model_sd': self.sd_to_cpu(self.model.state_dict()),
                   'model_state': self.model_state
                   }
        
        if save_cortex:
            results['cortex_sd'] = self.sd_to_cpu(self.model.cortex.state_dict())
        if save_w_steps > 0:
            results['parameters'] = self.nets
        if outsave_step > 0:
            results['output'] = outputs_memory
        if len(schedulers) > 0:
            results['lr_mem'] = {k:v for k,v in self.hyperparams.items() if k.endswith('_lr')}
        if record_gradients:
            results['gradients'] = self.gradients
            results['gradients_max'] = self.gradients_max
            if self.training_params.get('grad_clip'):
                results['clipped_gradients_max'] = self.clipped_gradients_max
        if record_changes:
            results['changes'] = self.changes
        performance = {
            'epochs': stop_pos if (self.training_params['early_stop'] and stop_pos is not None) else self.training_params['epochs'],
            'best_loss': best_loss,
            'tot_epochs': len(self.mse_losses)
        }
        return (results, performance)
    
    # ------------------------------------------------------------------ #
    #  Supporting methods                                                  #
    # ------------------------------------------------------------------ #
    
    
    def _setup_training(self, model_state, train_params, opt, logfile,
                        verbose, force, dataset_generator,
                        tensorboard_params, debug_step, kwargs):
        """Handles flags, params validation, deprecated checks, optimizer class
        selection. Returns the optimizer class to use."""
        if kwargs:
            self.printfunc(f"Warning: the following kwargs were not recognized and will be ignored: {kwargs.keys()}")

        self.force       = force
        self.tensorboard = True if tensorboard_params != {} else False
        self.logfile     = logfile
        self.verbose     = verbose

        if debug_step:
            self.printfunc("Starting training in 'train' mode")

        if model_state is not None:
            assert type(model_state) == dict and model_state.keys() == self.model_state.keys(), \
                "Please provide as model_state a dict with keys: %s" % str(list(self.model_state.keys()))
            self.model_state = model_state

        # updating training_params
        if not train_params and not self.training_params:
            self.printfunc('No training parameters provided. Using default settings')
        self.training_params = self.store_train_params(train_params)

        _deprecated = {
            'lr':           "use 'cx_lr' and 'cb_lr' instead",
            'lr_scheduler': "use 'cx_scheduler' and 'cb_scheduler' instead",
            'T_max':        "T_max is now encoded directly in 'cx_scheduler'/'cb_scheduler' values",
        }
        for _key, _msg in _deprecated.items():
            assert _key not in self.training_params, \
                f"Deprecated training param '{_key}' found: {_msg}"

        self.dataset_generator           = dataset_generator

        self.printfunc('Training initialization...\n', verbose=True)

        assert hasattr(self, 'training_params'), \
            "training parameters not assigned yet. Please provide it as kwarg ('train_params')" \
            " or use store_train_params(train_params) method."

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

        self.model.train(**self.model_state, pretraining=self.pretrain)
        return optimizer

    def _setup_dataset(self, dataset_generator, datatype, washout, debug_step, verbose):
        """Generates all datasets, handles
        washout. Sets self.dataset, self.unperturbed_dataset,
        self.correction_dataset, self.input_2_type, self.init_theta."""
        # Generating the dataset (works with reaching generator only)
        dt_text = datatype if datatype is not None else 'Default'
        self.printfunc(f"Using {dt_text} dataset.", verbose=True)

        correction_datatype = None
        input_2_type        = None
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
        self.input_2_type = input_2_type

        self.dataset = dataset_generator.generate_perturbed_dataset(datatype)
        self.unperturbed_dataset = dataset_generator.generate_unperturbed_dataset(datatype)
        if 'correction_fb' in self.training_params.keys() and self.training_params['correction_fb']:
            self.correction_dataset = dataset_generator.generate_perturbed_dataset(correction_datatype)[1]
        else:
            self.correction_dataset = None
        
        # after self.correction_dataset = ... assignments
        _target_space  = 'angles' if datatype == 'angles' else 'cart'
        _unperturbed_space = 'angles' if datatype == 'angles' else 'cart'  # same call
        _fb_space = (f"correction_dataset → {correction_datatype or 'cart'} "
                    f"(input_2_type: {input_2_type})"
                    if self.correction_dataset is not None
                    else "no feedback")
        self.printfunc(
            f"[dataset spaces] "
            f"target: {_target_space}"
            + (f" | unperturbed: {_unperturbed_space}")
            + f" | feedback: {_fb_space}"
        )

        if washout:
            datatype = datatype if datatype is not None or self.model.plant.name() == 'Identity' else 'cart'
            self.model.train(**self.model_state)
            self.dataset = dataset_generator.generate_unperturbed_dataset(datatype)
            self.unperturbed_dataset = dataset_generator.generate_perturbed_dataset(datatype)
            self.printfunc("Washout selected")

        if debug_step or verbose:
            self.printfunc("\nModel summary:", ['green', 'underline'])
            self.printfunc(self.model)
            self.printfunc("\nModel state:", ['green', 'underline'])
            self.printfunc(self.model.get_model_state())


    def _setup_optimizers(self, optimizer):
        """Builds optimizers dict and schedulers. Returns
        (optimizers OrderedDict, schedulers dict, T_maxes dict)."""
        ## SETTING UP OPTIMIZERs
        optimizers = {}
        cx_lr = self.training_params.get('cx_lr')
        fb_lr = self.training_params.get('fb_lr', None)
        cb_lr = self.training_params.get('cb_lr', None)
        assert self.training_params.get('lr', None) is None, \
            "lr is deprecated, please specify cx/fb/cb_lr explicitly."
        betas = tuple(self.training_params.get('betas', (0.9, 0.999)))

        # --- CORTEX optimizer ---
        cx_params, cx_names = self.plastic_params(self.model.cortex)

        # Extract fb params only if a dedicated fb_lr is requested
        if fb_lr:
            fb_indices = [i for i, n in enumerate(cx_names)
                        if n in ('rnn.input2h_2.weight', 'rnn.input2h_2.bias')]
            fb_params = [cx_params[i] for i in fb_indices]
            fb_names = [cx_names[i] for i in fb_indices]
            cx_params = [p for i, p in enumerate(cx_params) if i not in fb_indices]
            cx_names = [n for i, n in enumerate(cx_names) if i not in fb_indices]
            if not self.training_params.get('fb_lr_start'):
                self.training_params['fb_lr_start'] = 0
            self.printfunc(f"Using custom learning rate ({fb_lr:.2E}) for fb-weights (lr=0 until epoch {self.training_params['fb_lr_start']})", verbose=True)
            self._fb_group_idx = 1
            self._fb_lr = fb_lr
        else:
            self._fb_group_idx = None
            self._fb_lr = None

        if len(cx_params) > 0:
            assert cx_lr, "cx_lr must be provided with plastic parameters in the cortex"
            if not self.training_params.get('cx_lr_start'):
                self.training_params['cx_lr_start'] = 0
            self.printfunc(f"Using learning rate={cx_lr:.2E} for cortex", verbose=True)
            param_groups = [{'params': cx_params, 'lr': cx_lr, 'betas': betas, 'param_names': cx_names}]
            if fb_lr:
                param_groups.append({'params': fb_params, 'lr': 0.0, 'betas': betas, 'param_names': fb_names})
            optimizers['cx'] = optimizer(param_groups)

        # --- CEREBELLUM optimizer ---
        cb_params, cb_names = self.plastic_params(self.model.cerebellum)                       
        if len(cb_params) > 0:
            assert cb_lr, "cb_lr must be provided (lr is deprecated)"
            self.printfunc(f"Using learning rate={cb_lr:.2E} for cerebellum", verbose=True)
            optimizers['cb'] = optimizer([{'params': cb_params, 'lr': cb_lr,                  
                                            'betas': betas, 'param_names': cb_names}])

        # Reorder: cb before cx within each epoch (sorting alphabetically)
        optimizers = OrderedDict((k, optimizers[k]) for k in sorted(optimizers))

        # --- SCHEDULERS ---
        schedulers = {}
        T_maxes    = {}

        cx_sched, cx_T = self._init_scheduler(
            'cx', self.training_params.get('cx_scheduler'), optimizers, cx_lr)
        if cx_sched:
            schedulers['cx'] = cx_sched
            T_maxes['cx']    = cx_T

        cb_sched, cb_T = self._init_scheduler(
            'cb', self.training_params.get('cb_scheduler'), optimizers, cb_lr)
        if cb_sched:
            schedulers['cb'] = cb_sched
            T_maxes['cb']    = cb_T

        if self.training_params.get('cx_error', None) not in ['cx-only', 'cx+fb', None]:
            warnings.warn("\nOnly 'cx-only' and 'cx+fb' are supported for cx_error. This might raise errors.\n")

        if self.model_state['local_plasticity'] and 'cx' in optimizers.keys():
            self.printfunc("WARNING: Using cortex optmization with local_plasticity.",
                        formatting=['bold', 'underline', 'red'])
        
        return optimizers, schedulers, T_maxes


    def _pre_loop_init(self, optimizers, outsave_step, save_w_steps,
                    record_gradients, record_changes, debug_step):
        """Runs initial eval pass, initialises all tracking structures
        (metrics, hyperparams, net_buffer, changes, outputs_memory).
        Returns (output, unp_output, best_loss, outputs_memory, save_idx,
                early_stop_message, stop_pos)."""
        # initializing gradients storing variable
        if record_gradients:
            # derive fb_names directly from the already-built optimizers dict
            _fb_names = ([optimizers['cx'].param_groups[self._fb_group_idx]['param_names']]
                        if self._fb_group_idx is not None and 'cx' in optimizers else [])
            _fb_names = _fb_names[0] if _fb_names else []

            self.printfunc("\nGradients will be recorded.", verbose=True)
            self.gradients['cortex']       = {name: [] for (_, name) in zip(*self.plastic_params(self.model.cortex))}
            self.gradients['cerebellum']   = {name: [] for _, name in zip(*self.plastic_params(self.model.cerebellum))}
            self.gradients['feedback']     = {name: [] for name in _fb_names}

            self.gradients_max['cortex']     = {name: [] for (_, name) in zip(*self.plastic_params(self.model.cortex))}
            self.gradients_max['cerebellum'] = {name: [] for _, name in zip(*self.plastic_params(self.model.cerebellum))}
            self.gradients_max['feedback']   = {name: [] for name in _fb_names}
            if self.training_params.get('grad_clip'):
                self.clipped_gradients_max['cortex']     = {name: [] for (_, name) in zip(*self.plastic_params(self.model.cortex))}
                self.clipped_gradients_max['cerebellum'] = {name: [] for _, name in zip(*self.plastic_params(self.model.cerebellum))}
                self.clipped_gradients_max['feedback']   = {name: [] for name in _fb_names}

            self.gradients['cortex_local'] = []
        
        with torch.no_grad():
            self.model.eval(**self.model_state)
            output, activity, thetas = self._model_call(
                self.dataset[0], input_2=self.correction_dataset,
                input_2_type=self.input_2_type)
            if 'correction_fb' in self.training_params.keys() and self.training_params['correction_fb']:
                self.printfunc("\nUsing correction feedback.", ["bold", "underline"], verbose=True)
                self.prev_output = output.detach().clone()

            pretrain_model_state = {
                'cereb_feedback': False, 'cereb_readout': False,
                'modules_plasticity': None, 'local_plasticity': False,
                'plant': self.model_state['plant']}
            self.model.eval(**pretrain_model_state)
            if debug_step:
                self.printfunc("Pretrain model_state: %s" % str(pretrain_model_state), verbose=True)
                self.printfunc("Pretrain model: %s" % str(self.model), verbose=True)
            unp_output, _, _ = self._model_call(
                self.unperturbed_dataset[0],
                input_2=self.unperturbed_dataset[1],
                input_2_type=self.input_2_type)
            self.model.train(**self.model_state)
            self.printfunc("Current model: \n%s" % str(self.model), verbose=True)
            self.printfunc(
                f"\nTesting states:\ncx_only_state:\n{self.cx_only_state}\ncb_fb_state:\n{self.cb_fb_state}",
                verbose=True)

            if debug_step:
                if len(self.dataset[1].shape) == 3:
                    self.printfunc("params test:")
                    self.printfunc("rnn.h2h.weight: "     + str(self.model.cortex.rnn.h2h.weight.cpu().numpy()[0, :3]))
                    self.printfunc("rnn.h2h.bias: "       + str(self.model.cortex.rnn.h2h.bias.cpu().numpy()[:3]))
                    self.printfunc("rnn.input2h.weight: " + str(self.model.cortex.rnn.input2h.weight.cpu().numpy()[0, :3]))
                    self.printfunc("rnn.input2h.bias: "   + str(self.model.cortex.rnn.input2h.bias.cpu().numpy()[:3]))
                    self.printfunc("rnn.input2h_2.weight: " + str(self.model.cortex.rnn.input2h_2.weight.cpu().numpy()[0, :3]))
                    self.printfunc("fc.weight: "          + str(self.model.cortex.fc.weight.cpu().numpy()[0, :3]))
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
        best_loss       = loss.item()
        self.tot_losses = [loss.item()]
        self.mse_losses = [self.mse_loss]
        self.metrics = {
            'loss': [], 'mse': [], 'l1_loss': [], 'l2_loss': [], 'activity_loss': [], 'cx_lr': [], 'cb_lr': []
        }
        self.metrics['cortex_error'] = []
        self.metrics['nocb_error']   = []

        self.hyperparams = {item: [] for item in ['l1', 'l2', 'activity_penalty']
                            if self.training_params.get(item, 0.) != 0.}
        for key in optimizers:
            self.hyperparams[f'{key}_lr'] = []

        self.printfunc(
            f"Starting loss: {self.tot_losses[0]} (mse_loss: {self.mse_loss}, "
            f"l1_loss: {self.l1_loss.item()}, "
            f"l2_loss: {self.l2_loss.item()}, activity_loss: {self.activity_loss.item()})\n")

        self.printfunc("Unperturbed loss: {}".format(
            self.criterion(unp_output, self.unperturbed_dataset[1])))

        # local rule params settings
        if self.model_state['local_plasticity']:
            assert self.model_state['cereb_feedback'], "local rule requires 'cereb_feedback' to be True"
            if 'local_rule_settings' in self.training_params.keys():
                local_rule_step, local_rule_start, fb_switch = self.training_params['local_rule_settings']
                assert isinstance(local_rule_step, int), "local_rule_step must be an integer"
                assert local_rule_step > 0, "local_rule_step must be positive"
                assert isinstance(local_rule_step, int), "local_rule_step must be an integer"
                assert local_rule_step > 0, "local_rule_step must be positive"
                assert isinstance(fb_switch, bool), "fb_switch must be a boolean"
                self.training_params['local_rule_step']  = local_rule_step
                self.training_params['local_rule_start'] = local_rule_start
                self.training_params['fb_switch']        = fb_switch
            else:
                warnings.warn("local_rule_settings not provided. Using default values: step = 1, start = 0, fb switch = False")
                self.training_params['local_rule_step'] = 1
                self.training_params['local_rule_start'] = 0
                self.training_params['fb_switch'] = False

        metrics = {
            'loss': self.tot_losses[0], 'mse': self.mse_loss,
            'l1_loss': self.l1_loss.item(), 'l2_loss': self.l2_loss.item(),
            'activity_loss': self.activity_loss.item()
        }
            
        metrics['cortex_error'] = self.compute_cortex_loss(temp_state=self.cb_fb_state).detach().item()
        metrics['nocb_error']   = self.compute_cortex_loss(temp_state=self.cx_only_state).detach().item()
        for k, v in metrics.items():
            self.metrics[k].append(v)
            if self.tensorboard:
                self.writer.add_scalars(k, v, 0)

        if outsave_step:
            self.outputs = [output]

        # parameters buffer initialization
        self.net_buffer = Buffer(max(self.training_params['es_patience'], 100))
        if save_w_steps > 0:
            self.nets = [self.sd_to_cpu(deepcopy(self.model.state_dict()), 'cpu')]
        self.net_buffer.add(deepcopy(self.model.state_dict()))
        if record_changes:
            self.changes = {key: [] for key, _ in self.net_buffer.buf[-1].items()}

        outputs_memory = []
        save_idx       = 0
        if outsave_step > 0:
            n_saves        = self.training_params['epochs'] // outsave_step
            outputs_memory = torch.zeros(n_saves, *output.shape, device='cpu')

        return output, best_loss, outputs_memory, save_idx, "", None

    def init_model(self, model_class):
        args = self.net_params['args']
        kwargs = deepcopy(self.net_params)
        kwargs.pop('args')
        return model_class(*args, **kwargs, logfile=self.logfile, verbose=self.verbose)

    def _model_call(self, x, **kwargs):
        # Internal method to handle conditional 'init_state' argument, used
        if 'initial_theta' in kwargs:
            var = kwargs.pop('initial_theta')
            return self.model(x, **kwargs, var=var)
        else:
            return self.model(x, return_hidden=True, return_thetas=True, **kwargs)
    
    def store_train_params(self, train_params):
        """method storing training parameters"""
        final_train_params = self.default_training_params()
        if train_params:
            final_train_params.update(train_params)
            final_train_params['es_patience'] = train_params['es_params'][0]
            final_train_params['es_eps'] = float(train_params['es_params'][1])
            if final_train_params['reg_drop'] is not None:
                final_train_params['reg_drop_factor'] = final_train_params['reg_drop'] [0]
                final_train_params['reg_drop_time'] = final_train_params['reg_drop'] [1]
            else:
                final_train_params['reg_drop_factor'] = 0.
                final_train_params['reg_drop_time'] = -1
            if final_train_params['local_method'] is None:
                final_train_params['local_method'] = 'mean'
        return final_train_params
    
    def _init_scheduler(self, key, sched_val, optimizer, base_lr):
        """
        Initializes a CosineAnnealingLR scheduler if sched_val is truthy.
        Returns (scheduler, T_max) or (None, None).
        sched_val: False/None → no scheduler; True → T=epochs; int N → T=N
        """
        if not sched_val:
            return None, None
        if key not in optimizer:  # optimizer is the optimizers dict
            self.printfunc(f"Ignoring {key}_scheduler ({key} not in optimizers).",
                        formatting=['red', 'bold'])
            return None, None
        T = self.training_params['epochs'] if sched_val is True else int(sched_val)
        sched = CosineAnnealingLR(optimizer[key], T, eta_min=base_lr / 100)
        self.printfunc(f"CosineAnnealing LR scheduler for {key} (T_max={T}).")
        return sched, T
        
    def load_cortex(self, state_dict):
        """method loading pretrained cortex"""
        try:
            self.model.cortex.load_state_dict(state_dict)
            self.printfunc("Pretrained CXmodel correctly loaded.")
        except:
            current_state = self.model.cortex.state_dict()
            self.printfunc("The state_dict provided does not match the current model structure.\n")
            self.printfunc("Saved state_dict keys:")
            self.printfunc(str([(name,value.shape,value.device) for name, value in state_dict.items()]))
            self.printfunc("\n\nCurrent state_dict keys:")
            self.printfunc(str([(name,value.shape,value.device) for name, value in current_state.items()]))
            raise IOError

    def load_cereb(self, state_dict):
        """method loading pretrained cortex"""
        try:
            self.model.cerebellum.load_state_dict(state_dict)
            self.printfunc("Pretrained model correctly loaded.")
        except:
            current_state = self.model.cerebellum.state_dict()
            self.printfunc("The state_dict provided does not match the current model structure.\n")
            self.printfunc("Saved state_dict keys:")
            self.printfunc(str([(name,value.shape) for name, value in state_dict.items()]))
            self.printfunc("\n\nCurrent state_dict keys:")
            self.printfunc(str([(name,value.shape) for name, value in current_state.items()]))
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
        """Helper function, returning all plastic parameters in a give module. if none, module is the entire net"""
        if not module:
            module = self.model
        names = []
        params = []
        for name, param in module.named_parameters():
            # counting only plastic parameters
            if param.requires_grad:
                if flatten:
                    params.append(param.reshape(-1))
                else:
                    params.append(param)
                names.append(name)
        return params, names

    def all_params(self, module=None):
        """Helper function, returning all parameters in a give module. if none, module is the entire net"""
        if not module:
            module = self.model
        names = []
        params = []
        for name, param in module.named_parameters():
            params.append(param)
            names.append(name)
        return params, names

    @staticmethod
    def sd_to_cpu(sd, device='cpu'):
        """Move state_dict to the new device"""
        for k, v in sd.items():
            sd[k] = v.to(device)
        return sd

    @staticmethod
    def plot_distro(param, bins=50):
        from matplotlib import pyplot as plt
        assert hasattr(param, 'shape'), "Please pass a valid parameter (weight, bias, etc)"
        with torch.no_grad():
            if len(param.shape) > 1:
                param = param.detach().view(-1).numpy()
                n, bias, patches = plt.hist(param, bins)
                plt.show()
    
    @staticmethod
    def compute_parameter_changes(new_sd, prev_sd):
        """
        Computes the changes in the parameters between two state dictionaries.
        
        Args:
            new_sd (dict): The new state dictionary (current step).
            prev_sd (dict): The previous state dictionary (previous step).
        
        Returns:
            dict: A dictionary containing the changes for each parameter as W(new) - W(prev).
        """
        changes = {}

        # Iterate over each parameter in the new state dict
        for key in new_sd:
            # Ensure the same parameter exists in the previous state dict
            if key in prev_sd:
                # Compute the difference W(new) - W(prev)
                changes[key] = (new_sd[key] - prev_sd[key]).abs().mean().item()
            else:
                raise ValueError(f"Parameter {key} found in new_sd but not in prev_sd")

        return changes

    def _record_gradients(self, optimizers):
        """
        Extracts mean and max absolute gradient for every param in each optimizer.
        Requires 'param_names' stored in each param group (set in _setup_optimizers).

        Returns:
            grads     : {key: {param_name: mean_abs_grad}}
            grads_max : {key: {param_name: max_abs_grad}}
        where key is the optimizer key ('cx', 'cb', 'fb').
        """
        grads, grads_max = {}, {}
        for key, opt in optimizers.items():
            grads[key], grads_max[key] = {}, {}
            for pg in opt.param_groups:
                for name, param in zip(pg['param_names'], pg['params']):
                    if param.grad is not None:
                        g = param.grad.view(-1).abs()
                        grads[key][name]     = g.mean().item()
                        grads_max[key][name] = g.max().item()
                    else:
                        grads[key][name]     = 0.
                        grads_max[key][name] = 0.
        return grads, grads_max
    
    def _snapshot_grad_maxima(self, optimizers):
        """
        Snapshot per-parameter gradient max-abs values before clipping.
        Requires 'param_names' in each param group (set in _setup_optimizers).

        Returns:
            premax : {key: {param_name: max_abs_grad}}
        where key is the optimizer key ('cx', 'cb', 'fb').
        """
        premax = {}
        for key, opt in optimizers.items():
            premax[key] = {}
            for pg in opt.param_groups:
                for name, param in zip(pg['param_names'], pg['params']):
                    premax[key][name] = param.grad.view(-1).abs().max().item() \
                                        if param.grad is not None else 0.
        return premax
    
    def _store_gradients(self, grads, grads_max, delta_w, epoch, premax=None):
        """
        Appends gradient values to self.gradients / gradients_max /
        clipped_gradients_max and pushes to tensorboard.

        grads / grads_max : {key: {name: val}}, key in ('cx','cb','fb')
        premax            : {key: {name: val}} pre-clip max, or None
        grads_max         : post-clip max (equals raw max when no clipping)
        """
        _key_to_store = {'cx': 'cortex', 'cb': 'cerebellum', 'fb': 'feedback'}
        grad_clip = self.training_params.get('grad_clip')

        for key, name_vals in grads.items():
            store_key = _key_to_store.get(key, key)
            for name, val in name_vals.items():
                self.gradients[store_key][name].append(val)
                raw_max = (premax or {}).get(key, {}).get(name,
                           (grads_max or {}).get(key, {}).get(name, 0.))
                self.gradients_max[store_key][name].append(raw_max)
                if grad_clip and grads_max:
                    self.clipped_gradients_max[store_key][name].append(
                        grads_max.get(key, {}).get(name, 0.))

        if delta_w is not None:
            self.gradients['cortex_local'].append(delta_w)

        if self.tensorboard:
            for key, name_vals in grads.items():
                if name_vals:
                    self.writer.add_scalars(f'{_key_to_store.get(key, key)}_grads',
                                            name_vals, epoch)
            if delta_w is not None:
                self.writer.add_scalar('cortex_local_grads', delta_w, epoch)
    
    def _print_grad_summary(self, grads, grads_max, premax, delta_w,
                            thetas_mem, mean_hidden_activity, epoch):
        """
        Prints a compact gradient/activity/theta summary.
        grads / grads_max / premax : {key: {name: val}}, key in ('cx','cb','fb').
        """
        _key_to_label = {'cx': 'cortex', 'cb': 'cerebellum', 'fb': 'feedback'}
        self.printfunc(f"Epoch {epoch} — gradient summary:")
        for key, name_vals in grads.items():
            if name_vals:
                self.printfunc(f"  [{_key_to_label.get(key, key)}]")
                for name, val in name_vals.items():
                    self.printfunc(f"    {name}: {val:.3e}")
        for key, name_vals in (premax or {}).items():
            if name_vals:
                self.printfunc(f"  [{_key_to_label.get(key, key)} - max-pre-clip]")
                for name, val in name_vals.items():
                    self.printfunc(f"    {name}: {val:.3e}")
        for key, name_vals in (grads_max or {}).items():
            if name_vals:
                self.printfunc(f"  [{_key_to_label.get(key, key)} - max-post-clip]")
                for name, val in name_vals.items():
                    self.printfunc(f"    {name}: {val:.3e}")
        if delta_w is not None:
            self.printfunc(f"  [local rule] Δw abs mean: {delta_w:.3e}")
        self.printfunc(f"  thetas:          {thetas_mem}")
        self.printfunc(f"  hidden activity: {mean_hidden_activity / self.dataset[0].shape[1]:.4f}")
        self.printfunc(f"  plant: {self.model.plant.__class__.__name__}")

    def printfunc(self, output, formatting=None, end=None, verbose=False):
        """
        Prints output to stdout and/or to logfile.
        By default always prints (verbose=False means 'not verbose-only').
        Pass verbose=True to suppress stdout when self.verbose is False — use this
        for detailed diagnostic info (model repr, parameter shapes, etc.).
        """
        def _print_to(stream, use_formatting):
            if type(output) == dict:
                stream.write('{\n')
                for k, v in output.items():
                    stream.write(f"{k}: {v},\n")
                stream.write('}\n')
            else:
                if use_formatting and formatting is not None:
                    pr_print(formatting, output, end=end)
                else:
                    print(output, end=end, file=stream)

        # Always log
        if self.logfile is not None:
            with open(self.logfile, 'a') as f:
                _print_to(f, use_formatting=False)

        # Terminal: respect verbose flag
        if not verbose or self.verbose:
            _print_to(sys.stdout, use_formatting=True)

    
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
        self.mse_loss = 0.  # torch.tensor(0.).to(self.device)
        self.activity_loss = torch.tensor(0.).to(self.device)
        self.l1_loss = torch.tensor(0.).to(self.device)
        self.l2_loss = torch.tensor(0.).to(self.device)
    
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

if __name__ == '__main__':
    from ccdl.reaching import ReachingDataset
    from time import sleep
    
    # fixing seeds
    torch.manual_seed(1234)
    np.random.seed(1234)

    
    output_size = 2
    rnn_hidden_size = 100

    reaching = ReachingDataset()
    timesteps, reaching_directions, input_size = reaching.input_size
    output_size = reaching.output_size[2]


    pr_print(['bold','red'],"Dataset:")
    print(f"input_size: {input_size}")
    print(f"timesteps: {timesteps}")
    print(f"reaching_directions: {reaching_directions}")
    print(f"output_size: {output_size}")
    print(f"hidden_size: {rnn_hidden_size}")
    
    net_params={
        "args": [input_size, rnn_hidden_size],
        "ct": False,
        "hidden_layer_specs": CerebelloCorticalNet.default_hidden_layer_specs(),
        "rnn_specs": CerebelloCorticalNet.default_rnn_specs(),
        "cereb_specs": CerebelloCorticalNet.default_cereb_specs(),
        "inter_specs": CerebelloCorticalNet.default_inter_specs(),
        "nonlinearity_specs": CerebelloCorticalNet.default_nonlinearity_specs(),
        "device": "cpu"
    }
    
    training_params = CCNetTrainer.default_training_params()
    training_params.update({
        'epochs':        10,
        'cb_lr':         1e-4,
        'cx_lr':         1e-6,
        'cb_scheduler':  10,   # CosineAnnealing T_max=10 (replaces old lr_scheduler+T_max)
        'fb_lr_start':   4,
        'local_lr':      1e-4,
        'local_rule_step':  1,
        'local_rule_start': 0,
    })

    trainer = CCNetTrainer(net_params)
    errors = [
        ('- Standard',{
            'cx_error':None
        }),
        ('- Giving cortex info (cx-only)',{
            'cx_error':'cx-only',
            'cx_lr': 1e-5,
            'cx_lr_start': 5
        }),
        ('- Giving cortex info (cx+fb)',{
            'cx_error':'cx+fb',
            'cx_lr': 1e-5,
            'cx_lr_start': 5
        }),
        ('- Giving cortex info (same error)',{
            'cx_error':'same', #any other string will be assigned to this case
            'cx_lr': 1e-5,
            'cx_lr_start': 5
        })
    ]

    model_state = {
        'modules_plasticity':{
            'cereb':True,
            'cortex_hw':False,
            'cortex_hb':False,
            'cereb_to_cortex':False,
            'cortex_in': False, 
            'cortex_out': False
        },
        'cereb_feedback':True,
        'cereb_readout':True,
        'local_plasticity':True,
        'plant':True
    }

    pr_print(['bold','purple','underline'],"Testing training:")
    
    pert = reaching.default_perturbation_parameters()
    reaching.apply_perturbation(pert)

    for title, iter_params in errors:
        pr_print(['bold','yellow','underline'],title)
        train_params = training_params
        train_params.update(iter_params)
        state = deepcopy(model_state)
        if title == '- Only cortex learning':
            state['modules_plasticity']['cereb'] = False
        trainer.train(reaching, debug_step=1, train_params=train_params, model_state=state)
        for i in range(20 if sys.stdin.isatty() else 0):  # countdown only when run by hand
            for j in range(4):
                print(" "*60,end='\r')
                print(f"Starting new training in {20-i} seconds"+'.'*j,end='\r')
                sleep(0.25)

    pr_print(['bold','green','underline'], "In all cases, modules_plasticity keeps changes off as expected (writing down the result from tests on 04/11/2024).")
            
        