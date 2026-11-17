# -*- coding: utf-8 -*-
"""Damage-aware trainer.

:class:`CCNetDamageTrainer` subclasses :class:`~ccdl.trainer.CCNetTrainer` with
a training loop that tolerates (and skips) failed backward passes, as can happen
when training a lesioned/damaged network.
"""

from copy import deepcopy
import warnings

import numpy as np
import torch
import torch.nn as nn

from ccdl.trainer import CCNetTrainer
from ccdl.utils import selective_update


class CCNetDamageTrainer(CCNetTrainer):
    """
    Thin damage-aware version of CCNetTrainer.
    """

    def __init__(
        self,
        net_params,
        train_params=None,
        model_class='CerebelloCorticalNet',
        opt=torch.optim.Adam,
        criterion=nn.MSELoss,
        logfile=None,
        verbose=False,
        safe_backward=True,
    ):
        super().__init__(
            net_params=net_params,
            train_params=train_params,
            model_class=model_class,
            opt=opt,
            criterion=criterion,
            logfile=logfile,
            verbose=verbose,
        )

    def _training_loop(self, epoch, optimizers, record_gradients, debug_step):
        """Run one training epoch, guarding each ``backward()`` so a damaged
        model that fails to backpropagate is skipped rather than crashing."""
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
        thetas_mem = {
            'cortex': thetas[0].abs().mean().item(),
            'cereb': thetas[1].abs().mean().item()
        }
        if debug_step:
            mean_hidden_activity = activity.abs().mean().item()

        loss = self.compute_loss(outputs, targets, activity)

        grad_clip = self.training_params.get('grad_clip')
        grads, grads_max, premax = {}, {}, {}
        cx_error = self.training_params.get('cx_error', 'cx-only') or 'cx-only'
        retain_graph = cx_error not in ['cx-only', 'cx+fb']

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
                    self.printfunc(
                        f'cx optimizer, using error {cx_error}',
                        formatting=['purple'],
                        verbose=True
                    )

                print_state_flag = epoch % debug_step == 0 if debug_step else False
                if cx_error == 'cx-only':
                    cx_loss = self.compute_cortex_loss(
                        temp_state=self.cx_only_state,
                        debug=print_state_flag
                    )
                elif cx_error == 'cx+fb':
                    cx_loss = self.compute_cortex_loss(
                        temp_state=self.cb_fb_state,
                        debug=print_state_flag
                    )
                else:
                    cx_loss = loss.clone()

                self.cortex_error = cx_loss.clone().item()

                if np.abs(cx_loss.item()) > 0.:
                    try:
                        cx_loss.backward()
                    except:
                        pass
                else:
                    continue

            else:
                if np.abs(loss.item()) > 0.:
                    try:
                        loss.backward(retain_graph=retain_graph)
                    except:
                        pass
                else:
                    continue

            if grad_clip:
                if record_gradients:
                    premax.update(self._snapshot_grad_maxima({key: optimizer}))
                params_list = [p for pg in optimizer.param_groups for p in pg['params']]
                torch.nn.utils.clip_grad_norm_(params_list, max_norm=grad_clip)

            if record_gradients or debug_step:
                g, gmax = self._record_gradients({key: optimizer})
                grads.update(g)
                grads_max.update(gmax)

            optimizer.step()

        delta_w = None
        if (
            self.model_state['local_plasticity']
            and epoch > self.training_params['local_rule_start']
            and epoch % self.training_params['local_rule_step'] == 0
        ):
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
            self.cortex_error = self.compute_cortex_loss(
                thetas,
                input_2=input_2,
                input_2_type=self.input_2_type
            ).detach().item()

        return loss, outputs