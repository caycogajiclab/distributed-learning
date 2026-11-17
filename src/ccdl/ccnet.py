#!/usr/bin/env python
# coding: utf-8
"""Created on Thu Aug 25 15:11:02 2023
@author: Leonardo Agueci - leonardo.agueci@ens.psl.eu

Here, we define the CerebelloCorticalNet Network class, the central architecture of Distributed Learning.
"""
from typing import Dict, List, Optional, Tuple, Union, Any
import warnings
import copy

import torch
from torch import nn, Tensor

from ccdl import nonlinearities
from ccdl.utils import printsave, local_rule_update
from ccdl.models import Cortex, Cerebellum


class CerebelloCorticalNet(nn.Module):
    """
    Cerebello-Cortical Network combining a recurrent "cortex" and a feed-forward "cerebellum".

    This module pairs a cortical RNN (providing baseline dynamics) with a cerebellar FF network
    that computes corrective readouts. The cerebellum can provide both a readout contribution
    and feedback to the cortex. The class supports toggling plasticity for cortex/cerebellum,
    local update rules, and swapping the downstream plant (nonlinearity).

    Key concepts:
    - cortex: recurrent module (CTRNN2 wrapped in RNNet) producing baseline motor commands ($\theta$).
    - cerebellum: feed-forward module (FFNet) producing corrective signal for the motor command.
    - model_state: dict controlling network behavior (plasticity at the different sites, feedback/readout toggles, plant selection).
    - modules_plasticity: dict controlling plasticity of cortex and cerebellum sub-modules. contained in model_state.
    - Plasticity has to be changed by providing a modules_plasticity dict to the train() method.
    - DO NOT use set_plasticity() directly, use train() instead.
    - nonlinearity: maps the motor command to final readout ($y$); by default, it corresponds to the mapping from arm angles to hand position.
    - plant: the active motor commands-to-readout module (either nonlinearity or identity). This allows versatile usage of network.
    """
    def __init__(self, input_size, hidden_size, hidden_layer_specs: dict = {}, 
                 rnn_specs: dict = {}, cereb_specs: dict = {}, inter_specs: dict = {}, 
                 feedback_delay=0, nonlinearity_specs: dict = {}, 
                 device: str = 'cpu', dtype=None, verbose=False, **kwargs):
        """
        Initialize the Cerebello-Cortical network.

        Args:
            input_size (int): Dimensionality of upstream inputs to cortex and cerebellum modules.
            hidden_size (int): Number of hidden units in the cortical RNN.
            hidden_layer_specs (dict): Specifications for CTRNN2 layer settings (nonlinearity, initialization,
                time constant, biases, noise, eligibility propagation, self-connections). Defaults to empty dict.
            rnn_specs (dict): High-level RNN settings including output nonlinearity, initialization method,
                private gradients flag, and input noise. Defaults to empty dict.
            cereb_specs (dict): Cerebellum architecture including layer sizes, activation functions (hidden and output),
                initialization method, bias usage, and private gradients flag. Defaults to empty dict.
            inter_specs (dict): Connectivity initialization between cortex and cerebellum, including weight/bias
                distributions, autoencoder initialization flag, and feedback gain. Defaults to empty dict.
            feedback_delay (int): Integer delay (number of timesteps) applied to cerebellar feedback in cortex.
                Warning: delay implemented only for pretraining. Keep at 0 for normal training. Defaults to 0.
            nonlinearity_specs (dict): Specification of the plant/nonlinearity module (e.g., nonlinear plant),
                including module name and optional parameters. Defaults to empty dict.
            plasticity_specs (dict): Optional dictionary specifying which parameters are trainable in cortex and
                cerebellum modules. Defaults to empty dict.
            device (str): Device string for tensor operations (e.g., 'cpu', 'cuda'). Defaults to 'cpu'.
            dtype: PyTorch dtype for tensors. Defaults to torch.get_default_dtype() if None.
            verbose (bool): If True, print detailed initialization steps. Defaults to False.
            **kwargs: Additional keyword arguments (currently unused).
        """
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.fb_delay = feedback_delay
        
        # Factory kwargs for consistent device and dtype across modules
        if isinstance(device, str):
            device = torch.device(device)
        if isinstance(dtype, str):
            dtype = getattr(torch, dtype.split('.')[-1])
        self.factory_kwargs = {'device': device, 'dtype': dtype or torch.get_default_dtype()}
        
        self.inter_specs = self.default_inter_specs() if not inter_specs else copy.deepcopy(inter_specs)
        self.nonlinearity_specs = self.default_nonlinearity_specs() if not nonlinearity_specs else copy.deepcopy(nonlinearity_specs)
        
        # Initialize the plant (nonlinearity/biomechanics module)
        if verbose:
            self.printfunc(f"Initializing plant with specs: {self.nonlinearity_specs}")
        self.nonlinearity, self.identity = self.init_nonlinearity()
        self.plant = self.nonlinearity

        # Set readout dimensions
        self.net_readout_size = self.nonlinearity.input_size
        self.output_size = self.nonlinearity.output_size
        self.cereb_input_size = self.net_readout_size + self.input_size

        # Initialize main modules
        if verbose:
            self.printfunc(f"Initializing cortex")
        self.cortex = Cortex(input_size, hidden_size, self.net_readout_size, hidden_layer_specs, rnn_specs, self.factory_kwargs)
        if verbose:
            self.printfunc(f"Initializing cerebellum")
        self.cerebellum = Cerebellum(self.cereb_input_size, self.net_readout_size, cereb_specs, self.factory_kwargs)
        if verbose:
            self.printfunc(f"Initializing interconnectivity")
        self.init_interconnectivity()

        # Control flags for network behavior
        self.cereb_feedback = True
        self.cereb_readout = True
        self.local_plasticity = False
        
        # Dictionary used for storing network activity (used for local learning)
        self.activity = {}

        # Initialize plasticity configuration
        self.model_state = self.default_model_state()
        self.set_plasticity()
            
    def printfunc(self, output, formatting=None, end=None):
        """Helper function for formatted printing with optional logging."""
        printsave(output, None, True, formatting=formatting, end=end)
    
    @staticmethod
    def default_hidden_layer_specs():
        return Cortex.default_hidden_layer_specs()
    
    @staticmethod
    def default_rnn_specs():
        return Cortex.default_rnn_specs()
    
    @staticmethod
    def default_cereb_specs():
        return Cerebellum.default_specs()
    
    @staticmethod
    def default_inter_specs():
        return {
                'weight_distro': ['uniform', 0., 1.],
                'bias_distro': ['fixed', 0.],
                'autoencoder': False,
                'gain': 1,
            }
    
    @staticmethod
    def default_nonlinearity_specs():
        return {
            'module': 'TwoJointsArm',
        }
    
    # Next four methods are  for compatibility with previous version
    @property
    def plastic_cortex(self):
        return self.cortex.plastic

    @plastic_cortex.setter
    def plastic_cortex(self, value):
        self.cortex.plastic = value
        
    @property
    def plastic_cereb(self):
        return self.cerebellum.plastic

    @plastic_cereb.setter
    def plastic_cereb(self, value):
        self.cerebellum.plastic = value
    
    @staticmethod
    def default_cortex_plasticity(pretrain=False):
        r"""
        Returns the dictionary controlling cortex module plasticity during different training phases.
        
        Args:
            pretrain (bool): If True, enables plasticity for cortex parameters during pretraining (pretrain=True).
                Defaults to False.
        
        Returns:
            dict: Keys indicate different cortex parameter groups (input weights, hidden weights, biases,
                cerebellum feedback, output layer) with boolean values indicating plasticity.
        """
        return Cortex.default_plasticity_state(pretrain)
    
    def default_modules_plasticity(self, pretrain=False):
        r"""
        Returns the dictionary controlling overall module plasticity.
        
        If an entry is True, parameter plasticity will be determined by self.[module].plasticity_specs.
        
        Args:
            pretrain (bool): If True, disables cerebellum plasticity (only cortex trains).
                Defaults to False.
        
        Returns:
            dict: Contains plasticity flags for cerebellum and cortex sub-modules.
        """
        return {
            'cereb': not pretrain, 
            **Cortex.default_plasticity_state(pretrain)
        }
    
    @property
    def plasticity_specs(self):
        r"""
        Returns the current plasticity specifications for cortex and cerebellum modules.
        
        Returns:
            dict: Contains parameter names that are currently set as plastic in cortex and cerebellum.
        """
        return {'cortex': self.cortex.plasticity_specs, 'cereb': self.cerebellum.plasticity_specs}
    
    @staticmethod
    def default_plasticity_specs(n_layers, cx_in_plastic=True, cx_fb_plastic=True, cx_out_plastic=True):
        r"""
        Returns default specifications for trainable parameter sites in the network.
        
        Plasticity depends on both this specification and the training phase (pretraining vs. normal training).
        
        Args:
            n_layers (int): Number of layers in the cerebellum.
            cx_in_plastic (bool): If True, cortex input weights are plastic. Defaults to True.
            cx_fb_plastic (bool): If True, cortex feedback (cerebellum-to-cortex) weights are plastic. Defaults to True.
            cx_out_plastic (bool): If True, cortex output layer weights are plastic. Defaults to True.
        
        Returns:
            dict: Specifies parameter names that can be plastic in cortex, cerebellum, and feedback connections.
        """
        plasticity_specs = {
            'cortex': Cortex.default_plasticity_specs(cx_in_plastic, cx_fb_plastic, cx_out_plastic),
            'cereb': Cerebellum.default_plasticity_specs(n_layers),
        }
        return plasticity_specs

    @staticmethod
    def default_model_state():
        """
        Returns default model state dictionary describing network configuration.
        
        Returns:
            dict: Contains module plasticity flags, feedback/readout toggles, and plant selection.
        """
        return {'modules_plasticity': {'cereb': True, 'cortex_hw': False, 'cortex_hb': False, 'cereb_to_cortex': False, 'cortex_in': False, 'cortex_out': True},
                'cereb_feedback': True, 'cereb_readout': True, 'local_plasticity': False, 'plant': None}

    def get_model_state(self):
        """
        Returns the current model state reflecting network configuration and plasticity settings.
        
        Returns:
            dict: Current plasticity flags, feedback/readout status, and active plant module.
        """
        model_state = {
            'modules_plasticity': {
                'cereb': self.cerebellum.plastic,
                'cortex_hw': self.cortex.rnn.h2h.weight.requires_grad,
                'cortex_hb': self.cortex.rnn.h2h.bias.requires_grad if hasattr(self.cortex.rnn.h2h, 'bias') else False,
                'cereb_to_cortex': self.cortex.rnn.input2h_2.weight.requires_grad,
                'cortex_in': self.cortex.rnn.input2h.weight.requires_grad,
                'cortex_out': self.cortex.fc.weight.requires_grad
            },
            'cereb_feedback': self.cereb_feedback,
            'cereb_readout': self.cereb_readout,
            'local_plasticity': self.local_plasticity,
            'plant': (self.plant == self.nonlinearity)
            }
        return model_state
        
    def init_nonlinearity(self):
        """
        Initializes the plant/nonlinearity module (e.g., nonlinear plant or identity).
        
        Creates both an active nonlinearity and an identity wrapper for switching between them.
        
        Returns:
            tuple: (nonlinearity_module, identity_module) both instantiated and ready for use.
        """
        specs = copy.deepcopy(self.nonlinearity_specs)
        module_name = specs.pop('module')
        self.dt = specs.pop('dt', None)
        nonlinearity_cls = getattr(nonlinearities, module_name)
        nonlinearity = nonlinearity_cls(specs) if specs else nonlinearity_cls()

        # Identity used when plant is not active
        identity_cls = getattr(nonlinearities, 'Identity')
        identity = identity_cls(input_size=nonlinearity.input_size, output_size=nonlinearity.output_size)

        return nonlinearity, identity

    def init_interconnectivity(self):
        r"""
        Initialize weights connecting cortex and cerebellum modules.
        
        Supports two initialization modes: autoencoder-based (using pseudo-inverse of cortex output
        weights) or random initialization according to specified distributions.
        
        The feedback gain controlling cerebellar-to-cortex signal strength is also set here.
        """
        with torch.no_grad():
            if self.inter_specs['autoencoder']:
                self._init_autoencoder_weights()
            else:
                self._init_random_weights()
        
        self.gain = self.inter_specs.get('gain', 1.0)

    def _init_autoencoder_weights(self):
        """
        Initialize cerebellar-to-cortex feedback weights using pseudo-inverse of cortex output layer.
        
        This creates a pseudo-inverse mapping: if cortex.fc.weight maps hidden states to readout,
        its pseudo-inverse approximately maps readout back to hidden states for feedback.
        """
        pseudo_inverse = torch.pinverse(self.cortex.fc.weight).to(**self.factory_kwargs)
        self.cortex.rnn.input2h_2.weight.copy_(pseudo_inverse)
        if hasattr(self.cortex.rnn.input2h_2, 'bias'):
            self.cortex.rnn.input2h_2.bias.zero_()

    def _init_random_weights(self):
        weight_distro = self.inter_specs['weight_distro']
        bias_distro = self.inter_specs['bias_distro']
        
        self._initialize_tensor(self.cortex.rnn.input2h_2.weight, weight_distro)
        
        if hasattr(self.cortex.rnn.input2h_2, 'bias'):
            self._initialize_tensor(self.cortex.rnn.input2h_2.bias, bias_distro)

    def _initialize_tensor(self, tensor, distribution):
        """
        Initialize a tensor according to the specified distribution.
        
        Args:
            tensor (torch.Tensor): Tensor to initialize in-place.
            distribution (list): Distribution specification as [type, param1, param2, ...].
                Supported types: 'uniform' (params: min, max) or 'fixed' (param: value).
        
        Raises:
            NotImplementedError: If distribution type is not 'uniform' or 'fixed'.
        """
        dist_type = distribution[0]
        
        if dist_type == 'uniform':
            if len(distribution) == 3:
                nn.init.uniform_(tensor, distribution[1], distribution[2])
            else:
                scale = 1.0 / (self.hidden_size)
                tensor.uniform_(-scale, scale)
        elif dist_type == 'fixed':
            nn.init.constant_(tensor, distribution[1])
        else:
            raise NotImplementedError(f"Unsupported weight distribution: {dist_type}")

    def extra_repr(self) -> str:
        r"""Set the extra representation of the module, used for printing model infos through ``print(model)``
        """
        text = {}
        text['Modules info'] = f' - cereb_readout: {self.cereb_readout}\n - cereb_feedback: {self.cereb_feedback}\n - gain: {self.gain}\n - nonlinearity: {self.nonlinearity.__class__.__name__}\n'\
        f' - plant(the module effectively in use): {self.plant.__class__.__name__}'
        text['Modules Plasticity'] = f' - cortex: {self.cortex.plastic}\n - cereb: {self.cerebellum.plastic}\n - local plasticity: {self.local_plasticity}'
        text['Plasticity specs'] = f' - cortex: {sorted(self.cortex.plasticity_specs)}\n - cereb: {sorted(self.cerebellum.plasticity_specs)}'
        text['Feedback delay'] = f' {self.fb_delay}'
        return '\n'.join([f'{k}:\n{v}\n' for k, v in text.items()]) + '\nParameters:'

    def forward(
        self,
        input: torch.Tensor,
        hidden = None,
        input_2: Optional[torch.Tensor] = None,
        return_hidden: bool = False,
        return_thetas: bool = False,
        input_2_type: Optional[str] = None
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Forward pass through the Cerebello-Cortical network.
        
        Processes input through cortex and cerebellum in parallel, combines their outputs,
        and applies the plant nonlinearity. The cortex generates baseline motor commands while
        the cerebellum produces corrective adjustments. Thetas record both modules' outputs. 
        When input_2 is provided, it replaces cerebellar feedback as an external signal to the cortex.

        Args:
            input (torch.Tensor): External inputs with shape (timesteps, batch_size, input_size).
            hidden (torch.Tensor, optional): Initial hidden state for cortex RNN. 
                If None, initialized to zeros. Defaults to None.
            input_2 (torch.Tensor, optional): External feedback/target signal with shape 
                (timesteps, batch_size, input_size_2). Used only when cereb_feedback=False. 
                Defaults to None.
            return_hidden (bool): If True, return recorded cortex hidden states alongside readout. 
                Defaults to False.
            return_thetas (bool): If True, return internal theta (motor command) values from 
                cortex and cerebellum. Defaults to False.
            input_2_type (str, optional): Interpretation of input_2 signal:
                - 'output': feedback computed in output space (readout)
                - other (default): feedback computed in theta space (motor command)

        Returns:
            Union[torch.Tensor, Tuple]: Depending on return flags:
                - readout only: torch.Tensor of shape (timesteps, batch_size, output_size)
                - with return_hidden: tuple of (readout, hidden_states)
                - with return_thetas: tuple of (readout, thetas)
                - with both: tuple of (readout, hidden_states, thetas)

        Raises:
            AssertionError: If input shape doesn't match expected (timesteps, batch_size, input_size),
                or if input_2 dimensions are incorrect, or if input_2 used with cereb_feedback=True
        """
        batch_size = input.shape[1]
        T = input.size(0)
        timesteps = range(T)
        input_size_2 = self.cortex.rnn.input_size_2

        assert input.shape[-1] == self.input_size, "input must have shape (timesteps, batch_size, input_size)"

        # if self.batch_first:
        #     input = input.transpose(0, 1)
        
        if input_2 is not None:
            assert isinstance(input_2, torch.Tensor)
            assert input_2.shape[-1] == input_size_2, f"input_2 must have shape ({T}, {batch_size}, {input_size_2}) \
                but found shape {input_2.shape}"
            assert input_2.shape[0] == T, f"input_2 must have shape ({T}, {batch_size}, {input_size_2})"
            assert input_2.shape[1] == batch_size, f"input_2 must have shape ({T}, {batch_size}, {input_size_2})"
            assert not self.cereb_feedback, "input_2 cannot be used with cereb_feedback=True"
        
        assert self.fb_delay >= 0 and isinstance(self.fb_delay, int), "feedback_delay must be a non-negative integer"

        if self.cereb_feedback and self.gain == 0.:
            warnings.warn("Feedback gain set to 0, although cereb_feedback is set to True.")

        # Initialize storage tensors
        hidden_t = hidden if hidden is not None else self.cortex.rnn.init_hidden(input[0])
        thetas = input.new_zeros(2, len(timesteps)+1, batch_size, self.net_readout_size)
        theta_cerebellum_t = input.new_zeros(batch_size, self.net_readout_size)
        theta_cortex_t = input.new_zeros(batch_size, self.net_readout_size)
        prev_theta_cortex_t = input.new_zeros(batch_size, self.net_readout_size)
        readout = input.new_zeros((T, batch_size, self.output_size))
        feedback_t = input.new_zeros(batch_size, self.net_readout_size)
        self.activity_cb_output = torch.empty(input.shape[0], batch_size, self.net_readout_size, 
                                                      **self.factory_kwargs)
        self.activity_hidden = torch.empty(input.shape[0], batch_size, self.hidden_size, 
                                                    **self.factory_kwargs)
        
        for t in timesteps:
            # CORTEX: compute baseline motor command
            if input_2 is not None:
                if t > self.fb_delay:
                    if input_2_type == 'output':
                        feedback_t = self.gain * (input_2[t - self.fb_delay - 1] - readout[t - self.fb_delay - 1])
                    else:
                        feedback_t = self.gain * (input_2[t - self.fb_delay - 1] - theta_cortex_t)
            elif self.cereb_feedback:
                feedback_t = self.gain * (theta_cerebellum_t.detach() if self.cerebellum.private_gradients else theta_cerebellum_t)

            cortex_input_t = torch.cat([input[t], feedback_t], dim=-1)
            theta_cortex_t, hidden_t = self.cortex.forward(cortex_input_t, hidden_t)
            with torch.no_grad():
                thetas[0, t+1].copy_(theta_cortex_t)
                
            # CEREBELLUM: compute corrective signal
            if self.cereb_readout or self.cereb_feedback:
                cx_fb_t = prev_theta_cortex_t.detach() if self.cortex.private_gradients else prev_theta_cortex_t
                cereb_input_t = torch.cat([input[t], cx_fb_t], dim=-1)
                theta_cerebellum_t = self.cerebellum.forward(cereb_input_t)
                
                # updating prev_theta_cortex_t for the next timestep
                prev_theta_cortex_t.copy_(theta_cortex_t)
                with torch.no_grad():
                    thetas[1, t+1].copy_(theta_cerebellum_t)

            # Combine cortex and cerebellum outputs
            theta = theta_cortex_t + theta_cerebellum_t if self.cereb_readout else theta_cortex_t
            readout[t] = self.plant(theta)

            # Save activity for local learning rules
            with torch.no_grad():
                self.activity_hidden[t].copy_(hidden_t[0])
                self.activity_cb_output[t].copy_(theta_cerebellum_t) if input_2 is None else self.activity_cb_output[t].copy_(input_2[t].clone())

        # Store activity
        self.activity['hidden'] = self.activity_hidden
        self.activity['cb-output'] = self.activity_cb_output

        if not (return_hidden) and not (return_thetas):
            return readout

        outputs = [readout]
        if return_hidden:
            outputs.append(self.activity['hidden'].clone())
        if return_thetas:
            outputs.append(thetas[:, 1:].detach().clone())

        return tuple(outputs)
    
    def train(self, modules_plasticity=None, cereb_feedback=None, cereb_readout=None, plasticity_specs=None, local_plasticity=False, plant=None, pretraining=False):
        r"""Set training mode and control parameter plasticity across network modules.
        
        Overrides PyTorch's nn.Module.train() method to coordinate training mode with plasticity settings.
        This allows fine-grained control over which parameters are trainable during different training phases.

        Args:
            modules_plasticity (dict, optional): Dictionary specifying which module groups are plastic:
                Keys: 'cereb', 'cortex_hw' (hidden-to-hidden weights), 'cortex_hb' (hidden biases),
                'cereb_to_cortex' (feedback weights), 'cortex_in' (input weights), 'cortex_out' (output weights).
                Values: bool indicating if that module group is trainable. Defaults to None.
            cereb_feedback (bool, optional): Enable cerebellum-to-cortex feedback connection. 
                If None: False during pretraining, True during normal training. Defaults to None.
            cereb_readout (bool, optional): Enable cerebellar output contribution to motor command.
                If None: False during pretraining, True during normal training. Defaults to None.
            plasticity_specs (dict, deprecated): Use modules_plasticity instead. Defaults to None.
            local_plasticity (bool): Enable local learning rules for weight updates. Defaults to False.
            plant (bool, optional): Select plant module:
                - True: use nonlinearity (nonlinear plant)
                - False: use identity mapping
                - None: keep current selection
                Defaults to None.
            pretraining (bool): If True, apply pretraining defaults (cortex trains, cerebellum inactive).
                If False, apply normal training defaults. Defaults to False.

        Returns:
            None
        """
        if plasticity_specs:
            warnings.warn("'plasticity_specs' deprecated. Use 'modules_plasticity' instead.")
            modules_plasticity = plasticity_specs

        if modules_plasticity:
            assert all([key in modules_plasticity.keys() for key in self.default_modules_plasticity().keys()]), \
                f"modules_plasticity must contain keys {self.default_modules_plasticity().keys()}"
        else:
            modules_plasticity = {key: None for key in self.default_modules_plasticity().keys()}

        # Set defaults based on training phase
        if pretraining:
            cereb_feedback = False if cereb_feedback is None else cereb_feedback
            cereb_readout = False if cereb_readout is None else cereb_readout
            plant = False if plant is None else plant
            local_plasticity = False if local_plasticity is None else local_plasticity
        else:
            cereb_feedback = True if cereb_feedback is None else cereb_feedback
            cereb_readout = True if cereb_readout is None else cereb_readout
            plant = True if plant is None else plant
            local_plasticity = False if local_plasticity is None else local_plasticity
        
        # Configure cortex plasticity
        if pretraining:
            self.cortex.plastic = True if modules_plasticity and any([modules_plasticity[key] for key in self.cortex.default_plasticity_state().keys()]) else pretraining
        else:
            self.cortex.plastic = True if modules_plasticity and any([modules_plasticity[key] for key in self.cortex.default_plasticity_state().keys()]) else self.cortex.plastic
        self.cortex.training = self.cortex.plastic
        for module in self.cortex.children():
            module.train(self.cortex.plastic)

        # Configure cerebellum plasticity
        if pretraining:
            self.cerebellum.plastic = True if modules_plasticity and modules_plasticity['cereb'] else not pretraining
        else:
            if modules_plasticity:
                self.cerebellum.plastic = modules_plasticity['cereb'] if modules_plasticity['cereb'] else self.cerebellum.plastic
        self.cerebellum.training = self.cerebellum.plastic
        for module in self.cerebellum.children():
            module.train(self.cerebellum.plastic)

        self.cereb_feedback = cereb_feedback
        self.cereb_readout = cereb_readout
        self.set_plasticity(**modules_plasticity)
        self.local_plasticity = local_plasticity
        self.plant = self.nonlinearity if plant else self.identity

    def eval(self, cereb_feedback=None, cereb_readout=None, plant=None, **kwargs):
        r"""Set evaluation mode with optional control over feedback and plant selection.
        
        Overrides PyTorch's nn.Module.eval() method. Disables plasticity and configurable dropout/batch norm
        behavior while allowing independent control over feedback pathways and plant selection.

        Args:
            cereb_feedback (bool, optional): Enable cerebellum-to-cortex feedback. 
                If None, keeps current setting. Defaults to None.
            cereb_readout (bool, optional): Enable cerebellar readout contribution. 
                If None, keeps current setting. Defaults to None.
            plant (bool, optional): Select plant module:
                - True: use nonlinearity (nonlinear plant)
                - False: use identity mapping
                - None: keep current selection
                Defaults to None.
            **kwargs: Accepts but ignores additional arguments for compatibility with model_state dicts.

        Returns:
            None
        """
        self.cortex.plastic = False
        self.cortex.training = False
        for module in self.cortex.children():
            module.train(False)

        self.cerebellum.plastic = False
        self.cerebellum.training = False
        for module in self.cerebellum.children():
            module.train(False)

        self.cereb_feedback = cereb_feedback if cereb_feedback is not None else self.cereb_feedback
        self.cereb_readout = cereb_readout if cereb_readout is not None else self.cereb_readout
        
        if plant is not None:
            self.plant = self.nonlinearity if plant else self.identity

    def set_plasticity(self, cortex_in=None, cortex_out=None, cereb_to_cortex=None, cortex_hw=None, cortex_hb=None, cereb=None):
        r"""Control requires_grad for parameters to enable/disable learning at specific sites.
        
        Manages which parameters are trainable (requires_grad=True) in cortex and cerebellum modules.
        Can be configured via a specifications dictionary or individual boolean flags.
        Automatically initializes plasticity_specs if not previously set.

        Args:
            cortex_in (bool, optional): Enable plasticity for cortex input weights (input2h). 
                Defaults to None (no change).
            cortex_out (bool, optional): Enable plasticity for cortex output weights (fc layer). 
                Defaults to None (no change).
            cereb_to_cortex (bool, optional): Enable plasticity for cerebellum-to-cortex feedback weights (input2h_2). 
                Defaults to None (no change).
            cortex_hw (bool, optional): Enable plasticity for cortex recurrent weights (h2h). 
                Defaults to None (no change).
            cortex_hb (bool, optional): Enable plasticity for cortex recurrent biases (h2h.bias). 
                Defaults to None (no change).
            cereb (bool, optional): Override cerebellum plasticity directly (bypasses plasticity_specs). 
                Defaults to None (no change).
            (DEPRECATED) plasticity_specs (dict): Comprehensive specification with keys 'cortex' and 'cereb'
                mapping to sets/lists of parameter names to make plastic. Defaults to default_plasticity_specs if not provided.

        Returns:
            None
        """
        self.cortex.set_plasticity(cortex_in,cortex_out,cereb_to_cortex,cortex_hw,cortex_hb)

        self.cerebellum.set_plasticity(cereb)

        if 'rnn.input2h_2.weight' in self.cortex.plasticity_specs and self.inter_specs['autoencoder']:
            warnings.warn("Cerebellar-to-cortex feedback weights were initialized as autoencoding, but are set as plastic.")
    
    def delta_rule_on_cortex_(self, lr=1e-4, method='mean', return_deltaw=False, inplace=True, debug=False):
        r"""
        Apply delta rule for in-place weight updates on cortex recurrent connections.
        
        Args:
            lr (float): Learning rate scaling the weight updates. Defaults to 1e-4.
            method (str): Aggregation method for batch dimension. Options: 'mean', 'sum'. 
                Defaults to 'mean'.
            return_deltaw (bool): If True, return weights changes. Defaults to False.
            inplace (bool): If True, modify weights in-place. Defaults to True.

        Returns:
            torch.Tensor (optional): Weight update tensor (delta_w) if return_deltaw=True, else None.
        """
        
        assert self.local_plasticity, "local_plasticity must be True to use this method"
        assert hasattr(self, 'activity_hidden'), "Run forward() before calling this method"
        assert hasattr(self, 'activity_cb_output'), "Run forward() before calling this method"
        fb_weights = self.cortex.rnn.input2h_2.weight
        rec_weights = self.cortex.rnn.h2h.weight

        delta_w = local_rule_update(self.activity_hidden, self.activity_cb_output, fb_weights, 
                                    rec_weights, lr, method)

        if inplace:
            with torch.no_grad():
                rec_weights += delta_w.detach().clone()

        if return_deltaw:
            return delta_w
        
        if debug:
            return rec_weights, delta_w

    def delta_rule_on_cortex(self, lr=1e-4, method='mean', debug=False):
        r""" (deprecated)
        
        Compute weight updates via delta rule without modifying parameters.
        
        Similar to delta_rule_on_cortex_ but returns updated weights instead of modifying in-place.
        """
        return self.delta_rule_on_cortex_(lr, method, False, False, debug)



if __name__ == '__main__':
    from ccdl.trainer import CCNetTrainer
    from ccdl.reaching import ReachingDataset
    from ccdl.utils import *

    input_size = 10
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
        "hidden_layer_specs": Cortex.default_hidden_layer_specs(),
        "rnn_specs": Cortex.default_rnn_specs(),
        "cereb_specs": Cerebellum.default_specs(),
        "inter_specs": CerebelloCorticalNet.default_inter_specs(),
        "nonlinearity_specs": CerebelloCorticalNet.default_nonlinearity_specs(),
        "device": "cpu"
    }
    
    pretrain_params = {
        "epochs": 50,
        "lr": 1e-4,
        "l1": 1e-5,
        "l2": 1e-4,
        "activity_penalty": 1e-5,
        "reg_drop": [0.01, 2],
        "lr_scheduler": "cosine",
        "local_lr": 1e-4,
        "local_rule_settings": [1,0, False],
        "early_stop": False,
        "es_params": [300, 1e-9],
        "es_patience": 300,
        "es_eps": 1e-9
    }

    trainer = CCNetTrainer(net_params)
    model_state={
        "cereb_feedback": False,
        "cereb_readout": False,
        "modules_plasticity": {
            "cortex_hw": True,
            "cortex_hb": True,
            "cortex_in": True,
            "cortex_out": False,
            "cereb_to_cortex": False,
            "cereb": False
        },
        "local_plasticity": False,
        "plant": True
    }
    trainer.model.train(**model_state)

    if input("Testing pretraining? (y/n)") in ["y","yes"]:
        trainer.train(reaching, debug=1, pretrain=True, train_params=pretrain_params)
    else:
        pr_print(['bold','red'],"Model")
        print(trainer.model)
        
    model_state['local_plasticity'] = True
    trainer.model.train(**model_state)
    if input("Testing local rule? (y/n)") in ["y","yes"]:
        trainer.train(reaching, debug=1, pretrain=True, train_params=pretrain_params)
    else:
        pr_print(['bold','red'],"Model")
        print(trainer.model)