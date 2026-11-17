#!/usr/bin/env python
# coding: utf-8
"""
Created on Tue May 17 15:54:40 2022
@author: Leonardo Agueci - leonardo.agueci@ens.psl.eu
pytorch models used for defining the CerebelloCorticalNet class.
"""
import numpy as np

import torch
from torch import nn, Tensor, tanh, relu
from torch.nn import init
import torch.nn.functional as F
from copy import deepcopy

from ccdl.layers import Linear, CTLinear, CTRNN2  # , RNN, EILinear


# from torch.autograd import Variable

class FFNet(nn.Module):
    r"""
    Multi-layer FeedForward network with customizable n. of layers and size, as well as hidden/output nonlinearities.
    Adding dt as input, you will use a continuous-time version of the linear layer, where the activity of the neuron
    follows a differential equation instead (to finish).

    Note: In the ReLU output case, the output size is temporarily multiplied by two. This is because the correction could
    be negative too, but ReLU returns non-negative outputs. The two parts (+ & -) are summed to create the final output,
    meaning that this trick does not affect the final output.

    Warning: If regional output signals are not correctly normalized, using tanh can lead to issues.
    """

    def __init__(self, input_size: int, output_size: int, sizes: list, bias: bool = True, init_method: str = 'normal',
                 init_gain: float = 1.0, h_nlin=relu, out_nlin='linear', device=None, dtype=None, **kwargs) -> None:
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.h_sizes = sizes
        # n hidden layers
        self.n_layers = len(sizes)
        self.init = init_method
        self.bias = bias
        self.h_nlin_name = (h_nlin if type(h_nlin) == str else h_nlin.__name__) if h_nlin is not None else 'linear'
        self.h_nlin = getattr(torch, self.h_nlin_name) if self.h_nlin_name != 'linear' else (lambda x: x)
        self.h_nlin = getattr(F, self.h_nlin_name)
        self.device = device
        
        assert type(out_nlin) in (str, type(relu)), "out_nlin must be a string or a torch function"
        self.out_nlin_name = (out_nlin if type(out_nlin) == str else out_nlin.__name__) if out_nlin is not None else \
            'linear'
        self.out_nlin = getattr(torch, self.out_nlin_name) if self.out_nlin_name != 'linear' else (lambda x: x)

        # initialize network parameters
        n_in = input_size
        layers = []
        self.activations = []

        for layer_size in sizes:
            layers.append(Linear(n_in, layer_size, bias, init_method, init_gain, self.h_nlin, device, dtype))
            self.activations.append(self.h_nlin)
            n_in = layer_size

        # adding output layer
        layers.append(Linear(n_in, output_size, bias, init_method, init_gain, self.out_nlin, device, dtype))
        self.activations.append(self.out_nlin)

        self.layers = nn.ModuleList(layers)

    def forward(self, x: Tensor) -> Tensor:
        for i,layer in enumerate(self.layers):
            x = layer(x)
            if i < self.n_layers:
                x = self.activations[i](x)
            else:
                x = self.out_nlin(x)
        return x


class CTFFNet(nn.Module):
    r"""
    Continuous time Multi-layer FeedForward network with customizable n. of layers and size, as well as hidden/output
    nonlinearities. NOTICE that this network does not contain any time delay among units, signal propagation is still
    instantaneous.
    """
    
    def __init__(self, input_size: int, output_size: int, sizes: list, bias: bool = True, dt=None,
                 tau=100, sigma: float = 0., batch_first: bool = False, init_method: str = 'normal',
                 h_nlin=relu, out_nlin=torch.tanh, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.h_sizes = sizes
        self.n_layers = len(sizes)
        self.init = init_method
        self.bias = bias
        self.h_nlin_name = (h_nlin if type(h_nlin) == str else h_nlin.__name__) if h_nlin is not None else 'linear'
        self.h_nlin = getattr(torch, self.h_nlin_name)
        self.out_nlin_name = (
            out_nlin if type(out_nlin) == str else out_nlin.__name__) if out_nlin is not None else 'linear'
        self.out_nlin = getattr(torch, self.out_nlin_name)

        self.tau = tau
        self.dt = dt
        if dt is None:
            self.alpha = 1
        else:
            self.alpha = dt / self.tau
        self.sigma = sigma
        self._sigma = np.sqrt(2 * self.alpha) * sigma
        self.batch_first = batch_first

        # initializing network parameters
        n_in = input_size
        for layer in range(self.n_layers):
            n_out = sizes[layer]
            setattr(self, 'L%d' % layer, CTLinear(n_in, n_out, bias, init_method, self.h_nlin, dt, tau,
                                                  sigma=sigma, **factory_kwargs))
        # adding output layer
        self.L_out = CTLinear(n_in, output_size, bias, init_method, self.out_nlin, dt, tau,
                              sigma=sigma, **factory_kwargs)

    def init_layer(self, input, layer_name):
        layer = getattr(self, layer_name)
        batch_size = input.shape[1]
        return (
            torch.zeros(batch_size, layer.output_size).to(input.device),
            torch.zeros(batch_size, layer.output_size).to(input.device),
        )

    def forward_step(self, x: Tensor, hiddens) -> tuple:
        # hiddens contains the previous (state,output) tuples for all layers, including L_out
        for layer_id in range(self.n_layers):
            layer_name = 'L%d' % layer_id
            if hiddens[layer_id] is None:
                hiddens[layer_id] = self.init_layer(x, layer_name)
            layer = getattr(self, layer_name)
            x = layer(x, hiddens[layer_id])
            if self.h_nlin_name != 'linear':
                x = self.h_nlin(x)

        # output layer
        if hiddens[self.n_layers] is None:
            hiddens[self.n_layers] = self.init_layer(x, 'L_out')

        if self.out_nlin_name == 'linear':
            hiddens[self.n_layers] = self.L_out(x, hiddens[self.n_layers])
        else:
            hiddens[self.n_layers] = self.out_nlin(self.L_out(x, hiddens[self.n_layers]))

        return hiddens[self.n_layers][1], hiddens

    def forward(self, input: Tensor) -> Tensor:
        # hiddens contains the previous (state,output) tuples for all layers, including L_out
        if self.batch_first:
            input = input.transpose(0, 1)
        # batch_size = input.shape[1]
        timesteps = range(input.size(0))

        hiddens = [None for i in range(self.n_layers + 1)]  # +1 for the last layer
        readout = []

        for t in timesteps:
            output, hiddens = self.forward_step(input[t], hiddens)
            readout.append(output)

        readout = torch.stack(readout, dim=0)
        if self.batch_first:
            readout = readout.transpose(0, 1)
        return readout


class RNNet(nn.Module):
    """Recurrent network model.
    Args:
        input_size: int, input size
        hidden_size: int, hidden size
        output_size: int, output size
        pos_out: bool, set to True to have only Exc. output units
        htype: str, type of RNN (lstm, rnn, ctrnn)
        out_bias: bool, enable output layer biases
        out_sign: {1,-1,None}, it fix the output sign to positive/negative or real-valued
        sigma_in: float, std of the input (gaussian) noise that will be added
                            to the input
    kwArgs:
        nonlinearity: torch.functional, activation function
        init_method: str, choose between uniform and normal distribution
        dt: int, euler-integration time constant
        structure: FC(i.e. fully-connected),TM(i.e.Two-modules), or custom
                        (inputs: masks & indexes)
        CC_prop: float [0,1], cereb.neurons/tot.neurons (default=0.66, 2:1)
        in_bias: bool, enable input bias parameter
        h_bias: bool, enable hidden-layer bias parameter
        sigma_rec: float, defines the noise amplitude in each rec/hidden unit
        e_prop: float [0,1], E/I units ratio
        self_conn: bool, enable rec-units self-connections
        init_e_factor: Reset_parameter initialization resize factor, in case of
                        unstable dynamics DEFAULT 1
        batch_first: bool, indicates if batch dimension is at position 0 if
                        True, 1 otherwise
    Inputs:
        input: (seq_len, batch, input_size), network input
    """

    __constants__ = ["input_size", "hidden_size", "output_size", "out_bias"]

    def __init__(self, rnn, input_size, output_size, out_bias=True, sigma_in=0, init_method='uniform', nonlin=tanh,
                 device=None, dtype=None, **kwargs):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()

        # Continuous time RNN
        self.input_size = input_size
        self.sigma_in = sigma_in
        self.output_size = output_size
        self.out_bias = out_bias
        self.nonlin_name = (nonlin if type(nonlin) == str else nonlin.__name__) if nonlin is not None else 'linear'
        if self.nonlin_name != 'linear':
            self.nonlin = getattr(torch, self.nonlin_name)
        else:
            self.nonlin = lambda x: x

        self.rnn = rnn
        self.rnn_output_size = self.rnn.output_size
        self.fc = Linear(self.rnn_output_size, output_size, bias=out_bias, init_method=init_method, nonlinearity=nonlin,
                         **factory_kwargs)

    def forward_step(self, input, hidden):
        output, hidden = self.rnn(input, hidden)
        output = self.nonlin(self.fc(output))
        return output, hidden

    def forward(self, input, hidden):
        batch_size = input.shape[1]
        x = input + self.sigma_in * torch.randn_like(input)
        dims = input.ndim
        assert dims in [2, 3], f'input provided to {self.__class__.__name__} must have 2 or 3 dimensions, {dims} found'

        # single timestep input
        if dims == 2:
            output, hidden = self.forward_step(x, hidden)
        # entire sequence input
        else:
            timesteps = range(input.size(0))
            output = input.new_zeros(len(timesteps), batch_size, self.output_size)
            for t in timesteps:
                output[t], hidden = self.forward_step(x, hidden)

        return output, hidden

    def transparent_forward(self, x):
        x1 = x + self.sigma_in * torch.randn_like(x)
        in1, in2, tot_in, states1, outputs1, outputs2 = self.rnn.transparent_forward(x)
        # outputs3 = torch.masked_select(outputs2, self.rnn.OUT_mask.bool())
        x3 = self.out_sign * self.fc(outputs2)
        return [x, x1, in1, in2, tot_in, states1, outputs1, outputs2, x3]


class Cortex(RNNet):
    """
    Cortex = RNNet + plasticity policy + default specs
    """
    @staticmethod
    def default_hidden_layer_specs():
        return {
            "nonlinearity": 'relu',
            "init_method": "uniform",
            "dt": None,
            "tau": 100,
            "in_bias": True,
            "h_bias": True,
            "sigma_rec": 0,
            "e_prop": None,
            "self_conn": False,
            "init_e_factor": 1.0
        }

    @staticmethod
    def default_rnn_specs():
        return {
            "out_bias": True,
            "sigma_in": 0.,
            "init_method": 'uniform',
            "nonlin": 'relu',
            "private_gradients": False
        }

    @staticmethod
    def default_plasticity_specs(cx_in_plastic=True, cx_fb_plastic=True, cx_out_plastic=True):
        specs = set(['rnn.h2h.weight',
            'rnn.h2h.bias'])

        if cx_in_plastic:
            specs |= {'rnn.input2h.weight', 'rnn.input2h.bias'}
        if cx_fb_plastic:
            specs |= {'rnn.input2h_2.weight', 'rnn.input2h_2.bias'}
        if cx_out_plastic:
            specs |= {'fc.weight', 'fc.bias'}

        return specs
    
    @staticmethod
    def default_plasticity_state(pretrain=False):
        r"""
        Returns the dictionary controlling cortex module plasticity during different training phases.
        
        Args:
            pretrain (bool): If True, enables plasticity for cortex parameters during pretraining (pretrain=True).
                Defaults to False.
        
        Returns:
            dict: Keys indicate different cortex parameter groups (input weights, hidden weights, biases,
                cerebellum feedback, output layer) with boolean values indicating plasticity.
        """
        return {
            'cortex_hw': pretrain, 
            'cortex_hb': pretrain, 
            'cereb_to_cortex': False, 
            'cortex_in': pretrain, 
            'cortex_out': False
        }

    def __init__(
        self, 
        input_size: int, 
        hidden_size: int, 
        input_size_2: int, 
        hidden_layer_specs: dict = None, 
        rnn_specs: dict = None, 
        factory_kwargs=None
    ):
        """
        Initializes the cortical recurrent module (RNN).
        
        The cortex receives external inputs and optional feedback through dual input streams.
        Returns an RNNet wrapper around a CTRNN2 layer with configured plasticity settings.
        
        Returns:
            RNNet: Configured recurrent neural network module for the cortex
        """
        self.input_size = input_size
        self.hidden_size= hidden_size
        self.input_size_2 = input_size_2
        self.hidden_layer_specs = hidden_layer_specs or self.default_hidden_layer_specs
        self.rnn_specs = rnn_specs or self.default_rnn_specs
        self.plastic = False # prev. plastic_cortex
        
        assert 'private_gradients' in self.rnn_specs, "rnn_specs must contain key 'private_gradients' of type bool"
        self.private_gradients = rnn_specs.get('private_gradients',False)
        
        rnn_layer = CTRNN2(
            input_size,
            hidden_size,
            input_size_2=input_size_2,
            **hidden_layer_specs,
            **factory_kwargs
        )
        
        super().__init__(
            rnn_layer, 
            rnn_layer.total_input_size, 
            input_size_2, 
            **rnn_specs, 
            **factory_kwargs
        )

    def set_plasticity(self, cortex_in=None, cortex_out=None, cereb_to_cortex=None, cortex_hw=None, cortex_hb=None, **kwargs):
        r"""Control requires_grad for parameters to enable/disable learning at specific sites.
        
        Manages which parameters are trainable (requires_grad=True)
        Note: I left kwargs so that the function could be called directly providing the CerebelloCorticalNet modules_plasticity, without removing cereb-related args.
        """
        self.plasticity_specs = getattr(self, "plasticity_specs", None) or self.default_plasticity_specs(cortex_in,cortex_out,cereb_to_cortex)
        
        # Update based on individual flags
        if cortex_in is not None:
            params_names = ['rnn.input2h.weight', 'rnn.input2h.bias']
            if cortex_in:
                self.plasticity_specs.update(params_names)
            else:
                self.plasticity_specs.difference_update(params_names)
        if cortex_out is not None:
            params_names = ['fc.weight', 'fc.bias']
            if cortex_out:
                self.plasticity_specs.update(params_names)
            else:
                self.plasticity_specs.difference_update(params_names)
        if cereb_to_cortex is not None:
            params_names = ['rnn.input2h_2.weight', 'rnn.input2h_2.bias']
            if cereb_to_cortex:
                self.plasticity_specs.update(params_names)
            else:
                self.plasticity_specs.difference_update(params_names)
        if cortex_hw is not None:
            if cortex_hw:
                self.plasticity_specs.update(['rnn.h2h.weight'])
            else:
                self.plasticity_specs.discard('rnn.h2h.weight')
        if cortex_hb is not None:
            if cortex_hb:
                self.plasticity_specs.update(['rnn.h2h.bias'])
            else:
                self.plasticity_specs.discard('rnn.h2h.bias')

        # Apply plasticity to cortex
        for name, param in self.named_parameters():
            if name in self.plasticity_specs and self.plastic:
                param.requires_grad = True
            else:
                param.requires_grad = False
        
        return self.plasticity_specs


class Cerebellum(FFNet):
    """
    Feed-forward module used as the 'cerebellum'.
    
    Here, the cerebellum is a multi-layer feed-forward network that receives external inputs and
    cortical outputs, producing corrective motor command adjustments.
    """

    @staticmethod
    def default_specs():
        return {
            "sizes": [100]*2,
            "bias": True,
            "init_method": 'uniform',
            "h_nlin": 'relu',
            "out_nlin": 'linear',
            "private_gradients": False,
        }

    @staticmethod
    def default_plasticity_specs(n_layers: int):
        specs = set()
        for i in range(n_layers + 1):
            specs.add(f'layers.{i}.weight')
            specs.add(f'layers.{i}.bias')
        return specs

    def __init__(
        self,
        input_size: int,
        output_size: int,
        specs: dict = None,
        factory_kwargs: dict = None,
    ):
        self.input_size = input_size
        self.output_size = output_size
        assert 'sizes' in specs and isinstance(specs['sizes'], list), \
            "cereb_specs must contain key 'sizes' of type list of integers!"
        self.specs = deepcopy(specs) or self.default_specs
        sizes = self.specs.pop('sizes')
        self.factory_kwargs = factory_kwargs or {}
        self.plastic = False
        self.private_gradients = specs.get('private_gradients', False)

        super().__init__(
            input_size,
            output_size,
            sizes,
            **self.specs,
            **self.factory_kwargs,
        )

    def set_plasticity(self, cereb=None):
        self.plasticity_specs = getattr(self, "plasticity_specs", None) or self.default_plasticity_specs(self.n_layers)

        for name, param in self.named_parameters():
            if cereb is None:
                if name in self.plasticity_specs and self.plastic:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
            else:
                param.requires_grad = cereb
        return self.plasticity_specs



"""
    def show_rnn(self, CerebDF=None, legend=False):

        hiddenL = self.rnn

        if hasattr(hiddenL, "masks"):  # <=> its custom
            fig, axs = plt.subplots(1, 2, figsize=(14, 5), dpi=80)
            plot_CCmask(
                hiddenL.h2h.weight.detach().numpy(),
                "RNN's weights",
                ax=axs[0],
                legend=CerebDF,
            )
            plot_CCmask(
                hiddenL.h2h.effective_weight().detach().numpy(),
                "RNN's effective weights",
                ax=axs[1],
                legend=CerebDF,
            )
            plt.show()

            fig, axs = plt.subplots(1, 3, figsize=(12, 3), dpi=80)
            axs[0].set_title("CCRNN's biases")
            axs[0].plot(hiddenL.h2h.bias.detach().numpy())
            axs[1].set_title("CCRNN's effective biases")
            axs[1].plot(hiddenL.h2h.effective_bias().detach().numpy())
            axs[2].set_title("CCRNN's output mask")
            axs[2].plot(hiddenL.masks[6])
            plt.subplots_adjust(wspace=0.3)
            plt.show()
        else:
            plt.subplot(121)
            plt.title("Recurrent weights")
            plt.imshow(hiddenL.weight.detach().numpy())
            plt.colorbar()
            plt.subplot(122)
            plt.title("recurrent layer biases")
            plt.plot(hiddenL.bias.detach().numpy())
            plt.show()


Variables:
- input_size
- output_size
- Cortex(aka rnn) size
- Pons size
- GrCs layer size
- PjCs layer size
- DCNs layer size
"""
