# coding: utf-8
"""
Created on Tue May 25 16:58:10 2022
@author: Leonardo Agueci - leonardo.agueci@ens.psl.eu

This code implements different layers that could be assembled to create the 
network. Notice that these are not meant to be used outside a network class, 
particularly in the case of the continuous time one(s).

Most of 
CTRNN and EI layers are based on the code from: https://github.com/gyyang/nn-brain
"""
import math
import warnings
import numpy as np

import torch
from torch import Tensor, tensor, nn, relu
from torch.nn import init
from torch.nn.parameter import Parameter
import torch.nn.functional as F


# from torch.autograd import Variable

class Linear(nn.Module):
    r"""Applies a linear transformation to the incoming data: :math:`y = xA^T + b`

    This module is a custom version of torch.nn.Module.Linear`.

    Args:
        input_size: size of each input sample
        output_size: size of each output sample
        bias: If set to ``False``, the layer will not learn an additive bias.
            Default: ``True``

    Shape:
        - Input: :math:`(*, H_{in})` where :math:`*` means any number of
          dimensions including none and :math:`H_{in} = \text{in\_features}`.
        - Output: :math:`(*, H_{out})` where all but the last dimension
          are the same shape as the input and :math:`H_{out} = \text{out\_features}`.

    Attributes:
        weight: the learnable weights of the module of shape
            :math:`(\text{out\_features}, \text{in\_features})`. The values are
            initialized from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`, where
            :math:`k = \frac{1}{\text{in\_features}}`
        bias:   the learnable bias of the module of shape :math:`(\text{out\_features})`.
                If :attr:`bias` is ``True``, the values are initialized from
                :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
                :math:`k = \frac{1}{\text{in\_features}}`

    Examples::

        >>> m = nn.Linear(20, 30)
        >>> input = torch.randn(128, 20)
        >>> output = m(input)
        >>> print(output.size())
        torch.Size([128, 30])
    """
    __constants__ = ['input_size', 'output_size']
    input_size: int
    output_size: int
    weight: Tensor

    def __init__(self, input_size: int, output_size: int, bias: bool = True, init_method: str = 'uniform',
                 init_gain: float = 0.1, nonlinearity=relu, device=None, dtype=None) -> None:
        # assigning factory_kwargs values 
        device = 'cpu' if device is None else device
        dtype = torch.get_default_dtype() if dtype is None else dtype
        factory_kwargs = {'device': device, 'dtype': dtype}

        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.weight = Parameter(torch.empty((output_size, input_size), **factory_kwargs))
        if bias:
            self.bias = Parameter(torch.empty(output_size, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        
        if isinstance(nonlinearity, str):
            if nonlinearity == 'linear':
                self.nonlinearity = getattr(F, nonlinearity)
            else:
                self.nonlinearity = getattr(torch, nonlinearity)
        else:
            self.nonlinearity = nonlinearity

        self.nlin = self.nonlinearity
        self.nlin_name = self.nlin.__name__.lower() if nonlinearity is not None else 'linear'
        self.nlin_name = self.nlin_name if self.nlin_name != '<lambda>' else 'linear'
        self.init = init_method
        self.init_gain = init_gain
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(input_size), 1/sqrt(input_size)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109

        if self.init == 'normal':
            init.kaiming_normal_(
                self.weight, a=math.sqrt(5), nonlinearity=self.nlin_name
            )
        elif self.init == 'zero':
            init.zeros_(self.weight)
        elif self.init == 'xavier_normal':
            init.xavier_normal_(self.weight, gain=self.init_gain)
        elif self.init == 'xavier_uniform':
            init.xavier_uniform_(self.weight, gain=self.init_gain)

        else:
            init.kaiming_uniform_(
                self.weight, a=math.sqrt(5), nonlinearity=self.nlin_name
            )
        
        if self.bias is not None:
            if self.init in['xavier_normal', 'xavier_uniform']:
                # fan_mode = sum(init._calculate_fan_in_and_fan_out(self.weight))
                # bound = 1 / math.sqrt(fan_mode) if fan_mode > 0 else 0
                # init.uniform_(self.bias, -bound, bound)
                init.zeros_(self.bias)
            else:
                fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                init.uniform_(self.bias, -bound, bound)

    def forward(self, input: Tensor) -> Tensor:
        return F.linear(input, self.weight, self.bias)

    def extra_repr(self) -> str:
        return 'input_size={}, output_size={}, bias={}'.format(
            self.input_size, self.output_size, self.bias is not None
        )

class PosWLinear(nn.Module):
    """Applies a linear transformation to the incoming data: `y = xA^T + b`
    Same as Linear, except that weight matrix is constrained to be non-negative.

    """

    __constants__ = ["bias", "input_size", "output_size"]

    def __init__(self, input_size: int, output_size: int, bias: bool = True, init_method: str = 'uniform',
                 nonlinearity=relu, device=None, dtype=None) -> None:
        # assigning factory_kwargs values
        device = 'cpu' if device is None else device
        dtype = torch.get_default_dtype() if dtype is None else dtype
        factory_kwargs = {'device': device, 'dtype': dtype}

        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.weight = Parameter(torch.empty((output_size, input_size), **factory_kwargs))
        if bias:
            self.bias = Parameter(torch.empty(output_size, **factory_kwargs))
        else:
            self.register_parameter("bias", None)
        
        if isinstance(nonlinearity, str):
            if nonlinearity == 'linear':
                self.nonlinearity = getattr(F, nonlinearity)
            else:
                self.nonlinearity = getattr(torch, nonlinearity)
        else:
            self.nonlinearity = nonlinearity
        
        self.nlin = self.nonlinearity
        self.init = init_method
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.init == 'normal':
            init.kaiming_normal_(
                self.weight, a=math.sqrt(5), nonlinearity=self.nlin.__name__
            )
        else:
            init.kaiming_uniform_(
                self.weight, a=math.sqrt(5), nonlinearity=self.nlin.__name__
            )
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input) -> Tensor:
        # weight is non-negative
        return F.linear(input, torch.abs(self.weight), self.bias)


class EIRecLinear(nn.Module):
    """Recurrent E-I Linear transformation.
    Args:
        hidden_size: int, layer size
        e_prop: float [0,1], proportion of excitatory units
        bias: bool, enable bias parameters for the layer
        self_conn: bool, enable rec-units self-connections (usually not used)
    """

    __constants__ = ["hidden_size", "e_prop"]

    def __init__(self, hidden_size: int, e_prop, bias: bool = True, nonlinearity=relu, init_method: str =
    'uniform', init_e_factor=1, self_conn: bool = False, device=None, dtype=None) -> None:
        # assigning factory_kwargs values
        device = 'cpu' if device is None else device
        dtype = torch.get_default_dtype() if dtype is None else dtype
        factory_kwargs = {'device': device, 'dtype': dtype}

        super().__init__()
        self.hidden_size = hidden_size
        self.e_prop = e_prop
        self.self_conn = self_conn

        self.compute_ei_sizes()
        self.output_size = self.e_size

        self.weight = Parameter(torch.empty((hidden_size, hidden_size), **factory_kwargs))

        if bias:
            self.bias = Parameter(torch.empty(hidden_size, **factory_kwargs))
        else:
            self.register_parameter("bias", None)

        self.init_e_factor = init_e_factor
        self.reset_parameters()

        self.init_masks(**factory_kwargs)
        
        if isinstance(nonlinearity, str):
            if nonlinearity == 'linear':
                self.nonlinearity = getattr(F, nonlinearity)
            else:
                self.nonlinearity = getattr(torch, nonlinearity)
        else:
            self.nonlinearity = nonlinearity
        
        self.nlin = self.nonlinearity
        self.init = init_method

    def compute_ei_sizes(self):
        self.e_size = int(self.e_prop * self.hidden_size)
        self.i_size = self.hidden_size - self.e_size

    def init_masks(self, **factory_kwargs):
        size = self.hidden_size
        mask = np.tile([1] * self.e_size + [-1] * self.i_size, (size, 1))
        out_mask = mask[0, :] != -1
        if not self.self_conn:
            np.fill_diagonal(mask, 0)
        self.mask = Parameter(tensor(mask, dtype=torch.float32, **factory_kwargs), requires_grad=False)
        self.OUT_mask = Parameter(tensor(out_mask, dtype=torch.bool, **factory_kwargs), requires_grad=False)

    def reset_parameters(self) -> None:
        # WEIGHTS
        if self.init == 'normal':
            init.kaiming_normal_(
                self.weight, a=math.sqrt(5), nonlinearity=self.nlin.__name__
            )
        else:
            init.kaiming_uniform_(
                self.weight, a=math.sqrt(5), nonlinearity=self.nlin.__name__
            )

        # Scale E weight by E-I ratio (balancing E/I total signal)
        self.weight.data[:, : self.e_size] /= (self.e_size / self.i_size) * self.init_e_factor

        # BIASES
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def effective_weight(self) -> Tensor:
        return torch.abs(self.weight) * self.mask

    def forward(self, input: Tensor) -> Tensor:
        # weight is non-negative
        return F.linear(input, self.effective_weight(), self.bias)


class CustomRecLinear(nn.Module):
    """Custom Recurrent E-I Linear transformation, where you directly define
        the recurrent structure.

    Args:
        masks:   list, contains 2D-arrays defining the RNN layer structure
        indeces: list of tuples, defining indeces identifying cells groups
        e_prop: float, proportion of Excitatory units in the FC-RNN submodule
        nonlinearity: torch.functional, define the activation function used
        bias:    bool, enable bias parameters' usage
        init_e_factor: Reset_parameter initialization resize factor, in case
                            of instable dynamics DEFAULT 1

    Inputs:
        input: (seq_len, batch, input_size), network input
    """

    __constants__ = ["hidden_size"]

    def __init__(self, masks, indeces, e_prop, nonlinearity=F.relu, init_method: str = 'uniform', bias: bool = True,
                 init_e_factor=1, device=None, dtype=None) -> None:
        # assigning factory_kwargs values
        device = 'cpu' if device is None else device
        dtype = torch.get_default_dtype() if dtype is None else dtype
        factory_kwargs = {'device': device, 'dtype': dtype}

        super().__init__()

        if len(masks) != 7:
            raise IOError(
                "For the masks input (1st one) you must give me EXACTLY 6 masks, namely"
            )

        self.hidden_size = (masks[0]).shape[0]
        self.weight = Parameter(torch.empty((self.hidden_size, self.hidden_size), **factory_kwargs))
        self.indeces = indeces
        self.Nneocort = indeces[0][0][1]
        self.e_size = int(e_prop * self.Nneocort)
        self.ei_ratio = e_prop / (1 - e_prop)
        
        if isinstance(nonlinearity, str):
            if nonlinearity == 'linear':
                self.nonlinearity = getattr(F, nonlinearity)
            else:
                self.nonlinearity = getattr(torch, nonlinearity)
        else:
            self.nonlinearity = nonlinearity
        
        self.nlin = self.nonlinearity
        self.init = init_method
        S_mask = tensor(masks[0], dtype=torch.int32, device=factory_kwargs['device'])
        P_mask = tensor(masks[1], dtype=torch.int32, device=factory_kwargs['device'])
        self.SP_mask = Parameter(
            S_mask * P_mask, requires_grad=False
        )  # merging them together since fixed, for simplicity

        self.FW_mask = Parameter(tensor(masks[2], dtype=torch.float32, device=factory_kwargs['device']),
                                 requires_grad=False)
        self.PB_mask = Parameter(tensor(masks[3], dtype=torch.int32, device=factory_kwargs['device']),
                                 requires_grad=False)
        self.FB_mask = Parameter(tensor(masks[4], dtype=torch.float32, device=factory_kwargs['device']),
                                 requires_grad=False)
        self.IN_mask = Parameter(tensor(masks[5], dtype=torch.int32, device=factory_kwargs['device']),
                                 requires_grad=False)
        self.OUT_mask = Parameter(tensor(masks[6], dtype=torch.int32, device=factory_kwargs['device']),
                                  requires_grad=False)
        if bias:
            self.bias = Parameter(torch.empty(self.hidden_size, **factory_kwargs))
        else:
            self.register_parameter("bias", None)

        self.init_e_factor = init_e_factor
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Adapted from [2], extended to initialization customization"""
        if self.init == 'normal':
            init_function = init.kaiming_normal_
        else:
            init_function = init.kaiming_uniform_

        with torch.no_grad():
            # Not necessary (not initialized parts should be unused),
            #    only for a better visualization
            init.zeros_(self.weight)
            init.zeros_(self.bias)
            for i in range(len(self.indeces)):

                r_idx, c_idx = self.indeces[i]

                # ### Weights Part

                tempW = torch.zeros_like(
                    self.weight[r_idx[0]: r_idx[1], c_idx[0]: c_idx[1]]
                )
                init_function(
                    tempW, a=math.sqrt(5), nonlinearity=self.nlin.__name__
                )

                if i == 0:
                    # Neocortex (E/I) part only
                    # Scale E weight by E-I ratio (balancing E/I total signal)
                    #  (only the Neocortical part)
                    tempW[:, :self.e_size] /= self.ei_ratio * self.init_e_factor

                self.weight[
                r_idx[0]: r_idx[1], c_idx[0]: c_idx[1]
                ] = tempW.detach().clone()

                # ### Biases Part

                if self.bias is not None:
                    fan_in = self.weight[
                             r_idx[0]: r_idx[1], c_idx[0]: c_idx[1]
                             ].shape[1]
                    # fan_out = self.weight[r_idx[0]:r_idx[1],c_idx[0]:c_idx[1]].shape[0]
                    bound = 1 / (math.sqrt(fan_in) * self.init_e_factor)
                    init.uniform_(self.bias[r_idx[0]: r_idx[1]], -bound, bound)

    def effective_weight(self) -> Tensor:
        return torch.abs(self.weight) * self.SP_mask + self.FW_mask

    def effective_bias(self):
        if self.bias is not None:
            return self.bias * self.PB_mask + self.FB_mask
        else:
            return None

    def forward(self, input: Tensor) -> Tensor:
        return F.linear(input, self.effective_weight(), self.effective_bias())


class CTLinear(nn.Module):
    r"""Continuous-time Linear layer. Euler's integration of rate-network dynamics

    This layer computes the following function:

    .. math::
        a = dt/\tau \newline
        x_{t+1} = (1 - a)x_t + a W^{in}u_t) + \sqrt{2a\sigma^2}N(0,1) \newline
        r_{t+1} = \phi ([x_t]_+ ) with \phi the activation function
    
    Args:
        input_size: int, size of the input
        output_size: int, size of the output
        bias: bool, enable bias parameters' usage
        dt: float, time step
        tau: float, time constant of the 
        sigma: float, standard deviation of the noise
        nonlinearity: torch.nn, define the activation function

    Inputs:
        input: (batch, input_size), network input
        prev_step (optional): (batch, output_size), previous state of the network

    """

    def __init__(self, input_size: int, output_size: int, bias: bool = True, init_method: str = 'uniform',
                 nonlinearity=relu, dt=None, tau=100., sigma: float = 0., device=None, dtype=None, **kwargs) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size
        
        if isinstance(nonlinearity, str):
            if nonlinearity == 'linear':
                self.nonlinearity = getattr(F, nonlinearity)
            else:
                self.nonlinearity = getattr(torch, nonlinearity)
        else:
            self.nonlinearity = nonlinearity
        
        self.nonlin = nonlinearity
        self.init = init_method
        self.tau = tau
        self.dt = dt
        if dt is None:
            self.alpha = 1
        else:
            self.alpha = dt / self.tau
        self.sigma = np.abs(sigma)
        self._sigma = np.sqrt(2 * self.alpha) * sigma

        self.layer = Linear(input_size, output_size, bias, init_method, nonlinearity, **factory_kwargs)

        if "e_prop" in kwargs and kwargs["e_prop"] is not None:
            warnings.warn(
                "e_prop is not implemented yet! Using a Linear layer instead (signed)")

    def init_state(self, input):
        batch_size = input.shape[0]
        return (
            torch.zeros(batch_size, self.output_size).to(input.device),
            torch.zeros(batch_size, self.output_size).to(input.device),
        )

    def state_update(self, input, prev_step):
        """continuous time integration handler."""
        state, output = prev_step
        state = (1 - self.alpha) * state + self.alpha * self.layer(input)
        state += self._sigma * torch.randn_like(state)
        output = self.nonlin(state)
        return state, output

    def forward(self, input: Tensor, prev_step=None):
        """Forward method for single timestep"""

        assert input.shape[1] == self.input_size, f'Module {self.__class__.__name__} support only inputs with dimensions (batch, {self.input_size})'

        if prev_step is None:
            prev_step = self.init_state(input)

        state, output = self.state_update(input, prev_step)
        return state, output


class CTRNN(nn.Module):
    r"""Continuous-time RNN. Euler's integration of rate-neuron network dynamics

    Each layer computes the following function:

    .. math::
        a = dt/tau
        x_{t+1} = (1 - a)x_t + a(W^{rec} r_t + W^{in}u_t) + \sqrt{2a\sigma^2}N(0,1)
        r_{t+1} = [x_t]_+

    Args:
        input_size: Number of input neurons
        hidden_size: Number of hidden neurons
        nonlinearity: torch.functional, activation function
        dt: time-step for the Euler integration of the Continuous model
        tau: time-constant
        structure: FC(i.e. fully-connected) or custom(inputs: masks & indices)
        in_bias: bool, enable input bias parameter
        h_bias: bool, enable hidden-layer bias parameter
        sigma_rec: float, defines the noise amplitude in each rec/hidden unit
        in_sign: {1,-1,None}, it fix the input sign to pos/neg or real-valued
        e_prop: [0,1], defines numerosity of e/i neurons (#(e.u)/#(i.u) ratio)
        self_conn: bool, enable rec-units self-connections
        init_method: str, choose between uniform and normal distribution
        init_e_factor: Reset_parameter initialization resize factor, in case of
                        unstable dynamics DEFAULT 1

    Inputs:
        input: (seq_len, batch, input_size), network input
        hidden: (batch, hidden_size), initial hidden activity
    """

    __constants__ = [
        "input_size",
        "hidden_size",
        "tau",
        "in_bias",
        "h_bias",
        "self_conn",
    ]

    def __init__(self, input_size, hidden_size, nonlinearity=relu, init_method='uniform', dt=None, tau=100, in_bias=True,
                 h_bias=True, sigma_rec=0, e_prop=None, self_conn=False, init_e_factor=1,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        
        if isinstance(nonlinearity, str):
            if nonlinearity == 'linear':
                self.nonlinearity = getattr(F, nonlinearity)
            else:
                self.nonlinearity = getattr(torch, nonlinearity)
        else:
            self.nonlinearity = nonlinearity
        
        self.e_prop = e_prop

        self.tau = tau
        self.dt = dt
        if dt is None:
            self.alpha = 1
        else:
            self.alpha = dt / self.tau

        self._sigma_rec = np.sqrt(2 * self.alpha) * sigma_rec
        self.self_conn = self_conn
        # self.rec_signal = rec_signal
        self.init_e_factor = init_e_factor

        self.input2h = Linear(input_size, hidden_size, bias=in_bias, init_method=init_method,
                              nonlinearity=nonlinearity, **factory_kwargs)

        if e_prop is not None:
            self.h2h = EIRecLinear(hidden_size, e_prop, bias=h_bias, init_method=init_method,
                                   init_e_factor=self.init_e_factor, self_conn=self.self_conn, **factory_kwargs)
            self.OUT_mask = self.h2h.OUT_mask.detach().clone()
            self.output_size = self.OUT_mask.sum().item()
        else:
            self.h2h = Linear(hidden_size, hidden_size, bias=h_bias, init_method=init_method,
                              nonlinearity=nonlinearity, **factory_kwargs)
            self.output_size = hidden_size

    def init_hidden(self, input) -> tuple:
        batch_size = input.shape[0]
        return (
            input.new_zeros(batch_size, self.hidden_size),
            input.new_zeros(batch_size, self.hidden_size),
        )

    def recurrence(self, input, hidden):
        """Recurrence helper."""
        output, state = hidden
        total_input = self.input2h(input) + self.h2h(output)
        state = (1 - self.alpha) * state + self.alpha * total_input
        state += self._sigma_rec * torch.randn_like(state)
        output = self.nonlinearity(state)
        return output, state

    def forward(self, input: Tensor, hidden: tuple = (None, None)):
        """Propagate input through the network."""
        #assert input.shape[1] == self.input_size, f'Module {self.__class__.__name__} support only inputs with dimensions (batch, {self.input_size})'

        if hidden[0] is None:
            hidden = self.init_hidden(input)

        assert len(input.shape) == 2, f'Module {self.__class__.__name__} support only inputs with dimensions (batch,)'

        hidden = self.recurrence(input, hidden)
        output = hidden[0].clone()
        # selecting the correct output units (useful for EI nets, no effects in the )
        if self.e_prop:
            output = torch.masked_select(output, self.OUT_mask).view(batch_size, -1)

        return output, hidden

    def transparent_forward(self, input, hidden=None):
        """Propogate input through the network.
        Debugging method"""

        batch_size = input.shape[1]

        if hidden is None:
            hidden = self.init_hidden(input)

        output = []

        steps = range(input.size(0))

        in1 = []
        in2 = []
        tot_in = []
        states1 = []
        outputs1 = []
        outputs2 = []
        for i in steps:
            state, output = hidden
            input1 = self.input2h(input[i]) * self.IN_mask
            rec_input = self.h2h(output)
            total_input = input1 + rec_input
            state1 = (1 - self.alpha) * state + self.alpha * total_input
            state1 += self._sigma_rec * torch.randn_like(state1)
            output1 = self.nonlinearity(state1)
            output = torch.masked_select(output1, self.OUT_mask).view(batch_size, -1)

            in1.append(input1)
            in2.append(rec_input)
            tot_in.append(total_input)
            states1.append(state1)
            outputs1.append(output1)
            outputs2.append(output)
            hidden = state1, output1

        in1 = torch.stack(in1, dim=0)
        in2 = torch.stack(in2, dim=0)
        tot_in = torch.stack(tot_in, dim=0)
        states1 = torch.stack(states1, dim=0)
        outputs1 = torch.stack(outputs1, dim=0)
        outputs2 = torch.stack(outputs2, dim=0)


        return in1, in2, tot_in, states1, outputs1, outputs2
    

class CTRNN2(CTRNN):
    r""" Modified version of CTRNN, it allows to have two different input layers.
    The module contains two different sets of weights, one for each input.
    Notice that, for simplicity, the two inputs are passed as a merged (concatenated) tensor.

    Args:
        (CTRNN arguments, see CTRNN)
        input_size_2: size of the second input
    """

    __constants__ = [
        "input_size",
        "hidden_size",
        "tau",
        "in_bias",
        "h_bias",
        "self_conn",
    ]

    def __init__(self, input_size, hidden_size, nonlinearity='relu', init_method='uniform', dt=None, tau=100, in_bias=True,
                 h_bias=True, sigma_rec=0, e_prop=None, self_conn=False, init_e_factor=1, input_size_2=0,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(input_size, hidden_size, nonlinearity, init_method, dt, tau, in_bias, h_bias, sigma_rec, e_prop,
                         self_conn, init_e_factor, device, dtype)

        self.input_size_2 = int(input_size_2)
        self.total_input_size = self.input_size + self.input_size_2

        if self.input_size_2 > 0:
            self.input2h_2 = Linear(self.input_size_2, hidden_size, bias=in_bias, init_method=init_method,
                                    nonlinearity=nonlinearity, **factory_kwargs)

    def recurrence(self, input, hidden) -> tuple:
        """Recurrence helper."""
        output, state = hidden
        total_input = self.input2h(input[..., :self.input_size]) + self.h2h(output)
        # summing second input if present
        if self.input_size_2 > 0:
            total_input += self.input2h_2(input[..., self.input_size:])
        state = (1 - self.alpha) * state + self.alpha * total_input
        state += self._sigma_rec * torch.randn_like(state)
        output = self.nonlinearity(state)
        return output, state

    def forward(self, input, hidden=None) -> tuple:
        """Propagate input through the network."""
        assert input.shape[-1] == self.total_input_size, f"Module {self.__class__.__name__} should receive input of size" \
                                                            f"{self.total_input_size}, but {input.shape[-1]} found."
        if hidden is None:
            hidden = self.init_hidden(input)

        hidden = self.recurrence(input, hidden)
        output = hidden[0].clone()
        # selecting the correct output units (useful for EI nets, no effects in the )
        if self.e_prop:
            output = torch.masked_select(output, self.OUT_mask).view(batch_size, -1)

        return output, hidden



if __name__ == '__main__':
    # run some code (usually testing) if the module is run as a script
    import matplotlib.pyplot as plt

    choice = input('Do you wanna plot the weights and output distributions?')

    input_size = 20
    output_size = 50
    batch_size = 2
    input_tensor = torch.randn(batch_size, input_size)

    if input('Testing CTLinear?') in ['y','yes']:
        # Creating an instance of CTLinear
        layer = CTLinear(input_size, output_size)

        print(layer.layer.weight.shape)
        print(layer.layer.bias.shape)

        # Generating random test input tensor

        # Forward pass through the layer
        state, output = layer(input_tensor)

        # Printing the output state and output tensor
        print(f"State : {type(state)}, {state.shape}")
        print(f"Output: {type(output)}, {output.shape}")

        print('Testing CTLinear (more):')
        # Defining parameters and creating an instance of CTLinear
        init_method = 'normal'
        layer = CTLinear(input_size, output_size, init_method = init_method,
                    nonlinearity=torch.sigmoid, dt=None, 
                    sigma = 0.2, device=None, dtype=None, e_prop = 0.8)

        # Forward pass through the layer
        state, output = layer(input_tensor)
        if choice.lower() in ['y','yes']:
            # Generating an histogram of the weights
            hist = torch.histc(
                layer.layer.weight.detach().clone(), bins=20
            )
            plt.title(f'Plotting weights distro, {init} distro chosen')
            plt.plot(hist)
            plt.show()

        print(f'Check: {"passed" if torch.all(output>=0) and torch.all(output<=1) else "failed"}')

        print(f'Checking sigmoid activation: {"passed" if torch.all(output>=-1) else "failed"}')

        if choice.lower() in ['y','yes']:
            plt.title(f'Plotting output distro, {init} distro chosen')
            plt.plot(output.flatten().detach().clone())
            plt.show()


    print('Testing CTRNN:')
    dt = 10
    tau = 50
    layer = CTRNN(input_size, output_size, dt=dt, tau=tau)

    print(f'alpha ({dt}/{tau}) = {layer.alpha}')

    output, hidden = layer(input_tensor)
    print(output, hidden)
    print(f'output: {output.shape}, hidden: {hidden[1].shape}')

    print('Testing CTRNN2:')
    dt = 10
    tau = 50
    size_2 = 10

    input_tensor = torch.randn(batch_size, input_size + size_2)
    layer = CTRNN2(input_size, output_size, dt=dt, tau=tau, input_size_2=size_2)

    print(f'alpha ({dt}/{tau}) = {layer.alpha}')

    output, hidden = layer(input_tensor)
    
    print(f'output: {output.shape}, hidden: {hidden[1].shape}')
    print(f'input layer 1 shape: {layer.input2h.weight.shape}')
    print(f'input layer 2 shape: {layer.input2h_2.weight.shape}')
