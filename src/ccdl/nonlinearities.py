# -*- coding: utf-8 -*-
"""
Created on 23/02/2023
@author: Leonardo Agueci - leonardo.agueci@ens.psl.eu

Plant (nonlinearity) modules mapping controller outputs to hand positions.
Defines the base :class:`Nonlinearity` interface and the concrete plants used
by the reaching task (two-joint arm, identity, velocity wrapper).
"""
import torch
import copy
from torch import Tensor
from ccdl.reaching import angles_from_cartesian


class Nonlinearity:
    """Base class for plant modules mapping an input signal to an output signal.

    Subclasses fix ``input_size``/``output_size`` and implement ``forward``.
    Tunable parameters are attached as attributes via ``save_specs``.
    """

    def __init__(self, input_size: int = 2, output_size: int = 2, params: dict = None):
        # HARD-CODED FEATURES OF THE CLASS, they will change depending on the type of nonlinearity used
        self.input_size = input_size
        self.output_size = output_size

        # TUNABLE FEATURES
        self.save_specs(copy.deepcopy(params))

    def save_specs(self, params):
        if params is not None:
            assert type(params) == dict, 'Nonlinearity class accept only a dictionary as specs argument'
            for key, value in params.items():
                setattr(self, key, value)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def inverse(self):
        raise NotImplementedError

    def check_params(self):
        input = torch.randn(self.input_size)
        output = self(input)
        assert output.size() == torch.Size([self.output_size]), 'Something went wrong while checking nonlinearity ' \
                                                                'parameters, check them.'
    
    def forward(self, *args, **kwargs):
        raise NotImplementedError('This is just the parent structure. For using the forward module, define the ' \
                                    ' as one of the modules in nonlinearities.py.')


class TwoJointsArm(Nonlinearity):
    r"""Returns the hand position of a two joints arm stating from the joints angles. theta = [time,batch,angle]
     l1 = 0.145, l2 = 0.284, shoulder_position = [0.,0.] """

    def __init__(self, specs: dict = None):
        super(TwoJointsArm, self).__init__(input_size=2, output_size=2)
        # HARD-CODED FEATURES OF THE CLASS, they will change depending on the type of nonlinearity used
        self.input_size = 2
        self.output_size = 2
        #self.input_domain = torch.tensor([[-torch., torch.pi],[]])

        # TUNABLE FEATURES
        if specs is not None:
            self.save_specs(copy.deepcopy(specs))
        else:
            self.save_specs({'l1': 0.22, 'l2': 0.27, 'shoulder_position': torch.tensor([0., 0.])})
        self.check_params()

    @staticmethod
    def name():
        return 'TwoJointsArm'
    
    def inverse_mapping(self, x):
        return angles_from_cartesian(x, l1 = self.l1, l2 = self.l2, shoulder_position = self.shoulder_position)
    
    def forward(self, theta: Tensor):

        dims = theta.ndim

        assert dims < 4 and theta.shape[-1] == 2, "Invalid input: two_joints_arm_seq input can only be 1/2/3 " \
                                                  "dimensional, with last dimension = 2"
        theta1 = theta[..., 0]
        theta2 = theta[..., 1]

        x = self.l1 * torch.cos(theta1) + self.l2 * torch.cos(theta1 + theta2) + self.shoulder_position[0]
        y = self.l1 * torch.sin(theta1) + self.l2 * torch.sin(theta1 + theta2) + self.shoulder_position[1]
        readout = torch.stack((x, y), dim=dims - 1)

        return readout


class Identity(Nonlinearity):
    r"""Identity plant module"""

    def __init__(self, input_size=2, output_size=2, specs: dict = None):
        super(Identity, self).__init__(input_size=input_size, output_size=output_size)
        # HARD-CODED FEATURES OF THE CLASS, they will change depending on the type of nonlinearity used
        self.input_size = input_size
        self.output_size = output_size
        self.check_params()

    @staticmethod
    def name():
        return 'Identity'
    
    def inverse_mapping(self, x):
        return x
    
    def forward(self, theta: Tensor):
        dims = theta.ndim

        assert dims < 4 and theta.shape[-1] == 2, "Invalid input: two_joints_arm_seq input can only be 1/2/3 " \
                                                  "dimensional, with last dimension = 2"
        return theta
    

class VelocityWrapper:
    def __init__(self, module: Nonlinearity, dt: float):
        """
        Initializes the wrapper with a Nonlinearity instance and a time step.

        :param module: An instance of a Nonlinearity subclass
        :param dt: Time step for Euler integration
        """
        self.module = module
        self.dt = dt
        self.input_size = module.input_size
        self.output_size = module.output_size

        self._copy_module_attributes()

    def _copy_module_attributes(self):
        """
        Copy all attributes from the wrapped module to the wrapper.
        """
        for attr_name in dir(self.module):
            if not attr_name.startswith('__') and not hasattr(self, attr_name):
                setattr(self, attr_name, getattr(self.module, attr_name))

    def name(self):
        return 'VelocityWrapper_on_' + self.module.name()
    
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, theta_old: Tensor, theta_dot: Tensor) -> Tensor:
        """
        Applies Euler integration and then the module's forward function.

        :param theta_old: Previous input position
        :param theta_dot: Input velocity
        :return: Result of the module's forward function applied to the new position
        """
        # Euler integration
        theta = theta_old + self.dt * theta_dot
        # Apply the module's forward function
        return self.module.forward(theta), theta
