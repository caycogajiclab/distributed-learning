# -*- coding: utf-8 -*-
"""
Created on 06/03/2023
@author: Leonardo Agueci - leonardo.agueci@ens.psl.eu

Reaching task: two-joint arm kinematics and the :class:`ReachingDataset` that
generates the unperturbed and perturbed reaching trajectories used as network
inputs and targets. Provides both torch (batched) and numpy (scalar) versions
of the forward/inverse kinematics and the perturbation transform.
"""

import warnings
import copy

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib import cm
import torch


# Reaching task functions (torch version)

def cartesian_coordinates(theta, l1=0.145, l2=0.284, shoulder_position=[0., 0.]):
    """Forward kinematics (torch): joint angles -> hand Cartesian position.

    Angles are taken from the last dimension of ``theta``; the (x, y) hand
    position is returned with the same leading shape.
    """
    theta1 = theta[..., 0]
    theta2 = theta[..., 1]
    x = l1 * torch.cos(theta1) + l2 * torch.cos(theta1 + theta2) + shoulder_position[0]
    y = l1 * torch.sin(theta1) + l2 * torch.sin(theta1 + theta2) + shoulder_position[1]
    cart = torch.stack((x, y), dim=-1)
    cart.requires_grad = False
    return cart


def angles_from_cartesian(cart, l1=0.145, l2=0.284, shoulder_position=[0., 0.]):
    """Inverse kinematics (torch): hand Cartesian position -> joint angles.

    Inverse of :func:`cartesian_coordinates`. arccos raises benign
    divide-by-zero RuntimeWarnings at the boundary positions, suppressed here.
    """
    cart_x = cart[..., 0]
    cart_y = cart[..., 1]
    cart_x = cart_x - shoulder_position[0]
    cart_y = cart_y - shoulder_position[1]
    B2 = cart_x ** 2 + cart_y ** 2
    B = torch.sqrt(B2)
    ## using the selected positions, arccos gives some RuntimeWarnings (division by zero), but converges.
    ## the next two lines just avoid the warning messages to be shown
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        theta1 = torch.arccos(cart_x / B) - torch.arccos((l1 ** 2 - l2 ** 2 + B2) / (2 * B * l1))
        theta2 = torch.pi - torch.arccos((l1 ** 2 + l2 ** 2 - B2) / (2 * l1 * l2))

    theta = torch.stack((theta1, theta2), dim=-1)
    theta.requires_grad = False
    return theta


def apply_perturbation(initial_array: torch.Tensor, home_state: torch.Tensor,
                       angle: float = None, amplitude_factor: float = 1,
                       translation: torch.Tensor = torch.tensor([0., 0.])) -> torch.Tensor:
    """Apply a rotation/amplitude/translation perturbation to reach targets (torch).

    Rotates and scales trajectories about ``home_state`` then translates them.
    The x, y coordinates must be the last dimension of ``initial_array``.
    """
    device = initial_array.device
    dtype = initial_array.dtype
    with torch.no_grad():
        angle = np.deg2rad(angle.item())
        rotation_matrix = torch.tensor([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]], device=device,
                                       dtype=dtype)
        array = initial_array - home_state
        rotated_array = amplitude_factor * torch.permute(torch.matmul(rotation_matrix, torch.permute(array, (0, 2, 1))),
                                                         (0, 2, 1)) + translation
        rotated_array = rotated_array + home_state

    rotated_array.requires_grad = False

    return rotated_array


# Reaching Dataset Class
class ReachingDataset:
    """
    This class provides a comprehensive interface for generating, manipulating, 
    and visualizing a dataset related to reaching movements. 
    
    By changing the 'datatype' parameter, users can switch between cartesian 
    and the related angles coordinates, corresponding to the Two-Joints arm 
    model angles with joints length l1 and l2. 

    Properties:
        - dtype: Returns the data type of the dataset (e.g., float32), ensuring 
                consistency across computations.
        - device: Returns the computational device (e.g., GPU or CPU) where the 
                dataset is stored, facilitating efficient processing.
        - num_reach_combinations: Returns the total number of unique reach 
                combinations present in the dataset.
        - home_cart_state: Provides the home Cartesian state of the robotic 
                arm, representing its default or resting position.
        - input_size: Returns the dimensionality of the input data, informing 
                the structure required for model inputs.
        - output_size: Returns the dimensionality of the output data, informing 
                the structure required for model outputs.
        - unperturbed_cartesian_goal: Retrieves the goal positions in Cartesian 
                coordinates for unperturbed trajectories.
        - unperturbed_cartesian: Retrieves the entire set of unperturbed 
                Cartesian trajectories.
        - unperturbed_angles: Retrieves the entire set of unperturbed joint 
                angle trajectories.

    Methods:
        - angles_from_cartesian(cartesian_coords): Converts given Cartesian 
            coordinates to corresponding joint angles.
        - cartesian_coordinates(angle_coords): Converts given joint angles to 
            corresponding Cartesian coordinates.
        - apply_perturbation(perturbation_params): Applies specified 
            perturbations to the dataset, simulating variations or disturbances.
        - generate_unperturbed_dataset(): Generates a dataset of unperturbed 
            reaching movements, serving as a baseline for comparisons.
        - generate_perturbed_dataset(): Generates a dataset with applied 
            perturbations, useful for testing robustness and adaptability.
        - plot_dataset(): Visualizes the dataset including both unperturbed and 
            perturbed trajectories, providing insights into movement patterns.
        - plot_dataset2(): Offers an alternative visualization format, 
            displaying inputs and targets for selected tasks, aiding in 
            comprehensive analysis.

    Static Methods:
        - default_parameters(): Returns a dictionary of default parameters for 
            initializing the dataset, ensuring ease of use and consistency.
        - default_perturbation_parameters(): Returns a dictionary of default 
            perturbation parameters, providing standard settings for 
            perturbation applications.
    """

    def __init__(self, dataset_parameters=None):
        
        if dataset_parameters is not None:
            self.dataset_parameters = copy.deepcopy(dataset_parameters)
        else:
            self.dataset_parameters = self.default_parameters()
            print("ReachingDataset: using default parameters:\n", self.dataset_parameters)

        self.input_type = self.dataset_parameters['input_type']
        self.input_mode = self.dataset_parameters['input_mode'] if 'input_mode' in self.dataset_parameters.keys() else 'last'
        assert self.input_type in ['one-hot', 'autoenc'], 'input_type in ["one-hot", "autoenc"]. Provide correct input'
        self.joints_length = self.dataset_parameters['joints_length']
        self.hold_cue_time = self.dataset_parameters['hold_cue_time']
        self.target_input_duration = self.dataset_parameters['t_in_duration'] if 't_in_duration' in self.dataset_parameters.keys() else self.hold_cue_time
        assert isinstance(self.target_input_duration, int) and isinstance(self.hold_cue_time, int), 't_in_duration must be int'
        self.timesteps = self.dataset_parameters['timesteps']

        self.shoulder_position = torch.zeros(2)

        self.perturbed_cartesian = None
        self.perturbed_angles = None

    @property
    def dtype(self):
        try:
            dtype = self.dataset_parameters['dtype'] if type(self.dataset_parameters['dtype']) is str else self.dataset_parameters['dtype'].name
        except:
            dtype = 'float32'
        try:
            return getattr(torch, dtype)
        except:
            return getattr(torch,dtype.split('.')[1])

    @property
    def device(self):
        try:
            return self.dataset_parameters['device']
        except:
            return None

    @staticmethod
    def default_parameters():
        return {
            'tasks': 8,
            'small_ring': False,
            'target': 'moving',
            'interpolation': 'bell',
            'timesteps': 130,
            'hold_cue_time': 30,
            "t_in_duration": 30,
            'small_ring_rot': False,
            'rings_size': [0.1, 0.05],
            'joints_length': [0.22, 0.27],
            'input_type': 'autoenc',
            "input_mode": "full-go"
        }
    
    @staticmethod
    def default_perturbation_parameters():
        return {
            'angle': 10,
            'amplitude_factor': 1.,
            'translation': [0., 0.]
        }

    @property
    def num_reach_combinations(self):
        # + 1 for the no-movement trajectory, manually added at the end
        return len(self.unperturbed_cartesian_goal)

    @property
    def home_cart_state(self):
        return torch.tensor([-0.0055, 0.33], dtype=self.dtype, device=self.device, requires_grad=False)

    @property
    def input_size(self):
        if self.input_type == 'one-hot':
            return self.timesteps, self.num_reach_combinations, self.num_reach_combinations + 1
        else:
            if self.input_mode == 'full':
                return self.timesteps, self.num_reach_combinations, len(self.joints_length)
            elif self.input_mode == 'full-go':
                return self.timesteps, self.num_reach_combinations, len(self.joints_length) + 1
            else:
                return self.timesteps, self.num_reach_combinations, len(self.joints_length) + 1

    @property
    def output_size(self):
        return self.timesteps, self.num_reach_combinations, len(self.joints_length)

    @property
    def unperturbed_cartesian_goal(self):
        angles = [np.array(np.linspace(0, 360, self.dataset_parameters['tasks'] + 1))[:-1]]
        if self.dataset_parameters['small_ring']:
            rot = 0.
            if self.dataset_parameters['small_ring_rot']:
                rot = angles[0][1] / 2
            small_ring = angles[0] + rot
            angles.append(small_ring)

        x = self.dataset_parameters['rings_size'][0] * np.cos(np.deg2rad(angles[0]))
        y = self.dataset_parameters['rings_size'][0] * np.sin(np.deg2rad(angles[0]))

        # adding stay-still trajectory
        x = np.concatenate((np.array([0.0]), x))
        y = np.concatenate((np.array([0.0]), y))

        # small ring trajectories
        if self.dataset_parameters['small_ring']:
            x_s = self.dataset_parameters['rings_size'][1] * np.cos(np.deg2rad(angles[1]))
            y_s = self.dataset_parameters['rings_size'][1] * np.sin(np.deg2rad(angles[1]))

            x = np.concatenate((x, x_s))
            y = np.concatenate((y, y_s))

        return np.column_stack((x, y))

    def angles_from_cartesian(self, array):
        return angles_from_cartesian(array, *self.joints_length,
                                     shoulder_position=self.shoulder_position).type(self.dtype).to(self.device)

    def cartesian_coordinates(self, array):
        return cartesian_coordinates(array, *self.joints_length,
                                     shoulder_position=self.shoulder_position).type(self.dtype).to(self.device)

    @property
    def unperturbed_cartesian(self):
        interpolation = self.dataset_parameters['interpolation']
        time_seq = np.linspace(0.0, self.timesteps - 1, self.timesteps)
        target = np.zeros([time_seq.size, self.num_reach_combinations, len(self.joints_length)])
        home_cart = self.home_cart_state.cpu().numpy()

        # preparatory period
        target[:self.hold_cue_time, :, :] = home_cart

        # Movement period
        if self.dataset_parameters['target'] == 'fixed':
            target[self.hold_cue_time:] = self.unperturbed_cartesian_goal + home_cart
        else:
            if interpolation == 'linear':
                prop_vector = np.linspace(0, 1, time_seq.size - self.hold_cue_time)
            elif interpolation == 'bell':
                prop_vector = np.linspace(-1, 1, time_seq.size - self.hold_cue_time)
                # slope
                k = 10
                prop_vector = 1 / (1 + np.exp(-k * prop_vector))
            else:
                raise IOError("interpolation input could be either 'linear' or 'bell', no other options are supported!")

            prop_array = np.repeat(prop_vector[:, np.newaxis], self.num_reach_combinations, axis=1)
            prop_array = np.repeat(prop_array[:, :, np.newaxis], 2, axis=2)

            target[self.hold_cue_time:] = np.repeat(self.unperturbed_cartesian_goal[np.newaxis, :, :], self.timesteps -
                                                    self.hold_cue_time, axis=0)
            target[self.hold_cue_time:] = np.multiply(target[self.hold_cue_time:, :, :], prop_array) + home_cart
        return torch.tensor(target, dtype=self.dtype, device=self.device, requires_grad=False)

    @property
    def unperturbed_angles(self):
        return self.angles_from_cartesian(self.unperturbed_cartesian)

    def apply_perturbation(self, perturbation_parameters):

        # converting all parameters to tensors
        for key, param in perturbation_parameters.items():
            if type(param) == np.ndarray:
                perturbation_parameters[key] = torch.from_numpy(param).type(self.dtype).to(self.device)
            elif type(param) == torch.Tensor:
                pass
            else:
                perturbation_parameters[key] = torch.tensor(param).type(self.dtype).to(self.device)

        self.perturbed_cartesian = apply_perturbation(self.unperturbed_cartesian, self.home_cart_state,
                                                      **perturbation_parameters).type(self.dtype)

        self.perturbed_angles = self.angles_from_cartesian(self.perturbed_cartesian)
        self.pert_params = perturbation_parameters

    def generate_unperturbed_dataset(self, datatype=None, norm_inputs=True):
        inputs = torch.zeros(*self.input_size).to(self.device)
        if self.input_type == 'one-hot':
            # input size = # tasks + go cue input
            for task in range(self.num_reach_combinations):
                inputs[:self.target_input_duration, task, task] = 1
        else:
            if self.input_mode == 'last':
                inputs[:self.target_input_duration, :, :-1] = self.unperturbed_cartesian[-1,:, :].repeat(self.target_input_duration,1,1) 
                if norm_inputs:
                    inputs[:self.target_input_duration, :, :-1] = 1/(self.dataset_parameters['rings_size'][0]) * (inputs[:self.target_input_duration, :, :-1] - self.unperturbed_cartesian[0, 0, :])
            else:
                if self.input_mode == 'full-go':
                    inputs[:,:,:-1] = self.unperturbed_cartesian
                    if norm_inputs:
                        inputs[:,:,:-1] = 1/(self.dataset_parameters['rings_size'][0]) * (inputs[:,:,:-1] - self.unperturbed_cartesian[0, 0, :])
                else:
                    inputs = self.unperturbed_cartesian 
                    if norm_inputs:
                        inputs = 1/(self.dataset_parameters['rings_size'][0]) * (inputs - self.unperturbed_cartesian[0, 0, :])
                
        
        # adding go cue (if necessary)
        if self.input_type =='one-hot' or (self.input_type == 'autoenc' and self.input_mode == 'last'):
            inputs[self.hold_cue_time:, :, -1] = 1
        elif self.input_type == 'autoenc' and self.input_mode == 'full-go':
            # adding the cue as nogo=1, go=0
            inputs[:self.hold_cue_time, :, -1] = 1
        inputs = inputs.type(self.dtype).to(self.device)

        if datatype == 'angles':
            outputs = self.angles_from_cartesian(self.unperturbed_cartesian)
        else:
            outputs = self.unperturbed_cartesian

        return inputs.detach().clone(), outputs.detach().clone()

    def generate_perturbed_dataset(self, datatype='cart', update_inputs=False, norm_inputs=True):
        assert self.perturbed_cartesian is not None, "perturbed trajectories not assigned yet, use method " \
                                                     "ReachingDataset.apply_perturbation(perturbation_parameters)"
        
        if datatype == 'angles':
            outputs = self.angles_from_cartesian(self.perturbed_cartesian)
        else:
            outputs = self.perturbed_cartesian
        
        inputs = torch.zeros(*self.input_size).to(self.device)
        if self.input_type == 'one-hot':
            # input size = # tasks + go cue input
            for task in range(self.num_reach_combinations):
                inputs[:self.target_input_duration, task, task] = 1
        else:
            if not update_inputs:
                # the inputs must remain the same during the perturbation, otherwise the cortex could already know how to correct it
                if self.input_mode == 'last':
                    inputs[:self.target_input_duration, :, :-1] = self.unperturbed_cartesian[-1,:, :].repeat(self.target_input_duration,1,1)
                    if norm_inputs:
                        inputs[:self.target_input_duration, :, :-1] = 1/(self.dataset_parameters['rings_size'][0]) * (inputs[:self.target_input_duration, :, :-1] - self.unperturbed_cartesian[0,0,:])
                else:
                    if self.input_mode == 'full-go':
                        inputs[:,:,:-1] = self.unperturbed_cartesian
                        if norm_inputs:
                            inputs[:,:,:-1] = 1/(self.dataset_parameters['rings_size'][0]) * (inputs[:,:,:-1] - self.unperturbed_cartesian[0, 0, :])
                    else:
                        inputs = self.unperturbed_cartesian 
                        if norm_inputs:
                            inputs = 1/(self.dataset_parameters['rings_size'][0]) * (inputs - self.unperturbed_cartesian[0, 0, :])
            else:
                # we need to use the perturbed targets as inputs
                if self.input_mode == 'last':
                    inputs[:self.target_input_duration, :, :-1] = (self.perturbed_cartesian[-1,:, :].repeat(self.target_input_duration,1,1) - self.unperturbed_cartesian[0,0,:])
                    if norm_inputs:
                        inputs[:self.target_input_duration, :, :-1] = 1/(self.dataset_parameters['rings_size'][0]) * inputs[:self.target_input_duration, :, :-1] # should also be divided by *self.pert_params['amplitude_factor']
                else:
                    # not tested
                    if self.input_mode == 'full-go':
                        inputs[:,:,:-1] = (self.perturbed_cartesian - self.unperturbed_cartesian[0,0,:])
                        if norm_inputs:
                            inputs[:,:,:-1] = 1/(self.dataset_parameters['rings_size'][0]) * inputs[:,:,:-1] # should also be divided by *self.pert_params['amplitude_factor']
                    else:
                        inputs = (self.perturbed_cartesian - self.unperturbed_cartesian[0,0,:]) 
                        if norm_inputs:
                            inputs = 1/(self.dataset_parameters['rings_size'][0]) * inputs # should also be devided by *self.pert_params['amplitude_factor']

        # adding go cue (if necessary)
        if self.input_type =='one-hot' or (self.input_type == 'autoenc' and self.input_mode == 'last'):
            inputs[self.hold_cue_time:, :, -1] = 1
        elif self.input_type == 'autoenc' and self.input_mode == 'full-go':
            # adding the cue as nogo=1, go=0
            inputs[:self.hold_cue_time, :, -1] = 1
        inputs = inputs.type(self.dtype).to(self.device)

        return inputs.detach().clone(), outputs.detach().clone()
    
    def compute_signed_angular_error(self, outputs, targets, t_start=None, t_stop=None, mean=True):
        """
        Mean signed angular error between output and target trajectories,
        relative to the home position.
        Positive = output is CCW w.r.t. target; Negative = CW.
        
        outputs, targets : [T, N, 2]  (Cartesian x, y)
        t_start          : first timestep to include (e.g. hold_cue_time, to skip
                        the preparatory period where both sit at home)
        Returns a scalar (mean over time and all trials).
        """
        home = self.home_cart_state  # shape [2]
        if t_start is not None:
            outputs = outputs[t_start:]
            targets = targets[t_start:]
        if t_stop is not None:
            outputs = outputs[:t_stop]
            targets = targets[:t_stop]

        out_vec = outputs - home   # [T, N, 2]
        tgt_vec = targets  - home  # [T, N, 2]

        # 2D cross product (z-component) and dot product
        cross = tgt_vec[..., 0] * out_vec[..., 1] - tgt_vec[..., 1] * out_vec[..., 0]
        dot   = (tgt_vec * out_vec).sum(-1)

        signed_angle = torch.atan2(cross, dot)  # [T, N], radians
        
        return signed_angle.mean().item() if mean else signed_angle
    
    def plot_dataset(self, pert_params=None, figsize: tuple = (15, 8), skip_steps=3, selected_task=1, text=None,
                     lims1: list = None, lims2: list = None, msize=80, return_figure=False, legend=True, add_arms=True, grid=True):
        if pert_params:
            angle, ampl, transl = [pert_params['angle'], pert_params['amplitude_factor'], pert_params['translation']]
            self.apply_perturbation(pert_params)
            perturb_data = self.generate_perturbed_dataset()[1].numpy()
        else:
            perturb_data = None
        target = self.dataset_parameters['target']

        unperturbed = self.unperturbed_cartesian.detach().clone().numpy()

        def plot_arm(hand_coordinates, ax, color='k', label='arm', size_factor=1):
            from matplotlib.markers import MarkerStyle


            l1, l2 = self.joints_length
            angles = angles_from_cartesian_(hand_coordinates[0], hand_coordinates[1], l1, l2)

            x_l1_start = self.shoulder_position[0]
            y_l1_start = self.shoulder_position[1]

            x_l1_end = x_l1_start + l1 * np.cos(angles[0])
            y_l1_end = y_l1_start + l1 * np.sin(angles[0])

            x_l2_start = x_l1_end
            y_l2_start = y_l1_end

            x_l2_end = hand_coordinates[0]
            y_l2_end = hand_coordinates[1]

            arm_tract = np.array([[x_l1_start, y_l1_start], [x_l1_end, y_l1_end]])
            forearm_tract = np.array([[x_l2_start, y_l2_start], [x_l2_end, y_l2_end]])

            # if ax is None:
            #     fig, ax = plt.subplots()

            line_segments = LineCollection([arm_tract, forearm_tract], linewidths=10*size_factor, colors=color,
                                           linestyle='solid', label=label)
            ax.add_collection(line_segments)
            ax.scatter(arm_tract[:, 0], arm_tract[:, 1], s=200*size_factor, color=color, marker=MarkerStyle('o', fillstyle='full'), facecolors=color)
            ax.scatter(forearm_tract[1, 0], forearm_tract[1, 1], s=400*size_factor, color=color, marker=MarkerStyle('o', fillstyle='full'), facecolors=color, label='hand')

            # if ax is None:
            #     plt.show()
            return arm_tract, forearm_tract

        def single_plot(sequence, ax, x_lim=(-0.1, 0.25), y_lim=(-0.01, 0.4), title=None, selected_task=6, skip_steps=1,
                        perturb=None, msize=80, arm_factor=1, legend=True):

            if perturb is not None:
                unperturbed_seq = sequence.copy()
                sequence = perturb.copy()

            home_cart_state = sequence[0, 0]

            c = cm.rainbow(np.linspace(0, 1, sequence.shape[1]))
            # colors=[[c[i] for j in range(sequence.shape[0])] for i in range(sequence.shape[1])]

            ax.set_title(title, fontsize=16)

            for task in range(1,sequence.shape[1]):

                if perturb is not None:
                    x = unperturbed_seq[:, task, 0]
                    y = unperturbed_seq[:, task, 1]

                    ax.plot(x,y, color='k', alpha=0.09, linewidth=msize//100)
                    if task == selected_task:
                        # ax.scatter(x, y, color=c[task], alpha=0.4, edgecolor='k', s=msize,
                        #            linewidths=0.4)
                        ax.scatter(x[-1], y[-1], color='grey', label='initial targets', marker='X',
                                   edgecolor='k', s=1.2*msize, linewidths=0.5)
                    else:
                        # ax.scatter(x, y, color=c[task], alpha=0.4, edgecolor='grey', s=msize, linewidths=0.4)
                        ax.scatter(x[-1], y[-1], color=c[task], marker='X', edgecolor='k', s=1.2*msize, linewidths=0.5)

                x = sequence[::skip_steps, task, 0]
                y = sequence[::skip_steps, task, 1]

                if self.dataset_parameters['target'] == 'fixed':
                    ax.scatter(x, y, color=c[task], label='task-%d' % task, alpha=1, s=msize)
                else:
                    if task == selected_task:
                        alpha = 0.1
                        lw=0.4
                        label= 'adapted targets'
                        
                    else:
                        alpha = 0.2
                        lw=0.3
                        label = None
                    ax.plot(x,y, color=c[task], alpha=alpha, linewidth=msize//40)

                    ax.scatter(x[-1], y[-1], color=c[task], marker='o',label=label, edgecolor='k', s=1.5*msize, linewidths=lw*2, zorder=4)
                    
                    if task == selected_task:
                        ax.scatter(x, y, color=c[task], alpha=0.3, edgecolor='k', s=msize*0.25, linewidths=lw, zorder=2.5)

            ax.set_xlim(x_lim)
            ax.set_ylim(y_lim)

            if add_arms:
                arm_tract, forearm_tract = plot_arm(home_cart_state, ax, color='grey', label='initial-position',
                                                    size_factor=arm_factor)
                arm_tract_1, forearm_tract_1 = plot_arm(sequence[-1, selected_task], ax, label='final-position',
                                                        size_factor=arm_factor)

            if self.dataset_parameters['target'] == 'fixed':
                ax.plot([forearm_tract[1, 0], forearm_tract_1[1, 0]], [forearm_tract[1, 1], forearm_tract_1[1, 1]],
                        ls='-.', color='grey', lw=4, alpha=0.4, label='movement')

            #ax.scatter(home_cart_state[0], home_cart_state[1], color='k', label="home-state", s=msize)
            ax.tick_params(axis='both', labelsize=14)
            if legend:
                ax.legend(fontsize = 14)

        title = ""
        if pert_params:
            if float(angle) != 0.:
                title += "%d deg Rotation, " % angle
            if float(ampl) != 1.:
                title += "x%.2f amplification, " % ampl
            if transl != [0., 0.]:
                title += "(%.3f,%.3f) Translation, " % (transl[0], transl[1])
        if target == 'fixed':
            title += target
        else:
            title += 'moving target (%s)' % self.dataset_parameters['interpolation']

        fig, ax = plt.subplots(1, 2, figsize=figsize, num=1, gridspec_kw={'width_ratios': (1.2,1.2)})
        fig.suptitle(text, fontsize=26)

        if lims1 is None:
            lims1 = [(-0.2, 0.2), (0.17, 0.5)]

        single_plot(unperturbed, ax[0], *lims1, title, selected_task, int(skip_steps * 1.5), perturb_data, msize, 1.2, legend)
        ax[0].grid(grid)

        title = "Zoomed trajectories"
        if lims2 is None:
            lims2 = [(-0.2, 0.2), (0.17, 0.5)]
        single_plot(unperturbed, ax[1], *lims2, title, selected_task, skip_steps, perturb_data, msize*1.5, 1.4, legend)
        ax[1].grid(grid)
        plt.show()

        fig, ax = plt.subplots(1, 2, figsize=(14, 5))
        ax[0].set_title('X axis evolution', fontsize=20)
        ax[1].set_title('Y axis evolution', fontsize=20)

        ax[0].set_xlabel('time', fontsize=20)
        ax[1].set_xlabel('time', fontsize=20)
        c = cm.rainbow(np.linspace(0, 1, self.num_reach_combinations))
        
        
        ls = '--' if perturb_data is not None else '-'
        for i in range(self.num_reach_combinations): 
            ax[0].plot(unperturbed[:, i, 0], c = c[i], linestyle=ls, label = f'task {i}')
            ax[1].plot(unperturbed[:, i, 1], c = c[i], linestyle=ls, label = f'task {i}')
            if perturb_data is not None:
                ax[0].plot(perturb_data[:, i, 0], c = c[i], label = f'perturbed task {i}' if i ==5 else None)
                ax[1].plot(perturb_data[:, i, 1], c = c[i])
        ax[1].legend(loc='center left')

        
        plt.show()

        if return_figure:
            return fig

    def plot_dataset2(self, datatype='cart',dataset=None, pert=False, tasks=[1,2,5], figsize: tuple = None):
        
        if dataset is not None:
            inputs, targets = dataset
        else:
            if not pert:
                inputs, targets = self.generate_unperturbed_dataset(datatype)
            else:
                inputs, targets = self.generate_perturbed_dataset(datatype)
        task_idxs = np.arange(inputs.shape[1])
        tasks = [task_idxs[t] for t in tasks]
        n_tasks = len(tasks)

        if figsize is None:
            figsize = (12, 5*n_tasks)
        plt.figure(figsize=figsize)
        for i,j in enumerate(tasks):
            plt.subplot(n_tasks,2,i*2+1)
            plt.title('task %d inputs'%j)
            plt.imshow(inputs[:,j,:].numpy(),aspect = 'auto',interpolation = 'none')
            plt.ylabel("TIME")
            plt.xlabel("Input features")
            plt.colorbar()
            
            plt.subplot(n_tasks,2,i*2+2)
            plt.title('task %d target'%j)
            # plotting shadows of all targets
            for k in range(targets.shape[1]):
                alpha = 1 if k==j else 0.2
                plt.plot(targets[:,k,0], targets[:,k,1], lw = 3, alpha = alpha)

        plt.show()


# Reaching task functions (numpy version)
def cartesian_coordinates_(theta1, theta2, l1=0.145, l2=0.284, shoulder_position=np.array([0., 0.])):
    x = l1 * np.cos(theta1) + l2 * np.cos(theta1 + theta2) + shoulder_position[0]
    y = l1 * np.sin(theta1) + l2 * np.sin(theta1 + theta2) + shoulder_position[1]
    return x, y


def angles_from_cartesian_(x, y, l1=0.145, l2=0.284, shoulder_position=np.array([0., 0.])):
    x = x - shoulder_position[0]
    y = y - shoulder_position[1]
    B2 = x ** 2 + y ** 2
    B = np.sqrt(B2)
    ## using the selected positions, arccos gives some RuntimeWarnings (divisione by zero), but converges.
    ## the next two lines just avoid the warning messages to be shown
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        theta1 = np.arccos(x / B) - np.arccos((l1 ** 2 - l2 ** 2 + B2) / (2 * B * l1))
        theta2 = np.pi - np.arccos((l1 ** 2 + l2 ** 2 - B2) / (2 * l1 * l2))
    return theta1, theta2


def apply_perturbation_(initial_array: np.ndarray, home_state: np.ndarray, angle: float = None,
                        amplitude_factor: float = 1, translation: np.ndarray = np.array([0., 0.])) -> np.ndarray:
    ## in order to work, x,y dimension must be the last one!
    angle = np.deg2rad(angle)
    rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    array = initial_array - home_state
    rotated_array = amplitude_factor * np.transpose(np.matmul(rotation_matrix, np.transpose(array, (0, 2, 1))),
                                                    (0, 2, 1)) + translation
    rotated_array = rotated_array + home_state

    return rotated_array
