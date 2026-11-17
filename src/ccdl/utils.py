# -*- coding: utf-8 -*-
"""
Created on 07/02/2023
@author: Leonardo Agueci - leonardo.agueci@ens.psl.eu

General-purpose helpers shared across the library: pickle I/O, coloured
printing, nested-dict manipulation, parameter-grid expansion, a rolling
:class:`Buffer`, and small training/analysis utilities.
"""

import os, sys
import shutil
import pickle
import io
import contextlib

from datetime import datetime
import itertools
import torch
import copy
from typing import Any, Tuple, List

import numpy as np
from sklearn.model_selection import ParameterGrid


@contextlib.contextmanager
def _torch_storages_on_cpu():
    """Map torch tensor storages to CPU while unpickling: legacy experiments may
    embed tensors saved on a CUDA device, and the figure pipeline runs on CPU.
    Value-preserving (same tensor data, only the device changes)."""
    import torch.storage as _storage
    original = getattr(_storage, "_load_from_bytes", None)
    if original is None:
        yield
        return
    _storage._load_from_bytes = lambda b: torch.load(io.BytesIO(b), map_location="cpu")
    try:
        yield
    finally:
        _storage._load_from_bytes = original


class _CompatUnpickler(pickle.Unpickler):
    """Resolve classes pickled under the old flat module layout (e.g. ``utils``)
    to their current ``ccdl.*`` location, so legacy pickles load unchanged."""

    def find_class(self, module, name):
        try:
            return super().find_class(module, name)
        except (ModuleNotFoundError, AttributeError):
            parts = module.split(".")
            candidates = [
                # old top-level package renamed to `ccdl` (e.g. utils.utils -> ccdl.utils)
                "ccdl" + ("." + ".".join(parts[1:]) if len(parts) > 1 else ""),
                # old flat module now living under `ccdl` (e.g. utils -> ccdl.utils)
                "ccdl." + module,
            ]
            for candidate in candidates:
                try:
                    return super().find_class(candidate, name)
                except (ModuleNotFoundError, AttributeError):
                    continue
            raise


def load_var(filename):
    """Load and return a pickled object from ``filename``."""
    with open(filename, 'rb') as handle, _torch_storages_on_cpu():
        data = _CompatUnpickler(handle).load()
    return data

def pr_print(formats=None, text='', end=None):
    class Color:
        PURPLE = '\033[95m'
        CYAN = '\033[96m'
        DARKCYAN = '\033[36m'
        BLUE = '\033[94m'
        GREEN = '\033[92m'
        YELLOW = '\033[93m'
        RED = '\033[91m'
        BOLD = '\033[1m'
        UNDERLINE = '\033[4m'
        END = '\033[0m'

    if formats is None:
        print('The function must be used this way:\npr_print(list(formats),text\n\
            Available formats:\n-PURPLE\n-CYAN\n-DARKCYAN\n-BLUE\n-GREEN\n-YELLOW\n\
-RED\n-BOLD\n-UNDERLINE\n')
    else:
        formats = list(formats)
        formats = [getattr(Color, f.upper()) for f in formats]
        print(*formats, text, Color.END, end=end)


def printsave(output, file=None, flag=True, formatting=None, end=None):
    """
    Advanced print function. It can be used for normal printing, printing with formatting and
    saving what ou print in a file. If output is a tuple or a dict, 
    Args:
    - output: what you want to print. If iteerable, variables will be displayed line-by-line
    - file: path of the file where you want to save the output. 
    - flag: controls the file saving. USeful when you use this function with a decorator for the
                path
    - formatting: iterable, containing formatting options (see pr_print)
    - end: str, same as print() end kwarg
    """
    
    if type(output) == tuple:
        for k in output:
            printsave(k, file, flag)

    else:

        if type(output) == dict:
            print('{')
            for k in output.keys():
                print("%s: %s," % (k, output[k]))
            print('}\n', end=end)

        else:

            if formatting is None:
                print(output, end=end)
            else:
                pr_print(formatting, output, end=end)

        if flag and file is not None:
            original_stdout = sys.stdout  # Save a reference to the original standard output
            with open(file, 'a') as f:
                sys.stdout = f  # Change the standard output to the file we created.
                if type(output) == dict:
                    print('{')
                    for k in output.keys():
                        print(k, output[k])
                    print('}\n')
                else:
                    print(output, end=end)
                sys.stdout = original_stdout  # Reset the standard output to its original value


def save_var(var, filename):
    """Pickle ``var`` to ``filename``."""
    with open(filename, 'wb') as handle:
        pickle.dump(var, handle)


class Buffer(object):
    """
        Create a Buffer of n elements for saving data.
        """
    def __init__(self, n):
        self.buf = [None] * n
        self._valid = 0

    def add(self, obj):
        n = len(self.buf)
        self.buf.append(obj)
        self.buf = self.buf[-n:]
        if self._valid < n:
            self._valid += 1

    def __len__(self):
        return self._valid
    # could include other methods for accessing or modifying the contents


def generate_folder2(folder, name, force=False) -> Tuple[str, str]:
    new_folder = os.path.join(folder, name)

    if os.path.exists(new_folder):
        if force:
            shutil.rmtree(new_folder)
        else:
            choice = input(f"{new_folder} already exists. Do you want to erase the folder? if yes, data will be lost. (y/n)\n")
            if choice.lower() in ('y', 'yes'):
                shutil.rmtree(new_folder)
            elif choice.lower() in ('n', 'no'):
                choice2 = input("Do you want to use another name? (y/n)\n")
                if choice2.lower() in ('y', 'yes'):
                    name = input("Type the name you want to use: ")
                    return generate_folder2(folder, name)
                elif choice2.lower() in ('n', 'no'):
                    choice3 = input("Do you want to continue anyway (data could be overwritten, be careful)? (y/n)\n")
                    assert choice3.lower() in ('y', 'yes'), "exiting..."
                    return new_folder, name
                else:
                    print(f"invalid answer ({choice2.lower()}), exiting")
                    exit()
            else:
                print(f"invalid answer ({choice.lower()}), exiting")
                exit()

    os.makedirs(new_folder)
    return new_folder, name


def generate_folder(folder, name, force=False) -> str:
    new_folder = os.path.join(folder, name)
    if force:
        os.makedirs(new_folder)
        return new_folder
    try:
        os.makedirs(new_folder)
        return new_folder
    except:
        choice = input(f"{name} already exists. Do you want to erase the folder? if yes, data will be lost. (y/n)\n")
        if choice.lower() in ('y', 'yes'):
            shutil.rmtree(new_folder)
            os.makedirs(new_folder)
            return new_folder
        elif choice.lower() in ('n', 'no'):
            choice2 = input("Do you want to use another name? (y/n)\n")
            if choice2.lower() in ('y', 'yes'):
                name = input("Type the name you want to use: ")
                return generate_folder(folder, name)
            elif choice2.lower() in ('n', 'no'):
                choice3 = input("Do you want to continue anyway (data could be overwritten, be careful)? (y/n)\n")
                assert choice3.lower() in ('y', 'yes'), "exiting..."
                return new_folder

            else:
                print("invalid answer, exiting")
                exit()
        else:
            print("invalid answer, exiting")
            exit()


# training useful methods

def threshold_stopping(losses, patience=20, eps=10e-6):
    if len(losses) > patience:
        best = np.min(losses[-patience:])
        best_pos = np.argmin(losses)
        if len(losses) - best_pos >= patience and best <= eps:
            return True, best_pos
        else:
            return False, None
    else:
        return False, None


def plateau_stopping(losses, patience=100, eps=1e-7):
    if len(losses) > patience:
        best_pos = np.argmin(losses)
        last = losses[-1]
        last_pos = len(losses) - 1
        # exiting if there's no improvement
        if last_pos - best_pos == patience:
            return True, best_pos

        # exiting if the improvement is very low
        elif losses[-patience]-last < eps:
            """I should use abs(losses[-patience]-last) (high negative case will escape), but the negative case 
                            is already included in the previous condition"""
            return True, last_pos
        else:
            return False, None
    else:
        return False, None


class Experiment:
    r"""Basic class for saving experiments data as objects"""
    def __init__(self, folder, experiment_name):
        self.folder, self.name, self.path = self.generate_path(folder, experiment_name)
        self.variables = ['folder', 'name', 'path']
        self.timeline = {datetime.now(): 'Experiment initialization'}
        self.memorize()

    def update_variables_list(self, name):
        self.variables.append(name)

    @staticmethod
    def generate_path(folder, name):
        experiment_folder = generate_folder(folder, name)
        # generate_folder checks folder existence and allows changing name. Then, we need to extract the selected name
        name = os.path.basename(experiment_folder)
        data_path = os.path.join(experiment_folder, "experiment_data.pkl")
        return experiment_folder, name, data_path

    def add_event(self, text):
        self.timeline[datetime.now()] = text

    def memorize(self):
        # Using the module-level save_var, different from self.save_var
        save_var(self, self.path)

    def save_var(self, name, var):
        if hasattr(self, name):
            self.add_event("updating "+name)
        else:
            self.add_event("setting "+name)
            self.update_variables_list(name)
        setattr(self, name, var)
        self.memorize()

    def save_data(self, data):
        if type(data) == tuple:
            assert len(data) == 2 and type(data[0]) == str, "tuple must be provided as (str, variable)"
            self.save_var(*data)
        elif type(data) == dict:
            for name, var in data.items():
                self.save_var(name, var)
        else:
            raise IOError("The input provided doesn't have the correct form")


def L1_loss(net, reduction='mean'):
    names, modules = list(zip(*list(net.named_modules())))
    parameters_list = torch.cat([module.weight.view(-1) for module in modules[1:]])
    if reduction == 'mean':
        return torch.norm(parameters_list, 1) / parameters_list.shape[0]
    else:
        return torch.norm(parameters_list, 1)


def L2_loss(net, reduction='mean'):
    names, modules = list(zip(*list(net.named_modules())))
    parameters_list = torch.cat([module.weight.view(-1) for module in modules[1:]])
    if reduction == 'mean':
        return torch.norm(parameters_list, 2) / parameters_list.shape[0]
    else:
        return torch.norm(parameters_list, 2)


def local_rule_update(rec_activity: torch.Tensor, fb_signal: torch.Tensor,fb_weights: torch.Tensor, rec_weights: torch.Tensor = None, lr=1e-4, method='mean'):
    r"""Computes local delta rule for weight updates on RNN recurrent connections.
    
    Implements a local learning rule based on feedback signal (from feedback weights) and
    recorded hidden state activity. Updates weights in-place without returning modified weights.
    
    Reference: Aditya Gilra, Wulfram Gerstner (2017) "Predicting non-linear dynamics by 
    stable local learning in a recurrent spiking neural network" https://doi.org/10.7554/eLife.28295
    
    Args:
        rec_activity (torch.Tensor): Recorded RNN hidden state activity with shape 
            (timesteps, batch_size, hidden_size).
        fb_signal (torch.Tensor): Feedback signal from cerebellum with shape 
            (timesteps, batch_size, cereb_output_size).
        fb_weights (torch.Tensor): Feedback weights from cerebellum to RNN with shape 
            (hidden_size, cereb_output_size).
        rec_weights (torch.Tensor, optional): RNN recurrent weights with shape 
            (hidden_size, hidden_size). Defaults to None.
        lr (float): Learning rate scaling the weight updates. Defaults to 1e-4.
        method (str): Aggregation method for batch dimension. Options: 'mean', 'sum'. 
            Defaults to 'mean'.
    
    Returns:
        torch.Tensor: Weight update tensor (delta_w) to be applied to recurrent weights.
    """
    assert method in ['mean', 'sum'], "method must be 'mean' or 'sum'"
    assert rec_activity.dim() == fb_signal.dim(), "rec_activity and fb_signal must have same number of dimensions"
    dims = rec_activity.dim()
    assert 2 <= dims <= 3, "hidden activity must be 2D or 3D tensor"
    if dims == 2:
        rec_timesteps, hidden_size = rec_activity.shape
        fb_timesteps, fb_out_size = fb_signal.shape
    else:
        rec_timesteps, rec_batch_size, hidden_size = rec_activity.shape
        fb_timesteps, fb_batch_size, fb_out_size = fb_signal.shape
        assert rec_batch_size == fb_batch_size, "activity tensors must have same batch size"

    assert rec_timesteps == fb_timesteps, "activity tensors must have same timesteps"
    assert fb_weights.shape[0] == hidden_size, "feedback weight height must match RNN size activity size"
    assert fb_weights.shape[1] == fb_out_size, "feedback weight width must match feedback signal size"
    if rec_weights is not None:
        assert hidden_size == rec_weights.shape[0] == rec_weights.shape[1], "recurrent weights must be square, and match hidden size"
    
    with torch.no_grad():
        function = getattr(torch, method)
        error_signal = torch.matmul(fb_signal, fb_weights.T)
        delta_w = lr * error_signal.unsqueeze(dims) @ rec_activity.unsqueeze(dims - 1)

        # Apply aggregation function
        delta_w = function(delta_w, dim=0) # Aggregate across time
        if dims == 3:
            delta_w = function(delta_w, dim=0) # Aggregate across batch
        return delta_w



# support training script functions

def get_iterable_params(params):
    iter_params = {}
    # extracting params to be iterated and counting occurrencies
    for key, value in params.items():
        if type(value) in [list, tuple]:
            iter_params[key] = value

    # generating the permutation params list
    iter_params = list(ParameterGrid(iter_params))
    return iter_params


def generate_name(params: Any, iters: tuple):
    iteration, total_iterations = iters
    text = []
    for k, v in params.items():
        if type(v) == int or v is None:
            text.append('.'.join([k, str(v)]))
        elif type(v) == str:
            text.append('.'.join([k, v]))
        else:
            if v != 0.:
                text.append('.'.join([k, "{:.1e}".format(v)]))
            else:
                text.append('.'.join([k, str(0)]))
    if total_iterations > 1:
        text.append(str(iteration))
    text = ".".join(text)
    return text


def update_params_dict(default_params: Any, params: dict) -> dict:
    # update default_params with provided params
    assert type(default_params) in [type(None), dict]
    if default_params:
        for k, v in params.items():
            if v:
                default_params[k] = v
    return default_params


def generate_params_combinations(starting_dictionary: dict, iter_params: Any) -> List[dict]:
    """
    Generate distinct updated dictionaries for all combinations of values from grid_dictionary.

    Args:
        starting_dictionary (dict): The dictionary to be updated.
        grid_dictionary (dict): The dictionary containing keys and lists of values.

    Returns:
        list: A list of updated dictionaries with all combinations.
    """
    
    if isinstance(iter_params, dict):
        combinations = list(itertools.product(*iter_params.values()))
        combo_keys = [iter_params.keys() for i in range(len(combinations))]
    elif isinstance(iter_params, list):
        combinations = [list(iter_param_i.values()) for iter_param_i in iter_params]
        combo_keys = [iter_param_i.keys() for iter_param_i in iter_params]
    else:
        raise IOError('Please provide either grid_dictionary or train_list')
    updated_dicts = []

    for combo, keys in zip(combinations, combo_keys):
        updated_dict = copy.deepcopy(starting_dictionary)
        for i, key in enumerate(keys):
            keys = key.split('.')
            current_dict = updated_dict
            for subkey in keys[:-1]:
                if subkey not in current_dict:
                    current_dict[subkey] = {}
                current_dict = current_dict[subkey]
            current_dict[keys[-1]] = combo[i]

        updated_dicts.append(updated_dict)

    return updated_dicts


def generate_names_list(params) -> List[str]:
    if isinstance(params, dict):
        combinations = list(itertools.product(*params.values()))
        combo_keys = [params.keys() for i in range(len(combinations))]
    elif isinstance(params, list):
        combinations = [list(iter_param_i.values()) for iter_param_i in params]
        combo_keys = [iter_param_i.keys() for iter_param_i in params]
    else:
        raise IOError('Please provide a list of parameters (dict.s), or a single parameters dict with'\
                                        'grid=True, in the case of grid search mode')
    dicts_names = []

    for combo, keys in zip(combinations, combo_keys):
        name = []
        for i, key in enumerate(keys):
            value = combo[i]
            keys_list = key.split('.')
            if isinstance(combo[i], list):
                value = '.'.join([str(val) for val in value])
            # Correctly formatting the value
            elif isinstance(value, float):
                value = "{:.1e}".format(value) if value!=0. else str(0)
            elif isinstance(value, bool):
                value = str(int(value))
            else:
                value = str(value)
            
            name.append( '.'.join([keys_list[-1], value]))
        dicts_names.append('_'.join(name))
    return dicts_names


def selective_update(original, new_update):
    assert isinstance(original, dict)
    assert isinstance(new_update, dict)
    updated_dict = copy.deepcopy(original)  # Create a deep copy to avoid modifying the original
    for key, value in new_update.items():
        if isinstance(value, dict):  # If it's a nested dictionary, recurse
            updated_dict[key] = selective_update(updated_dict.get(key, {}), value)
        else:
            # Update only if the original value is True
            if key in updated_dict and updated_dict[key]:  
                updated_dict[key] = value  # True remains True if new is True, otherwise becomes False
    return updated_dict


class Experiment2:
    r"""Basic class for saving experiments data as objects"""
    def __init__(self, folder: str, experiment_name: str, load: bool = False, 
                 force: bool = False) -> None:
        """
        Initialize the Experiment object.

        Args:
            folder:           The folder path where the experiment data will be 
                                saved.
            experiment_name:  The name of the experiment.
            load:             Boolean used to control initialization behavior 
                                (create new experiment vs. load an existent one)
        """
        self.folder, self.name, self.path = self._generate_folder(folder, 
                                                experiment_name, load, force)

        # variable enabling/disabling memorization
        self._locked_mem = False

        if not load:
            self.timeline = {datetime.now(): 'Experiment initialization'}
            self.variables = ['folder', 'name', 'path']
            self.parameters = {}
            self._update_variables_list('parameters')
            self.performance = {}
            self._update_variables_list('performance')
            self._memorize()
        else:
            self._locked_mem = True

    def _generate_folder(self, folder: str, name: str, load: bool, force: bool) -> Tuple[str, str, str]:
        """
        Generate the folder path and data path for saving the experiment, and creates the folder. If load is True, the
        method will try to load data from the specified path.

        Args:
            folder: The folder path where the experiment data will be saved.
            name: The name of the experiment.
            load: boolean used to control initialization behavior (create new experiment vs. load an existent one)

        Returns:
            The folder path, experiment name, and data path.
        """
        # generate_folder checks folder existence and allows changing name
        
        

        if not load:
            experiment_folder, name = generate_folder2(folder, name, force)
            data_path = os.path.join(experiment_folder, "experiment.pkl")
        else:
            experiment_folder = os.path.join(folder, name)
            data_path = os.path.join(experiment_folder, "experiment.pkl")
            self._load_experiment(data_path)

        

        return experiment_folder, name, data_path
       

    def _update_variables_list(self, name):
        """This method updates the variables saved in the object"""
        self.variables.append(name)

    def _add_event(self, text: str) -> None:
        """
        Add an event to the Experiment object's timeline.

        Args:
            text: The event description.
        """
        self.timeline[datetime.now()] = text

    def _memorize(self) -> None:
        """Save the Experiment object to the predefined location."""
        if not self._locked_mem:
            with open(self.path, 'wb') as handle:
                pickle.dump(self, handle)

    def save_parameter(self, name: str, var: Any) -> None:
        """
        Save a parameter to the parameters' dictionary.

        Args:
            name: The name of the parameter.
            var: The value of the parameter.
        """
        assert not self._locked_mem, "You are trying to save/update the experiment opened in safe. Declare " \
                                     "self.self._locked_mem=False first (not recommended, data could be corrupted)"
        if not isinstance(var, dict):
            var = var.__name__ if callable(var) else var
            self.parameters[name] = var
            self._add_event("Setting parameter: " + name)
        else:
            for key, val in var.items():
                key = '.'.join((name, key))
                self.save_parameter(key, val)

    def save(self, name: str, var: Any, is_parameter: bool = False) -> None:
        """
        Add new data to the Experiment object.

        Args:
            name: The name of the variable.
            var: The value of the variable.
            is_parameter: Whether the variable is a parameter.

        Raises:
            ValueError: If the variable name is invalid or if var is a parameter and neither a value nor a dictionary.
            TypeError: If the data doesn't have the correct form.
        """
        var = copy.deepcopy(var)
        assert not self._locked_mem, "You are trying to save/update the experiment opened in safe. Declare " \
                                     "self.self._locked_mem=False first (not recommended, data could be corrupted)"
        if not isinstance(name, str) or not name.isidentifier():
            raise ValueError("Invalid variable name provided.")

        if hasattr(self, name):
            self._add_event("Updating " + name)
        else:
            self._add_event("Setting " + name)
            self._update_variables_list(name)
        setattr(self, name, var)

        # If it is a parameter, add it to parameters
        if is_parameter:
            self.save_parameter(name, var)

        # saving changes
        self._memorize()

    def _load_experiment(self, path: str) -> None:
        """Load the Experiment object from the saved file."""
        try:
            with open(path, "rb") as file, _torch_storages_on_cpu():
                loaded_experiment = _CompatUnpickler(file).load()
            self.__dict__.update(loaded_experiment.__dict__)
        except FileNotFoundError:
            raise FileNotFoundError("Experiment file does not exist.")
        except Exception as e:
            raise RuntimeError("An error occurred while loading the experiment: " + str(e))

    def update_performance(self, name: str, var: Any) -> None:
        """
        Update the performance dictionary.

        Args:
            name: The name of the performance metric.
            var: The value of the performance metric.
        """
        if isinstance(var, dict):
            self.performance.update(var)
            self._add_event("Setting performance variables: " + ', '.join(var.keys()))
        else:
            self.performance[name] = var
            self._add_event("Setting performance variable: " + name)

        # saving changes
        self._memorize()

def fb_nuc_norm(model):
    return torch.linalg.matrix_norm(model.cortex.rnn.input2h_2.weight, ord='nuc').item()

# New metrics
def compute_take_off_angle(outputs, targets, start, threshold_coefficient=0.2):
    
    # compute radius and threshold distance
    radius = torch.norm(targets[-1,1]-targets[0,1])
    threshold_distance = threshold_coefficient * radius

    # define target directions (shape (tasks x 2)), normalize
    targets = targets[:,1:]
    target_directions = targets[-1]-targets[0]
    target_directions = target_directions / torch.norm(target_directions, dim=1, keepdim=True)
    
    ntasks = targets.shape[1]

    # IMPORTANT: check if you can do this. each Task timeseries will get different size on step (*)
    # take movement part, remove task 0 (stay still)
    outputs = outputs[start:,1:]
    # compute relative trajectories (centering trajectories)
    outputs = outputs[1:]-outputs[0]
    # compute distance for all timesteps
    distance = torch.norm(outputs, dim=2, keepdim=True)

    distance[:,2] += 0.01

    # (*) find all timesteps where distance > threshold distance, select them
    timesteps = (distance > threshold_distance)[:,:,0]
    first_valid_indices = torch.argmax(timesteps.int(), dim=0)  # First index where condition is True
    directions = torch.stack([outputs[first_valid_indices[t], t] for t in range(ntasks)])
    
    # Normalize the directions
    directions = directions / torch.norm(directions, dim=1, keepdim=True)
    
    # compute angle
    dot_product = torch.sum(directions * target_directions, dim=1).clamp(-1.0, 1.0)
    angles = torch.acos(dot_product)
    
    return angles

def compute_endpoint_distance(outputs, targets, endpoints=1, norm=False, std=None):
    distance = torch.mean((outputs[-endpoints:] - targets[-endpoints:]) **2,axis=(0,2))
    if norm:
        assert std is not None, "In order to normalize distance std must be provided"
        distance = distance/std
    return distance


if __name__ == '__main__':
    print('Testing update_nested_dict')
    starting_dict = {'a':{'c':0,'d':5},'h': 9}
    grid_dict = {
        'a.c': [1, 2],
        'a.b': [3, 4]
    }

    updated_dicts = generate_params_combinations(starting_dict, grid_dict)
    names = generate_names_list(grid_dict)

    print(f"Starting dict: {starting_dict}")
    for i, updated_dict in enumerate(updated_dicts, 1):
        name = names[i-1]
        print(i,', name: ', name)
        print(f"Combination {i}: {updated_dict}")