#!/usr/bin/python3
"""
Utilities for model
"""

# Version Identification and Dependency Specification
__author__="Harsh Bhate"
__copyright__ = "Copyright 2024, Harsh Bhate"

__license__ = "CC 1.0"
__version__ = "0.1"
__email__ = "bhateharsh@gmail.com"
__status__ = "Prototype"

# Libraries
import torch
from modules.utilities.dataset_utility import *

def _get_average(data):
    """
    Function to return list of normalized map
    """
    total = float(sum(data.values()))
    weight_list = []
    for k,v in data.items():
        weight_list.append(v/total)
    return weight_list

def get_weight_tensor(meta_data):
    """
    Function to compute the weight tensor from meta-data
    """
    class_map, anomaly_map = get_statistics(meta_data=meta_data)
    class_weight_list = _get_average(class_map)
    anomaly_weight_list = _get_average(anomaly_map)
    class_weights = torch.tensor(class_weight_list, dtype=torch.float32)
    anomaly_weights = torch.tensor(anomaly_weight_list, dtype=torch.float32)
    return class_weights, anomaly_weights
