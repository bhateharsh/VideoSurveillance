#!/usr/bin/python3
"""
Test dataset parsing

TODO: Use google test or other framework and create unit tests
"""

# Version Identification and Dependency Specification
__author__="Harsh Bhate"
__copyright__ = "Copyright 2024, Harsh Bhate"

__license__ = "CC 1.0"
__version__ = "0.1"
__email__ = "bhateharsh@gmail.com"
__status__ = "Prototype"

# Libraries
from torch.utils.data import Dataset, DataLoader
from torchvision.models.video import r3d_18, R3D_18_Weights

from modules.utilities.dataset_utility import *
from modules.training.VideoDataset import VideoDataset

def test_dataset(split=0.2, use_r3d_transform=True):
    # Get Training set
    meta_data = get_filenames()
    test_meta_data, train_meta_data = get_split(meta_data=meta_data, 
                                                split=split)
    # Get transforms
    if use_r3d_transform:
        weights = R3D_18_Weights.DEFAULT
        model = r3d_18(weights=weights)
        model.eval()
        # Initialize inference transforms
        preprocess = weights.transforms()
    else:
        preprocess = None
    # Initialize dataset
    v_dataset = VideoDataset(meta_data=train_meta_data, 
                             transform=preprocess)
    dataloader = DataLoader(v_dataset, 
                            batch_size=32, 
                            shuffle=True)
    # Accessing a sample
    video, class_label, anomaly_label = v_dataset[0]
    print("Video shape: ", video.shape)
    print("Class Label: ", class_label)
    print("Anomaly Label: ", anomaly_label)

if __name__=="__main__":
    test_dataset(split=0.2, use_r3d_transform=True)