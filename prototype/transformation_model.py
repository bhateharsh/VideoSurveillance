#!/usr/bin/python3
"""
Test of pre-trained video classification transformer from PyTorch and basic 
demo. 

"""

# Version Identification and Dependency Specification
__author__="Harsh Bhate"
__copyright__ = "Copyright 2024, Harsh Bhate"

__license__ = "CC 1.0"
__version__ = "0.1"
__email__ = "bhateharsh@gmail.com"
__status__ = "Prototype"

# Libraries
from torchvision.io.video import read_video
from torchvision.models.video import r3d_18, R3D_18_Weights

# Constants
DEMO_SET = {
    "shoplifting":"demo/shoplifting.mp4", 
    "road_accident":"demo/road_accidents.mp4", 
    "vandalism":"dataset/Vandalism/Vandalism034_x264.mp4/Vandalism034_x264_1.mp4"
}
DEMO_OPTION = "vandalism" 

# Get data
vid, _, _ = read_video(filename=DEMO_SET[DEMO_OPTION], 
                       output_format="TCHW")
print("Video loaded.")

# Initialize Model with best available weights
weights = R3D_18_Weights.DEFAULT
model = r3d_18(weights=weights)
model.eval()

# Initialize inference transforms
preprocess = weights.transforms()
# Apply inference pre-processing transforms
batch = preprocess(vid).unsqueeze(0)

# Info
print ("Video Shape: ", vid.shape)
print ("Transformation: ", preprocess)
print ("Batch: ", batch.shape)
# # Use models to print predicted category
# prediction = model(batch).squeeze(0).softmax(0)
# label = prediction.argmax().item()
# score = prediction[label].item()
# category_name = weights.meta["categories"][label]
# print(f"Category: {DEMO_OPTION}, Predicted Category: {category_name}: {100 * score}%")
