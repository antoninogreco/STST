<p align="center">
<img src="logo.png" width="70%">
</p>


# Spatiotemporal Style Transfer (STST)
The STST algorithm is a dynamic visual stimulus generation framework designed for vision research. 
It is based on a two-stream deep neural network model that factorizes spatial and temporal features to generate dynamic visual stimuli whose model layer activations are matched to those of input videos.
This makes it a powerful tool for studying object recognition and other areas of vision science in both biological and artificial systems.

<p align="center">
<img src="fig1.png" width="70%">
</p>



## Features
Independent Spatiotemporal Factorization: Generates video stimuli with isolated spatial and temporal features, allowing for targeted testing of human and artificial vision models.
Versatile Stimulus Generation: Creates stimuli for various visual research applications, particularly useful for studying biases in spatial and temporal encoding.
Optimized for Deep Vision Models: Tested on models like PredNet, STST preserves low-level visual features, making it a valuable tool for machine learning research on spatiotemporal perception.

## Repository Structure
SpaceTimeStyleTransfer.py: Main class implementing the STST algorithm.
configurations/: Example configurations for running STST with various options for spatiotemporal factorization.
data/: Placeholder folder for input videos formatted as 4D numpy arrays.
output/: Default output folder for generated videos.
Getting Started
## Prerequisites
Python: Ensure you have Python 3.6 or higher installed.
TensorFlow: Required for neural network operations.
NumPy: For numerical operations on input video arrays.
Additional Libraries: Check requirements.txt for other dependencies.
## Installation
Clone the repository and install dependencies:

bash
Copy code
git clone https://github.com/username/STST
cd STST
pip install -r requirements.txt
Usage
