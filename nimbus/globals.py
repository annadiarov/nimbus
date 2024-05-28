"""
Contains default global variables used throughout the project
# TODO Update global variables when using config files or command line arguments
# When running the project, this parameters can be overwritten by passing them as
# arguments to the script.
"""
import os
import torch

# System
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED = 42
NIMBUS_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(NIMBUS_BASE_DIR, '..', 'data')
CHECKPOINT_PATH = os.path.join(NIMBUS_BASE_DIR, '..', 'checkpoints')

# Logger
LOGGER_LEVEL = 'INFO'

# Data
N_WORKERS = 20 # Number of workers for data loading
