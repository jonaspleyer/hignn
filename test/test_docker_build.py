"""
  import all of the packages required by the package for checking the compatibility of docker image
"""

import os
import hignn
import numpy as np
import sys
from mpi4py import MPI
import time
import h5py
import vtk
import torch
import torch_scatter

os.system("clear")

print("Success of loading all required packages")