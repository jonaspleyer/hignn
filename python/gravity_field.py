import os
import hignn
import numpy as np
import sys
from mpi4py import MPI
import time
import h5py
import torch
import torch.nn as nn
from torch_scatter import scatter
from HIGNN.model_structure import HIGNN_model

os.system("clear")

# # Create Edge
# # 3body edge: (j, k, i), attr: F_j, from j to k to i
# # 2body_self edge: (j, i), attr: F_i, from i to j to i
def build_neighbor_list(X, three_body_cutoff):
    
    neighbor_list = hignn.NeighborLists()
    neighbor_list.set_three_body_epsilon(three_body_cutoff)
    neighbor_list.update_coord(X)
    neighbor_list.build_three_body_info()
    three_body_edge_info = neighbor_list.get_three_body_edge_info()
    three_body_edge_info = torch.tensor(three_body_edge_info, dtype=torch.long)
    # print(three_body_edge_info)
    two_body_edge_self_info = neighbor_list.get_three_body_edge_self_info()
    two_body_edge_self_info = torch.tensor(two_body_edge_self_info, dtype=torch.long)
    # print(two_body_edge_self_info)
    del neighbor_list
    return three_body_edge_info, two_body_edge_self_info

# calculate velocity without 2body
def velocity_update_without_2body(X, force, device):

    edge_3body, edge_2bodySelf = build_neighbor_list(X, 5.0)
    edge_3body = edge_3body.to(device)
    edge_2bodySelf = edge_2bodySelf.to(device)
    edge_2body = torch.zeros((2, 0), dtype=torch.long).to(device)
    force = torch.tensor(force, dtype = torch.float32).to(device)

    edge_attr_2body = force[edge_2body[0, :], :]
    edge_attr_3body = force[edge_3body[0, :], :]
    edge_attr_2bodySelf = force[edge_2bodySelf[1, :], :]
    Nc = X.shape[0]
    edge_1body = torch.arange(0, Nc, 1).reshape((1, Nc))
    edge_1body = edge_1body.long().to(device)
    edge_attr_1body = force
    X = torch.tensor(X, dtype = torch.float32)
    X = X.to(device)
    with torch.no_grad():
        velocity = original_hignn(X, edge_2body, edge_3body, edge_2bodySelf, edge_1body, edge_attr_2body, edge_attr_3body, edge_attr_2bodySelf, edge_attr_1body).cpu().numpy()
    torch.cuda.empty_cache()
    
    return velocity


def velocity_update(t, position):

    
    # update coordinates
    hignn_model.update_coord(position[:, 0:3])
    
    # update force/potential
    force = np.zeros((position.shape[0], 3), dtype=np.float32)
    force[:, 2] = -1.0
    
    # create velocity array
    velocity = np.zeros((position.shape[0], 3), dtype=np.float32)
    
    hignn_model.dot(velocity, force)
    # print(velocity)
    velocity = velocity + velocity_update_without_2body(position[:, 0:3], force, device)
    # print(velocity)
    return velocity

if __name__ == '__main__':
    # initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    hignn.Init()
    
    # initial position
    nx = 50
    ny = 50
    nz = 50
    dx = 3.0
    x = np.arange(0, nx * dx, dx)
    y = np.arange(0, ny * dx, dx)
    z = np.arange(0, nz * dx, dx)
    xx, yy, zz = np.meshgrid(x, y, z)
    X = np.concatenate(
        (xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)), axis=1)
    X = X.astype(np.float32)
    NN = X.shape[0]

    # load original hignn model for 3-body and 2-body self calculation 
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    original_hignn = torch.load('python/Saved_Model/Unbounded_try1/HIGNN.pkl', weights_only = False).to(device)
    print(original_hignn)

    # initialize hierarchical_hignn_model for 2body interaction
    hignn_model = hignn.HignnModel(X, 50)
    hignn_model.load_two_body_model('nn/two_body_unbounded_updated')
    
    # set parameters for far dot, the following parameters are default values
    hignn_model.set_epsilon(0.1)
    hignn_model.set_max_iter(15)
    hignn_model.set_mat_pool_size_factor(30)
    hignn_model.set_post_check_flag(False)
    hignn_model.set_use_symmetry_flag(True)
    hignn_model.set_max_far_dot_work_node_size(10000)
    hignn_model.set_max_relative_coord(1000000)
    
    
    # setup time integrator
    time_integrator = hignn.ExplicitEuler()
    
    time_integrator.set_time_step(0.005)
    time_integrator.set_final_time(1.0)
    time_integrator.set_num_rigid_body(NN)
    time_integrator.set_output_step(10)
    
    time_integrator.set_velocity_func(velocity_update)
    time_integrator.initialize(X)
    
    t1 = time.time()
    
    time_integrator.run()
    
    del hignn_model
    del time_integrator

    hignn.Finalize()