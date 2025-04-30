import os
import hignn
import json
import numpy as np
import sys
from mpi4py import MPI
import time
import h5py


os.system("clear")

def chain_bending(X, k_b):
    """
    Compute the bending forces for a single filament chain.

    Parameters:
    -----------
    X : np.ndarray
        Coordinates of the particles in the filament, of shape (N, 3).
        
    k_b : float
          Bending stiffness parameter.

    Returns:
    --------
    F: np.ndarray
       Bending forces for each particle, of shape (N, 3).
    """
    N = X.shape[0]
    F = np.zeros((N, 3))

    arm1 = X[1:(N - 1), :] - X[0:(N - 2), :]
    arm2 = X[2:N, :] - X[1:(N - 1), :]
    arm1_norm = np.linalg.norm(arm1, ord=2, axis=1).reshape((-1, 1))
    arm2_norm = np.linalg.norm(arm2, ord=2, axis=1).reshape((-1, 1))
    arm1 = arm1 / arm1_norm
    arm2 = arm2 / arm2_norm
    theta_cos = np.sum(arm1 * arm2, axis=1)
    tmp = 1 - theta_cos * theta_cos
    tmp[(tmp < 0).nonzero()] = 0.0
    theta_sin = np.sqrt(tmp).reshape((-1, 1))
    uv = np.cross(arm1, arm2)
    torque = - uv * theta_sin
    f21 = np.cross(torque, arm1)
    f23 = np.cross(torque, arm2)
    F[0:(N - 2), :] += f21
    F[1:(N - 1), :] -= f21 + f23
    F[2:N, :] += f23
    F = k_b * F
    
    return F

def chain_tension(X, k_t, rest_length):
    """
    Compute the tension forces for a single filament chain.

    Parameters:
    -----------
    X : np.ndarray
        Coordinates of the particles in the filament, of shape (N, 3).
        
    k_t : float
          Tension stiffness parameter.
        
    rest_length : float
                  Rest length between consecutive particles in the chain.

    Returns:
    --------
    F: np.ndarray
       Tension forces for each particle, of shape (N, 3).
    """
    N = X.shape[0]
    F = np.zeros((N, 3))

    r = X[0: (N - 1), :] - X[1: N, :]
    r_norm = np.linalg.norm(r, ord=2, axis=1)
    r_norm = r_norm.reshape((N - 1, 1))
    r = r / r_norm
    Fm = k_t * (r_norm - rest_length)
    nodal_force = r * Fm
    F[0: (N - 1), :] -= nodal_force
    F[1:N, :] += nodal_force
    
    return F

def velocity_update(t, position, f_g, n_filament, n_chain, rest_length, k_t, k_b):
    """
    Update the velocity of each particle based on external forces, chain tension, and bending forces.

    Parameters:
    -----------
    t : float
        Current simulation time.
        
    position : np.ndarray
               Current positions of all particles, of shape (N, 3).
        
    f_g : float, optional, default=-1
          Gravitational force applied in the z-direction.
        
    n_filament : int, optional
                 Number of filaments in the system.
        
    n_chain : int, optional
              Number of particles in each filament chain.
        
    rest_length : float, optional
                  Rest length between consecutive particles in the chain.
        
    k_t : float, optional
          Tension stiffness parameter.
        
    K_b : float, optional
          Bending stiffness parameter.

    Returns:
    --------
    velocity: np.ndarray
              Updated velocities for each particle, of shape (N, 3).
    """
    if rank == 0:
        print("t = {t:.4f}".format(t = t))
    hignn_model.update_coord(position[:, 0:3])
    velocity = np.zeros((position.shape[0], 3), dtype=np.float32)
    force = np.zeros((position.shape[0], 3), dtype=np.float32)
    force[:, 2] = f_g
    
    # Filaments
    if (n_filament > 0):
        for i in range(n_filament):
            force[n_chain * i:n_chain * i + n_chain, :] += chain_bending(position[n_chain * i:n_chain * i + n_chain, :], k_b)
            force[n_chain * i:n_chain * i + n_chain, :] += chain_tension(position[n_chain * i:n_chain * i + n_chain, :], k_t, rest_length)
    
    hignn_model.dot(velocity, force)
    
    return velocity

if __name__ == '__main__':    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    # Initialize hignn
    hignn.Init()
    
    # Load simulation parameters from JSON
    working_directory = 'python/filament_particle_cloud'
    with open(f"{working_directory}/config.json", "r") as f:
        config = json.load(f)
    
    simulation_params = config["simulation"]
    f_g = simulation_params["gravitational_force"]             # Gravitational force
    dt = simulation_params["dt"]                               # Time step
    t_max = simulation_params["t_max"]                         # Total number of iterations
    t_meas = simulation_params["t_meas"]                       # interval after which data is written
    
    # Load cloud parameters from JSON
    cloud_params = config['cloud']
    cloud_type = cloud_params['type']
    filament_cloud_params = cloud_params['filament']
    
    if filament_cloud_params:
        n_filament = filament_cloud_params['n_filament']       # Number of filaments
        n_chain = filament_cloud_params['n_chain']             # Particles per filament
        rest_length = filament_cloud_params['rest_length']     # Rest length between particles
        k_t = filament_cloud_params['k_t']                     # Tension stiffness parameter
        k_b = filament_cloud_params['k_b']                     # Bending stiffness parameter
    else:
        n_filament = n_chain = rest_length = k_t = k_b = 0
        
    X = np.loadtxt(working_directory+'/'+cloud_type +'.pos')
    
    NN = X.shape[0]
    
    # HIGNN model
    hignn_model = hignn.HignnModel(X, 100 if X.shape[0] > 100 else X.shape[0] // 2) # The minimal number of particles in a leaf node
    hignn_model_params = config['model']
    two_body_model = hignn_model_params['two_body_model']
    hignn_model.load_two_body_model(two_body_model)
    
    # Set parameters for far dot, the following parameters are default values
    hignn_model.set_epsilon(0.01)
    hignn_model.set_max_iter(50)
    hignn_model.set_mat_pool_size_factor(200)
    hignn_model.set_post_check_flag(False)
    hignn_model.set_use_symmetry_flag(True)
    hignn_model.set_max_far_dot_work_node_size(10000)
    hignn_model.set_max_relative_coord(100000)
    
    rank_range = np.linspace(0, NN, comm.Get_size() + 1, dtype=np.int32)
    
    # Main loop
    ts = 0
    ite = 0
    
    t1 = time.time()
    os.makedirs(f"{working_directory}/output/hdf5", exist_ok=True)
    for i in range(t_max):
        if i % t_meas == 0:
            with h5py.File(working_directory+'/output/hdf5/pos_rank_'+str(rank)+"_"+str(int(i/t_meas))+'.h5', 'w') as f:
                f.create_dataset('pos', data=X[rank_range[rank]:rank_range[rank+1], :])
        
        tt1 = time.time()
        V = velocity_update(ts, X, f_g, n_filament, n_chain, rest_length, k_t, k_b)
        if rank == 0:
            print("Time for velocity_update: {t:.4f}s".format(t = time.time() - tt1))

        if i % t_meas == 0:
            with h5py.File(working_directory+'/output/hdf5/vel_rank_'+str(rank)+"_"+str(int(i/t_meas))+'.h5', 'w') as f:
                f.create_dataset('vel', data=V[rank_range[rank]:rank_range[rank+1], :])
        
        X = X + dt * V
        ts = ts + dt

    if rank == 0:
        print("Time for simulation: {t:.4f}s".format(t = time.time() - t1))
    
    del hignn_model
    
    # Finalize hignn
    hignn.Finalize()