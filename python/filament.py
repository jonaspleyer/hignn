import os
import hignn
import numpy as np
import sys
from mpi4py import MPI
import time
import h5py

os.system("clear")

def chain_bending(X, k_b):
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

def velocity_update(t, position):
    if rank == 0:
        print("t = {t:.4f}".format(t = t))
    hignn_model.update_coord(position[:, 0:3])
    velocity = np.zeros((position.shape[0], 3), dtype=np.float32)
    force = np.zeros((position.shape[0], 3), dtype=np.float32)
    force[:, 2] = -1.0
    
    # filaments
    n_chain = 100
    num_chains = int(position.shape[0] / n_chain)
    k_t = 100 # tension stiffness coefficient
    k_b = 30 # bending stiffness coefficient
    rest_length = 2.4
    for i in range(num_chains):
        force[n_chain * i:n_chain * i + n_chain, :] += chain_bending(position[n_chain * i:n_chain * i + n_chain, :], k_b)
        force[n_chain * i:n_chain * i + n_chain, :] += chain_tension(position[n_chain * i:n_chain * i + n_chain, :], k_t, rest_length)
    
    hignn_model.dot(velocity, force)
    
    return velocity

if __name__ == '__main__':    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    hignn.Init()
    
    np.random.seed(0)
    
    n_chain = 100
    nx = 5
    ny = 5
    nz = n_chain
    dx = 3
    dy = 3
    dz = 2.4
    x = np.arange(0, nx * dx, dx)
    y = np.arange(0, ny * dy, dy)
    z = np.arange(0, nz * dz, dz)
    zz, yy, xx = np.meshgrid(x, y, z)
    X = np.concatenate(
        (xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)), axis=1)
    X = X.astype(np.float32)
    X += np.random.rand(X.shape[0], X.shape[1]) * 0.5
    
    NN = X.shape[0]

    hignn_model = hignn.HignnModel(X, 100)
    
    hignn_model.load_two_body_model('nn/3D_force_UB_max600_try2')
    
    # set parameters for far dot, the following parameters are default values
    hignn_model.set_epsilon(0.01)
    hignn_model.set_max_iter(50)
    hignn_model.set_mat_pool_size_factor(200)
    hignn_model.set_post_check_flag(False)
    hignn_model.set_use_symmetry_flag(False)
    hignn_model.set_max_far_dot_work_node_size(10000)
    hignn_model.set_max_relative_coord(100000)
    
    rank_range = np.linspace(0, NN, comm.Get_size() + 1, dtype=np.int32)
    
    ts = 0
    dt = 0.001
    ite = 0
    
    t1 = time.time()

    for i in range(50000):
        if i % 100 == 0:
            with h5py.File('Result/pos'+str(int(i/100))+'rank'+str(rank)+'.h5', 'w') as f:
                f.create_dataset('pos', data=X[rank_range[rank]:rank_range[rank+1], :])
        
        tt1 = time.time()
        V = velocity_update(ts, X)
        if rank == 0:
            print("Time for velocity_update: {t:.4f}s".format(t = time.time() - tt1))

        if i % 100 == 0:
            with h5py.File('Result/vel'+str(int(i/100))+'rank'+str(rank)+'.h5', 'w') as f:
                f.create_dataset('vel', data=V[rank_range[rank]:rank_range[rank+1], :])
        
        X = X + dt * V
        ts = ts + dt          

    if rank == 0:
        print("Time for simulation: {t:.4f}s".format(t = time.time() - t1))
    
    del hignn_model
    
    hignn.Finalize()