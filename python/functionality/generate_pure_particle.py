import os
import numpy as np
import scipy.spatial as KDTreed

def check_self_intersection(X, min_dist):
    tree = KDTreed.KDTree(X)
    d, _ = tree.query(X, k=2)
    idx = np.where(d[:, 1] > min_dist)[0]
    return idx

def check_intersection(X0, X, min_dist):
    if X0.shape[0] == 0:
        return False
    
    tree = KDTreed.KDTree(X0)
    d, _ = tree.query(X)
    idx = np.where(d > min_dist)[0]
    if idx.shape[0] < X.shape[0]:
        return True
    else:
        return False

def generate_fiber(Xc, chain):
    n_fiber = Xc.shape[0]
    n_chain = chain.shape[0]
    alpha = 2 * np.random.rand(n_fiber) * np.pi - np.pi
    beta = np.random.rand(n_fiber) * np.pi
    gamma = 2 * np.random.rand(n_fiber) * np.pi - np.pi
    
    X = np.zeros((n_chain * n_fiber, 3), dtype=np.float32)
    
    for i in range(n_fiber):
        rotation_matrix = np.zeros((3, 3), dtype=np.float32)
        rotation_matrix[0, 0] = np.cos(alpha[i]) * np.cos(beta[i])
        rotation_matrix[0, 1] = np.cos(alpha[i]) * np.sin(beta[i]) * np.sin(gamma[i]) - np.sin(alpha[i]) * np.cos(gamma[i])
        rotation_matrix[0, 2] = np.cos(alpha[i]) * np.sin(beta[i]) * np.cos(gamma[i]) + np.sin(alpha[i]) * np.sin(gamma[i])
        rotation_matrix[1, 0] = np.sin(alpha[i]) * np.cos(beta[i])
        rotation_matrix[1, 1] = np.sin(alpha[i]) * np.sin(beta[i]) * np.sin(gamma[i]) + np.cos(alpha[i]) * np.cos(gamma[i])
        rotation_matrix[1, 2] = np.sin(alpha[i]) * np.sin(beta[i]) * np.cos(gamma[i]) - np.cos(alpha[i]) * np.sin(gamma[i])
        rotation_matrix[2, 0] = -np.sin(beta[i])
        rotation_matrix[2, 1] = np.cos(beta[i]) * np.sin(gamma[i])
        rotation_matrix[2, 2] = np.cos(beta[i]) * np.cos(gamma[i])
        
        X[i * n_chain:(i + 1) * n_chain, :] = (rotation_matrix @ chain.T).T + Xc[i, :]
    
    return X

if __name__ == '__main__':
    os.system('clear')
    
    np.random.seed(0)
    
    scale = 10
    dr = 3                                                # rest length between particles
    n_chain = 21                                            # number of particles in a chain        
    n_fiber0 = 1024 * scale
    n_fiber = 512 * scale
    len_fiber = dr * (n_chain - 1)                          # length of a fiber
    R1 = 3 * len_fiber * scale**(1/3)
    R2 = 4 * len_fiber * scale**(1/3)
    n_particle = n_fiber0 * n_chain
    
    X = np.zeros((n_particle, 3), dtype=np.float32)
    
    min_dist = 2.1
    
    n = 0
    while n < n_particle:
        print('N = ', n, end='\r')
        # generate candidate particles
        X_candidate = 2*np.random.rand(10240*3)*R2-R2
        X_candidate = X_candidate.reshape((10240, 3))
        r = np.linalg.norm(X_candidate, axis=1)
        X_candidate = np.reshape(X_candidate[(r < R2).nonzero(), :], (-1, 3))
        
        # remove too close particles
        idx = check_self_intersection(X_candidate, min_dist)
        X_candidate = np.reshape(X_candidate[idx, :], (-1, 3))
        
        selected_num = min(n_particle - n, X_candidate.shape[0])
        X[n:n + selected_num, :] = X_candidate[:selected_num, :]
        n += selected_num
    
    np.savetxt('initial/pure_particle_20480.txt', X)