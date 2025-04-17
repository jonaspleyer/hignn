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
    
    dr = 3                                                  # rest length between particles
    n_chain = 11                                            # number of particles in a chain        
    n_fiber = int(1024*10*21/11)                                  # number of fibers            
    len_fiber = dr * (n_chain - 1)                          # length of a fiber
    R = 4 * dr * 20 * 10**(1/3)
    
    X = np.zeros((n_chain * n_fiber, 3), dtype=np.float32)
    
    min_dist = 2.1
    
    chain = np.zeros((n_chain, 3), dtype=np.float32)
    chain[:, 2] = np.linspace(0, len_fiber, n_chain) - 0.5 * len_fiber
    
    n = 0
    while n < n_fiber:
        print('N =', n, end='\r')
        # generate candidate center points
        X_candidate = 2*np.random.rand(1024*3)*R-R
        X_candidate = X_candidate.reshape((1024, 3))
        r = np.linalg.norm(X_candidate, axis=1)
        X_candidate = np.reshape(X_candidate[(r < R).nonzero(), :], (-1, 3))
        
        # remove too close centers
        idx = check_self_intersection(X_candidate, min_dist)
        X_candidate = np.reshape(X_candidate[idx, :], (-1, 3))
        
        X_chain = generate_fiber(X_candidate, chain)
        
        for i in range(X_candidate.shape[0]):
            if not check_intersection(X[:n_chain * n, :], X_chain[n_chain * i:n_chain * i + n_chain, :], min_dist):
                X[n_chain * n:n_chain * n + n_chain, :] = X_chain[n_chain * i:n_chain * i + n_chain, :]
                n += 1
            if n >= n_fiber:
                break
    
    np.savetxt('initial/filament_l_'+str(n_chain)+'_'+str(n_fiber)+'.txt', X)