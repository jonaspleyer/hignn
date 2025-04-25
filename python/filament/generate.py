import os
import numpy as np
import scipy.spatial as KDTreed
import time

os.system("clear")

def check_center_proximity(X, min_dist):
    tree = KDTreed.KDTree(X)
    d, _ = tree.query(X, k=2)
    idx = np.where(d[:, 1] > min_dist)[0]
    
    return idx

def find_intersection(X0, X, n_chain, min_dist):
    if X0.shape[0] == 0:
        return np.array([])

    tree = KDTreed.KDTree(X0)
    d, _ = tree.query(X)
    idx = np.where(d < min_dist)[0]

    return np.unique(idx // n_chain)

def remove_intersections(X_chain, n_chain, min_dist):
    tree = KDTreed.KDTree(X_chain)
    d, _ = tree.query(X_chain, k=2)
    intersected_chain_idx = np.unique(np.sort(np.where(d[:, 1] < min_dist)[0] // n_chain))
    non_intersected_chain_idx = np.setdiff1d(np.arange(X_chain.shape[0] // n_chain), intersected_chain_idx)
    non_intersected_particle_idx = np.arange(X_chain.shape[0], dtype=np.int32).reshape(-1, n_chain)[non_intersected_chain_idx, :].flatten()
  
    return X_chain[non_intersected_particle_idx, :]

def generate_fiber(Xc, chain):
    n_fiber = Xc.shape[0]
    n_chain = chain.shape[0]
    direction = np.random.randn(n_fiber, 3)
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)

    X = np.zeros((n_chain * n_fiber, 3), dtype=np.float32)

    for i in range(n_fiber):
      X[i * n_chain:(i + 1) * n_chain, :] = Xc[i] + chain * direction[i]

    return X

if __name__ == '__main__':
    os.system('clear')

    np.random.seed(0)

    dr = 3                                                  # rest length between particles
    n_chain = int(11)                                            # number of particles in a chain
    n_fiber = int(100000)                            # number of fibers
    len_fiber = dr * (n_chain - 1)                          # length of a fiber
    R = 4 * dr * 20 * 10**(1/3) * (n_fiber/(1024*10*21/11))

    X = np.zeros((n_chain * n_fiber, 3), dtype=np.float32)

    min_dist = 2.1

    chain = np.linspace(-0.5, 0.5, n_chain).reshape(-1, 1) * len_fiber

    n = 0
    remaining_fibers = n_fiber
    batch_size = int(1024)

    print(f"Generating filaments...\n\n")

    start_time = time.time()
    
    while n < n_fiber:
        # generate candidate center points
        X_candidate = 2*np.random.rand(batch_size*3)*R-R
        X_candidate = X_candidate.reshape((batch_size, 3))
        r = np.linalg.norm(X_candidate, axis=1)
        X_candidate = np.reshape(X_candidate[(r < R).nonzero(), :], (-1, 3))

        # remove too close centers
        idx = check_center_proximity(X_candidate, min_dist)
        X_candidate = np.reshape(X_candidate[idx, :], (-1, 3))

        # generate fibers
        X_chain = generate_fiber(X_candidate, chain)

        # remove generated filaments having intersection
        X_chain = remove_intersections(X_chain, n_chain, min_dist)

        # find and remove generated filaments intersecting with existing filaments
        idx_chain = np.setdiff1d(np.arange(X_chain.shape[0]//n_chain, dtype=np.int32), find_intersection(X[:n*n_chain, :], X_chain, n_chain, min_dist).astype(np.int32))
        idx_chain = idx_chain[:min(idx_chain.shape[0], remaining_fibers)]
        idx_particle = np.arange(X_chain.shape[0], dtype=np.int32).reshape(-1, n_chain)[idx_chain, :].flatten()
        X_chain = X_chain[idx_particle, :]
        
        # save valid generated filaments
        m = idx_chain.shape[0]
        X[n * n_chain:(n+m) * n_chain, :] = X_chain
        remaining_fibers -= m
        n += m
        print(f"\033[2A\033[KSuccessfully generated {n:8d} filaments!\n\033[K{remaining_fibers:8d} filaments remain!", flush=True)


    end_time = time.time()

    elapsed_time = end_time - start_time
    print("------------------------------------------------------------------")
    print(f"- Elapsed time: {elapsed_time:6.2f} seconds")
    print("------------------------------------------------------------------")
    
    np.savetxt('initial/filament_l_'+str(n_chain)+'_'+str(n_fiber)+'.txt', X)