import os
import numpy as np
import scipy.spatial as KDTreed
import time

def check_center_proximity(X, min_dist):
    tree = KDTreed.KDTree(X)
    d, _ = tree.query(X, k=2)
    idx = np.where(d[:, 1] > min_dist)[0]
    return idx

def check_intersection(X0, X, min_dist):
    if X0.shape[0] == 0:
        return False

    tree = KDTreed.KDTree(X0)
    d, _ = tree.query(X)
    idx = np.where(d < min_dist)[0]

    return True if idx.shape[0] > 0 else False

def find_intersection(X0, X, n_chain, min_dist):
    if X0.shape[0] == 0:
        return np.array([])

    tree = KDTreed.KDTree(X0)
    d, _ = tree.query(X)
    idx = np.where(d < min_dist)[0]

    return np.unique(idx//n_chain)

def remove_intersections(X_chain, n_chain, min_dist):
  n_fiber = X_chain.shape[0] // n_chain
  X_chain = X_chain.reshape((n_fiber, n_chain, 3))
  for i in range(n_fiber):
    if i >= X_chain.shape[0]:
      break

    if check_intersection(np.delete(X_chain, i, axis=0).reshape(-1, 3), X_chain[i], min_dist):
      X_chain = np.delete(X_chain, i, axis=0)
      i -= 1

  return X_chain.reshape(-1, 3)

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
    n_chain = 11                                            # number of particles in a chain
    n_fiber = int(100000)                            # number of fibers
    len_fiber = dr * (n_chain - 1)                          # length of a fiber
    R = 4 * dr * 20 * 10**(1/3) * (n_fiber/(1024*10*21/11))

    X = np.zeros((n_chain * n_fiber, 3), dtype=np.float32)

    min_dist = 2.1

    chain = np.linspace(-0.5, 0.5, n_chain).reshape(-1, 1) * len_fiber

    n = 0
    remaining_fibers = n_fiber
    batch_size = int(1024)

    print(f"Generating filaments...")

    start_time = time.time()
    
    while n < n_fiber:
        # generate candidate center points
        X_candidate = 2*np.random.rand(batch_size*3)*R-R
        X_candidate = X_candidate.reshape((batch_size, 3))
        r = np.linalg.norm(X_candidate, axis=1)
        X_candidate = np.reshape(X_candidate[(r < R).nonzero(), :], (-1, 3))

        # remove too close centers
        idx = check_center_proximity(X_candidate, 5 * min_dist)
        X_candidate = np.reshape(X_candidate[idx, :], (-1, 3))

        # genearate fibers
        X_chain = generate_fiber(X_candidate, chain)

        # remove generated filaments having intersection
        X_chain = remove_intersections(X_chain, n_chain, min_dist)

        # find and remove generatd filaments intersecting with existing filaments
        idx_chain = find_intersection(X, X_chain, n_chain, min_dist)
        X_chain = X_chain.reshape(-1, n_chain, 3)
        X_chain = np.delete(X_chain, idx_chain, axis=0).reshape(-1, 3)
        
        # save valid generated filaments
        m = X_chain.shape[0]//n_chain if X_chain.shape[0]//n_chain <= remaining_fibers else remaining_fibers
        X[n * n_chain:(n+m) * n_chain, :]  = X_chain[0:m * n_chain, :]
        remaining_fibers -= m
        n += m
        print(f"Successfully generated {n} filaments!")
        print(f"{remaining_fibers} filaments remain!")
        print(".\n.\n.\n")


    end_time = time.time()

    elapsed_time = end_time - start_time
    print("------------------------------------------------------------------")
    print(f"- Elapsed time: {elapsed_time:.2f} seconds")
    print("------------------------------------------------------------------")
    
    np.savetxt('initial/filament_l_'+str(n_chain)+'_'+str(n_fiber)+'.txt', X)