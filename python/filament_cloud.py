import os
import hignn
import numpy as np
import sys
from mpi4py import MPI
import time
import h5py   

# HiGNN Model Configuration
LEAF_MINIMUM_NUMBER_OF_PARTICLES = 100         # Must be < total number of particles 
MODEL = 'nn/3D_force_UB_max600_try2'           # Path to the pre-trained two-body model

# Filament Properties
TENSION_STIFFNESS = 20                         # Tension stiffness of filaments
BENDING_STIFFNESS = 300                        # Bending stiffness of filaments
REST_LENGTH = 2.4                              # Equilibrium length between adjacent particles

# Filament and Domain Setup
N_CHAIN = 31                                   # Number of particles per filament
N_FILAMENTS = 1024                             # Total number of filaments in the system
FILAMENT_LENGTH = (N_CHAIN - 1) * REST_LENGTH  # Physical length of each filament

# Define the radius of the spherical domain 
# Filaments should initially be placed inside a sphere where their length is 32% of the sphere's radius.
R = FILAMENT_LENGTH / 0.32 

# Simulation Parameters
DT = 0.001                                     # Time step
GRAVITATIONAL_FORCE = -1.0                     # Constant gravitational force
MAXIMUM_NUMBER_OF_ITERATIONS = 100000          # Total simulation iterations
MAXIMUM_TRIES = 100                            # Max attempts to place a filament without intersection

# Clear the terminal screen 
os.system("clear")  

# Function to calculate bending force on filaments
def chain_bending(X, k_b):
    """
    Compute bending forces acting on a filament.

    Parameters:
    X  : np.ndarray (N, 3) - Positions of filament particles.
    k_b: float - Bending stiffness coefficient.

    Returns:
    F  : np.ndarray (N, 3) - Bending forces on each particle.
    """
    N = X.shape[0]  # Number of particles in the filament
    F = np.zeros((N, 3))  # Initialize force array

    # Compute vectors between consecutive position vectors
    arm1 = X[1:(N - 1), :] - X[0:(N - 2), :]
    arm2 = X[2:N, :] - X[1:(N - 1), :]

    # Normalize vectors
    arm1_norm = np.linalg.norm(arm1, axis=1, keepdims=True)
    arm2_norm = np.linalg.norm(arm2, axis=1, keepdims=True)
    arm1 = arm1 / arm1_norm
    arm2 = arm2 / arm2_norm

    # Compute the cosine of the bending angle
    theta_cos = np.sum(arm1 * arm2, axis=1)

    # Compute the sine of the bending angle (handle numerical errors)
    tmp = 1 - theta_cos**2
    tmp[tmp < 0] = 0.0  # Ensure non-negative values
    theta_sin = np.sqrt(tmp).reshape((-1, 1))

    # Compute torque direction
    uv = np.cross(arm1, arm2)

    # Compute torque
    torque = -uv * theta_sin

    # Compute force contributions
    f21 = np.cross(torque, arm1)
    f23 = np.cross(torque, arm2)

    # Distribute forces along the filament
    F[0:(N - 2), :] += f21
    F[1:(N - 1), :] -= f21 + f23
    F[2:N, :] += f23

    # Scale by bending stiffness
    F = k_b * F

    return F

# Function to calculate tension force on filaments
def chain_tension(X, k_t, rest_length):
    """
    Compute tension forces acting on a filament.

    Parameters:
    X          : np.ndarray (N, 3) - Positions of filament particles.
    k_t        : float - Tension stiffness coefficient.
    rest_length: float - Resting length between adjacent particles.

    Returns:
    F          : np.ndarray (N, 3) - Tension forces on each particle.
    """
    N = X.shape[0]  # Number of particles in the filament
    F = np.zeros((N, 3))  # Initialize force array

    # Compute vectors between consecutive position vectors
    r = X[:-1, :] - X[1:, :]

    # Compute vectors lengths
    r_norm = np.linalg.norm(r, axis=1, keepdims=True)

    # Normalize vectors
    r = r / r_norm

    # Compute force magnitude 
    Fm = k_t * (r_norm - rest_length)

    # Compute force contributions at each node
    nodal_force = r * Fm

    # Apply forces to adjacent particles
    F[:-1, :] -= nodal_force
    F[1:, :] += nodal_force

    return F

# Function to update velocity based on forces
def velocity_update(t, position):
    """
    Compute new velocities for all particles.

    Parameters:
    t        : float - Current simulation time.
    position : np.ndarray (N, 3) - Current positions of all particles.

    Returns:
    velocity : np.ndarray (N, 3) - Updated velocities of all particles.
    """
    # Print current simulation time (only rank 0 prints to reduce redundancy)
    if rank == 0:
        print("t = {t:.4f}".format(t=t))

    # Update particle positions in the HiGNN model
    hignn_model.update_coord(position[:, :3])

    # Initialize velocity and force arrays
    velocity = np.zeros((position.shape[0], 3), dtype=np.float32)
    force = np.zeros((position.shape[0], 3), dtype=np.float32)

    # Apply constant gravitational force in the z-direction
    force[:, 2] = GRAVITATIONAL_FORCE

    # Filament properties
    n_chain = N_CHAIN                          # Number of particles per filament
    num_chains = position.shape[0] // n_chain  # Total number of filaments
    k_t = TENSION_STIFFNESS                    # Tension stiffness coefficient
    k_b = BENDING_STIFFNESS                    # Bending stiffness coefficient
    rest_length = REST_LENGTH                  # Resting length between adjacent particles

    # Compute forces for each filament
    for i in range(num_chains):
        # Extract filament positions
        filament_positions = position[n_chain * i : n_chain * (i + 1), :]

        # Compute and apply bending forces
        force[n_chain * i : n_chain * (i + 1), :] += chain_bending(filament_positions, k_b)

        # Compute and apply tension forces
        force[n_chain * i : n_chain * (i + 1), :] += chain_tension(filament_positions, k_t, rest_length)

    # Compute new velocity using HiGNN model
    hignn_model.dot(velocity, force)

    return velocity

# Function to check if filaments have intersection
def check_intersection(filament1, filament2):
    """Check if two filaments intersect using NumPy broadcasting."""
    # Compute pairwise distances between all particles in filament1 and filament2
    dists = np.linalg.norm(filament1[:, np.newaxis, :] - filament2[np.newaxis, :, :], axis=2)
    
    # Check if any distance is less than REST_LENGTH
    return np.any(dists < REST_LENGTH)

# Function to generate initial position of filaments
def generate_random_filament():
    """
    Generate a single filament,
    with random center of gravity,
    and random orientation within an sphere of radius R.
    """
    # Trying to place filament 
    for _ in range(MAXIMUM_TRIES):
        print(f"Attempt: {_+1}/{MAXIMUM_TRIES}")
        # Generate a random center within sphere
        center = np.random.uniform(-R, R, 3)
        if np.linalg.norm(center) > R:
            continue
        
        # Generate a random orientation
        direction = np.random.randn(3)
        direction /= np.linalg.norm(direction)
        
        # Compute filament particles positions
        displacements = np.linspace(-0.5, 0.5, N_CHAIN).reshape(-1, 1) * FILAMENT_LENGTH
        positions = center + displacements * direction
        
        # Check if filament intersects with existing ones
        if all(not check_intersection(positions, existing) for existing in X):
            return positions
        
        return None

# Setup simulation
if __name__ == '__main__':    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    # Initialize HIGNN
    hignn.Init()
    
    np.random.seed(0)
    
    X = []
    
    # Generate the initial position of filament cloud
    for _ in range(N_FILAMENTS):
        print(f"Generating filament {_+1}/{N_FILAMENTS}...")
        
        new_filament = generate_random_filament()
        
        # Check if new filament isn't None
        if new_filament is not None:
            X.append(new_filament)
            print(f"Filament placed successfully!\n")
        else:
            print(f"Warning: Could not place filament {_+1}/{N_FILAMENTS} after max tries.\n")
            
    print("Starting simulation...\n")
        
    X = np.array(X).reshape(-1, 3) # Shape: (N_FILAMENTS * N_CHAIN, 3)

    # Convert X to float32 for efficient computation
    X = X.astype(np.float32)

    # Number of particles in the system
    NN = X.shape[0]

    # Initialize the HiGNN model with the particle positions
    hignn_model = hignn.HignnModel(X, LEAF_MINIMUM_NUMBER_OF_PARTICLES)

    # Load the pre-trained two-body model
    hignn_model.load_two_body_model(MODEL)

    # Set parameters for the far-field dot product calculations (default values)
    hignn_model.set_epsilon(0.01)                      
    hignn_model.set_max_iter(50)                       # Maximum number of iterations
    hignn_model.set_mat_pool_size_factor(200)          # Matrix pool size factor
    hignn_model.set_post_check_flag(False)             # Disable post-checking
    hignn_model.set_use_symmetry_flag(False)           # Disable symmetry optimizations
    hignn_model.set_max_far_dot_work_node_size(10000)  # Max size for far-field dot product
    hignn_model.set_max_relative_coord(100000)         # Max relative coordinate value

    # Divide particles among MPI processes for parallel execution
    rank_range = np.linspace(0, NN, comm.Get_size() + 1, dtype=np.int32)

    # Initialize simulation time and iteration counter
    ts = 0        # Initial time
    dt = DT       # Time step
    ite = 0       # Iteration counter

    # Start timing the entire simulation
    t1 = time.time()

    # Main simulation loop
    for i in range(MAXIMUM_NUMBER_OF_ITERATIONS):

        # Every 100 iterations, save particle positions to an HDF5 file
        if i % 100 == 0:
            with h5py.File(f'Result/pos{int(i/100)}rank{rank}.h5', 'w') as f:
                f.create_dataset('pos', data=X[rank_range[rank]:rank_range[rank+1], :])

        # Measure time for velocity update step
        tt1 = time.time()
        V = velocity_update(ts, X)

        # Print velocity update time only on rank 0 (to avoid redundant logging)
        if rank == 0:
            print("Time for velocity_update: {t:.4f}s".format(t=time.time() - tt1))

        # Every 100 iterations, save particle velocities to an HDF5 file
        if i % 100 == 0:
            with h5py.File(f'Result/vel{int(i/100)}rank{rank}.h5', 'w') as f:
                f.create_dataset('vel', data=V[rank_range[rank]:rank_range[rank+1], :])

        # Update particle positions using velocity
        X = X + dt * V

        # Advance simulation time
        ts = ts + dt          

    # Print total simulation time (only for rank 0)
    if rank == 0:
        print("Time for simulation: {t:.4f}s".format(t=time.time() - t1))

    # Cleanup: Delete HiGNN model to free memory
    del hignn_model

    # Finalize HiGNN
    hignn.Finalize()