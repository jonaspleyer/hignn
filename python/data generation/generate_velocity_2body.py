import numpy as np
import time
from functions_shared import add_sphere_rotations_to_positions, same_setup_as
from functions_timestepping import euler_timestep, did_something_go_wrong_with_dumbells, euler_timestep_rotation, \
    orthogonal_proj, do_we_have_all_size_ratios, generate_output_FTSUOE, are_some_of_the_particles_too_close

# ############ load position ############
data_pos = np.loadtxt('sample_2cir_box32.txt')
# data in a shape of (number, 6)
# in the form of (x1, y1, z1, x2, y2, z2)
# You need to generate the sample that you want.
Nc = 2
print data_pos.shape
print data_pos[0, :]

# ############ input mode ############
# change this part according to the boundary condition. 
# Please carefully read and understand this part of the original code in https://github.com/Pecnut/stokesian-dynamics.

# unbounded
# box_bottom_left = np.array([0, 0, 0])
# box_top_right = np.array([0, 0, 0])

# periodic
box_bottom_left = np.array([-16, -16, -16])
box_top_right = np.array([16, 16, 16])

# ############ Don't change ############
num_spheres = int(data_pos.shape[1] / 3)
num_data = data_pos.shape[0]
sphere_sizes = np.array([1.0 for i in range(num_spheres)])
dumbbell_sizes = np.array([])
dumbbell_positions = np.empty([0, 3])
dumbbell_deltax = np.empty([0, 3])
input_form = 'fte'
use_drag_Minfinity = False
use_Minfinity_only = False
extract_force_on_wall_due_to_dumbbells = False
last_velocities = (np.zeros((num_spheres, 3)), np.array([]), np.array([]), np.zeros((num_spheres, 3)))
last_velocity_vector = [0] * num_spheres * 11
checkpoint_start_from_frame = 0
feed_every_n_timesteps = 0
frameno = 0
last_generated_Minfinity_inverse = np.array([0])
regenerate_Minfinity = True
timestep = 0.1
cutoff_factor = 2
printout = 0
use_XYZd_values = True


# ############ Output ############
data_vel = np.zeros((2 * num_data, 9))
t1 = time.time()
for i in range(data_pos.shape[0]):
    sphere_positions = data_pos[i, :].reshape((num_spheres, 3))
    #    print(sphere_positions)

    # IMPORTANT: input number is defined in input_setups.py
    # it is related to the boundary condition. Please make sure you understand it.
    input_number = i % 3 + 10

    sphere_rotations = add_sphere_rotations_to_positions(sphere_positions, sphere_sizes,
                                                         np.array([[1, 0, 0], [0, 0, 1]]))
    posdata = (sphere_sizes, sphere_positions, sphere_rotations, dumbbell_sizes, dumbbell_positions, dumbbell_deltax)
    Fa_out_k1, Ta_out_k1, Sa_out_k1, Fb_out_k1, DFb_out_k1, Ua_out_k1, Oa_out_k1, Ea_out_k1, Ub_out_k1, DUb_out_k1, \
    last_generated_Minfinity_inverse, gen_times, U_infinity_k1, O_infinity_k1, centre_of_background_flow, force_on_wall_due_to_dumbbells_k1, \
    last_velocity_vector, grand_resistance_matrix = generate_output_FTSUOE(
        posdata, frameno, timestep, input_number, last_generated_Minfinity_inverse, regenerate_Minfinity, input_form,
        cutoff_factor, printout, use_XYZd_values, use_drag_Minfinity, use_Minfinity_only,
        extract_force_on_wall_due_to_dumbbells, last_velocities, last_velocity_vector, checkpoint_start_from_frame,
        box_bottom_left, box_top_right, feed_every_n_timesteps=feed_every_n_timesteps)

    
    Ua_out_k1[:, 0] = 6 * np.pi * Ua_out_k1[:, 0]
    Ua_out_k1[:, 1] = 6 * np.pi * Ua_out_k1[:, 1]
    Ua_out_k1[:, 2] = 6 * np.pi * Ua_out_k1[:, 2]
    # print Ua_out_k1
    
    Ua_out_k1 = Ua_out_k1.reshape(-1)
    # print Ua_out_k1
    # Here we can duplicate the data by symmetry.
    vel = np.concatenate((data_pos[i, 3:6], Ua_out_k1), axis=0)
    vel1 = np.concatenate((-data_pos[i, 3:6], Ua_out_k1), axis=0)
    # print vel
    # velocity_temp = np.concatenate((posdata[1], Ua_out_k1), axis=1)
    # velocity_temp = velocity_temp.reshape(1, 6 * num_spheres)
    data_vel[2 * i, :] = vel
    data_vel[2 * i + 1, :] = vel1
    
    if i % 1000 == 0:
        print i

t2 = time.time()
print t2 - t1
print data_vel.shape
# print data_vel
np.savetxt('data_output/data_2cir_box32.txt', data_vel)
