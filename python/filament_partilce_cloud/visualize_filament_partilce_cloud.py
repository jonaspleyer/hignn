import json
import numpy as np
import vtk
from vtk.util import numpy_support as VN
import os
import h5py

# Load cloud parameters from JSON
with open("config.json", "r") as f:
    config = json.load(f)
    
cloud_params = config['cloud']
filament_cloud_params = cloud_params['filament']
    
if filament_cloud_params:
    n_filament = filament_cloud_params['n_filament']       # Number of filaments
    n_chain = filament_cloud_params['n_chain']             # Particles per filament
    rest_length = filament_cloud_params['rest_length']     # Rest length between particles
    k_t = filament_cloud_params['k_t']                     # Tension stiffness parameter
    k_b = filament_cloud_params['k_b']                     # Bending stiffness parameter
    len_fiber = n_filament * (n_chain - 1)                 # Filament length

# Convert .h5 files to .vtp files
for N in range(500):
    os.system('clear')
    
    print('N =', N, end='\r')
    
    points = vtk.vtkPoints()
    velocity = vtk.vtkFloatArray()
    velocity.SetNumberOfComponents(3)
    vertices = vtk.vtkCellArray()
    
    move_forward = True
    
    count = 0
    points_array = None
    velocity_array = None
    for rank in range(4):
        try:
            points_array_rank = h5py.File('output/particle_filament_l_'+str(n_chain)+'_kt_'+str(k_t)+'_kb_'+str(k_b)+'/pos_rank_'+str(rank)+'_N_'+str(n_filament)+"_"+str(N)+'.h5', 'r')['pos'][:]
            if points_array is None:
                points_array = points_array_rank
            else:
                points_array = np.concatenate((points_array, points_array_rank), axis=0)
            velocity_array_rank = h5py.File('output/particle_filament_l_'+str(n_chain)+'_kt_'+str(k_t)+'_kb_'+str(k_b)+'/vel_rank_'+str(rank)+'_N_'+str(n_filament)+"_"+str(N)+'.h5', 'r')['vel'][:]
            if velocity_array is None:
                velocity_array = velocity_array_rank
            else:
                velocity_array = np.concatenate((velocity_array, velocity_array_rank), axis=0)
        except:
            move_forward = False
            break
    
    if not move_forward:
        break
    
    # for i in range(3):
    #     original_coord = points_array[:, i]
    #     coord_mean = np.mean(original_coord)
    #     new_coord = original_coord - coord_mean
    #     coord_real_mean = np.mean(original_coord[new_coord<R])
    #     points_array[:, i] -= coord_real_mean

    for i in range(points_array.shape[0]):
        points.InsertNextPoint(points_array[i, :])
    
    for i in range(velocity_array.shape[0]):
        velocity.InsertNextTuple3(velocity_array[i, 0], velocity_array[i, 1], velocity_array[i, 2])

    particle_start = n_filament * n_chain
    for i in range(particle_start, points_array.shape[0]):
        vertices.InsertNextCell(1)
        vertices.InsertCellPoint(i)
    
    count = points_array.shape[0]

    velocity.SetName('u')
        
    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)
    polydata.GetPointData().SetVectors(velocity)
    polydata.SetVerts(vertices)
    
    lines = vtk.vtkCellArray()
    
    for n in range(n_filament):
        line = vtk.vtkPolyLine()
        line.GetPointIds().SetNumberOfIds(n_chain)
        for i in range(n_chain):
            line.GetPointIds().SetId(i, i+n*n_chain)
        
        lines.InsertNextCell(line)
    
    polydata.SetLines(lines)

    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetInputData(polydata)
    
    if os.path.exists('visualization/particle_filament_l_'+str(n_chain)+'_kt_'+str(k_t)+'_kb_'+str(k_b)) == False:
        os.makedirs('visualization/particle_filament_l_'+str(n_chain)+'_kt_'+str(k_t)+'_kb_'+str(k_b))
    # writer.SetFileName('visualization/particle_filament_l_'+str(n_chain)+'_kt_'+str(k_t)+'_kb_'+str(k_b)+'/original_N_'+str(n_filament)+"_"+str(N)+'.vtp')
    writer.SetFileName('visualization/particle_filament_l_'+str(n_chain)+'_kt_'+str(k_t)+'_kb_'+str(k_b)+'/original_filament_only_N_'+str(n_filament)+"_"+str(N)+'.vtp')
    writer.Write()