import numpy as np
import vtk
from vtk.util import numpy_support as VN
import os
import h5py

num_particle = 215040
k_t = 20
k_b = 300
n_chain = 21
dr = 3
len_fiber = dr * (n_chain - 1)
R = 4 * dr * 20 * 10**(1/3)

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
            points_array_rank = h5py.File('Result/pure_particle_'+str(num_particle)+'/pos_rank_'+str(rank)+"_"+str(int(N))+'.h5', 'r')['pos'][:]
            if points_array is None:
                points_array = points_array_rank
            else:
                points_array = np.concatenate((points_array, points_array_rank), axis=0)
            velocity_array_rank = h5py.File('Result/pure_particle_'+str(num_particle)+'/vel_rank_'+str(rank)+"_"+str(int(N))+'.h5', 'r')['vel'][:]
            if velocity_array is None:
                velocity_array = velocity_array_rank
            else:
                velocity_array = np.concatenate((velocity_array, velocity_array_rank), axis=0)
        except:
            move_forward = False
            break
    
    if not move_forward:
        break
    
    for i in range(3):
        original_coord = points_array[:, i]
        coord_mean = np.mean(original_coord)
        new_coord = original_coord - coord_mean
        coord_real_mean = np.mean(original_coord[new_coord<R])
        points_array[:, i] -= coord_real_mean

    for i in range(points_array.shape[0]):
        points.InsertNextPoint(points_array[i, :])
    
    for i in range(velocity_array.shape[0]):
        velocity.InsertNextTuple3(velocity_array[i, 0], velocity_array[i, 1], velocity_array[i, 2])

    for i in range(points_array.shape[0]):
        vertices.InsertNextCell(1)
        vertices.InsertCellPoint(i+count)
    
    count = points_array.shape[0]

    velocity.SetName('u')
        
    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)
    polydata.GetPointData().SetVectors(velocity)
    polydata.SetVerts(vertices)

    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetInputData(polydata)
    
    if os.path.exists('visualization/pure_particle_'+str(num_particle)) == False:
        os.makedirs('visualization/pure_particle_'+str(num_particle))
    writer.SetFileName('visualization/pure_particle_'+str(num_particle)+'/original_N_'+str(N)+'.vtp')
    writer.Write()