import numpy as np
import vtk
from vtk.util import numpy_support as VN
import os
import h5py

for N in range(500):
    os.system('clear')
    
    print('N =', N, end='\r')
    
    points = vtk.vtkPoints()
    velocity = vtk.vtkFloatArray()
    velocity.SetNumberOfComponents(3)
    vertices = vtk.vtkCellArray()
    
    move_forward = True
    
    count = 0
    for rank in range(1):
        try:
            points_array = h5py.File('Result/pos'+str(N)+'rank'+str(rank)+'.h5', 'r')['pos'][:]
            velocity_array = h5py.File('Result/vel'+str(N)+'rank'+str(rank)+'.h5', 'r')['vel'][:]
        except:
            move_forward = False
            break

        for i in range(points_array.shape[0]):
            points.InsertNextPoint(points_array[i, :])
        
        for i in range(velocity_array.shape[0]):
            velocity.InsertNextTuple3(velocity_array[i, 0], velocity_array[i, 1], velocity_array[i, 2])

        for i in range(points_array.shape[0]):
            vertices.InsertNextCell(1)
            vertices.InsertCellPoint(i+count)
            
        count += points_array.shape[0]
    
    if not move_forward:
        break

    velocity.SetName('u')
        
    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)
    polydata.GetPointData().SetVectors(velocity)
    # polydata.SetVerts(vertices)
    
    lines = vtk.vtkCellArray()
    
    n_chain = 31
    num_chains = int(points_array.shape[0] / n_chain)
    for n in range(num_chains):
        line = vtk.vtkPolyLine()
        line.GetPointIds().SetNumberOfIds(n_chain)
        for i in range(n_chain):
            line.GetPointIds().SetId(i, i+n*n_chain)
        
        lines.InsertNextCell(line)
    
    polydata.SetLines(lines)

    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetInputData(polydata)
    writer.SetFileName('Result/w_multi_filament_kt_20_kb_40_'+str(N)+'.vtp')
    writer.Write()