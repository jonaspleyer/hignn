import numpy as np
import vtk
from vtk.util import numpy_support as VN
import os
import h5py
from filament import MAXIMUM_NUMBER_OF_ITERATIONS, N_CHAIN
  
# MAXIMUM_NUMBER_OF_ITERATIONS = 100000
# N_CHAIN = 31

for N in range(int(MAXIMUM_NUMBER_OF_ITERATIONS/100)):
    os.system('clear')
    print(f'N = {N}', end='\r')

    points = vtk.vtkPoints()
    velocity = vtk.vtkFloatArray()
    velocity.SetNumberOfComponents(3)
    velocity.SetName('u')

    move_forward = True
    count = 0

    for rank in range(1):  # Adjust if you have multiple ranks
        try:
            with h5py.File(f'Result/pos{N}rank{rank}.h5', 'r') as f_pos:
                points_array = f_pos['pos'][:]
            with h5py.File(f'Result/vel{N}rank{rank}.h5', 'r') as f_vel:
                velocity_array = f_vel['vel'][:]
        except Exception as e:
            print(f"Error reading file for N={N}, rank={rank}: {e}")
            move_forward = False
            break

        # Insert points and velocity data
        for i in range(points_array.shape[0]):
            points.InsertNextPoint(points_array[i, :])
            velocity.InsertNextTuple3(*velocity_array[i])

        count += points_array.shape[0]

    if not move_forward:
        break

    # Create polydata
    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)
    polydata.GetPointData().SetVectors(velocity)

    # Create line connectivity
    lines = vtk.vtkCellArray()
    n_chain = N_CHAIN  # Particles per filament
    num_chains = points_array.shape[0] // n_chain  # Number of filaments

    if num_chains > 0:
        for n in range(num_chains):
            line = vtk.vtkPolyLine()
            line.GetPointIds().SetNumberOfIds(n_chain)

            for i in range(n_chain):
                global_index = n * n_chain + i  
                line.GetPointIds().SetId(i, global_index)

            lines.InsertNextCell(line)

        polydata.SetLines(lines)
    else:
        print(f"⚠ Warning: No lines created for N={N}")

    # Debugging output
    print(f"Frame {N}: Points = {polydata.GetNumberOfPoints()}, Lines = {polydata.GetNumberOfLines()}")

    # Write to VTP file
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetInputData(polydata)
    writer.SetFileName(f'Result/w_multi_filament{N}.vtp')
    writer.Write()