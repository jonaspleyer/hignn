import json
import numpy as np
import vtk
import os
import h5py

# Load cloud parameters from JSON
working_directory = "python/filament_particle_cloud"
with open(f"{working_directory}/config.json", "r") as f:
    config = json.load(f)

simulation_params = config["simulation"]
t_max = simulation_params["t_max"]
t_meas = simulation_params["t_meas"]

cloud_params = config['cloud']
cloud_type = cloud_params['type']
particle_cloud_params = cloud_params['particle']
filament_cloud_params = cloud_params['filament']

if particle_cloud_params:
    n_particle = particle_cloud_params['n_particle']
else:
    n_particle = 0

if filament_cloud_params:
    n_filament = filament_cloud_params['n_filament']
    n_chain = filament_cloud_params['n_chain']
    rest_length = filament_cloud_params['rest_length']
    k_t = filament_cloud_params['k_t']
    k_b = filament_cloud_params['k_b']
    len_fiber = n_filament * (n_chain - 1)
else:
    n_filament = n_chain = rest_length = k_t = k_b = 0

print(f"Converting HDF5 files to VTP files..")

# Function to write .vtp file
def write_vtp(filename, points_data, velocity_data, lines_data=None, vertex_indices=None):
    points = vtk.vtkPoints()
    for p in points_data:
        points.InsertNextPoint(p)

    velocity = vtk.vtkFloatArray()
    velocity.SetNumberOfComponents(3)
    for v in velocity_data:
        velocity.InsertNextTuple3(*v)
    velocity.SetName('u')

    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)
    polydata.GetPointData().SetVectors(velocity)

    if vertex_indices is not None:
        vertices = vtk.vtkCellArray()
        for i in vertex_indices:
            vertices.InsertNextCell(1)
            vertices.InsertCellPoint(i)
        polydata.SetVerts(vertices)

    if lines_data is not None:
        polydata.SetLines(lines_data)

    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetInputData(polydata)
    writer.SetFileName(filename)
    writer.Write()

# Total number of steps
n_total = int(t_max / t_meas)
for N in range(n_total):
    move_forward = True
    points_array = None
    velocity_array = None

    # Load HDF5 data from all MPI ranks
    for rank in range(1):
        try:
            points_array_rank = h5py.File(f'{working_directory}/output/hdf5/pos_rank_{rank}_{N}.h5', 'r')['pos'][:]
            velocity_array_rank = h5py.File(f'{working_directory}/output/hdf5/vel_rank_{rank}_{N}.h5', 'r')['vel'][:]
            if points_array is None:
                points_array = points_array_rank
                velocity_array = velocity_array_rank
            else:
                points_array = np.concatenate((points_array, points_array_rank), axis=0)
                velocity_array = np.concatenate((velocity_array, velocity_array_rank), axis=0)
        except:
            move_forward = False
            break

    if not move_forward:
        break

    # Split data into filament and particle parts
    filament_end = n_filament * n_chain
    filament_points = points_array[:filament_end]
    filament_velocity = velocity_array[:filament_end]
    particle_points = points_array[filament_end:]
    particle_velocity = velocity_array[filament_end:]

    # Write particle-only file
    if particle_cloud_params:
        particle_vertex_indices = list(range(particle_points.shape[0]))
        write_vtp(f'{working_directory}/output/N_{n_particle}_{N}.vtp',
                  particle_points, particle_velocity,
                  vertex_indices=particle_vertex_indices)

    # Write filament-only file
    if filament_cloud_params:
        lines = vtk.vtkCellArray()
        for n in range(n_filament):
            line = vtk.vtkPolyLine()
            line.GetPointIds().SetNumberOfIds(n_chain)
            for i in range(n_chain):
                line.GetPointIds().SetId(i, i + n * n_chain)
            lines.InsertNextCell(line)

        write_vtp(f'{working_directory}/output/filament_only_N_{n_filament}_{N}.vtp',
                  filament_points, filament_velocity,
                  lines_data=lines)

    # Progress bar
    progress = (N + 1) / n_total
    bar_length = 50
    filled_length = int(bar_length * progress)
    bar = '█' * filled_length + ' ' * (bar_length - filled_length)
    print(f"\rConvert status: |{bar}| {N+1}/{n_total} files ({progress*100:.2f}%)", end='', flush=True)

print("\nSuccessfully converted files!")
