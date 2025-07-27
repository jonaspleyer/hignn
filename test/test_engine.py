import numpy as np
import subprocess
import pytest
import json
import shutil
import os

from container import get_cmd, get_num_device

# Test that engine.py fails to run without a config file
def test_cant_run_without_config():
    test_cmd = "python3 python/engine.py --generate"
    result = subprocess.run(
        get_cmd(test_cmd),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        shell=True
    )
    print("\nStderr:", result.stderr)

    assert result.returncode !=0
    
# Test that cloud generation works correctly
def test_can_generate_correctly():
    test_cmd = "python3 python/engine.py python/config/config_template.json --generate"
    process = subprocess.Popen(
        get_cmd(test_cmd),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        shell=True
    )
    
    for line in process.stdout:
        print(line, end="") 
    
    process.stdout.close()
    process.wait()
    print("\nStderr:", process.stderr)
        
    # Parse config
    with open("python/config/config_template.json", "r") as f:
        config = json.load(f)

    cloud_params = config['cloud']
    filament_cloud_params = cloud_params.get('filament')
    particle_cloud_params = cloud_params.get('particle')

    n_filament = filament_cloud_params.get('n_filament', 0) if filament_cloud_params else 0
    n_chain = filament_cloud_params.get('n_chain', 0) if filament_cloud_params else 0
    n_particle = particle_cloud_params.get('n_particle', 0) if particle_cloud_params else 0

    expected_count = n_filament * n_chain + n_particle

    # Load generated .pos file
    positions = np.loadtxt("python/cloud/filament_particle_cloud.pos")

    assert positions.shape[0] == expected_count
    
# Test that simulation works correctly
def test_can_simulate_correctly():
    test_cmd = "python3 python/engine.py python/config/config_template.json --simulate"
    process = subprocess.Popen(
        get_cmd(test_cmd),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        shell=True
    )
    
    for line in process.stdout:
        print(line, end="") 
         
    process.stdout.close()
    process.wait()
    print("\nStderr:", process.stderr)
        
    assert process.returncode == 0

# Test that simulation works correctly
def test_can_simulate_parallel_correctly():
    num_device = get_num_device()
    test_cmd = "mpirun -n " + str(num_device) + " python3 python/engine.py python/config/config_template.json --simulate"
    process = subprocess.Popen(
        get_cmd(test_cmd),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        shell=True
    )
    
    for line in process.stdout:
        print(line, end="") 
         
    process.stdout.close()
    process.wait()
    print("\nStderr:", process.stderr)
        
    assert process.returncode == 0
    
# Test that visualization works correctly
def test_can_visualize_correctly():
    test_cmd = "python3 python/engine.py python/config/config_template.json --visualize"
    process = subprocess.run(
        get_cmd(test_cmd),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        shell=True
    )
    
    print(process.stdout, end="")
    print("\nStderr:", process.stderr)
    
    assert process.returncode == 0

test_cant_run_without_config()
test_can_generate_correctly()
test_can_simulate_correctly()
test_can_simulate_parallel_correctly()
test_can_visualize_correctly()