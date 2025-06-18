import subprocess
import os

def running_in_docker():
    try:
        with open("/proc/1/cgroup", "rt") as f:
            content = f.read()
        return "docker" in content or "containerd" in content
    except FileNotFoundError:
        return False

def image_exists_locally(image_name):
    try:
        result = subprocess.run(
            ["docker", "image", "inspect", image_name],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return True
    except FileNotFoundError:
        raise False

def get_cmd(cmd):
    result = running_in_docker()
    if result:
        return cmd
    
    result = image_exists_locally("panlabuwmadison/hignn:latest")
    if result:
        prefix_cmd = "docker run --privileged -i --rm -v $PWD:/local -w /local --entrypoint /bin/bash --gpus=all --shm-size=4g panlabuwmadison/hignn:latest -c "
        return prefix_cmd + "\"" + cmd + "\""
    
    result = image_exists_locally("panlabuwmadison/hignn:amprere")
    if result:
        prefix_cmd = "docker run --privileged -i --rm -v $PWD:/local -w /local --entrypoint /bin/bash --gpus=all --shm-size=4g panlabuwmadison/hignn:amprere -c "
        return prefix_cmd + "\"" + cmd + "\""
    
    result = image_exists_locally("panlabuwmadison/hignn:adalovelace")
    if result:
        prefix_cmd = "docker run --privileged -i --rm -v $PWD:/local -w /local --entrypoint /bin/bash --gpus=all --shm-size=4g panlabuwmadison/hignn:adalovelace -c "
        return prefix_cmd + "\"" + cmd + "\""
    
    result = image_exists_locally("panlabuwmadison/hignn:cpu")
    if result:
        prefix_cmd = "docker run --privileged -i --rm -v $PWD:/local -w /local --entrypoint /bin/bash --shm-size=4g panlabuwmadison/hignn:cpu -c "
        return prefix_cmd + "\"" + cmd + "\""
    
    raise False

def get_num_device():
    try:
        result = subprocess.run(["nvidia-smi", "-L"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode == 0 and result.stdout.strip():
            gpus = result.stdout.strip().split('\n')
            return len(gpus)
    except FileNotFoundError:
        pass  # nvidia-smi not found

    cpu_count = os.cpu_count()
    return cpu_count