#!/bin/bash
#SBATCH --job-name=sphere_filament           # Job name
#SBATCH --output=sphere_filament.out       # Output file (%j will be replaced with job ID)
#SBATCH --error=sphere_filament.err        # Error file (%j will be replaced with job ID)
#SBATCH --ntasks=1                       # Number of tasks (processes)
#SBATCH --cpus-per-task=40                # Number of CPU cores per task
#SBATCH --gres=gpu:2                     # Request 1 GPU
#SBATCH --mem=16G                        # Memory per node
#SBATCH --time=72:00:00                  # Time limit (hh:mm:ss)
#SBATCH --partition=hignn         # Partition to submit to

DOCKER_IMAGE="hignn:latest"
DOCKER_CMD="mpirun -n 2 python3 python/multiple_filament.py"

srun --gres=gpu:2 --mem=15G docker run --gpus all --rm -v $PWD:/local -w /local --entrypoint /bin/bash --shm-size=4g -e TERM=xterm $DOCKER_IMAGE -c "$DOCKER_CMD"