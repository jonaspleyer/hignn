# Prerequisites
[Nvidia driver](https://www.nvidia.com/en-us/drivers/) (currently, it requires >= 11.8), [docker engine](https://docs.docker.com/engine/install/) and [Nvidia container toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

# Prepare the docker environment
One can build the docker image locally
```shell
cd script
docker build --rm -f Dockerfile.hignn.Ampreme -t hignn .
```
or pull the image from the docker hub. By default, it assumes the Ampreme architecture
```shell
docker pull zishengy/hignn:latest
```
On Ada Lovelace architecture
```shell
docker pull zishengy/hignn:adalovelace
```
If using CPU only, one can pull the image via
```
docker pull zishengy/hignn:cpu
```

# Initialize
Rename the image with
```shell
docker image tag zishengy/hignn:{ARCH} hignn
```
where {ARCH} can be latest/adalovelace/cpu.

On Linux
```shell
docker run --privileged -it --rm -v $PWD:/local -w /local --entrypoint /bin/bash --gpus=all --shm-size=4g --hostname hignn hignn
```
On Windows
```shell
docker run --privileged -it --rm -v ${PWD}:/local -w /local --entrypoint /bin/bash --gpus=all --shm-size=4g --hostname hignn hignn
```
If using cpu only, please use
```shell
docker run --privileged -it --rm -v $PWD:/local -w /local --entrypoint /bin/bash --shm-size=4g --hostname hignn hignn
```

# Compile the code

After entering the docker, you can run the Python script at the root directory as follows:
```shell
python3 python/compile.py --rebuild
```

The above command only need once, the argument ``--rebuild`` is no more needed after the first time.  Also, re-entering to the docker environment won't need to compile the code again if the code is unchanged.

# Run the code

```shell
python3 python/engine.py $config_file $new_cmd
```

One can select ``$new_cmd`` from one of ``--generate``, ``--simulate`` and ``--visualize``.