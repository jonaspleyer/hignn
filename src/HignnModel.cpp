#include <execution>

#include "HignnModel.hpp"

using namespace std;
using namespace std::chrono;

/**
 * @brief Initialize MPI (if needed) and Kokkos runtime for parallel
 * computations.
 *
 * Sets up the computational environment for either CPU or GPU based on
 * compile-time flags. This is required before any parallel computation using
 * HignnModel.
 */
void Init() {
  int flag;

  MPI_Initialized(&flag);

  if (!flag)
    MPI_Init(NULL, NULL);

  int mpiRank;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank);

#ifdef USE_GPU
  int deviceCount;
  cudaGetDeviceCount(&deviceCount);

  auto settings = Kokkos::InitializationSettings()
                      .set_num_threads(10)
                      .set_device_id(mpiRank % deviceCount)
                      .set_disable_warnings(false);
#else
  auto settings =
      Kokkos::InitializationSettings().set_num_threads(10).set_disable_warnings(
          false);
#endif

  Kokkos::initialize(settings);
}

/**
 * @brief Finalize Kokkos runtime.
 */
void Finalize() {
  Kokkos::finalize();

  // MPI_Finalize();
}

/**
 * @brief Get the number of particles in the model.
 *
 * @return size_t The number of particles in the model.
 */
size_t HignnModel::GetCount() {
  return mCoordPtr->extent(0);
}

/**
 * @brief Computes the minimum and maximum values for each dimension (x, y, z)
 *        over a specified range of particle indices.
 *
 * This function iterates over the range of particles and updates the auxiliary
 * vector such that for each spatial dimension d, aux[2 * d] stores the minimum
 * and aux[2 * d + 1] stores the maximum coordinate value found in that
 * dimension across all particles in the specified range.
 *
 * @param first The index of the first particle in the range.
 * @param last The index one past the last particle in the range.
 * @param aux A vector of size (2 * mDim) which will be filled with the minimum
 *            and maximum values for each spatial dimension.
 */
void HignnModel::ComputeAux(const std::size_t first,
                            const std::size_t last,
                            std::vector<float> &aux) {
  auto &mVertexMirror = *mCoordMirrorPtr;

  for (int d = 0; d < mDim; d++) {
    aux[2 * d] = std::numeric_limits<float>::max();
    aux[2 * d + 1] = -std::numeric_limits<float>::max();
  }

  // Iterate over the specified range and update min/max.
  for (size_t i = first; i < last; i++)
    for (int d = 0; d < mDim; d++) {
      aux[2 * d] =
          (aux[2 * d] > mVertexMirror(i, d)) ? mVertexMirror(i, d) : aux[2 * d];
      aux[2 * d + 1] = (aux[2 * d + 1] < mVertexMirror(i, d))
                           ? mVertexMirror(i, d)
                           : aux[2 * d + 1];
    }
}

/**
 * @brief Divides particles and reorders them based on their spatial
 * coordinates over a specified range of particle indices.
 *
 * This function partitions the specified range of particle indices [first,
 * last) and reorders them based on spatial coordinates using principal
 * component analysis (PCA). The process computes the mean position, centers the
 * coordinates, performs SVD to find the dominant direction, and then sorts the
 * particles along this direction. The particles are thus grouped into clusters
 * for efficient processing. If parallelFlag is true, parts of the computation
 * are performed in parallel for performance.
 *
 * @param first [in] The index of the first particle in the range to consider.
 * @param last [in] The index one past the last particle in the range.
 * @param reorderedMap [in,out] A vector that holds the reordered particle
 * indices after processing. On output, it contains the new ordering for indices
 * in [first, last).
 * @param parallelFlag [in] If true, enables parallel computation where
 * possible.
 *
 * @return The index in reorderedMap where the division ends.
 */
size_t HignnModel::Divide(const std::size_t first,
                          const std::size_t last,
                          std::vector<std::size_t> &reorderedMap,
                          const bool parallelFlag) {
  auto &mVertexMirror = *mCoordMirrorPtr;

  const std::size_t L = last - first;

  std::vector<float> mean(mDim, 0.0);
  std::vector<float> temp(L);

  for (size_t i = first; i < last; i++)
    for (int d = 0; d < mDim; d++)
      mean[d] += mVertexMirror(reorderedMap[i], d);
  for (int d = 0; d < mDim; d++)
    mean[d] /= (float)L;

  Eigen::MatrixXf vertexHat(mDim, L);
  for (int d = 0; d < mDim; d++) {
    if (parallelFlag)
#pragma omp parallel for schedule(static, 1024)
      for (size_t i = 0; i < L; i++) {
        vertexHat(d, i) = mVertexMirror(reorderedMap[i + first], d) - mean[d];
      }
    else
      for (size_t i = 0; i < L; i++) {
        vertexHat(d, i) = mVertexMirror(reorderedMap[i + first], d) - mean[d];
      }
  }

  Eigen::JacobiSVD<Eigen::MatrixXf> svdHolder(
      vertexHat, Eigen::ComputeThinU | Eigen::ComputeThinV);
  Eigen::MatrixXf U = svdHolder.matrixU();

  for (size_t i = 0; i < L; i++) {
    temp[i] = 0.0;
    for (int d = 0; d < mDim; d++) {
      temp[i] += U(d, 0) * vertexHat(d, i);
    }
  }

  std::vector<std::size_t> newIndex;
  newIndex.resize(temp.size());
  iota(newIndex.begin(), newIndex.end(), 0);

  if (parallelFlag) {
    sort(std::execution::par_unseq, newIndex.begin(), newIndex.end(),
         [&temp](int i1, int i2) { return temp[i1] < temp[i2]; });
    sort(std::execution::par_unseq, temp.begin(), temp.end());
  } else {
    sort(newIndex.begin(), newIndex.end(),
         [&temp](int i1, int i2) { return temp[i1] < temp[i2]; });
    sort(temp.begin(), temp.end());
  }

  // Currently, the reordering is a very stupid implementation. Need to
  // improve it.

  std::vector<std::size_t> copyIndex(L);

  if (parallelFlag) {
#pragma omp parallel for schedule(static, 1024)
    for (size_t i = 0; i < L; i++)
      copyIndex[i] = reorderedMap[i + first];

#pragma omp parallel for schedule(static, 1024)
    for (size_t i = 0; i < L; i++)
      reorderedMap[i + first] = copyIndex[newIndex[i]];
  } else {
    for (size_t i = 0; i < L; i++)
      copyIndex[i] = reorderedMap[i + first];

    for (size_t i = 0; i < L; i++)
      reorderedMap[i + first] = copyIndex[newIndex[i]];
  }

  auto result = std::upper_bound(temp.begin(), temp.end(), 0);

  return (std::size_t)(result - temp.begin()) + first;
}

/**
 * @brief Reorders the particle coordinates in both host and device memory based
 * on the provided index mapping.
 *
 * The host-side coordinates are first copied to a temporary buffer, then
 * rearranged according to the order specified in reorderedMap. After
 * reordering, the i-th row of the coordinates corresponds to the particle
 * originally at index reorderedMap[i]. The device-side coordinates are then
 * updated to reflect this new order.
 *
 * @param reorderedMap A vector of size (num_particles) specifying the new order
 * of particle indices.
 */
void HignnModel::Reorder(const std::vector<std::size_t> &reorderedMap) {
  auto &mVertexMirror = *mCoordMirrorPtr;
  auto &mVertex = *mCoordPtr;

  // Create a temporary copy of current coordinates.
  HostFloatMatrix copyVertex;
  Kokkos::resize(copyVertex, reorderedMap.size(), 3);

  // Copy original coordinates to the temporary buffer.
  for (size_t i = 0; i < reorderedMap.size(); i++) {
    copyVertex(i, 0) = mVertexMirror(i, 0);
    copyVertex(i, 1) = mVertexMirror(i, 1);
    copyVertex(i, 2) = mVertexMirror(i, 2);
  }

  // Rearrange coordinates in mVertexMirror according to reorderedMap.
  for (size_t i = 0; i < reorderedMap.size(); i++) {
    mVertexMirror(i, 0) = copyVertex(reorderedMap[i], 0);
    mVertexMirror(i, 1) = copyVertex(reorderedMap[i], 1);
    mVertexMirror(i, 2) = copyVertex(reorderedMap[i], 2);
  }

  // Copy reordered coordinates from host to device
  Kokkos::deep_copy(mVertex, mVertexMirror);
}

/**
 * @brief Reorders the rows of the given device matrix v based on the provided
 * index mapping.
 *
 * Produces a reordered copy of the matrix v so that row i of the result is
 * taken from row reorderedMap[i] of the original matrix. The matrix v is
 * updated in place in device memory.
 *
 * @param reorderedMap A vector of size (num_particles) specifying the new order
 * for the rows of v.
 * @param v The device matrix of size (num_particles, 3) to be reordered in
 * place.
 */
void HignnModel::Reorder(const std::vector<size_t> &reorderedMap,
                         DeviceDoubleMatrix v) {
  // Create a device matrix for the reordered result
  DeviceDoubleMatrix vCopy("vCopy", v.extent(0), v.extent(1));

  // Transfer the reorderedMap to device memory
  DeviceIndexVector deviceReorderedMap("reorderedMap", reorderedMap.size());
  auto hostReorderedMap = Kokkos::create_mirror_view(deviceReorderedMap);

  for (size_t i = 0; i < reorderedMap.size(); i++)
    hostReorderedMap(i) = reorderedMap[i];

  Kokkos::deep_copy(deviceReorderedMap, hostReorderedMap);

  // Rearrange rows of v on device according to reorderedMap
  Kokkos::parallel_for(
      Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, v.extent(0)),
      KOKKOS_LAMBDA(const int i) {
        vCopy(i, 0) = v(deviceReorderedMap(i), 0);
        vCopy(i, 1) = v(deviceReorderedMap(i), 1);
        vCopy(i, 2) = v(deviceReorderedMap(i), 2);
      });

  // Copy reordered matrix back into original v
  Kokkos::deep_copy(v, vCopy);
}

/**
 * @brief Reverses the reordering of the rows of the given matrix v based on the
 * provided index mapping.
 *
 * This function restores the original row order of the matrix v (of size
 * [num_particles, 3]) using the mapping provided by reorderedMap (vector of
 * size num_nodes). The mapping is applied on the device using Kokkos
 * parallelism for efficient performance.
 *
 * @param reorderedMap A vector of size (num_particles) specifying the original
 * row order to restore in v.
 * @param v The device matrix of size (num_particles, 3) to be reordered back.
 * After the function, v will have its rows placed back into the original order.
 */
void HignnModel::BackwardReorder(const std::vector<size_t> &reorderedMap,
                                 DeviceDoubleMatrix v) {
  // Create a temporary copy of v with the same dimensions
  DeviceDoubleMatrix vCopy("vCopy", v.extent(0), v.extent(1));

  // Prepare a device vector storing the reordered indices
  DeviceIndexVector deviceReorderedMap("reorderedMap", reorderedMap.size());
  auto hostReorderedMap = Kokkos::create_mirror_view(deviceReorderedMap);

  // Copy the reorderedMap indices to the Kokkos host view
  for (size_t i = 0; i < reorderedMap.size(); i++)
    hostReorderedMap(i) = reorderedMap[i];

  // Copy the index mapping from host to device
  Kokkos::deep_copy(deviceReorderedMap, hostReorderedMap);

  // Perform the actual reordering in parallel on the device
  Kokkos::parallel_for(
      Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, v.extent(0)),
      KOKKOS_LAMBDA(const int i) {
        vCopy(deviceReorderedMap(i), 0) = v(i, 0);
        vCopy(deviceReorderedMap(i), 1) = v(i, 1);
        vCopy(deviceReorderedMap(i), 2) = v(i, 2);
      });

  // Copy the reordered result back into v
  Kokkos::deep_copy(v, vCopy);
}

/**
 * @brief Constructor for the HignnModel class.
 *
 * Initializes the model by setting default parameters and allocating memory for
 * the particle coordinates. Sets up the coordinate arrays on both host and
 * device, copies the input coordinates to internal storage, and initializes MPI
 * rank/size and other model parameters.
 *
 * @param coord A 2D numpy array (dimension: num_particles × 3) containing the
 * (x, y, z) positions of all particles.
 * @param blockSize The maximum number of particles allowed in a leaf node (a
 * cluster in the tree that is not further subdivided) during spatial division.
 */
HignnModel::HignnModel(pybind11::array_t<float> &coord, const int blockSize) {
  // default values
  mPostCheckFlag = false;
  mUseSymmetry = true;

  mMaxFarDotWorkNodeSize = 5000;

  mMaxRelativeCoord = 500000;

  mMaxFarFieldDistance = 1000;

  // Get shape and access data from numpy array (num_particles × 3)
  auto data = coord.unchecked<2>();
  auto shape = coord.shape();

  // Allocate device and host memory for particle coordinates
  mCoordPtr = std::make_shared<DeviceFloatMatrix>(
      DeviceFloatMatrix("mCoordPtr", (size_t)shape[0], (size_t)shape[1]));
  mCoordMirrorPtr = std::make_shared<DeviceFloatMatrix::HostMirror>();
  *mCoordMirrorPtr = Kokkos::create_mirror_view(*mCoordPtr);

  // Fill the host coordinates with data from the input numpy array
  auto &hostCoord = *mCoordMirrorPtr;

  for (size_t i = 0; i < (size_t)shape[0]; i++) {
    hostCoord(i, 0) = data(i, 0);
    hostCoord(i, 1) = data(i, 1);
    hostCoord(i, 2) = data(i, 2);
  }

  // Copy the initialized host coordinates to device memory
  Kokkos::deep_copy(*mCoordPtr, hostCoord);

  // Store user-specified block size and set model to 3D
  mBlockSize = blockSize;
  mDim = 3;

  MPI_Comm_rank(MPI_COMM_WORLD, &mMPIRank);
  MPI_Comm_size(MPI_COMM_WORLD, &mMPISize);

  // Initialize default solver/model parameters
  mEpsilon = 0.05;
  mEta = 1.0;
  mMaxIter = 100;
  mMatPoolSizeFactor = 40;

#if USE_GPU
  int deviceCount;
  cudaGetDeviceCount(&deviceCount);
  mCudaDevice = mMPIRank % deviceCount;
#endif
}

/**
 * @brief Loads a pre-trained two-body interaction model from the given file
 * path.
 *
 * Loads the model using TorchScript, selecting the appropriate device (CPU or
 * GPU) based on compile-time configuration. For GPU builds, appends the CUDA
 * device ID to the filename to support multi-GPU execution. After loading, the
 * model is moved to the selected device, and a test forward pass is performed
 * with a dummy input tensor of shape (50000, 3) to ensure the model is ready
 * for inference.
 *
 * @param modelPath The base name (without ".pt" extension) of the two-body
 * model file.
 */
void HignnModel::LoadTwoBodyModel(const std::string &modelPath) {
  // load script model
#if USE_GPU
  mTwoBodyModel =
      torch::jit::load(modelPath + "_" + std::to_string(mCudaDevice) + ".pt");
  mDeviceString = "cuda:" + std::to_string(mCudaDevice);
  mTwoBodyModel.to(mDeviceString);
#else
  mTwoBodyModel = torch::jit::load(modelPath + ".pt");
  mTwoBodyModel.to(torch::kCPU);
#endif

  // default options for torch. As no backward propagation is not required,
  // gradient is not required to be stored and requires_grad is set to false.
#if USE_GPU
  auto options = torch::TensorOptions()
                     .dtype(torch::kFloat32)
                     .device(torch::kCUDA, mCudaDevice)
                     .requires_grad(false);
#else
  auto options = torch::TensorOptions()
                     .dtype(torch::kFloat32)
                     .device(torch::kCPU)
                     .requires_grad(false);
#endif
  // Use a dummy input tensor for model warm-up with a forward pass
  torch::Tensor testTensor = torch::ones({50000, 3}, options);
  std::vector<c10::IValue> inputs;
  inputs.push_back(testTensor);

  auto testResult = mTwoBodyModel.forward(inputs);
}

/**
 * @brief Loads a pre-trained three-body interaction model from the given file
 * path.
 *
 * Currently, three body model is loaded on the python side as no acceleration
 * via C++/Kokkos when doing dot product w.r.t. three-body model.
 *
 * @param modelPath The path to the three-body model file (currently unused).
 */
void HignnModel::LoadThreeBodyModel([
    [maybe_unused]] const std::string &modelPath) {
}

/**
 * @brief Updates the model's state by rebuilding the cluster tree and updating
 * close/far pair information.
 *
 * This method should be called after the coordinates are changed. It rebuilds
 * the internal clustering structure (via Build), then identifies which node
 * pairs are close and which are far for subsequent computations (via
 * CloseFarCheck).
 */
void HignnModel::Update() {
  Build();

  CloseFarCheck();
}

/**
 * @brief Determines if two nodes are considered 'far' based on bounding boxes
 * and relative distances.
 *
 * Uses the domain of the bounding box of the node and a distance criterion to
 * decide if node1 and node2 should be treated as 'far' pairs, which enables the
 * use of matrix acceleration in subsequent computations. The function first
 * checks if the bounding boxes (defined by minimum and maximum coordinates) of
 * node1 and node2 are disjoint. If they are, the nodes are considered 'far'.
 * Otherwise, it further checks the relative distance and size of their bounding
 * boxes.
 *
 * @param aux      HostFloatMatrix of shape (num_particles, 6), where each row
 * contains the min and max coordinates for x, y, z: [xmin, xmax, ymin, ymax,
 * zmin, zmax].
 * @param node1    Index of the first node (row in aux).
 * @param node2    Index of the second node (row in aux).
 * @return         true if the nodes are considered 'far', false otherwise.
 */
bool HignnModel::CloseFarCheck(HostFloatMatrix aux,
                               const std::size_t node1,
                               const std::size_t node2) {
  float diam0 = 0.0;
  float diam1 = 0.0;
  float dist = 0.0;
  float tmp = 0.0;

  bool isFar = false;
  // AABB bounding box intersection check
  if (aux(node1, 0) > aux(node2, 1) || aux(node1, 1) < aux(node2, 0) ||
      aux(node1, 2) > aux(node2, 3) || aux(node1, 3) < aux(node2, 2) ||
      aux(node1, 4) > aux(node2, 5) || aux(node1, 5) < aux(node2, 4))
    isFar = true;
  else
    return false;

  // Compute squared diameters and center distance for bounding boxes
  for (int j = 0; j < 3; j++) {
    tmp = aux(node1, 2 * j) - aux(node1, 2 * j + 1);
    diam0 += tmp * tmp;
    tmp = aux(node2, 2 * j) - aux(node2, 2 * j + 1);
    diam1 += tmp * tmp;
    tmp = aux(node1, 2 * j) + aux(node1, 2 * j + 1) - aux(node2, 2 * j) -
          aux(node2, 2 * j + 1);
    dist += tmp * tmp;
  }

  dist *= 0.25;
  if ((dist > diam0) && (dist > diam1)) {
    isFar = true;
  } else {
    isFar = false;
  }

  return isFar;
}

/**
 * @brief Computes hydrodynamic interaction and update the velocities from the
 * given forces using hierarchical matrix acceleration to the mobility tensor.
 *
 * This function calculates the particle velocities (mobility problem) by
 * applying the hierarchical matrix acceleration when calculating the product
 * between the mobility tensor and the input forces. It supports distributed
 * execution based on MPI and optionally runs a post-processing check. The
 * function performs the following steps:
 *   - Initializes local velocity and force arrays (both with dimensions
 * [num_particles, 3]).
 *   - Copies force data from Python array into device memory.
 *   - Reorders the force array aligning with the node ordering of the
 * clustering tree.
 *   - Computes the close- and far-field contributions to the velocities.
 *   - Collects and sums the results across all MPI ranks.
 *   - Optionally verifies the result if post-check is enabled.
 *   - Restores the velocity array to the original ordering.
 *   - Copies the computed velocities back to the output Python array.
 *
 * @param uArray [out] Output array (num_particles, 3) to be filled with
 * computed velocities for each particle.
 * @param fArray [in]  Input array (num_particles, 3) of forces acting on each
 * particle.
 */
void HignnModel::Dot(pybind11::array_t<float> &uArray,
                     pybind11::array_t<float> &fArray) {
  if (mMPIRank == 0)
    std::cout << "start of Dot" << std::endl;

  std::chrono::high_resolution_clock::time_point t1 =
      std::chrono::high_resolution_clock::now();

  auto shape = fArray.shape();

  DeviceDoubleMatrix u("u", shape[0], 3);
  DeviceDoubleMatrix f("f", shape[0], 3);

  // initialize u
  Kokkos::parallel_for(
      Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, u.extent(0)),
      KOKKOS_LAMBDA(const int i) {
        u(i, 0) = 0.0;
        u(i, 1) = 0.0;
        u(i, 2) = 0.0;
      });
  Kokkos::fence();

  // Access data from Python arrays
  auto fData = fArray.unchecked<2>();
  auto uData = uArray.mutable_unchecked<2>();

  // Host mirror for force array
  DeviceDoubleMatrix::HostMirror hostF = Kokkos::create_mirror_view(f);

  // Copy force data from Python array to host mirror
  Kokkos::parallel_for(
      Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0, f.extent(0)),
      [&](const int i) {
        hostF(i, 0) = fData(i, 0);
        hostF(i, 1) = fData(i, 1);
        hostF(i, 2) = fData(i, 2);
      });
  Kokkos::fence();

  // Transfer force data to device memory
  Kokkos::deep_copy(f, hostF);

  // Reorders the force array aligning with the node ordering of the clustering
  // tree.
  Reorder(mReorderedMap, f);

  // Compute close- and far-range velocity contributions
  CloseDot(u, f);
  FarDot(u, f);

  // Copy result velocities back to host
  DeviceDoubleMatrix::HostMirror hostU = Kokkos::create_mirror_view(u);

  Kokkos::deep_copy(hostU, u);

  // Perform an all-reduce to sum the velocity results from all MPI ranks
  MPI_Allreduce(MPI_IN_PLACE, hostU.data(), u.extent(0) * u.extent(1),
                MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

  Kokkos::deep_copy(u, hostU);

  // Optional post-processing check for accuracy
  if (mPostCheckFlag) {
    PostCheckDot(u, f);
  }

  // Restores the velocity array to the original ordering
  BackwardReorder(mReorderedMap, u);

  Kokkos::deep_copy(hostU, u);

  // Copy final velocities to Python output array
  Kokkos::parallel_for(
      Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0, u.extent(0)),
      [&](const int i) {
        uData(i, 0) = hostU(i, 0);
        uData(i, 1) = hostU(i, 1);
        uData(i, 2) = hostU(i, 2);
      });
  Kokkos::fence();

  std::chrono::high_resolution_clock::time_point t2 =
      std::chrono::high_resolution_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();

  if (mMPIRank == 0)
    printf("End of Dot. Dot time: %.4fs\n", (double)duration / 1e6);
}

/**
 * @brief Computes hydrodynamic interaction and update the velocities from the
 * given forces using the dense mobility tensor.
 *
 * This function applies the dense (without hierarchical matrix acceleration)
 * mobility tensor to the input force array to compute the resulting velocities.
 * The result is aggregated across all MPI ranks.
 *
 * Workflow:
 * - Allocates velocity and force arrays (each of shape [num_particles, 3]).
 * - Initializes the velocity array to zero.
 * - Copies the input force array into device memory.
 * - Reorders the force array aligning with the node ordering of the
 * clustering tree.
 * - Calls DenseDot to compute velocities.
 * - Collects the result across MPI ranks using all-reduce.
 * - Restores the velocity array to the original ordering.
 * - Copies the result back into the output Python array.
 *
 * @param uArray [out] The computed velocities, numpy array of shape
 * (num_particles, 3).
 * @param fArray [in]  The input forces, numpy array of shape (num_particles,
 * 3).
 */
void HignnModel::DenseDot(pybind11::array_t<float> &uArray,
                          pybind11::array_t<float> &fArray) {
  auto shape = fArray.shape();

  DeviceDoubleMatrix u("u", shape[0], 3);
  DeviceDoubleMatrix f("f", shape[0], 3);

  // initialize u
  Kokkos::parallel_for(
      Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, u.extent(0)),
      KOKKOS_LAMBDA(const int i) {
        u(i, 0) = 0.0;
        u(i, 1) = 0.0;
        u(i, 2) = 0.0;
      });
  Kokkos::fence();

  // Access data from Python arrays
  auto fData = fArray.unchecked<2>();
  auto uData = uArray.mutable_unchecked<2>();

  // Host mirror for force array
  DeviceDoubleMatrix::HostMirror hostF = Kokkos::create_mirror_view(f);

  Kokkos::parallel_for(
      Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0, f.extent(0)),
      [&](const int i) {
        hostF(i, 0) = fData(i, 0);
        hostF(i, 1) = fData(i, 1);
        hostF(i, 2) = fData(i, 2);
      });
  Kokkos::fence();

  // Transfer force data to device memory
  Kokkos::deep_copy(f, hostF);

  // Reorders the force array aligning with the node ordering of the clustering
  // tree.
  Reorder(mReorderedMap, f);

  // Compute velocities using the dense mobility tensor (on device)
  DenseDot(u, f);

  // Copy velocities to host for MPI reduction
  DeviceDoubleMatrix::HostMirror hostU = Kokkos::create_mirror_view(u);

  Kokkos::deep_copy(hostU, u);

  // Perform an all-reduce to sum the velocity results from all MPI ranks
  MPI_Allreduce(MPI_IN_PLACE, hostU.data(), u.extent(0) * u.extent(1),
                MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

  Kokkos::deep_copy(u, hostU);

  // Restores the velocity array to the original ordering
  BackwardReorder(mReorderedMap, u);

  Kokkos::deep_copy(hostU, u);

  // Copy result velocities to output Python array
  Kokkos::parallel_for(
      Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0, u.extent(0)),
      [&](const int i) {
        uData(i, 0) = hostU(i, 0);
        uData(i, 1) = hostU(i, 1);
        uData(i, 2) = hostU(i, 2);
      });
  Kokkos::fence();
}

/**
 * @brief Update the internal coordinates with new particle positions and
 * trigger model update.
 *
 * This function copies the provided coordinate data (shape: num_particles × 3)
 * into the internal host-side coordinate array, then transfers the data to the
 * device-side storage. After updating the coordinates, it calls Update() to
 * rebuild the clustering tree and update all structures dependent on the
 * particle positions.
 *
 * @param coord [in] 2D numpy array of shape (num_particles, 3) containing  the
 * new coordinates for the particles Note: Currently, only nDim = 3 is supported
 * in this implementation.
 */
void HignnModel::UpdateCoord(pybind11::array_t<float> &coord) {
  auto data = coord.unchecked<2>();

  auto shape = coord.shape();

  auto &hostCoord = *mCoordMirrorPtr;

  // Copy each (x, y, z) position from input to host coordinate storage
  for (size_t i = 0; i < (size_t)shape[0]; i++) {
    hostCoord(i, 0) = data(i, 0);
    hostCoord(i, 1) = data(i, 1);
    hostCoord(i, 2) = data(i, 2);
  }

  // Copy updated coordinates from host to device memory
  Kokkos::deep_copy(*mCoordPtr, hostCoord);

  Update();
}

/**
 * @brief Sets the value of epsilon used by the adaptive cross approximation
 *
 * This function sets the value of the internal `mEpsilon` variable, which
 * controls the convergence criteria of the adaptive cross approximation used
 * in function FarDot.
 *
 * @param epsilon [in] The value to set for epsilon.
 */
void HignnModel::SetEpsilon(const double epsilon) {
  mEpsilon = epsilon;
}

/**
 * @brief Set the eta parameter used in the clustering tree.
 *
 * This updates the mEta variable, affecting how close/far pairs are determined
 * in the clustering tree.
 *
 * @param eta The value to set for eta.
 */
void HignnModel::SetEta(const double eta) {
  mEta = eta;
}

/**
 * @brief Set the maximum number of iterations for the adaptive cross
 * approximation.
 *
 * This updates mMaxIter, controlling how many iterations are performed by the
 * algorithm.
 *
 * @param maxIter The maximum number of iterations to perform.
 */
void HignnModel::SetMaxIter(const int maxIter) {
  mMaxIter = maxIter;
}

/**
 * @brief Set the factor that determines the size of the matrix pool.
 *
 * This updates mMatPoolSizeFactor, which is used to preallocate memory for
 * FarDot and optimize performance.
 *
 * @param factor The matrix pool size factor.
 */
void HignnModel::SetMatPoolSizeFactor(const int factor) {
  mMatPoolSizeFactor = factor;
}

/**
 * @brief Enable or disable post-check operations.
 *
 * Sets the mPostCheckFlag variable, which controls whether post-checking is
 * performed after computation.
 *
 * @param flag Boolean flag: true to enable post-checking, false to disable.
 */
void HignnModel::SetPostCheckFlag(const bool flag) {
  mPostCheckFlag = flag;
}

/**
 * @brief Enable or disable symmetry in the model.
 *
 * Sets the mUseSymmetry variable, allowing the algorithm to optimize
 * calculations if symmetry is present.
 *
 * @param flag Boolean flag: true to use symmetry, false to disable.
 */
void HignnModel::SetUseSymmetryFlag(const bool flag) {
  mUseSymmetry = flag;
}

/**
 * @brief Set the maximum number of node pairs for far-range interactions.
 *
 * Updates mMaxFarDotWorkNodeSize, which controls the upper limit of node pairs
 * for FarDot computations.
 *
 * @param size The maximum number of node pairs that can work simultaneously.
 */
void HignnModel::SetMaxFarDotWorkNodeSize(const int size) {
  mMaxFarDotWorkNodeSize = size;
}

/**
 * @brief Set the maximum number of relative coordinates that can be passed
 * forward to the two-body model.
 *
 * Sets mMaxRelativeCoord, which determines the upper bound for storage and
 * queries in CloseDot and FarDot.
 *
 * @param size The maximum number of relative coordinates that can be used in
 * single forward pass when querying the two-body model.
 */
void HignnModel::SetMaxRelativeCoord(const size_t size) {
  mMaxRelativeCoord = size;
}

/**
 * @brief Set the far-field cut-off distance for interactions (not used
 * anymore).
 *
 * Updates mMaxFarFieldDistance, the cut-off for far-range interactions.
 *
 * @param distance The cut-off distance.
 */
void HignnModel::SetMaxFarFieldDistance(const double distance) {
  mMaxFarFieldDistance = distance;
}