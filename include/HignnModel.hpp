#ifndef _HignnModel_Hpp_
#define _HignnModel_Hpp_

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <future>
#include <iostream>
#include <memory>
#include <queue>
#include <stack>
#include <vector>
#include <string>

#include <torch/script.h>

using namespace std::chrono;

#include <Eigen/Core>
#include <Eigen/LU>
#include <Eigen/SVD>

#include <mpi.h>

#include "Typedef.hpp"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

namespace py = pybind11;

void Init();

void Finalize();

/**
 * @class HignnModel
 * @brief Class that represents the hydrodynamic interaction graph neural
 * network (HIGNN) model.
 *
 * This class manages the coordinates, cluster tree, close/far range
 * interactions, and other settings for running the HIGNN model. It provides
 * methods for building the model, updating the coordinates, and performing
 * various dot operations for calculating particle interactions and velocities.
 */

class HignnModel {
protected:
  std::shared_ptr<DeviceFloatMatrix> mCoordPtr;
  //!< A shared pointer to the matrix storing particle coordinates on the device
  std::shared_ptr<DeviceFloatMatrix::HostMirror> mCoordMirrorPtr;
  //!< A shared pointer to the host-side mirror of the particle coordinates
  //!< matrix.

  std::shared_ptr<DeviceIndexMatrix> mClusterTreePtr;
  //!< A shared pointer to the cluster tree matrix on the device
  std::shared_ptr<DeviceIndexMatrix::HostMirror> mClusterTreeMirrorPtr;
  //!< A shared pointer to the host-side mirror of the cluster tree matrix.

  std::shared_ptr<DeviceIndexVector> mCloseMatIPtr;
  //!< Pointer to device matrix for close-range interaction indices (I).
  std::shared_ptr<DeviceIndexVector> mCloseMatJPtr;
  //!< Pointer to device matrix for close-range interaction indices (J).

  std::shared_ptr<DeviceIndexVector> mFarMatIPtr;
  //!< Pointer to device matrix for far-range interaction indices (I).
  std::shared_ptr<DeviceIndexVector> mFarMatJPtr;
  //!< Pointer to device matrix for far-range interaction indices (J).

  std::shared_ptr<DeviceIndexVector> mLeafNodePtr;
  //!< A shared pointer to the device vector storing leaf node indices.

  std::vector<std::size_t> mLeafNodeList;
  //!< List of all leaf nodes.

  std::vector<std::size_t> mReorderedMap;
  //!< A vector holding the reordered node indices, used for
  //!< optimized processing of nodes.

  unsigned int mBlockSize;
  //!< Defines the number of nodes processed together in each batch during
  //!< computations.
  int mDim;
  //!< Dimensionality of the system. 3 for (x, y, z); 6 for (x, y, z, rx, ry,
  //!< rz)

  int mMPIRank;
  int mMPISize;
#if USE_GPU
  int mCudaDevice;
#endif

  double mEpsilon;
  //!< A small value used for numerical stability in calculations.
  double mEta;
  //!< A parameter related to the algorithm's convergence behavior.

  double mMaxFarFieldDistance;
  //!< Maximum distance considered for far-field interactions.

  int mMaxIter;
  //!< Maximum number of iterations for certain operations.

  int mMatPoolSizeFactor;
  //!< Factor controlling the matrix pool size.

  int mMaxFarDotWorkNodeSize;
  //!< Maximum number of nodes to be processed at once in the far dot product.
  int mMaxCloseDotBlockSize;
  //!< Maximum block size for close-range dot product calculations.

  size_t mMaxRelativeCoord;
  //!< Maximum size for storing relative coordinates.

  std::size_t mClusterTreeSize;
  torch::jit::script::Module mTwoBodyModel;
  //!< The two-body interaction model loaded using Torch model.

#if USE_GPU
  std::string mDeviceString;
  //!< String representing the GPU device configuration (if using GPU). This is
  //!< used for loading the right Torch model. When the model is converted by
  //!< python/convert.py, it has already decided the working device. Therefore,
  //!< it has to be converted onto different devices for ensuring load-balance.
#endif

  bool mPostCheckFlag;
  //!< Flag to enable/disable post-checking after computations.

  bool mUseSymmetry;
  //!< Flag to enable/disable the use of symmetry in calculations.

protected:
  /**
   * @brief Returns the total count of elements in the coordinate matrix.
   * @return Size of the coordinate matrix (num_nodes).
   */
  std::size_t GetCount();

  /**
   * @brief Computes the minimum and maximum values for each dimension (x, y, z)
   * over a specified range of particle indices.
   *
   * This function iterates over a specified range of particles and updates
   * the auxiliary array to store the minimum and maximum values for each
   * dimension (x, y, z).
   *
   * @param first The index of the first particle in the range.
   * @param last The index of the last particle in the range.
   * @param aux A vector to store the minimum and maximum values for each
   * dimension. The size of this vector will be (2 * mDim) (min and max for each
   * dimension).
   */
  void ComputeAux(const std::size_t first,
                  const std::size_t last,
                  std::vector<float> &aux);

  /**
   * @brief Divides a range of particles and reorders them based on their
   * spatial coordinates.
   *
   * This function partitions the given range of particle indices (`first` to
   * `last`) and reorders them based on their spatial coordinates. The particles
   * are grouped into clusters for efficient processing. This function also
   * provides an option to perform the computation in parallel by setting the
   * `parallelFlag` to `true`.
   *
   * @param first [in] The index of the first particle in the range.
   * @param last [in] The index of the last particle in the range.
   * @param reorderedMap [in/out] A vector that holds the reordered particle
   * indices after processing. The size of this vector will be the number of
   * particles in the range.
   * @param parallelFlag [in] A flag indicating whether to perform the
   * computation in parallel.
   *
   * @return The index in the reordered map where the division ends.
   */
  std::size_t Divide(const std::size_t first,
                     const std::size_t last,
                     std::vector<std::size_t> &reorderedMap,
                     const bool parallelFlag);

  /**
   * @brief Reorders the particle coordinates based on the specified index map.
   *
   * This function uses the input vector `reorderedMap`, which contains indices
   * that define the new order of the particles. It rearranges the particle
   * coordinates in both the mirrored host and device representations according
   * to the order specified in this map.
   *
   * @param reorderedMap [in] A vector of size (num_nodes) specifying the new
   * order of particle indices for reordering the coordinates.
   */
  void Reorder(const std::vector<std::size_t> &reorderedMap);

public:
  /**
   * @brief Constructor for the HignnModel class.
   *
   * Initializes the model with the provided particle coordinates and sets the
   * batch size for computations. The constructor allocates memory for particle
   * coordinates and sets up necessary configurations.
   *
   * @param coord A 2D numpy array containing the coordinates of the particles.
   * The dimensions are (num_nodes, 3), where each row represents the (x, y, z)
   * position of a particle.
   * @param blockSize The number of maximum number of particles of the leaf
   * nodes when doing division of particles.
   */
  HignnModel(pybind11::array_t<float> &coord, const int blockSize);

  /**
   * @brief Loads a pre-trained two-body interaction model from the specified
   * file path.
   *
   * This function loads a model to the appropriate device (CPU or GPU) based on
   * the system configuration. It initializes the model for inference by
   * performing a test forward pass.
   *
   * @param modelPath The base name of the model file, excluding the ".pt"
   * extension.
   */
  void LoadTwoBodyModel(const std::string &modelPath);

  /**
   * @brief Loads a pre-trained three-body interaction model from the specified
   * file path.
   *
   * This function is currently a placeholder and does not utilize the provided
   * model path. It is intended for future implementation of loading a
   * three-body model.
   *
   * @param modelPath The path to the model file. Currently unused.
   */
  void LoadThreeBodyModel(const std::string &modelPath);

  /**
   * @brief Updates the model's state by invoking internal update operations.
   * This is called when the coords are updated by default.
   */
  void Update();

  /**
   * @brief Builds the clustering tree.
   */
  void Build();

  /**
   * @brief Determines if two particles are considered `far` based on their
   * bounding boxes and relative distances.
   *
   * This function evaluates whether two sets of nodes (particles) are
   * sufficiently distant from each other by checking their bounding boxes and
   * calculating the relative distance between them. It returns true if the
   * particles are considered far; false otherwise.
   *
   * @param aux A matrix containing auxiliary data used to compute distances and
   * diameters. Size: [num_nodes, 6], where each node constains the the minimum
   * and maximum values for each dimension.
   * @param node1 The index of the first node in the check.
   * @param node2 The index of the second node in the check.
   * @return Returns true if the sets of particles are considered `far`; false
   * otherwise.
   */
  bool CloseFarCheck(HostFloatMatrix aux,
                     const std::size_t node1,
                     const std::size_t node2);

  /**
   * @brief Splits all of the node pairs in the clustering tree into `close` or
   * `far` pairs.
   *
   * This function evaluates whether two sets of particles are considered to be
   * `close` or `far` based on their relative positions of the bounding box of
   * the set. It helps in determining whether the computational model should
   * treat them as interacting closely or not.
   */
  void CloseFarCheck();

  /**
   * @brief Performs a post-processing check after computation.
   *
   * This function compares the difference between the dense matrix and the
   * hierarchical matrix acceleration in Frobenius norm. It evaluates the norm
   * by sequentially traversing all of the node pairs in the clustering tree.
   */
  void PostCheck();

  /**
   * @brief Performs a post-processing check after computation.
   *
   * This function compares the difference between the dense dot and the
   * hierarchical matrix accelerated dot results in the norm of `u`. The dense
   * dot result of the velocity is obtained within this function using the input
   * force.
   *
   * @param u [in] The velocities obtained by the hierarchical matrix
   * accelerated dot, size (num_particles, 3).
   * @param f [in] The forces, size (num_particles, 3).
   */
  void PostCheckDot(DeviceDoubleMatrix u, DeviceDoubleMatrix f);

  /**
   * @brief Computes the close-range hydrodynamic interacted velocities with the
   * acting force.
   *
   * @param u [in,out] The velocities, size (num_particles, 3). It is
   * modified to store updated velocities.
   * @param f [in] The forces, size (num_particles, 3).
   */
  void CloseDot(DeviceDoubleMatrix u, DeviceDoubleMatrix f);

  /**
   * @brief Computes the far-range hydrodynamic interacted velocities with the
   * acting force.
   *
   * @param u [in,out] The velocities, size (num_nodes, 3). It is updated with
   * the new velocities.
   * @param f [in] The forces, size (num_nodes, 3).
   */
  void FarDot(DeviceDoubleMatrix u, DeviceDoubleMatrix f);

  /**
   * @brief Computes the hydrodynamic interacted velocities with the acting
   * force.
   *
   * This function evaluates the mobility tensor without distinguish the close
   * or far range interactions. It assembles the mobility tensor in a dense
   * manner and updates the velocities of particles using the dense mobility
   * tensor with the input force.
   *
   * @param u [out] The particle velocities. Size: (num_particles, 3). It is
   * updated with the new velocities after the interaction is computed.
   * @param f [in] The forces acting on each particle. Size: (num_particles, 3).
   * It contains the forces acting on each particle, and is used in the
   * calculation of the interaction.
   */
  void DenseDot(DeviceDoubleMatrix u, DeviceDoubleMatrix f);

  /**
   * @brief Computes the hydrodynamic interacted velocities with the acting
   * force using the hierarchical matrix acceleration.
   *
   * This function evaluates the mobility tensor by taking advantage of the
   * hierarchical matrix acceleration. It updates the particle velocities
   * based on the hierarchical matrix accelerated mobility tensor and performs
   * communication between MPI ranks to aggregate the results. Post-checking is
   * also performed if enabled.
   *
   * @param uArray [out] The particle velocities. Size: (num_particles, 3). It
   * is updated with the new velocities using the hierarchical matrix
   * accelerated mobility tensor and the input forces.
   * @param fArray [in] The forces acting on each particles. Size:
   * (num_particles, 3). It is used in the calculation of particle velocities.
   */
  void Dot(pybind11::array_t<float> &uArray, pybind11::array_t<float> &fArray);

  /**
   * @brief Computes the hydrodynamic interacted velocities with the acting
   * force with the dense mobility tensor
   *
   * This function evaluates the mobility tensor in a dense manner. It updates
   * the particle velocities with the dense mobility tensor and the input acting
   * force. It performs communication between MPI ranks to aggregate the
   * results.
   *
   * @param uArray [out] The particle velocities. Size: (num_particles, 3). It
   * is updated with new velocity after the dense mobility tensor is assembled
   * for each node pair on the clustering tree.
   * @param fArray [in] The forces acting on each particles. Size
   * (num_particles, 3).
   */
  void DenseDot(pybind11::array_t<float> &uArray,
                pybind11::array_t<float> &fArray);

  /**
   * @brief Updates the particle coordinates and triggers an internal update of
   * the clustering tree.
   *
   * This function updates the internally stored particle coordinates
   * `mCoordPtr` with the values from the input `coord`. After the update,
   * it calls the `Update()` function to perform the rebuilding of the
   * clustering tree w.r.t. the new particle coordinates.
   *
   * @param coord [in] The array representing the new coordinates for the
   * particles. Size (num_particles, nDim). If nDim = 3, it stands for (x, y,
   * z). If nDim = 6, it stands for (x, y, z, rx, ry, rz) which is not
   * implemented yet.
   */
  void UpdateCoord(pybind11::array_t<float> &coord);

  /**
   * @brief Sets the value of epsilon used by the adaptive cross approximation
   *
   * This function sets the value of the internal `mEpsilon` variable, which
   * controls the convergence criteria of the adaptive cross approximation used
   * in function FarDot.
   *
   * @param epsilon [in] The value to set for epsilon.
   */
  void SetEpsilon(const double epsilon);

  /**
   * @brief Sets the value of eta used by the clustering tree.
   *
   * This function sets the value of the internal `mEta` variable, which
   * controls the identification of `close` or `far` pair in the clustering
   * tree.
   *
   * @param eta [in] The value to set for eta.
   */
  void SetEta(const double eta);

  /**
   * @brief Sets the maximum number of iterations for the adaptive across
   * approximation.
   *
   * This value determines how many iterations of the adaptive across
   * approximation will run. This values influences the spare storage space on
   * GPUs.
   *
   * @param maxIter [in] The maximum number of iterations to perform.
   */
  void SetMaxIter(const int maxIter);

  /**
   * @brief Sets the factor that determines the size of the matrix pool.
   *
   * FarDot requires a spare storage space allocated in front to avoid the
   * frequent allocation and deallocation of memory on GPU used for storing C
   * and Q used by the adaptive cross approximation.
   *
   * @param factor [in] The factor to adjust the matrix pool size.
   */
  void SetMatPoolSizeFactor(const int factor);

  /**
   * @brief Sets the flag to enable or disable the post-check operation.
   *
   * @param flag [in] A boolean flag indicating whether the post-check operation
   * is enabled (true) or disabled (false).
   */
  void SetPostCheckFlag(const bool flag);

  /**
   * @brief Sets the flag to enable or disable the use of symmetry in the model.
   *
   * The mobility tensor has a symmetric pattern by definition. Enabling this
   * flag can save the number of queries when doing CloseDot and FarDot.
   *
   * @param flag [in] A boolean flag indicating whether symmetry is enabled
   * (true) or disabled (false).
   */
  void SetUseSymmetryFlag(const bool flag);

  /**
   * @brief Sets the maximum number of node pairs for far-range interaction
   * computations in function FarDot.
   *
   * This function sets the limit on the number of node pairs that can
   * simultaneously involved in far-range interaction computations. This value
   * helps in managing computational resources.
   *
   * @param size [in] The maximum number of node pairs to be considered for
   * far-range interaction computations.
   */
  void SetMaxFarDotWorkNodeSize(const int size);

  /**
   * @brief Sets the maximum number of relative coordinates can be
   * simultaneously processed.
   *
   * This function sets the upper limit on the number of relative coordinates
   * that can be handled by the function CloseDot and FarDot. It also determines
   * the maximum number of queries can be performed by the two-body interaction
   * model when evaluating the entries in the mobility tensor.
   *
   * @param size [in] The maximum number of relative coordinates that can be
   * processed.
   */

  void SetMaxRelativeCoord(const size_t size);

  /**
   * @brief Sets the cut-off distance in the model. (not used anymore)
   *
   * @param distance [in] The  cut-off distance for far-range interactions.
   */

  void SetMaxFarFieldDistance(const double distance);

  /**
   * @brief Reorders the rows of the given matrix `v` based on the provided
   * index mapping.
   *
   * This function rearranges the rows of the input matrix `v` according to the
   * order specified in the `reorderedMap`. It is used to reorder matrices such
   * as velocities, forces, or other quantities before further computations.
   *
   * @param reorderedMap [in] A vector of size (num_nodes) that specifies the
   * new order of the rows in `v`.
   * @param v [in,out] The matrix to be reordered, size (num_nodes, 3). It is
   * updated in place with the new row order.
   */
  void Reorder(const std::vector<std::size_t> &reorderedMap,
               DeviceDoubleMatrix v);

  /**
   * @brief Reverses the reordering of the rows of the given matrix `v` based on
   * the provided index mapping.
   *
   * This function undoes the row reordering performed by the `Reorder` function
   * by applying the inverse of the order specified in the `reorderedMap` to the
   * input matrix `v`. It restores the original order of rows in `v`.
   *
   * @param reorderedMap [in] A vector of size (num_nodes) that specifies the
   * original order of the rows in `v`.
   * @param v [in,out] The matrix to be reordered back, size (num_nodes, 3). It
   * is updated in place with the reversed row order.
   */
  void BackwardReorder(const std::vector<std::size_t> &reorderedMap,
                       DeviceDoubleMatrix v);
};

#endif