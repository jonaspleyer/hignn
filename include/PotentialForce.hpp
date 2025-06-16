#ifndef _PotentialForce_Hpp_
#define _PotentialForce_Hpp_

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <memory>
#include <vector>
#include <chrono>

#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <mpi.h>

#include "Typedef.hpp"
#include "PointCloudSearch.hpp"

/**
 * @class PotentialForce
 * @brief Class for computing the potential force between particles in periodic
 * or non-periodic domains.
 *
 * This class handles the calculation of potential forces between particles. It
 * supports for periodic boundaries. It computes forces
 * (CalculatePotentialForce) based on the distances between particles.
 */
class __attribute__((visibility("default"))) PotentialForce {
private:
  HostFloatMatrix mSourceSites;
  //!< The positions of source/background particles in space.
  HostFloatMatrix mTargetSites;
  //!< The positions of target particles in space.
  HostFloatMatrix mPotentialForce;
  //!< The calculated potential forces on the target particles based on
  //!< interactions with the source particles.
  HostFloatVector mEpsilonLists;
  //!< The radius of the neighbor list of the target particles.
  HostIndexVector mSourceIndex;
  //!< The global index of source particles.
  HostIndexMatrix mTwoBodyEdgeInfo;
  //!< Truncated two body particle pairs between the target and source
  //!< particles. Size: (mTwoBodyEdgeNum, 2). The first entry is the index of
  //!< target particle in the pair; the second entry is the index of the source
  //!< particle in pair. Truncation distance is managed by mTwoBodyEpsilon.
  HostIndexMatrix mTargetNeighborLists;
  //!< Neighbor list of target particles.

  HostIndexVector mTwoBodyEdgeNum;
  //!< Number of two body particle pairs for each target particle.
  HostIndexVector mTwoBodyEdgeOffset;
  //!< The cumulative offsets for each target particle in mTwoBodyEdgeInfo.

  bool mIsPeriodicBoundary;
  //!< Flag of periodic boundary condition
  float mTwoBodyEpsilon, mThreeBodyEpsilon;
  //!< Truncation radius of two-body and three-body pairs/couples.
  float mDomainLow[3], mDomainHigh[3];
  //!< The lower and upper bounds of the domain for periodic boundary
  //!< conditions.
  int mDim;
  //!< Dimensionality

  std::shared_ptr<PointCloudSearch<HostFloatMatrix>> mPointCloudSearch;
  //!< Pointer to the point cloud search object used for searching neighbors.

public:
  /**
   * @brief Constructor for the PotentialForce class.
   *
   * Initializes the member variables with default values. The periodic boundary
   * flag is set to false by default, and the two-body truncation radius is set
   * to 0.0.
   */
  PotentialForce() : mIsPeriodicBoundary(false), mTwoBodyEpsilon(0.0), mDim(3) {
  }

  /**
   * @brief Destructor for the PotentialForce class.
   *
   * This destructor cleans up any resources allocated by the class.
   */
  ~PotentialForce() {
  }

  /**
   * @brief Calculates the potential force between two particles based on the
   * Morse potential.
   *
   * This function calculates the force between two particles using the Morse
   * potential formula, which models the interaction between a pair of particles
   * based on their center-to-center distance. The resulting force vector is
   * stored in the provided `f` array.
   *
   * The force is computed using the following steps:
   * 1. The relative position vector `r` between the particles is used to
   * compute the distance (norm).
   * 2. The force magnitude is calculated using the Morse potential formula.
   * 3. The force vector is then scaled by the ratio of force magnitude to
   * distance.
   *
   * The force will be attractive or repulsive depending on the distance between
   * the particles.
   *
   * @param r [in] A float array of size 3 representing the relative position
   * vector between the two particles. The components are the differences in the
   * x, y, and z coordinates.
   * @param f [out] A float array of size 3 where the calculated force vector is
   * stored. The force vector corresponds to the interaction between the
   * particles.
   */
  inline void CalculatePotentialForce(float *r, float *f) {
    const float De = 1.0;
    //!< Depth of the attractive potential.
    const float a = 1.0;
    //!< Parameter controlling the interaction range.
    const float re = 2.5;
    //!< Equilibrium distance.

    float rNorm = 0.0;
    //!< the center-to-center distance between particles.
    float fMag = 0.0;
    //!< Magnitude of the force between the particles.

    rNorm = sqrt(r[0] * r[0] + r[1] * r[1] + r[2] * r[2]);
    //!< Compute the distance between particles.
    fMag =
        -2.0 * a * De * (exp(-a * (rNorm - re)) - exp(-2.0 * a * (rNorm - re)));
    //!< Calculate the force magnitude using Morse potential.

    float ratio = fMag / rNorm;
    //!< Calculate the ratio of force magnitude to distance.
    for (int j = 0; j < 3; j++) {
      f[j] = r[j] * ratio;
      //!< Scale the relative position vector to get the force vector.
    }
  }

  /**
   * @brief Sets whether the domain is periodic.
   *
   * This function sets the flag for periodic boundary conditions, which affects
   * the neighbor particle searching.
   *
   * @param isPeriodicBoundary [in] A boolean flag indicating whether the domain
   * should have periodic boundary conditions.
   */
  void SetPeriodic(bool isPeriodicBoundary = false) {
    mIsPeriodicBoundary = isPeriodicBoundary;
  }

  /**
   * @brief Sets the domain when enabled with periodic boundary conditions.
   *
   * This function defines the lower and upper bounds of the periodic domain. It
   * also enables periodic boundary conditions by setting the
   * `mIsPeriodicBoundary` flag to true.
   *
   * @param domain [in] A 2D NumPy array of shape (2, 3), where:
   * - The first row specifies the lower bounds of the domain.
   * - The second row specifies the upper bounds of the domain.
   */
  void SetDomain(pybind11::array_t<float> domain) {
    pybind11::buffer_info buf = domain.request();

    if (buf.ndim != 2) {
      throw std::runtime_error("Rank of domain must be two");
    }

    auto data = domain.unchecked<2>();

    mDomainLow[0] = data(0, 0);
    mDomainLow[1] = data(0, 1);
    mDomainLow[2] = data(0, 2);

    mDomainHigh[0] = data(1, 0);
    mDomainHigh[1] = data(1, 1);
    mDomainHigh[2] = data(1, 2);

    mIsPeriodicBoundary = true;
  }

  /**
   * @brief Updates the target coordinates for the particles.
   *
   * This function updates the target sites (particles)' positions in the
   * simulation. The new coordinates are provided via a NumPy array with each
   * row representing the position of a single particle.
   *
   * @param coord [in] A 2D NumPy array of size (num_nodes, 3) representing the
   * new coordinates of the target particles.
   */
  void UpdateCoord(pybind11::array_t<float> coord) {
    pybind11::buffer_info buf = coord.request();

    if (buf.ndim != 2) {
      throw std::runtime_error("Rank of target sites must be two");
    }

    mTargetSites =
        decltype(mTargetSites)("target sites", (std::size_t)coord.shape(0),
                               (std::size_t)coord.shape(1));

    auto data = coord.unchecked<2>();

    Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(
                             0, mTargetSites.extent(0)),
                         [=](size_t i) {
                           mTargetSites(i, 0) = data(i, 0);
                           mTargetSites(i, 1) = data(i, 1);
                           mTargetSites(i, 2) = data(i, 2);
                         });
    Kokkos::fence();

    mTargetNeighborLists = decltype(mTargetNeighborLists)(
        "two body neighbor lists", mTargetSites.extent(0), 2);
    mEpsilonLists =
        decltype(mEpsilonLists)("epsilon lists", mTargetSites.extent(0));
  }

  /**
   * @brief Sets the truncation radius of two-body pairs.
   *
   * @param epsilon [in] The truncation radius of two-body pairs.
   */
  void SetTwoBodyEpsilon(float epsilon) {
    mTwoBodyEpsilon = epsilon;
  }

  /**
   * @brief Builds the source positions for particles, considering periodic
   * boundaries. The source positions are built for neighbor particle search.
   *
   * This function calculates the positions of particles and adjusts them based
   * on the periodic boundary conditions, if enabled. If periodic boundaries are
   * not used, it simply copies the target positions to the source positions.
   *
   * When periodic boundaries are enabled, particles near the boundaries (within
   * the truncation radius of two-body pairs) are duplicated and adjusted to
   * the opposite side of the domain. This ensures that particles can find the
   * neighboring particles on the other side of the simulation box, creating a
   * continuous simulation space.
   */
  void BuildSourceSites() {
    if (!mIsPeriodicBoundary) {
      mSourceSites = mTargetSites;

      mSourceIndex =
          decltype(mSourceIndex)("source index", mSourceSites.extent(0));
      Kokkos::parallel_for(
          Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(
              0, mSourceIndex.extent(0)),
          [&](size_t i) { mSourceIndex(i) = i; });
      Kokkos::fence();
    } else {
      float epsilon = mTwoBodyEpsilon;
      // Only consider the two-body pair now. In the future, the three-body
      // couple should also be included. Probably, it would be the maximum
      // between mTwoBodyEpsilon and mThreeBodyEpsilon.

      float coreDomainLow[3], coreDomainHigh[3], domainSize[3];

      // Adjust the domain size based on the truncation radius for periodic
      // boundaries
      for (int i = 0; i < 3; i++)
        coreDomainLow[i] = mDomainLow[i] + epsilon;
      for (int i = 0; i < 3; i++)
        coreDomainHigh[i] = mDomainHigh[i] - epsilon;
      for (int i = 0; i < 3; i++)
        domainSize[i] = mDomainHigh[i] - mDomainLow[i];

      const std::size_t numTarget = mTargetSites.extent(0);

      // Calculate duplicates for periodic boundary adjustments
      HostIndexVector numSourceDuplicate =
          HostIndexVector("num source duplicate", numTarget);
      HostIntMatrix axisSourceDuplicate =
          HostIntMatrix("axis source duplicate", numTarget, 3);

      Kokkos::parallel_for(
          Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0, numTarget),
          [&](size_t i) {
            std::size_t num = 1;
            for (int j = 0; j < 3; j++) {
              axisSourceDuplicate(i, j) = 1;
              // Check if the target site is within the domain that needs
              // reflection due to periodic BCs
              if (mTargetSites(i, j) < coreDomainLow[j]) {
                axisSourceDuplicate(i, j) = -2;
                num *= 2;
              } else if (mTargetSites(i, j) > coreDomainHigh[j]) {
                axisSourceDuplicate(i, j) = 2;
                num *= 2;
              } else {
                axisSourceDuplicate(i, j) = 1;
              }
            }

            numSourceDuplicate(i) = num;
          });
      Kokkos::fence();

      HostIndexVector numSourceOffset =
          HostIndexVector("num source offset", numTarget + 1);

      numSourceOffset(0) = 0;
      for (size_t i = 0; i < numTarget; i++) {
        numSourceOffset(i + 1) = numSourceOffset(i) + numSourceDuplicate(i);
      }
      std::size_t numSource = numSourceOffset(numTarget);

      mSourceSites = decltype(mSourceSites)("source sites", numSource, 3);
      mSourceIndex = decltype(mSourceIndex)("source index", numSource);

      // Update the source positions with the adjusted coordinates, considering
      // the duplicates
      Kokkos::parallel_for(
          Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0, numTarget),
          [&](size_t i) {
            std::vector<float> offset;
            offset.resize(3 * numSourceDuplicate(i));

            const std::size_t num = numSourceDuplicate(i);
            std::size_t stride1 = numSourceDuplicate(i);

            // Adjust positions based on the periodic boundary condition and
            // epsilon
            for (int j = 0; j < 3; j++) {
              if (axisSourceDuplicate(i, j) == 1) {
                for (size_t n = 0; n < num; n++) {
                  offset[n * 3 + j] = 0;
                }
              }
              if (axisSourceDuplicate(i, j) == 2) {
                for (size_t m = 0; m < num; m += stride1) {
                  for (size_t n = m; n < m + stride1 / 2; n++) {
                    offset[n * 3 + j] = 0;
                  }
                  for (size_t n = m + stride1 / 2; n < m + stride1; n++) {
                    offset[n * 3 + j] = -domainSize[j];
                  }
                }
                stride1 /= 2;
              }
              if (axisSourceDuplicate(i, j) == -2) {
                for (size_t m = 0; m < num; m += stride1) {
                  for (size_t n = m; n < m + stride1 / 2; n++) {
                    offset[n * 3 + j] = 0;
                  }
                  for (size_t n = m + stride1 / 2; n < m + stride1; n++) {
                    offset[n * 3 + j] = domainSize[j];
                  }
                }
                stride1 /= 2;
              }
            }

            for (size_t m = numSourceOffset[i]; m < numSourceOffset[i + 1];
                 m++) {
              for (int j = 0; j < 3; j++) {
                mSourceSites(m, j) = mTargetSites(i, j) +
                                     offset[(m - numSourceOffset[i]) * 3 + j];
              }
              mSourceIndex(m) = i;
            }
          });
      Kokkos::fence();
    }

    // Create the PointCloudSearch object for fast search of neighboring
    // particles
    mPointCloudSearch = std::make_shared<PointCloudSearch<HostFloatMatrix>>(
        CreatePointCloudSearch(mSourceSites, 3));
  }

  /**
   * @brief Builds the neighbor lists for target particles based on their
   * proximity.
   *
   * This function calculates and organizes the neighboring particles for each
   * target particle by performing a radius search. The search is based on the
   * distance between the target particles and their potential neighbors,
   * considering a specified epsilon (threshold) value. If periodic boundary
   * conditions are enabled, it handles duplicate particles near the boundaries
   * and updates the lists accordingly.
   *
   * @param epsilon [in] A float value representing the cutoff distance for
   * finding neighboring particles. This value determines the proximity within
   * which particles are considered neighbors.
   */
  void BuildTargetNeighborLists(float epsilon) {
    BuildSourceSites();

    auto numTarget = mTargetSites.extent(0);
    //!< Get the number of target particles.

    // Assign epsilon value to each target particle's list.
    Kokkos::parallel_for(
        Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0, numTarget),
        [&](size_t i) { mEpsilonLists(i) = epsilon; });
    Kokkos::fence();

    // Perform a 2D radius search to find neighboring particles for each target
    // particle
    auto numNeighbor =
        1 + mPointCloudSearch->generate2DNeighborListsFromRadiusSearch(
                true, mTargetSites, mTargetNeighborLists, mEpsilonLists, 0.0,
                epsilon);

    // Resize the neighbor list if more neighbors are found than the current
    // capacity
    if (numNeighbor > mTargetNeighborLists.extent(1))
      Kokkos::resize(mTargetNeighborLists, numTarget, numNeighbor);

    mPointCloudSearch->generate2DNeighborListsFromRadiusSearch(
        false, mTargetSites, mTargetNeighborLists, mEpsilonLists, 0.0, epsilon);

    Kokkos::parallel_for(
        Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0, numTarget),
        [&](size_t i) {
          // Change the target index to the real index
          size_t counter = 0;
          while (counter < mTargetNeighborLists(i, 0)) {
            mTargetNeighborLists(i, counter + 1) =
                mSourceIndex(mTargetNeighborLists(i, counter + 1));
            counter++;
          }
        });
    Kokkos::fence();

    Kokkos::parallel_for(
        Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0, numTarget),
        [&](size_t i) {
          // Ensure the current target index appears at the
          // beginning of the list
          size_t counter = 0;
          while (counter < mTargetNeighborLists(i, 0)) {
            if (mTargetNeighborLists(i, counter + 1) == i) {
              std::swap(mTargetNeighborLists(i, 1),
                        mTargetNeighborLists(i, counter + 1));
              break;
            }
            counter++;
          }
        });
    Kokkos::fence();
  }

  /**
   * @brief Updates the particle positions and calculates the potential forces.
   *
   * This function updates the particle coordinates, builds the target neighbor
   * lists, and computes the potential forces between the particles based on
   * their interactions. It handles both the communication of the particle data
   * across MPI processes and the synchronization of threads using Kokkos for
   * parallel computing.
   *
   * @param coord [in] A 2D NumPy array representing the updated positions of
   * the particles. It is a matrix of size (num_nodes, 3), where num_nodes is
   * the number of target particles.
   *
   * @return A NumPy array of size (num_nodes, 3) containing the calculated
   * potential forces for each particle.
   */
  pybind11::array_t<float> GetPotentialForce(pybind11::array_t<float> coord) {
    int mpiRank;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank);

    // Update the particle positions with the new data  passed in `coord`.
    UpdateCoord(coord);

    if (mpiRank == 0) {
      std::cout << "  Potential force updated coord" << std::endl;
    }

    // Build target neighbor lists considering the epsilon for two-body
    // interactions.
    BuildTargetNeighborLists(mTwoBodyEpsilon);

    if (mpiRank == 0) {
      std::cout << "  Potential force built target neighbor lists" << std::endl;
    }

    std::size_t numTarget = mTargetSites.extent(0);
    //!< Get the number of target sites (particles).

    // Resize and initialize two-body interaction variables
    Kokkos::resize(mTwoBodyEdgeNum, numTarget);
    Kokkos::parallel_for(
        Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0, numTarget),
        [=](std::size_t i) {
          mTwoBodyEdgeNum(i) = mTargetNeighborLists(i, 0) - 1;
        });
    Kokkos::fence();

    // Resize and initialize two-body edge offsets
    Kokkos::resize(mTwoBodyEdgeOffset, numTarget);

    // Pre-scan for counting the offset of two-body edge
    Kokkos::parallel_scan(
        Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0, numTarget),
        [&](const int i, size_t &sum, [[maybe_unused]] bool final) {
          mTwoBodyEdgeOffset(i) = sum;
          sum += mTwoBodyEdgeNum(i);
        });
    Kokkos::fence();

    // Calculate the total number of two-body edges (pa)
    const int numEdge =
        mTwoBodyEdgeOffset(numTarget - 1) + mTwoBodyEdgeNum(numTarget - 1);

    // Resize and initialize matrices for storing two-body edge information and
    // potential forces
    mTwoBodyEdgeInfo =
        decltype(mTwoBodyEdgeInfo)("two body edge info", numEdge, 2);
    mPotentialForce =
        decltype(mPotentialForce)("potential force", numTarget, 3);

    if (mpiRank == 0) {
      std::cout << "  Start of calculating potential force" << std::endl;
    }

    std::chrono::high_resolution_clock::time_point t1 =
        std::chrono::high_resolution_clock::now();
    // Calculate the potential force for each target particle based on its
    // neighbors
    Kokkos::parallel_for(
        Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0, numTarget),
        [&](size_t i) {
          std::size_t offset = 0;
          for (int j = 0; j < 3; j++)
            mPotentialForce(i, j) = 0.0;

          // Loop over each neighbor of the target particle and compute the
          // interaction force
          for (std::size_t j = 1; j < mTargetNeighborLists(i, 0); j++) {
            std::size_t neighborIdx = mTargetNeighborLists(i, j + 1);
            mTwoBodyEdgeInfo(mTwoBodyEdgeOffset(i) + offset, 0) = i;
            mTwoBodyEdgeInfo(mTwoBodyEdgeOffset(i) + offset, 1) =
                mSourceIndex(neighborIdx);

            float r[3], f[3];
            //!< Arrays to hold the relative positions and calculated force.
            for (int k = 0; k < 3; k++)
              r[k] = mTargetSites(i, k) - mSourceSites(neighborIdx, k);

            float rNorm = sqrt(r[0] * r[0] + r[1] * r[1] + r[2] * r[2]);

            // If distance is non-zero, calculate the force based on the
            // potential
            if (rNorm > 1e-3) {
              CalculatePotentialForce(r, f);
              for (int k = 0; k < 3; k++)
                mPotentialForce(i, k) += f[k];
              //!< Accumulate the force on the target particle.
            }

            offset++;
          }
        });
    Kokkos::fence();

    std::chrono::high_resolution_clock::time_point t2 =
        std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();

    if (mpiRank == 0)
      printf("  End of calculating potential force. Time: %.4fs\n",
             (double)duration / 1e6);
    //!< Print out the time taken for force calculation.

    // Create a NumPy array to store the potential forces
    pybind11::array_t<float> potentialForce(
        {mPotentialForce.extent(0), mPotentialForce.extent(1)});

    pybind11::buffer_info buf = potentialForce.request();
    auto ptr = (float *)buf.ptr;

    // Copy the calculated potential forces into the NumPy array
    Kokkos::parallel_for(
        Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0, numTarget),
        [=](size_t i) {
          for (int j = 0; j < 3; j++)
            ptr[i * 3 + j] = mPotentialForce(i, j);
        });
    Kokkos::fence();

    return potentialForce;
  }
};

#endif