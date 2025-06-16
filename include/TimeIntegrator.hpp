#ifndef _TimeIntegrator_Hpp_
#define _TimeIntegrator_Hpp_

#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <Quaternion.hpp>
#include <Vec3.hpp>

#include <fstream>
#include <string>
#include <vector>
#include <sstream>

#include <mpi.h>

#include <vtkSmartPointer.h>
#include <vtkPoints.h>
#include <vtkPolyData.h>
#include <vtkVertex.h>
#include <vtkCellArray.h>
#include <vtkFloatArray.h>
#include <vtkPointData.h>
#include <vtkXMLPolyDataWriter.h>

/**
 * @class TimeIntegrator
 * @brief Time-stepping driver for particulate suspension simulations.
 *
 * Manages the integration loop by:
 *  - Calling a Python callback to compute particle velocities.
 *  - Updating positions and orientations via Euler or RK4 schemes.
 *  - Handling MPI partitioning and periodic boundary conditions.
 *  - Exporting particle positions and velocities in VTK format for
 * visualization.
 */
class __attribute__((visibility("default"))) TimeIntegrator {
protected:
  int mMpiRank, mMpiSize;

  int mFuncCount;             //!< Count of integration steps already executed
  int mOutputStep;            //!< Interval (in steps) between VTK outputs
  std::size_t mNumRigidBody;  //!< Total number of particles in the simulation

  double mTimeStep;
  double mFinalTime;  // Simulation end time

  bool mIsPeriodicBoundary;  //!< Enable or disable periodic boundary handling

  std::string mOutputFilePrefix;  //!< Directory/prefix for all output files

  std::vector<Vec3> mPosition0;           // Initial particle positions
  std::vector<Quaternion> mOrientation0;  // Initial particle orientations

  std::vector<Vec3> mPosition;           // Current particle positions
  std::vector<Quaternion> mOrientation;  // Current particle orientations

  std::vector<Vec3> mVelocity;  //!< Current linear velocities
  std::vector<Vec3>
      mAngularVelocity;  //!< Current angular velocities (unused by default)

  Vec3 mDomainLimit[2];  //!< [min,max] bounds for periodic wrapping
  std::vector<Vec3>
      mPositionOffset;  //!< Accumulated offsets to apply when wrapping

  /**
   * @brief Python callback to compute particle velocities.
   * @param t [in] Current time
   * @param position [in] Particle positions and orientations, size
   * (num_particles, 6) with [x, y, z, roll, pitch, yaw].
   * @return The updated linear velocities, size (num_particles, 3)
   */
  std::function<pybind11::array_t<float>(float, pybind11::array_t<float>)>
      mVelocityUpdateFunc;
  std::function<void(pybind11::array_t<float>, pybind11::array_t<float>)>
      mOutputFunc;
  bool mCustomOutputFlag;

  /**
   * @brief Convert C++ particle positions and orientations into a Python array,
   * invokes the Python velocity_update callback, and writes back the
   * computed velocities.
   * @param t [in] Current simulation time.
   * @param position [in] Vector of particle positions, size (num_particles, 3).
   * @param orientation [in] Vector of particle orientations (unused here).
   * @param velocity [out] Output vector to be filled with each particle's
   * linear velocity, size (num_particles, 3).
   * @param angularVelocity [out] Output vector for angular velocities (unused).
   */
  void VelocityUpdate(
      const float t,
      const std::vector<Vec3> &position,
      [[maybe_unused]] const std::vector<Quaternion> &orientation,
      std::vector<Vec3> &velocity,
      [[maybe_unused]] std::vector<Vec3> &angularVelocity) {
    // convert
    pybind11::array_t<float> input;
    input.resize({(int)mNumRigidBody, 6});

    auto inputData = input.mutable_unchecked<2>();
    // deep copy
#pragma omp parallel for schedule(static)
    for (std::size_t num = 0; num < mNumRigidBody; num++) {
      // First three columns: positions
      for (int j = 0; j < 3; j++) {
        if (mIsPeriodicBoundary) {
          // periodic boundary condition
          if (position[num][j] > mDomainLimit[1][j]) {
            mPositionOffset[num][j] = mDomainLimit[0][j] - mDomainLimit[1][j];
          } else if (position[num][j] < mDomainLimit[0][j]) {
            mPositionOffset[num][j] = mDomainLimit[1][j] - mDomainLimit[0][j];
          } else {
            mPositionOffset[num][j] = 0.0;
          }
          inputData(num, j) = position[num][j] + mPositionOffset[num][j];
        } else
          inputData(num, j) = position[num][j];
      }

      // Next three columns: Euler angles extracted from quaternion
      float row, pitch, yaw;
      mOrientation[num].to_euler_angles(row, pitch, yaw);
      inputData(num, 3) = row;
      inputData(num, 4) = pitch;
      inputData(num, 5) = yaw;
    }
    // Invoke the Python callback: returns the array of velocities
    auto result = mVelocityUpdateFunc(t, input);

    auto result_data = result.unchecked<2>();
    // deep copy
#pragma omp parallel for schedule(static)
    for (std::size_t num = 0; num < mNumRigidBody; num++) {
      for (int j = 0; j < 3; j++) {
        velocity[num][j] = result_data(num, j);
        // angularVelocity[num][j] = result_data(num, j + 3);
      }
    }
  }

  /**
   * @brief Copy positions and velocities to Python output func
   */
  void Output() {
    pybind11::array_t<float> position, velocity;
    position.resize({(int)mNumRigidBody, 3});
    velocity.resize({(int)mNumRigidBody, 3});

    auto positionData = position.mutable_unchecked<2>();
    auto velocityData = velocity.mutable_unchecked<2>();

#pragma omp parallel for schedule(static)
    for (std::size_t num = 0; num < mNumRigidBody; num++) {
      for (int j = 0; j < 3; j++) {
        positionData(num, j) = mPosition[num][j];
        velocityData(num, j) = mVelocity[num][j];
      }
    }

    mOutputFunc(position, velocity);
  }

  /**
   * @brief Write VTK output for current particle positions and velocities.
   *
   * Each MPI rank writes its own .vtp file of the subset of particles it owns,
   * and rank 0 also generates a .pvd collection file to reference all per-rank
   * files.
   */
  void Output(const std::string &outputFilename) {
    // Build the per-rank filename
    std::string rankOutputFilename =
        outputFilename + "_Rank" + std::to_string(mMpiRank) + ".vtp";
    std::string fullRankOutputFilename = mOutputFilePrefix + rankOutputFilename;
    // On rank 0, write the .pvd file that collects all .vtp pieces
    if (mMpiRank == 0) {
      std::ofstream pvdFile(mOutputFilePrefix + outputFilename + ".pvd");
      pvdFile << "<?xml version=\"1.0\"?>\n"
              << "<VTKFile type=\"Collection\" version=\"0.1\" "
                 "byte_order=\"LittleEndian\">\n"
              << "  <Collection>\n";

      for (int i = 0; i < mMpiSize; ++i)
        pvdFile << "    <DataSet timestep=\"0\" group=\"\" "
                << "part=\"" << i << "\" file=\""
                << outputFilename + "_Rank" + std::to_string(i) + ".vtp\"/>\n";

      pvdFile << "  </Collection>\n"
              << "</VTKFile>\n";

      pvdFile.close();
    }

    /** Compute this rank's particle range. */
    auto size = mPosition.size();
    size_t localSize = ceil((float)size / mMpiSize);
    size_t start = mMpiRank * localSize;
    size_t end = std::min(start + localSize, size);

    // Set up VTK structures
    vtkSmartPointer<vtkPolyData> polyData = vtkSmartPointer<vtkPolyData>::New();
    vtkSmartPointer<vtkPoints> points = vtkSmartPointer<vtkPoints>::New();
    vtkSmartPointer<vtkCellArray> vertices =
        vtkSmartPointer<vtkCellArray>::New();
    vtkSmartPointer<vtkFloatArray> velocityArray =
        vtkSmartPointer<vtkFloatArray>::New();

    velocityArray->SetNumberOfComponents(3);
    velocityArray->SetName("velocity");

    // Insert each particle's point and velocity
    for (size_t i = start; i < end; i++) {
      vtkIdType pid[1];

      pid[0] = points->InsertNextPoint(mPosition[i][0], mPosition[i][1],
                                       mPosition[i][2]);

      vertices->InsertNextCell(1, pid);

      velocityArray->InsertNextTuple3(mVelocity[i][0], mVelocity[i][1],
                                      mVelocity[i][2]);
    }
    // Assemble and write the .vtp file
    polyData->SetPoints(points);
    polyData->SetVerts(vertices);
    polyData->GetPointData()->AddArray(velocityArray);

    vtkSmartPointer<vtkXMLPolyDataWriter> writer =
        vtkSmartPointer<vtkXMLPolyDataWriter>::New();
    writer->SetFileName(fullRankOutputFilename.c_str());
    writer->SetInputData(polyData);

    writer->SetCompressorTypeToZLib();
    writer->SetDataModeToAppended();

    writer->Write();
    writer->Update();
  }

public:
  /**
   * @brief Default constructor: set up MPI and default simulation parameters.
   */
  TimeIntegrator()
      : mFuncCount(0),
        mOutputStep(1),
        mNumRigidBody(0),
        mTimeStep(1.0),
        mFinalTime(10.0),
        mIsPeriodicBoundary(false),
        mOutputFilePrefix("Result/"),
        mCustomOutputFlag(false) {
    MPI_Comm_rank(MPI_COMM_WORLD, &mMpiRank);
    MPI_Comm_size(MPI_COMM_WORLD, &mMpiSize);

    // Default periodic domain bounds: [-1,1] in x, y, z
    mDomainLimit[0][0] = -1;
    mDomainLimit[1][0] = 1;
    mDomainLimit[0][1] = -1;
    mDomainLimit[1][1] = 1;
    mDomainLimit[0][2] = -1;
    mDomainLimit[1][2] = 1;
  }

  /**
   * @brief Set the integration time‐step.
   * @param timeStep [in] New time‐step value.
   */
  void SetTimeStep(const float timeStep) {
    mTimeStep = timeStep;
  }

  /**
   * @brief Set the simulation’s final time.
   * @param finalTime [in] Time at which to stop the simulation.
   */
  void SetFinalTime(const float finalTime) {
    mFinalTime = finalTime;
  }

  /**
   * @brief Set the total number of particles and allocate per-particle storage.
   * @param numRigidBody [in] Total number of particles in the simulation.
   *
   * Resizes internal vectors to hold positions, orientations, velocities,
   * and boundary offsets for each particle.
   */
  void SetNumRigidBody(const std::size_t numRigidBody) {
    mNumRigidBody = numRigidBody;

    mPosition0.resize(mNumRigidBody);
    mOrientation0.resize(mNumRigidBody);

    mPosition.resize(mNumRigidBody);
    mOrientation.resize(mNumRigidBody);

    mVelocity.resize(mNumRigidBody);
    mAngularVelocity.resize(mNumRigidBody);

    mPositionOffset.resize(mNumRigidBody);
  }

  /**
   * @brief Register the Python velocity‐update callback.
   * @param func [in] The Python function to call for computing velocities.
   */
  void SetVelocityUpdateFunc(
      const std::function<
          pybind11::array_t<float>(float, pybind11::array_t<float>)> &func) {
    mVelocityUpdateFunc = func;
  }

  /**
   * @brief Set Python output func
   */

  void SetOutputFunc(
      const std::function<void(pybind11::array_t<float>,
                               pybind11::array_t<float>)> &func) {
    mOutputFunc = func;
    mCustomOutputFlag = true;
  }

  /**
   * @brief Enable periodic boundary in X using the given limits.
   * @param xLim [in] Two‐element list specifying [xmin, xmax].
   */
  void SetXLim(pybind11::list xLim) {
    mDomainLimit[0][0] = pybind11::cast<float>(xLim[0]);
    mDomainLimit[1][0] = pybind11::cast<float>(xLim[1]);

    mIsPeriodicBoundary = true;
  }

  /**
   * @brief Enable periodic boundary in Y using the given limits.
   * @param yLim [in] Two‐element list specifying [ymin, ymax].
   */
  void SetYLim(pybind11::list yLim) {
    mDomainLimit[0][1] = pybind11::cast<float>(yLim[0]);
    mDomainLimit[1][1] = pybind11::cast<float>(yLim[1]);

    mIsPeriodicBoundary = true;
  }

  /**
   * @brief Enable periodic boundary in Z using the given limits.
   * @param zLim [in] Two‐element list specifying [zmin, zmax].
   */
  void SetZLim(pybind11::list zLim) {
    mDomainLimit[0][2] = pybind11::cast<float>(zLim[0]);
    mDomainLimit[1][2] = pybind11::cast<float>(zLim[1]);

    mIsPeriodicBoundary = true;
  }

  /**
   * @brief Configure how often to write output files.
   * @param outputStep [in] Number of simulation steps between each VTK write.
   */
  void SetOutputStep(const int outputStep) {
    mOutputStep = outputStep;
  }

  /**
   * @brief Initialize particle positions and orientations from Python.
   * @param initPosition [in] Particle positions and orientations, size
   * (num_particles, 6) with [x, y, z, roll, pitch, yaw] for each particle.
   *
   * Validates dimensions, then deep‐copies into internal storage:
   *  - mPosition0, mOrientation0 as initial state
   *  - mPosition, mOrientation for ongoing simulation
   */
  void Init(pybind11::array_t<float> initPosition) {
    pybind11::buffer_info buf = initPosition.request();

    if (buf.ndim != 2) {
      throw std::runtime_error(
          "Number of dimensions of input positions must be two");
    }

    if ((std::size_t)initPosition.shape(0) != mNumRigidBody) {
      throw std::runtime_error("Inconsistent number of rigid bodies");
    }

    auto data = initPosition.unchecked<2>();

    // deep copy
#pragma omp parallel for schedule(static)
    for (std::size_t num = 0; num < mNumRigidBody; num++) {
      for (int j = 0; j < 3; j++)
        mPosition0[num][j] = data(num, j);
      mOrientation0[num] = Quaternion(data(num, 3), data(num, 4), data(num, 5));

      mPosition[num] = mPosition0[num];
      mOrientation[num] = mOrientation0[num];
    }
  }

  /**
   * @brief Get the number of integration steps completed.
   * @return  The current step count.
   */
  int GetFuncCount() {
    return mFuncCount;
  }
};

// As one pybind11 object is stored in a shared library, it is necessary to
// control the visibility of the object.
/**
 * @class ExplicitEuler
 * @brief First-order explicit Euler time integrator for particle simulations.
 *
 * In each time step, this class:
 *  - Computes velocity at t + dt via the Python callback.
 *  - Updates particle positions by x += v * dt.
 *  - Updates orientations via a quaternion rotation of magnitude dt.
 *  - Applies periodic boundary corrections if enabled.
 *  - Writes output at configured intervals.
 */
class __attribute__((visibility("default"))) ExplicitEuler
    : public TimeIntegrator {
public:
  ExplicitEuler() {
  }

  /** @brief Run the time integration until mFinalTime. */
  void Run() {
    mFuncCount = 0;
    double t = 0;
    double h0 = mTimeStep;

    int mMpiRank;
    MPI_Comm_rank(MPI_COMM_WORLD, &mMpiRank);

    while (t < mFinalTime - 1e-5) {
      if (mMpiRank == 0) {
        std::cout << "--------------------------------" << std::endl;
        printf("Time: %.6f, dt: %.6f\n", t, h0);
      }

      double dt = std::min(h0, mFinalTime - t);

      // Fetch velocities at the upcoming time
      VelocityUpdate(t + dt, mPosition, mOrientation, mVelocity,
                     mAngularVelocity);

#pragma omp parallel for schedule(static)
      for (std::size_t num = 0; num < mNumRigidBody; num++) {
        mPosition[num] = mPosition[num] + mVelocity[num] * dt;
        mOrientation[num].Cross(mOrientation[num],
                                Quaternion(mAngularVelocity[num], dt));

        if (mIsPeriodicBoundary) {
          for (int j = 0; j < 3; j++) {
            // periodic boundary condition
            if (mPosition[num][j] > mDomainLimit[1][j]) {
              mPositionOffset[num][j] = mDomainLimit[0][j] - mDomainLimit[1][j];
            } else if (mPosition[num][j] < mDomainLimit[0][j]) {
              mPositionOffset[num][j] = mDomainLimit[1][j] - mDomainLimit[0][j];
            } else {
              mPositionOffset[num][j] = 0.0;
            }
          }
        }
      }

      t += dt;

      // Write output if it's the right step
      if (mFuncCount % mOutputStep == 0) {
        if (mCustomOutputFlag)
          Output();
        else
          Output("output" + std::to_string(mFuncCount));
      }

      if (mIsPeriodicBoundary)
        for (std::size_t num = 0; num < mNumRigidBody; num++) {
          mPosition[num] = mPosition[num] + mPositionOffset[num];
        }

      mFuncCount++;
    }
  }
};

/**
 * @class ExplicitRk4
 * @brief Adaptive Runge-Kutta-Fehlberg integrator for particle simulations.
 *
 * Implements a 7-stage RKF45 method with error control:
 *  - Predefined coefficients (a_ij, b_i, dc_i, c_i) describe the scheme.
 *  - Automatically adjusts dt to meet error threshold mThreshold.
 *  - Supports periodic boundaries and VTK output as in TimeIntegrator.
 */
class __attribute__((visibility("default"))) ExplicitRk4
    : public TimeIntegrator {
protected:
  // constants for integration
  const float a21 = static_cast<float>(1) / static_cast<float>(5);

  const float a31 = static_cast<float>(3) / static_cast<float>(40);
  const float a32 = static_cast<float>(9) / static_cast<float>(40);

  const float a41 = static_cast<float>(44) / static_cast<float>(45);
  const float a42 = static_cast<float>(-56) / static_cast<float>(15);
  const float a43 = static_cast<float>(32) / static_cast<float>(9);

  const float a51 = static_cast<float>(19372) / static_cast<float>(6561);
  const float a52 = static_cast<float>(-25360) / static_cast<float>(2187);
  const float a53 = static_cast<float>(64448) / static_cast<float>(6561);
  const float a54 = static_cast<float>(-212) / static_cast<float>(729);

  const float a61 = static_cast<float>(9017) / static_cast<float>(3168);
  const float a62 = static_cast<float>(-355) / static_cast<float>(33);
  const float a63 = static_cast<float>(46732) / static_cast<float>(5247);
  const float a64 = static_cast<float>(49) / static_cast<float>(176);
  const float a65 = static_cast<float>(-5103) / static_cast<float>(18656);

  const float b1 = static_cast<float>(35) / static_cast<float>(384);
  const float b3 = static_cast<float>(500) / static_cast<float>(1113);
  const float b4 = static_cast<float>(125) / static_cast<float>(192);
  const float b5 = static_cast<float>(-2187) / static_cast<float>(6784);
  const float b6 = static_cast<float>(11) / static_cast<float>(84);

  const float dc1 = b1 - static_cast<float>(5179) / static_cast<float>(57600);
  const float dc3 = b3 - static_cast<float>(7571) / static_cast<float>(16695);
  const float dc4 = b4 - static_cast<float>(393) / static_cast<float>(640);
  const float dc5 =
      b5 - static_cast<float>(-92097) / static_cast<float>(339200);
  const float dc6 = b6 - static_cast<float>(187) / static_cast<float>(2100);
  const float dc7 = static_cast<float>(-1) / static_cast<float>(40);

  const float c[7] = {static_cast<float>(0),
                      static_cast<float>(1) / static_cast<float>(5),
                      static_cast<float>(3) / static_cast<float>(10),
                      static_cast<float>(4) / static_cast<float>(5),
                      static_cast<float>(8) / static_cast<float>(9),
                      static_cast<float>(1),
                      static_cast<float>(1)};

  // Storage for runge-kutta stages
  std::vector<Vec3> mVelocityK1;
  std::vector<Vec3> mVelocityK2;
  std::vector<Vec3> mVelocityK3;
  std::vector<Vec3> mVelocityK4;
  std::vector<Vec3> mVelocityK5;
  std::vector<Vec3> mVelocityK6;
  std::vector<Vec3> mVelocityK7;

  std::vector<Vec3> mAngularVelocityK1;
  std::vector<Vec3> mAngularVelocityK2;
  std::vector<Vec3> mAngularVelocityK3;
  std::vector<Vec3> mAngularVelocityK4;
  std::vector<Vec3> mAngularVelocityK5;
  std::vector<Vec3> mAngularVelocityK6;
  std::vector<Vec3> mAngularVelocityK7;

  double mThreshold;  //!< Error tolerance for adaptive stepping

public:
  ExplicitRk4() : mThreshold(1e-3) {
  }

  /**
   * @brief Configure number of particles and allocate stage buffers.
   * @param numRigidBody [in] Number of particles.
   *
   * Resizes all stage vectors (K1...K7) and angular stages to match.
   */
  void SetNumRigidBody(std::size_t numRigidBody) {
    TimeIntegrator::SetNumRigidBody(numRigidBody);

    mVelocityK1.resize(mNumRigidBody);
    mVelocityK2.resize(mNumRigidBody);
    mVelocityK3.resize(mNumRigidBody);
    mVelocityK4.resize(mNumRigidBody);
    mVelocityK5.resize(mNumRigidBody);
    mVelocityK6.resize(mNumRigidBody);
    mVelocityK7.resize(mNumRigidBody);

    mAngularVelocityK1.resize(mNumRigidBody);
    mAngularVelocityK2.resize(mNumRigidBody);
    mAngularVelocityK3.resize(mNumRigidBody);
    mAngularVelocityK4.resize(mNumRigidBody);
    mAngularVelocityK5.resize(mNumRigidBody);
    mAngularVelocityK6.resize(mNumRigidBody);
    mAngularVelocityK7.resize(mNumRigidBody);
  }
  /**
   * @brief Set the maximum allowed local error for adaptive RK.
   * @param threshold [in] Desired error tolerance per time step.
   */
  void SetThreshold(double threshold) {
    mThreshold = threshold;
  }

  void Run() {
    mFuncCount = 0;
    double t = 0;
    double h0 = mTimeStep;

    // initial velocity evaluation at t=0
    VelocityUpdate(t, mPosition, mOrientation, mVelocity, mAngularVelocity);

    // main time‐stepping loop
    while (t < mFinalTime - 1e-5) {
      if (mMpiRank == 0) {
        std::cout << "--------------------------------" << std::endl;
        printf("Time: %.6f, dt: %.6f\n", t, h0);
      }
      // choose dt so we never overshoot final time
      double dt = std::min(h0, mFinalTime - t);

// copy current state into “base” arrays before stages
#pragma omp parallel for schedule(static)
      for (std::size_t num = 0; num < mNumRigidBody; num++) {
        if (mIsPeriodicBoundary)
          mPosition0[num] = mPosition[num] + mPositionOffset[num];
        else
          mPosition0[num] = mPosition[num];
        mOrientation0[num] = mOrientation[num];
      }

      double err = 100;
      while (err > mThreshold) {
        for (int i = 1; i < 7; i++) {
          switch (i) {
            case 1:
#pragma omp parallel for schedule(static)
              for (std::size_t num = 0; num < mNumRigidBody; num++) {
                mPosition[num] =
                    mPosition0[num] + mVelocityK1[num] * (a21 * dt);
                mOrientation[num].Cross(
                    mOrientation0[num],
                    Quaternion(mAngularVelocityK1[num] * a21, dt));
              }
              break;
            case 2:
#pragma omp parallel for schedule(static)
              for (std::size_t num = 0; num < mNumRigidBody; num++) {
                mPosition[num] = mPosition0[num] +
                                 mVelocityK1[num] * (a31 * dt) +
                                 mVelocityK2[num] * (a32 * dt);
                mOrientation[num].Cross(
                    mOrientation0[num],
                    Quaternion((mAngularVelocityK1[num] * a31 +
                                mAngularVelocityK2[num] * a32),
                               dt));
              }
              break;
            case 3:
#pragma omp parallel for schedule(static)
              for (std::size_t num = 0; num < mNumRigidBody; num++) {
                mPosition[num] = mPosition0[num] +
                                 mVelocityK1[num] * (a41 * dt) +
                                 mVelocityK2[num] * (a42 * dt) +
                                 mVelocityK3[num] * (a43 * dt);
                mOrientation[num].Cross(
                    mOrientation0[num],
                    Quaternion((mAngularVelocityK1[num] * a41 +
                                mAngularVelocityK2[num] * a42 +
                                mAngularVelocityK3[num] * a43),
                               dt));
              }
              break;
            case 4:
#pragma omp parallel for schedule(static)
              for (std::size_t num = 0; num < mNumRigidBody; num++) {
                mPosition[num] = mPosition0[num] +
                                 mVelocityK1[num] * (a51 * dt) +
                                 mVelocityK2[num] * (a52 * dt) +
                                 mVelocityK3[num] * (a53 * dt) +
                                 mVelocityK4[num] * (a54 * dt);
                mOrientation[num].Cross(
                    mOrientation0[num],
                    Quaternion((mAngularVelocityK1[num] * a51 +
                                mAngularVelocityK2[num] * a52 +
                                mAngularVelocityK3[num] * a53 +
                                mAngularVelocityK4[num] * a54),
                               dt));
              }
              break;
            case 5:
#pragma omp parallel for schedule(static)
              for (std::size_t num = 0; num < mNumRigidBody; num++) {
                mPosition[num] = mPosition0[num] +
                                 mVelocityK1[num] * (a61 * dt) +
                                 mVelocityK2[num] * (a62 * dt) +
                                 mVelocityK3[num] * (a63 * dt) +
                                 mVelocityK4[num] * (a64 * dt) +
                                 mVelocityK5[num] * (a65 * dt);
                mOrientation[num].Cross(
                    mOrientation0[num],
                    Quaternion((mAngularVelocityK1[num] * a61 +
                                mAngularVelocityK2[num] * a62 +
                                mAngularVelocityK3[num] * a63 +
                                mAngularVelocityK4[num] * a64 +
                                mAngularVelocityK5[num] * a65),
                               dt));
              }
              break;
            case 6:
#pragma omp parallel for schedule(static)
              for (std::size_t num = 0; num < mNumRigidBody; num++) {
                mPosition[num] =
                    mPosition0[num] + mVelocityK1[num] * (b1 * dt) +
                    mVelocityK3[num] * (b3 * dt) +
                    mVelocityK4[num] * (b4 * dt) +
                    mVelocityK5[num] * (b5 * dt) + mVelocityK6[num] * (b6 * dt);
                mOrientation[num].Cross(
                    mOrientation0[num],
                    Quaternion((mAngularVelocityK1[num] * b1 +
                                mAngularVelocityK3[num] * b3 +
                                mAngularVelocityK4[num] * b4 +
                                mAngularVelocityK5[num] * b5 +
                                mAngularVelocityK6[num] * b6),
                               dt));
              }
              break;
          }

          VelocityUpdate(t + c[i] * dt, mPosition, mOrientation, mVelocity,
                         mAngularVelocity);

          // update velocity
          switch (i) {
            case 1:
#pragma omp parallel for schedule(static)
              for (std::size_t num = 0; num < mNumRigidBody; num++) {
                mVelocityK2[num] = mVelocity[num];
                mAngularVelocityK2[num] = dexpinv(
                    mAngularVelocityK1[num] * a21 * dt, mAngularVelocity[num]);
              }
              break;
            case 2:
#pragma omp parallel for schedule(static)
              for (std::size_t num = 0; num < mNumRigidBody; num++) {
                mVelocityK3[num] = mVelocity[num];
                mAngularVelocityK3[num] =
                    dexpinv((mAngularVelocityK1[num] * a31 +
                             mAngularVelocityK2[num] * a32) *
                                dt,
                            mAngularVelocity[num]);
              }
              break;
            case 3:
#pragma omp parallel for schedule(static)
              for (std::size_t num = 0; num < mNumRigidBody; num++) {
                mVelocityK4[num] = mVelocity[num];
                mAngularVelocityK4[num] =
                    dexpinv((mAngularVelocityK1[num] * a41 +
                             mAngularVelocityK2[num] * a42 +
                             mAngularVelocityK3[num] * a43) *
                                dt,
                            mAngularVelocity[num]);
              }
              break;
            case 4:
#pragma omp parallel for schedule(static)
              for (std::size_t num = 0; num < mNumRigidBody; num++) {
                mVelocityK5[num] = mVelocity[num];
                mAngularVelocityK5[num] =
                    dexpinv((mAngularVelocityK1[num] * a51 +
                             mAngularVelocityK2[num] * a52 +
                             mAngularVelocityK3[num] * a53 +
                             mAngularVelocityK4[num] * a54) *
                                dt,
                            mAngularVelocity[num]);
              }
              break;
            case 5:
#pragma omp parallel for schedule(static)
              for (std::size_t num = 0; num < mNumRigidBody; num++) {
                mVelocityK6[num] = mVelocity[num];
                mAngularVelocityK6[num] =
                    dexpinv((mAngularVelocityK1[num] * a61 +
                             mAngularVelocityK2[num] * a62 +
                             mAngularVelocityK3[num] * a63 +
                             mAngularVelocityK4[num] * a64 +
                             mAngularVelocityK5[num] * a65) *
                                dt,
                            mAngularVelocity[num]);
              }
              break;
            case 6:
#pragma omp parallel for schedule(static)
              for (std::size_t num = 0; num < mNumRigidBody; num++) {
                mVelocityK7[num] = mVelocity[num];
                mAngularVelocityK7[num] =
                    dexpinv((mAngularVelocityK1[num] * b1 +
                             mAngularVelocityK3[num] * b3 +
                             mAngularVelocityK4[num] * b4 +
                             mAngularVelocityK5[num] * b5 +
                             mAngularVelocityK6[num] * b6) *
                                dt,
                            mAngularVelocity[num]);
              }
              break;
          }
        }

        // error check
        {
          err = 0.0;

#pragma omp parallel for schedule(static) reduction(+ : err)
          for (std::size_t num = 0; num < mNumRigidBody; num++) {
            Vec3 velocity_err =
                (mVelocityK1[num] * dc1 + mVelocityK3[num] * dc3 +
                 mVelocityK4[num] * dc4 + mVelocityK5[num] * dc5 +
                 mVelocityK6[num] * dc6 + mVelocityK7[num] * dc7) *
                dt;

            err += velocity_err.SquareMag();
          }
          err = sqrt(err);

          if (err > mThreshold) {
            // reduce time step
            dt = dt * std::max(0.8 * pow(err / mThreshold, -0.2), 0.1);
            dt = std::max(dt, 1e-6);
          }
        }
      }

      t += dt;

      // increase time step
      {
        double temp = 1.25 * pow(err / mThreshold, 0.2);
        if (temp > 0.2) {
          dt = dt / temp;
        } else {
          dt *= 5.0;
        }
      }
      dt = std::min(dt, h0);

      if (mFuncCount % mOutputStep == 0) {
        if (mCustomOutputFlag)
          Output();
        else
          Output("output" + std::to_string(mFuncCount));
      }

      // reset k1 for next step
#pragma omp parallel for schedule(static)
      for (std::size_t num = 0; num < mNumRigidBody; num++) {
        mVelocityK1[num] = mVelocityK7[num];
        mAngularVelocityK1[num] = mAngularVelocityK7[num];
      }

      mFuncCount++;
    }
  }
};

#endif