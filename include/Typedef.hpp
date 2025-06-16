/**
 * @file Typedef.hpp
 * @brief Scalar types and Kokkos::View aliases used throughout the project.
 *
 * Defines fundamental floating-point scalars (`Double`, `Float`) and 1D/2D
 * Kokkos::View aliases for floats, doubles, ints, and std::size_t in both
 * host and device execution spaces.
 */
#ifndef _Typedef_Hpp_
#define _Typedef_Hpp_

#include <Kokkos_Core.hpp>

// Scalar types
typedef double Double;
typedef float Float;

// Float matrix views
typedef Kokkos::View<Float **, Kokkos::DefaultHostExecutionSpace>
    HostFloatMatrix;
typedef Kokkos::View<Float **, Kokkos::DefaultExecutionSpace> DeviceFloatMatrix;

// Float vector views
typedef Kokkos::View<Float *, Kokkos::DefaultHostExecutionSpace>
    HostFloatVector;
typedef Kokkos::View<Float *, Kokkos::DefaultExecutionSpace> DeviceFloatVector;

// Double matrix views
typedef Kokkos::View<Double **, Kokkos::DefaultHostExecutionSpace>
    HostDoubleMatrix;
typedef Kokkos::View<Double **, Kokkos::DefaultExecutionSpace>
    DeviceDoubleMatrix;

// Double vector views
typedef Kokkos::View<Double *, Kokkos::DefaultHostExecutionSpace>
    HostDoubleVector;
typedef Kokkos::View<Double *, Kokkos::DefaultExecutionSpace>
    DeviceDoubleVector;

// Int vector views
typedef Kokkos::View<int *, Kokkos::DefaultHostExecutionSpace> HostIntVector;
typedef Kokkos::View<int *, Kokkos::DefaultExecutionSpace> DeviceIntVector;

// Int matrix views
typedef Kokkos::View<int **, Kokkos::DefaultHostExecutionSpace> HostIntMatrix;
typedef Kokkos::View<int **, Kokkos::DefaultExecutionSpace> DeviceIntMatrix;

// Index matrix views
typedef Kokkos::View<std::size_t **, Kokkos::DefaultHostExecutionSpace>
    HostIndexMatrix;
typedef Kokkos::View<std::size_t **, Kokkos::DefaultExecutionSpace>
    DeviceIndexMatrix;

// Index vector views
typedef Kokkos::View<std::size_t *, Kokkos::DefaultHostExecutionSpace>
    HostIndexVector;
typedef Kokkos::View<std::size_t *, Kokkos::DefaultExecutionSpace>
    DeviceIndexVector;

#endif