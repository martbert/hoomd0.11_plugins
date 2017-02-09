/*
Highly Optimized Object-oriented Many-particle Dynamics -- Blue Edition
(HOOMD-blue) Open Source Software License Copyright 2008, 2014 Ames Laboratory
Iowa State University and The Regents of the University of Michigan All rights
reserved.
*/

// $Id: RTForceComputeGPU.cuh martbert $
// Maintainer: martbert

#include <hoomd/hoomd_config.h>
// need to include the particle data definition
#include <hoomd/HOOMDMath.h>
#include <hoomd/ParticleData.cuh>

/*! \file RTForceCompute.cuh
*/

#ifndef _RTFORCECOMPUTE_CUH_
#define _RTFORCECOMPUTE_CUH_

//! Kernel driver
cudaError_t gpu_compute_rt_forces(float4* d_force,
								 const unsigned int N,
								 const Scalar4 *d_pos,
								 const unsigned int *d_rtag,
								 const BoxDim box,
								 const uint2 *d_bond_table,
								 const unsigned int *d_bond_types,
								 const unsigned int nbonds,
								 const Scalar4 *d_params,
								 int *d_state,
								 unsigned int timestep,
								 unsigned int block_size);

#endif

