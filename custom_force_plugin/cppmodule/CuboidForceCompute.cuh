/*
Highly Optimized Object-oriented Many-particle Dynamics -- Blue Edition
(HOOMD-blue) Open Source Software License Copyright 2008, 2009 Ames Laboratory
Iowa State University and The Regents of the University of Michigan All rights
reserved.
*/

// $Id: CuboidForceGPU.cuh 2587 2010-01-08 17:02:54Z martbert $
// Maintainer: martbert

#include <hoomd/hoomd_config.h>
// need to include the particle data definition
#include <hoomd/HOOMDMath.h>
#include <hoomd/ParticleData.cuh>
// #include <hoomd/ForceCompute.cuh>

/*! \file CuboidForceCompute.cuh
    \brief Declares GPU kernel code for calculating the Cuboid Constant Force. Used by CuboidForceComputeGPU.
*/

#ifndef _CUBOIDFORCECOMPUTE_CUH_
#define _CUBOIDFORCECOMPUTE_CUH_

//! Kernel driver that computes Cuboid forces for CuboidForceComputeGPU
cudaError_t gpu_compute_cuboid_forces(float4* d_force,
										 const Scalar4* d_pos,
										 const unsigned int N,
                                         const Scalar3& f,
										 const Scalar3& min,
										 const Scalar3& max,
                                         int block_size);

#endif

