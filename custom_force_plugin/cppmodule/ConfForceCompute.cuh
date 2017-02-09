/*
Highly Optimized Object-oriented Many-particle Dynamics -- Blue Edition
(HOOMD-blue) Open Source Software License Copyright 2008, 2009 Ames Laboratory
Iowa State University and The Regents of the University of Michigan All rights
reserved.
*/

// $Id: ConfForceGPU.cuh 2587 2010-01-08 17:02:54Z martbert $
// Maintainer: martbert

#include <hoomd/hoomd_config.h>
// need to include the particle data definition
#include <hoomd/HOOMDMath.h>
#include <hoomd/ParticleData.cuh>
// #include <hoomd/ForceCompute.cuh>

/*! \file ConfForceCompute.cuh
    \brief Declares GPU kernel code for calculating the Cuboid Constant Force. Used by ConfForceComputeGPU.
*/

#ifndef _CONFFORCECOMPUTE_CUH_
#define _CONFFORCECOMPUTE_CUH_

//! Kernel driver
cudaError_t gpu_compute_conf_forces(float4* d_force,
									 const unsigned int *d_group_members,
									 unsigned int group_size,
									 const unsigned int N,
									 const Scalar4 *d_pos,
									 Scalar k,
									 Scalar roff,
									 int dflag,
									 unsigned int block_size);

#endif

