/*
Highly Optimized Object-oriented Many-particle Dynamics -- Blue Edition
(HOOMD-blue) Open Source Software License Copyright 2008, 2009 Ames Laboratory
Iowa State University and The Regents of the University of Michigan All rights
reserved.
*/

// $Id: FreeDrainForceCompute.cuh 2011-11-19 martbert $
// Maintainer: martbert

#include <hoomd/hoomd_config.h>
// need to include the particle data definition
#include <hoomd/HOOMDMath.h>
#include <hoomd/ParticleData.cuh>
#include <hoomd/Index1D.h>
// #include <hoomd/ForceCompute.cuh>

/*! \file CuboidForceCompute.cuh
    \brief Declares GPU kernel code for calculating the Cuboid Constant Force. Used by CuboidForceComputeGPU.
*/

#ifndef _FREEDRAINFORCECOMPUTE_CUH_
#define _FREEDRAINFORCECOMPUTE_CUH_

//! Kernel driver that computes Cuboid forces for CuboidForceComputeGPU
cudaError_t gpu_compute_freedrain_forces(float4* d_force,
										 float* d_virial,
										 const unsigned int N,
										 const Scalar4 *d_pos,
										 float *d_charge,
										 float *d_org_charge,
										 const BoxDim& box,
										 const unsigned int *d_group1_members,
										 unsigned int group1_size,
										 const unsigned int *d_group2_members,
										 unsigned int group2_size,
										 Scalar Ex,
										 Scalar Ey,
										 Scalar Ez,
										 Scalar lD,
										 Scalar bj,
										 Scalar qt,
										 Scalar cut2,
										 unsigned int block_size);

#endif

