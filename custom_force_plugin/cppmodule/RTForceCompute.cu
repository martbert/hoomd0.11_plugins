/*
Highly Optimized Object-oriented Many-particle Dynamics -- Blue Edition
(HOOMD-blue) Open Source Software License Copyright 2008, 2014 Ames Laboratory
Iowa State University and The Regents of the University of Michigan All rights
reserved.
*/

// $Id: RTForceGPU.cu martbert $
// Maintainer: martbert

#include "RTForceCompute.cuh"

#include <hoomd/saruprngCUDA.h>

#include <assert.h>


/*! \file RTForceComputeGPU.cu
    \brief Defines GPU kernel code for calculating the run and tumble forces. Used by RTForceComputeGPU.
*/

//! Kernel for caculating R/T forces on the GPU
extern "C" __global__ 
void gpu_compute_rt_forces_kernel(float4* d_force,
                                 const unsigned int N,
                                 const Scalar4 *d_pos,
                                 const unsigned int *d_rtag,
                                 const BoxDim box,
                                 const uint2 *d_bond_table,
                                 const unsigned int *d_bond_types,
                                 const unsigned int nbonds,
                                 const Scalar4 *d_params,
                                 int *d_state,
                                 unsigned int timestep)
    {
	
    // start by identifying which particle we are to handle
    int pair_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (pair_idx >= nbonds)
    	return;
    
    unsigned int i = d_rtag[d_bond_table[pair_idx].x];
    unsigned int j = d_rtag[d_bond_table[pair_idx].y];
    unsigned int type = d_bond_types[pair_idx];
    float4 params = make_float4(d_params[type].x,d_params[type].y,d_params[type].z,d_params[type].w);
    
    // Verify that the swimmer will be affected:
    // Get swimmer's state
    int current_state = d_state[pair_idx];
    // Generate a random number
    SaruGPU s(i, j, timestep); // 3 dimensional seeding
    float ran = s.f(0,1);

    // If it's running see if changes to tumbling
    // And vice versa
    if (current_state == 0 && ran < 0.5f*params.x)
        current_state = 1;
    else if (current_state == 0 && ran < params.x)
        current_state = -1;
    else if ((current_state == 1 || current_state == -1) && ran < params.y)
        current_state = 0;
    else
        ;
    d_state[pair_idx] = current_state;

    // read in position, velocity, net force, and mass
    float3 pi = make_float3(d_pos[i].x, d_pos[i].y, d_pos[i].z);
    float3 pj = make_float3(d_pos[j].x, d_pos[j].y, d_pos[j].z);

    // Calculate dr
    float3 dr = pi - pj;

    // Calculate min image
    dr = box.minImage(dr);
    
    // Calculate rsq
    float rsq = dot(dr,dr);
    float rinv = rsqrt(rsq);

    // initialize the force to 0
    float4 f = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

    // If random number generated is smaller than prob then run else tumble
    if (current_state == 0)
    {
    	// Run: apply force on both i and j in direction of dr
    	f.x = dr.x * rinv * params.z;
      f.y = dr.y * rinv * params.z;
      f.z = dr.z * rinv * params.z;
      d_force[i].x += f.x;
      d_force[i].y += f.y;
      d_force[i].z += f.z;
      d_force[j].x += f.x;
      d_force[j].y += f.y;
      d_force[j].z += f.z;
    }
    else
    {
    	// Tumble: apply force on both i and j perpendicular to dr
    	float3 dt = make_float3(current_state*dr.y,-current_state*dr.x,0.0f);
      float tsq = dot(dt,dt);
      float tinv = rsqrt(tsq);
    	f.x = dt.x * tinv * params.w;
    	f.y = dt.y * tinv * params.w;
    	d_force[i].x += f.x;
		  d_force[i].y += f.y;
		  d_force[j].x += -f.x;
		  d_force[j].y += -f.y;
    }
}

/*! \param force_data Force data on GPU to write forces to
    \returns Any error code resulting from the kernel launch
    \note Always returns cudaSuccess in release builds to avoid the cudaThreadSynchronize()
*/
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
                                 unsigned int block_size)
    {
    // check that block_size is valid
    assert(block_size != 0);
    
    // setup the grid to run the kernel
   dim3 grid( (int)ceil((double)nbonds / (double)block_size), 1, 1);
   dim3 threads(block_size, 1, 1);
    
   cudaMemset(d_force, 0, sizeof(float4)*N);
   gpu_compute_rt_forces_kernel<<< grid, threads>>>(d_force,
                                                    N,
                                                    d_pos,
                                                    d_rtag,
                                                    box,
                           													d_bond_table,
                                                    d_bond_types,
                           													nbonds,
                                                    d_params,
                                                    d_state,
                           													timestep);

    return cudaSuccess;
    }

