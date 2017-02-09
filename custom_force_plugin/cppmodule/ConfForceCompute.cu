/*
Highly Optimized Object-oriented Many-particle Dynamics -- Blue Edition
(HOOMD-blue) Open Source Software License Copyright 2008, 2009 Ames Laboratory
Iowa State University and The Regents of the University of Michigan All rights
reserved.
*/

// $Id: ConfForceGPU.cu martbert $
// Maintainer: martbert

#include "ConfForceCompute.cuh"

#include <assert.h>


/*! \file ConfForceGPU.cu
    \brief Defines GPU kernel code for calculating the harmonic confinement forces. Used by ConfForceComputeGPU.
*/

//! Kernel for caculating confinement forces on the GPU
extern "C" __global__ 
void gpu_compute_conf_forces_kernel(float4* d_force,
									 const unsigned int *d_group_members,
									 unsigned int group_size,
									 const unsigned int N,
									 const Scalar4 *d_pos,
									 Scalar k,
									 Scalar roff,
									 int dflag)
    {
	
    // start by identifying which particle we are to handle
    int group_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (group_idx >= group_size)
        return;
    
    unsigned int idx = d_group_members[group_idx];
                
    // read in position, velocity, net force, and mass
    float4 pos = d_pos[idx];
        
    // initialize the force to 0
    float4 force = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    
    if (dflag == 1)
	{
		// get pos (FLOPS: 3)
		float y = pos.y;
		float z = pos.z;
		
		// Calculate distance to center
		float r = sqrtf(y*y + z*z);
		
		//Verify that the particle is outside the offset radius
		if (r > roff)
		{
			float forcedivr = - k * (r - roff) / r;
			force.y += forcedivr * y;
			force.z += forcedivr * z;
		}
	} else if (dflag == 2)
	{
		// get pos (FLOPS: 3)
		float x = pos.x;
		float z = pos.z;
		
		// Calculate distance to center
		float r = sqrtf(x*x + z*z);
		
		//Verify that the particle is outside the offset radius
		if (r > roff)
		{
			float forcedivr = - k * (r - roff) / r;
			force.x += forcedivr * x;
			force.z += forcedivr * z;
		}
	} else if (dflag == 3)
	{
		// get pos (FLOPS: 3)
		float y = pos.y;
		float x = pos.x;
		
		// Calculate distance to center
		float r = sqrtf(y*y + x*x);
		
		//Verify that the particle is outside the offset radius
		if (r > roff)
		{
			float forcedivr = - k * (r - roff) / r;
			force.y += forcedivr * y;
			force.x += forcedivr * x;
		}
	}		
	// now that the force calculation is complete, write out the result (MEM TRANSFER: 20 bytes);
    d_force[idx] = force;
}

/*! \param force_data Force data on GPU to write forces to
    \returns Any error code resulting from the kernel launch
    \note Always returns cudaSuccess in release builds to avoid the cudaThreadSynchronize()
*/
cudaError_t gpu_compute_conf_forces(float4* d_force,
									 const unsigned int *d_group_members,
									 unsigned int group_size,
									 const unsigned int N,
									 const Scalar4 *d_pos,
									 Scalar k,
									 Scalar roff,
									 int dflag,
									 unsigned int block_size)
    {
    // check that block_size is valid
    assert(block_size != 0);
    
    // setup the grid to run the kernel
   dim3 grid( (int)ceil((double)group_size / (double)block_size), 1, 1);
    dim3 threads(block_size, 1, 1);
    
   cudaMemset(d_force, 0, sizeof(float4)*N);
   gpu_compute_conf_forces_kernel<<< grid, threads>>>(d_force,d_group_members,group_size,N,d_pos,k,roff,dflag);

    return cudaSuccess;
    }

