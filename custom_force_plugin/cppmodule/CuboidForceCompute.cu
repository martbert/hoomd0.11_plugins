/*
Highly Optimized Object-oriented Many-particle Dynamics -- Blue Edition
(HOOMD-blue) Open Source Software License Copyright 2008, 2009 Ames Laboratory
Iowa State University and The Regents of the University of Michigan All rights
reserved.
*/

// $Id: CuboidForceGPU.cu 3299 2010-08-04 16:26:59Z martbert $
// Maintainer: martbert

#include "CuboidForceCompute.cuh"

#include <assert.h>


/*! \file CuboidForceGPU.cu
    \brief Defines GPU kernel code for calculating the Cuboid forces. Used by CuboidForceComputeGPU.
*/

//! Texture for reading particle positions
texture<float4, 1, cudaReadModeElementType> pdata_pos_tex;

//! Kernel for caculating Cuboid forces on the GPU
/*! \param force_data Data to write the compute forces to
    \param pdata Particle data arrays to calculate forces on
    \param box Box dimensions for periodic boundary condition handling
*/
extern "C" __global__ 
void gpu_compute_cuboid_forces_kernel(float4* d_force,
										const Scalar4 *d_pos,
										const unsigned int N,
										Scalar3 f,
										Scalar3 min,
										Scalar3 max)
    {
	
    // start by identifying which particle we are to handle
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= N)
        return;
                    
    // initialize the force to 0
    float4 force = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    
	// get pos (FLOPS: 3)
    float x = d_pos[idx].x;
    float y = d_pos[idx].y;
    float z = d_pos[idx].z;
	
	// Check if particle is in the cuboid and if so apply force
	if (x >= min.x && x < max.x && y >= min.y && y < max.y && z >= min.z && z < max.z)
	{
		force.x += f.x;
		force.y += f.y;
		force.z += f.z;
	}
	
	// now that the force calculation is complete, write out the result (MEM TRANSFER: 20 bytes);
    d_force[idx] = force;
}

/*! \param force_data Force data on GPU to write forces to
    \param pdata Particle data on the GPU to perform the calculation on
    \param box Box dimensions (in GPU format) to use for periodic boundary conditions
    \param f
	\param min
	\param max
    \param block_size Block size to use when performing calculations

    \returns Any error code resulting from the kernel launch
    \note Always returns cudaSuccess in release builds to avoid the cudaThreadSynchronize()
*/
cudaError_t gpu_compute_cuboid_forces(float4* d_force,
										const Scalar4 *d_pos,
										const unsigned int N,
                                         const Scalar3& f,
										 const Scalar3& min,
										 const Scalar3& max,
                                         int block_size)
    {
    assert(f);
	assert(min);
	assert(max);
    // check that block_size is valid
    assert(block_size != 0);
    
    // setup the grid to run the kernel
    dim3 grid( (int)ceil((double)N / (double)block_size), 1, 1);
    dim3 threads(block_size, 1, 1);
    
    cudaMemset(d_force, 0, sizeof(float4) * N);
    // run the kernel
    gpu_compute_cuboid_forces_kernel<<< grid, threads>>>(d_force, d_pos, N, f,min,max);
            
    return cudaSuccess;
    }

