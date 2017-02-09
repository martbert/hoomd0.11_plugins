/*
Highly Optimized Object-oriented Many-particle Dynamics -- Blue Edition
(HOOMD-blue) Open Source Software License Copyright 2008, 2009 Ames Laboratory
Iowa State University and The Regents of the University of Michigan All rights
reserved.
*/

// $Id: FreeDrainForceCompute.cu 2011-11-19 martbert $
// Maintainer: martbert

#include "FreeDrainForceCompute.cuh"
#include <assert.h>
#include <stdio.h>

/*! \file FreeDrainForceCompute.cu
    \brief Defines GPU kernel code for calculating the free draining forces. Used by FreeDrainForceComputeGPU.
*/

/*! Sequence of kernels related to the first loop:
	- calculate charges imparted locally while accumulating the total
	- reduce the total
	- normalize local charges and accumulate in total
*/

//! Kernel calculating charges on neighbouring fluid particles
extern "C" __global__ 
void gpu_group2_compute_charges_kernel(const unsigned int N,
								 	   const Scalar4 *d_pos,
								 	   float *d_charge,
                                       float *d_org_charge,
								 	   float *l_charge,
								 	   BoxDim box,
								 	   const unsigned int *d_group1_members,
								 	   unsigned int group1_size,
								 	   const unsigned int *d_group2_members,
								 	   unsigned int group2_size,
								 	   unsigned int m_id,
								 	   Scalar lD,
								 	   Scalar cut2,
								 	   float *per_block_counter)
{
	
	// prepare counting the number of solvent particles affected
	extern __shared__ float sdata[];
	
    // start by identifying which particle we are to handle
    int group_idx = blockIdx.x * blockDim.x + threadIdx.x;
    float factor = 0.0f;

	if (group_idx < group2_size)
    {
		unsigned int idx = d_group2_members[group_idx];
		unsigned int cur_j = d_group1_members[m_id];

		// set local and global charge
		l_charge[idx] = 0.0f;
		if (m_id == 0)
			d_charge[idx] = 0.0f;

		// get monomer (group 2) particle info
    	float4 posj = d_pos[cur_j];
    	float qj = d_org_charge[m_id];
    	// float qj = 1.0f;
	
	    if (idx < N)
        {  
			// read in the position of our particle. (MEM TRANSFER: 16 bytes)
			float4 pos = d_pos[idx];
					
			// calculate dr (with periodic boundary conditions) (FLOPS: 3)
			Scalar3 dr;
			dr.x = pos.x - posj.x;
			dr.y = pos.y - posj.y;
			dr.z = pos.z - posj.z;
				
			// apply periodic boundary conditions: (FLOPS 12)
			dr = box.minImage(dr);
				
			// calculate r squared (FLOPS: 5)
			float rsq = dot(dr, dr);
			// add force if particle is within debye length
			if (rsq < cut2)
			{
				float r = sqrtf(rsq);
				factor = -qj * expf( -r / lD ) / r;
				l_charge[idx] += factor;
			}
		}
	}
	
	// update counter sum
	int i = threadIdx.x;
	 sdata[i] = factor;
  	__syncthreads();
  	
  	// contiguous range pattern
	for(int offset = blockDim.x / 2; offset > 0; offset >>= 1)
	{
		if(threadIdx.x < offset)
		{
			// add a partial sum upstream to our own
			sdata[i] += sdata[i + offset];
		}
		
		// wait until all threads in the block have
		// updated their partial sums
		__syncthreads();
	}
	
	// thread 0 writes the final result
	if(threadIdx.x == 0)
	{
		per_block_counter[blockIdx.x] = sdata[0];
	}
}


//! Kernel for normalizing and adding charges
extern "C" __global__ 
void gpu_group2_normalize_add_charges_kernel(const unsigned int N,
								 	   float *d_charge,
                                       float *d_org_charge,
								 	   float *l_charge,
								 	   const unsigned int *d_group2_members,
								 	   unsigned int group2_size,
								 	   float factor,
								 	   float *factors,
								 	   unsigned int m_id)
{
	// start by identifying which particle we are to handle
    int group_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
	if (group_idx >= group2_size)
        return;
    
    unsigned int idx = d_group2_members[group_idx];
	
    if (idx >= N)
        return;
    
    // normalize charge
    factor *= fabs(d_org_charge[m_id]);
    l_charge[idx] *= factor;

	// store factor for later use
    factors[m_id] = factor;

    // add charge
    d_charge[idx] += l_charge[idx];

}

/*! Squence of kernels related to the second part:
	- condense and compute forces for group2
	- compute forces for group1 taking into account condensation
*/


//! Kernel to calculate condensation and compute forces for group2
extern "C" __global__ 
void gpu_group2_condense_compute_forces_kernel(float4* d_force,
										 	 const unsigned int N,
											 float *d_charge,
											 const unsigned int *d_group2_members,
											 unsigned int group2_size,
											 unsigned int *condensed_flags,
											 Scalar Ex,
											 Scalar Ey,
											 Scalar Ez,
											 Scalar qt)
{
	// start by identifying which particle we are to handle
    int group_idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (group_idx < group2_size)
    {
		unsigned int idx = d_group2_members[group_idx];
	
	    if (idx < N)
        {
        	// Check for condensation
        	if (fabsf(d_charge[idx]) >= fabsf(qt))
        	{
        		d_charge[idx] = 0.0f;
        		condensed_flags[group_idx] = 1;
        	} else {
        		condensed_flags[group_idx] = 0;
        	}

        	// Compute forces
        	d_force[idx].x = d_charge[idx]*Ex;
        	d_force[idx].y = d_charge[idx]*Ey;
        	d_force[idx].z = d_charge[idx]*Ez;
        }
	}
}



//! Kernel calculating charges and forces on group1
extern "C" __global__ 
void gpu_group1_compute_charges_forces_kernel(float4* d_force,
										 const unsigned int N,
										 const Scalar4 *d_pos,
										 float *d_charge,
                                         float *d_org_charge,
										 BoxDim box,
										 const unsigned int *d_group1_members,
										 unsigned int group1_size,
										 const unsigned int *d_group2_members,
										 unsigned int group2_size,
										 unsigned int *condensed_flags,
										 float *factors,
										 Scalar Ex,
										 Scalar Ey,
										 Scalar Ez,
										 Scalar lD,
										 Scalar cut2)
{
	// start by identifying which particle we are to handle
    int group_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
	if (group_idx >= group1_size)
        return;
    
    unsigned int idx = d_group1_members[group_idx];
	
    if (idx >= N)
        return;
    
    // read in the position of our particle. (MEM TRANSFER: 16 bytes)
	float4 posi = d_pos[idx];
        
    // Get charge of particle idx
	float org_qi = d_org_charge[group_idx];
    float qi = org_qi;
	// float qi = 1.0f;
	float qtemp = 0.0f;
	
	// Take care of condensation
	for (unsigned int j = 0; j < group2_size; j++)
	{
		// If condensed then subtract charge
		if (condensed_flags[j] == 1)
		{
			unsigned int cur_j = d_group2_members[j];
			float4 posj = d_pos[cur_j];

			// calculate dr (with periodic boundary conditions) (FLOPS: 3)
			Scalar3 dr;
			dr.x = posi.x - posj.x;
			dr.y = posi.y - posj.y;
			dr.z = posi.z - posj.z;
				
			// apply periodic boundary conditions: (FLOPS 12)
			dr = box.minImage(dr);
				
			// calculate r squared (FLOPS: 5)
			float rsq = dot(dr, dr);
			// subtract charge if it is within cutoff
			if (rsq < cut2)
			{
				float r = sqrtf(rsq);
				qtemp = org_qi * expf( -r / lD ) / r;
				qtemp *= factors[group_idx];
				qi -= qtemp;
			}
		}
	}

	// Add force due to field
	d_force[idx].x = qi*Ex;
	d_force[idx].y = qi*Ey;
	d_force[idx].z = qi*Ez;
	d_charge[idx] = qi;
}

//! Kernel calculating debye-huckel forces
extern "C" __global__ 
void gpu_group1_compute_dh_forces_kernel(float4* d_force,
										 const unsigned int N,
										 const Scalar4 *d_pos,
										 float *d_charge,
										 BoxDim box,
										 const unsigned int *d_group1_members,
										 unsigned int group1_size,
										 Scalar lD,
										 Scalar bj,
										 Scalar cut2)
{
	// start by identifying which particle we are to handle
    int group_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
	if (group_idx >= group1_size)
        return;
    
    unsigned int idx = d_group1_members[group_idx];
	
    if (idx >= N)
        return;
    
    // read in the position of our particle. (MEM TRANSFER: 16 bytes)
	float4 posi = d_pos[idx];
        
    // Get charge of particle idx
	float qi = d_charge[idx];

	// Initialize force to 0
	float4 forcei = make_float4(0.0f,0.0f,0.0f,0.0f);

	// Add force due to Debye-Huckel type interactions
	//if (1.0f == 0.0f)
	for (unsigned int j = 0; j < group1_size; j++)
	{
		unsigned int cur_j = d_group1_members[j];
		
		if (cur_j != idx) 
		{			
			// get the polymer's position (MEM TRANSFER: 16 bytes)
			float4 posj = d_pos[cur_j];
			float qj = d_charge[cur_j];
			
			// calculate dr (with periodic boundary conditions) (FLOPS: 3)
			Scalar3 dr;
			dr.x = posi.x - posj.x;
			dr.y = posi.y - posj.y;
			dr.z = posi.z - posj.z;
				
			// apply periodic boundary conditions: (FLOPS 12)
			dr = box.minImage(dr);
				
			// calculate r squared (FLOPS: 5)
			float rsq = dot(dr, dr);
			// add force if particle is within debye length
			if (rsq < cut2)
			{
				float r = sqrtf(rsq);
				float energy = bj * qi * qj * expf( -r / lD ) / r;
				float forcedivr = energy / r / r / lD * (r + lD);
				forcei.x += forcedivr * dr.x;
				forcei.y += forcedivr * dr.y;
				forcei.z += forcedivr * dr.z;
				forcei.w += 0.5*energy;
			}
		}
	}
	d_force[idx].x += forcei.x;
	d_force[idx].y += forcei.y;
	d_force[idx].z += forcei.z;
	d_force[idx].w += forcei.w;
}
		
//! Kernel to calculate sum on the GPU
extern "C" __global__ 
void block_sum(const float *input,
                          float *per_block_results,
                          const size_t n)
{
  extern __shared__ float ssdata[];

  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

  // load input into __shared__ memory
  float x = 0;
  if(i < n)
  {
    x = input[i];
  }
  ssdata[threadIdx.x] = x;
  __syncthreads();

  // contiguous range pattern
  for(int offset = blockDim.x / 2;
      offset > 0;
      offset >>= 1)
  {
    if(threadIdx.x < offset)
    {
      // add a partial sum upstream to our own
      ssdata[threadIdx.x] += ssdata[threadIdx.x + offset];
    }

    // wait until all threads in the block have
    // updated their partial sums
    __syncthreads();
  }

  // thread 0 writes the final result
  if(threadIdx.x == 0)
  {
    per_block_results[blockIdx.x] = ssdata[0];
  }
}


/*! \param d_force Force data on GPU to write forces to
    \param pdata Particle data on the GPU to perform the calculation on
    \param d_group1_members Members of group1
    \param d_group2_members Members of group2 generating EOF around group1
	\param Ex Field in x
	\param Ey Field in y
	\param Ez Field in z
	\param lD Debye length
    \param block_size Block size to use when performing calculations

    \returns Any error code resulting from the kernel launch
    \note Always returns cudaSuccess in release builds to avoid the cudaThreadSynchronize()
*/
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
										 unsigned int block_size)
    {

    // check that block_size is valid
    assert(block_size != 0);
	
    // zero force
    cudaMemset(d_force, 0, sizeof(float4)*N);

    // setup the grid to run the kernel1
    size_t num_blocks1 = (unsigned int)ceil((double)group2_size / (double)block_size);
    size_t threads1 = block_size;

	//setup the grid to run kernel2 - round num threads to upper power of 2
	size_t threads2 = num_blocks1;
	threads2--;
    threads2 |= threads2 >> 1;
    threads2 |= threads2 >> 2;
    threads2 |= threads2 >> 4;
    threads2 |= threads2 >> 8;
    threads2 |= threads2 >> 16;
    threads2++;

	// prepare the sum per block
    float *d_partial_sums_and_total = 0;
  	cudaMalloc((void**)&d_partial_sums_and_total, sizeof(float) * (threads2 + 1));
  	// local charges array
  	float *l_charge = 0;
  	cudaMalloc((void**)&l_charge, sizeof(float) * N);
  	// condensation array
  	unsigned int *condensed_flags = 0;
  	cudaMalloc((void**)&condensed_flags, sizeof(unsigned int) * group2_size);
  	// Total charge
  	float totalCharge;
  	float factor;
  	float *factors = 0;
  	cudaMalloc((void**)&factors, sizeof(unsigned int) * group1_size);

  	// Loop over monomers for charge calculation of group2
  	for (unsigned int i = 0; i < group1_size; i++)
  	{
  		// Set d_partial_sums_and_total to 0
  		cudaMemset(d_partial_sums_and_total, 0, sizeof(float)*(threads2 + 1));
  		// Set local charges to zero
  		cudaMemset(l_charge, 0, sizeof(float)*N);
  		// Calculate local charges
        // printf("%s\n","Compute charges kernel");
  		gpu_group2_compute_charges_kernel<<< num_blocks1, threads1, threads1*sizeof(float)>>>(
  										N,
										d_pos,
										d_charge,
                                        d_org_charge,
										l_charge,
										box,
										d_group1_members,
										group1_size,
										d_group2_members,
										group2_size,
										i,
										lD,
										cut2,
										d_partial_sums_and_total);
  		// Sum the total charge
  		block_sum<<<1,threads2,threads2*sizeof(float)>>>(d_partial_sums_and_total, d_partial_sums_and_total + threads2, threads2);
  		cudaMemcpy(&totalCharge, d_partial_sums_and_total + threads2, sizeof(float), cudaMemcpyDeviceToHost);
  		// Normalize and add the charges
        // printf("%s\n","Calculating factor" );
  		if (totalCharge != 0)
  		{
  			factor = 1.0f / fabs(totalCharge);
  		} else {
  			factor = 1.0f;
  		}
        // printf("%s\n","Normalize and add charges" );
  		gpu_group2_normalize_add_charges_kernel<<< num_blocks1, threads1, threads1*sizeof(float)>>>(
  										N,
  										d_charge,
                                        d_org_charge,
										l_charge,
										d_group2_members,
										group2_size,
										factor,
										factors,
										i);
  	}
	
  	// Condense and compute forces for group2
    // printf("%s\n","Condense compute");
  	gpu_group2_condense_compute_forces_kernel<<< num_blocks1, threads1, threads1*sizeof(float)>>>(
  											 d_force,
										 	 N,
											 d_charge,
											 d_group2_members,
											 group2_size,
											 condensed_flags,
											 Ex,
											 Ey,
											 Ez,
											 qt);

  	num_blocks1 = (unsigned int)ceil((double)group1_size / (double)block_size);

  	// Compute charges and forces on group1
    // printf("%s\n","Compute charges kernel on group 1");
  	gpu_group1_compute_charges_forces_kernel<<< num_blocks1, threads1>>>(
  											 d_force,
											 N,
											 d_pos,
											 d_charge,
                                             d_org_charge,
											 box,
											 d_group1_members,
											 group1_size,
											 d_group2_members,
											 group2_size,
											 condensed_flags,
											 factors,
											 Ex,
											 Ey,
											 Ez,
											 lD,
											 cut2);

  	// Compute Debye-Huckel forces on group1
    // printf("%s\n","Compute DH forces");
  	gpu_group1_compute_dh_forces_kernel<<< num_blocks1, threads1>>>(
  	  											 d_force,
  												 N,
  												 d_pos,
  												 d_charge,
  												 box,
  												 d_group1_members,
  												 group1_size,
  												 lD,
  												 bj,
  												 cut2);
  	
	// Free memory
	cudaFree(d_partial_sums_and_total);
	cudaFree(l_charge);
	cudaFree(condensed_flags);
	cudaFree(factors);

    return cudaSuccess;
    }

