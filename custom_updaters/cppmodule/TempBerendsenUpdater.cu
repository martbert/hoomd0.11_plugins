#include "TempBerendsenUpdater.cuh"

/*! \file TempBerendsenUpdater.cu
    \brief CUDA kernels for TempBerendsenUpdater
*/

// First, kernel code for Berendsen temperature rescaling GPU
/*! \param d_vel array of particle velocties
    \param d_group_members Device array listing the indicies of the members of the group to integrate
    \param group_size Number of members in the group
    \param box Box dimensions for applying periodic boundary conditions
    \param lambda Intermediate variable computed on the host and used in integrating the velocity
    \param deltaT Length of one timestep

    This kernel executes one thread per particle and applies the theromstat to each each. It can be
    run with any 1D block size as long as block_size * num_blocks is >= the number of particles.
*/
extern "C" __global__ 
void gpu_temp_berendsen_kernel(Scalar4 *d_vel,
							   unsigned int *d_group_members,
							   unsigned int group_size,
							   float lambda)
{	
    // start by identifying which particle we are to handle
    int group_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
	if (group_idx < group_size)
	{
        unsigned int idx = d_group_members[group_idx];

        // read the particle velocity and acceleration
        float4 vel = d_vel[idx];
		
		// update velocity according to berendsen scheme
        vel.x = lambda * vel.x;
        vel.y = lambda * vel.y;
        vel.z = lambda * vel.z;

        // write the results
        d_vel[idx] = vel;
	}
}

extern "C" __global__ 
void gpu_temp_berendsen_rigid_kernel (Scalar4 *d_com_vel,
							   Scalar4 *d_com_angmom,
							   unsigned int n_bodies,
								float lambda)
{
	// start by identifying which particle we are to handle
    int body_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (body_idx < n_bodies)
	{
        // get center of mass of rigid body
        float4 vel = d_com_vel[body_idx];
        float4 angmom = d_com_angmom[body_idx];
		
		// update velocity and angular momentum according to berendsen scheme
        vel.x = lambda * vel.x;
        vel.y = lambda * vel.y;
        vel.z = lambda * vel.z;
        
        angmom.x = lambda * angmom.x;
        angmom.y = lambda * angmom.y;
        angmom.z = lambda * angmom.z;

        // write the results
        d_com_vel[body_idx] = vel;
        d_com_angmom[body_idx] = angmom;
	}
}


/*! 
    This is just a driver for gpu_temp_berendsen_kernel(), see it for the details
*/
cudaError_t gpu_temp_berendsen(Scalar4 *d_vel,
							   unsigned int *d_group_members,
							   unsigned int group_size,
							   unsigned int block_size,
							   float lambda)
{
    // setup the grid to run the kernel
    dim3 grid( (group_size / block_size) + 1, 1, 1);
    dim3 threads(block_size, 1, 1);
    
	// run the kernel
    gpu_temp_berendsen_kernel<<< grid, threads, block_size * sizeof(float) >>>(d_vel,
																	   d_group_members,
																	   group_size,
																	   lambda);
																	   
    // cudaSuccess
    return cudaSuccess;
}

/*! 
    This is just a driver for gpu_temp_berendsen_rigid_kernel(), see it for the details
*/
cudaError_t gpu_temp_berendsen_rigid(Scalar4 *d_com_vel,
							   Scalar4 *d_com_angmom,
							   unsigned int n_bodies,
								unsigned int block_size,
								float lambda)
{
    // setup the grid to run the kernel
    dim3 grid( n_bodies + 1, 1, 1);
    dim3 threads(block_size, 1, 1);
    
	// run the kernel
    gpu_temp_berendsen_rigid_kernel<<< grid, threads, block_size * sizeof(float) >>>(d_com_vel,
							  												 d_com_angmom,
							   												 n_bodies,
																			 lambda);
																	   
    // cudaSuccess
    return cudaSuccess;
}
