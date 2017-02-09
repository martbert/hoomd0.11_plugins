#include "PressBerendsenUpdater.cuh"

/*! \file PressBerendsenUpdater.cu
    \brief CUDA kernels for PressBerendsenUpdater
*/

// First, kernel code for Berendsen pressure rescaling GPU
/*! \param d_pos array of particle positions
    \param d_group_members Device array listing the indicies of the members of the group to integrate
    \param group_size Number of members in the group
    \param mux Intermediate variable computed on the host
    \param muy Intermediate variable computed on the host
    \param muz Intermediate variable computed on the host
    \param deltaT Length of one timestep

    This kernel executes one thread per particle and applies the theromstat to each each. It can be
    run with any 1D block size as long as block_size * num_blocks is >= the number of particles.
*/
extern "C" __global__ 
void gpu_press_berendsen_kernel(Scalar4 *d_pos,
							   unsigned int *d_group_members,
							   unsigned int group_size,
							   float mux,
							   float muy,
							   float muz,
							   float deltaT)
{	
    // start by identifying which particle we are to handle
    int group_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
	if (group_idx < group_size)
	{
        unsigned int idx = d_group_members[group_idx];

        // read the particle position
        float4 pos = d_pos[idx];
		
		// update position according to berendsen scheme
        pos.x = mux * pos.x;
        pos.y = muy * pos.y;
        pos.z = muz * pos.z;

        // write the results
        d_pos[idx] = pos;
	}
}

/*! 
    This is just a driver for gpu_temp_berendsen_kernel(), see it for the details
*/
cudaError_t gpu_press_berendsen(Scalar4 *d_pos,
							   unsigned int *d_group_members,
							   unsigned int group_size,
							   unsigned int block_size,
							   float mux,
							   float muy,
							   float muz,
							   float deltaT)
{
    // setup the grid to run the kernel
    dim3 grid( (group_size / block_size) + 1, 1, 1);
    dim3 threads(block_size, 1, 1);
    
	// run the kernel
    gpu_press_berendsen_kernel<<< grid, threads, block_size * sizeof(float) >>>(d_pos,
																	   d_group_members,
																	   group_size,
																	   mux,
																	   muy,
																	   muz,
																	   deltaT);
																	   
    // cudaSuccess
    return cudaSuccess;
}


// Second, kernel code for Berendsen pressure rescaling of rigid bodies GPU
/*! \param d_com array of particle positions
    \param n_bodies Number of rigid bodies
    \param mux Intermediate variable computed on the host
    \param muy Intermediate variable computed on the host
    \param muz Intermediate variable computed on the host
    \param deltaT Length of one timestep

    This kernel executes one thread per particle and applies the theromstat to each each. It can be
    run with any 1D block size as long as block_size * num_blocks is >= the number of particles.
*/
extern "C" __global__ 
void gpu_press_berendsen_rigid_kernel(Scalar4 *d_com,
							   unsigned int n_bodies,
							   float mux,
							   float muy,
							   float muz,
							   float deltaT)
{	
    // start by identifying which body we are to handle
    int body_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
	if (body_idx < n_bodies)
	{
        // get center of mass of rigid body
        float4 com = d_com[body_idx];
		
		// update position according to berendsen scheme
        com.x = mux * com.x;
        com.y = muy * com.y;
        com.z = muz * com.z;

        // write the results
        d_com[body_idx] = com;
	}
}

/*! 
    This is just a driver for gpu_press_berendsen_rigid_kernel(), see it for the details
*/
cudaError_t gpu_press_berendsen_rigid(Scalar4 *d_com,
							   unsigned int n_bodies,
							   unsigned int block_size,
							   float mux,
							   float muy,
							   float muz,
							   float deltaT)
{
    // setup the grid to run the kernel
    dim3 grid( n_bodies + 1, 1, 1);
    dim3 threads(block_size, 1, 1);
    
	// run the kernel
    gpu_press_berendsen_rigid_kernel<<< grid, threads, block_size * sizeof(float) >>>(d_com,
																	   n_bodies,
																	   mux,
																	   muy,
																	   muz,
																	   deltaT);
																	   
    // cudaSuccess
    return cudaSuccess;
}