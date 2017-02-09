#ifndef _TEMPBERENDSENUPDATER_CUH_
#define _TEMPBERENDSENUPDATER_CUH_

// there is no convenient header to include all GPU related headers, we need to include those that are needed
#include <hoomd/hoomd_config.h>
#include <hoomd/HOOMDMath.h>
// need to include the particle data definition
#include <hoomd/ParticleData.cuh>

/*! \file TempBerendsenUpdater.cuh
    \brief Declaration of CUDA kernels for TempBerendsenUpdater
*/

// A C API call to run a CUDA kernel is needed for TempBerendsenUpdaterGPU to call
cudaError_t gpu_temp_berendsen(Scalar4 *d_vel,
								unsigned int *d_group_members,
								unsigned int group_size,
								unsigned int block_size,
								float lambda);

cudaError_t gpu_temp_berendsen_rigid(Scalar4 *d_com_vel,
							   Scalar4 *d_com_angmom,
							   unsigned int n_bodies,
								unsigned int block_size,
								float lambda);

#endif // _TEMPBERENDSENUPDATER_CUH_

