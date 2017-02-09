#ifndef _PRESSBERENDSENUPDATER_CUH_
#define _PRESSBERENDSENUPDATER_CUH_

// there is no convenient header to include all GPU related headers, we need to include those that are needed
#include <hoomd/hoomd_config.h>
#include <hoomd/HOOMDMath.h>
// need to include the particle data definition
#include <hoomd/ParticleData.cuh>
// need to include the rigid data definition

/*! \file PressBerendsenUpdater.cuh
    \brief Declaration of CUDA kernels for PressBerendsenUpdater
*/

// A C API call to run a CUDA kernel is needed for PressBerendsenUpdaterGPU to call
cudaError_t gpu_press_berendsen(Scalar4 *d_pos,
								unsigned int *d_group_members,
								unsigned int group_size,
								unsigned int block_size,
								float mux,
								float muy,
								float muz,
								float deltaT);

cudaError_t gpu_press_berendsen_rigid(Scalar4 *d_com,
									unsigned int n_bodies,
									unsigned int block_size,
									float mux,
									float muy,
									float muz,
									float deltaT);
#endif // _PRESSBERENDSENUPDATER_CUH_

