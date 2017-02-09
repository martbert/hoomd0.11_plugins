// Every Evaluator in this plugin needs a corresponding function call here. This function call is responsible for
// performing the pair force computation with that evaluator on the GPU. (See AllDriverPairExtGPU.cu)

#ifndef __ALL_DRIVER_POTENTIAL_PAIR_EXT_GPU_CUH__
#define __ALL_DRIVER_POTENTIAL_PAIR_EXT_GPU_CUH__

#include "hoomd/hoomd_config.h"
#include "hoomd/PotentialPairGPU.cuh"

//! Compute lj pair forces on the GPU with PairEvaluatorLJM
cudaError_t gpu_compute_ljm_forces(const pair_args_t& pair_args, const float4 *d_params);

//! Compute soft pair forces on the GPU with PairEvaluatorCoulombM
cudaError_t gpu_compute_coulombm_forces(const pair_args_t& pair_args, const float4 *d_params);
#endif

