// Every Evaluator in this plugin needs a corresponding function call here. This function call is responsible for
// performing the pair force computation with that evaluator on the GPU. (See AllDriverPairExtGPU.cu)

#ifndef __ALL_DRIVER_POTENTIAL_PAIR_EXT_GPU_CUH__
#define __ALL_DRIVER_POTENTIAL_PAIR_EXT_GPU_CUH__

#include "hoomd/hoomd_config.h"
#include "hoomd/PotentialPairGPU.cuh"
#include "PotentialPairLoweThermoGPU.cuh"
#include "EvaluatorPairLoweThermo.h"

//! Compute lj pair forces on the GPU with PairEvaluatorLJ2
cudaError_t gpu_compute_lj2_forces(const pair_args_t& pair_args, const float4 *d_params);
// the last argument on the previous line is a pointer to the per type pair parameters passed into the pair potential
// it must be the same as the param_type in the evaluator this function uses

//! Compute soft pair forces on the GPU with PairEvaluatorSoft
cudaError_t gpu_compute_soft_forces(const pair_args_t& pair_args, const float3 *d_params);

//! Compute soft pair forces on the GPU with PairEvaluatorCoulomb
cudaError_t gpu_compute_coulomb_forces(const pair_args_t& pair_args, const float4 *d_params);

//! Compute shifted Gauss pair forces on the GPU with PairEvaluatorSGauss
cudaError_t gpu_compute_sgauss_forces(const pair_args_t& pair_args, const float2 *d_params);

//! Compute shifted Yukawa pair forces on the GPU with PairEvaluatorSYukawa
cudaError_t gpu_compute_syukawa_forces(const pair_args_t& pair_args, const float2 *d_params);

//! Compute lowe thermostat on GPU with PairEvaluatorLoweThermo 
cudaError_t gpu_compute_lowethermolowe_forces(const lowe_pair_args_t& args,
                                            const float2 *d_params);

//! Compute lowe conservative force on GPU with PairEvaluatorLoweThermo
cudaError_t gpu_compute_lowethermo_forces(const pair_args_t& pair_args,
                                         const float2 *d_params);
#endif
