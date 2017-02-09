#include "AllDriverPotentialPairExtGPU.cuh"
#include "EvaluatorPairSoft.h"
#include "EvaluatorPairLJ2.h"
#include "EvaluatorPairCoulomb.h"
#include "EvaluatorPairSGauss.h"
#include "EvaluatorPairSYukawa.h"
#include "PotentialPairLoweThermoGPU.cuh"
#include "EvaluatorPairLoweThermo.h"

// Every evaluator needs a function in this file. The functions are very simple, containing a one line call to
// a template that does all of the work. To add a additional function, copy and paste this one, change the 
// template argument to the correct evaluator <EvaluatorPairMine>, and update the type of the 2nd argument to the
// param_type of the evaluator

cudaError_t gpu_compute_lj2_forces(const pair_args_t& pair_args,
                                   const float4 *d_params)
    {
    return gpu_compute_pair_forces<EvaluatorPairLJ2>(pair_args, d_params);
    }

cudaError_t gpu_compute_soft_forces(const pair_args_t& pair_args,
                                   const float3 *d_params)
    {
    return gpu_compute_pair_forces<EvaluatorPairSoft>(pair_args, d_params);
    }

cudaError_t gpu_compute_coulomb_forces(const pair_args_t& pair_args,
                                   const float4 *d_params)
    {
    return gpu_compute_pair_forces<EvaluatorPairCoulomb>(pair_args, d_params);
    }

cudaError_t gpu_compute_sgauss_forces(const pair_args_t& pair_args,
                                   const float2 *d_params)
    {
    return gpu_compute_pair_forces<EvaluatorPairSGauss>(pair_args, d_params);
    }

cudaError_t gpu_compute_syukawa_forces(const pair_args_t& pair_args,
                                   const float2 *d_params)
    {
    return gpu_compute_pair_forces<EvaluatorPairSYukawa>(pair_args, d_params);
    }

cudaError_t gpu_compute_lowethermolowe_forces(const lowe_pair_args_t& args,
                                            const float2 *d_params)
    {
    return gpu_compute_lowe_forces<EvaluatorPairLoweThermo>(args,
                                                          d_params);
    }

cudaError_t gpu_compute_lowethermo_forces(const pair_args_t& pair_args,
                                         const float2 *d_params)
    {
    return gpu_compute_pair_forces<EvaluatorPairLoweThermo>(pair_args,
                                                           d_params);
    }
