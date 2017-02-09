#ifndef __PAIR_EXT_POTENTIALS__H__
#define __PAIR_EXT_POTENTIALS__H__

// need to include hoomd_config and PotentialPair here
#include "hoomd/hoomd_config.h"
#include "hoomd/PotentialPair.h"

// include all of the evaluators that the plugin contains
#include "EvaluatorPairLJM.h"
#include "EvaluatorPairCoulombM.h"

#ifdef ENABLE_CUDA
// PotentialPairGPU is the class that performs the pair computations on the GPU
#include "hoomd/PotentialPairGPU.h"
// AllDriverPotentialPairExtGPU.cuh is a header file containing the kernel driver functions for computing the pair
// potentials defined in this plugin. See it for more details
#include "AllDriverPotentialPairExtGPU.cuh"
#endif

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

//! Pair potential force compute for LJ forces
typedef PotentialPair<EvaluatorPairLJM> PotentialPairLJM;
typedef PotentialPair<EvaluatorPairCoulombM> PotentialPairCoulombM;

#ifdef ENABLE_CUDA
//! Pair potential force compute for LJ forces on the GPU
typedef PotentialPairGPU< EvaluatorPairLJM, gpu_compute_ljm_forces > PotentialPairLJMGPU;
typedef PotentialPairGPU< EvaluatorPairCoulombM, gpu_compute_coulombm_forces > PotentialPairCoulombMGPU;
#endif

#endif // __PAIR_EXT_POTENTIALS_H__

