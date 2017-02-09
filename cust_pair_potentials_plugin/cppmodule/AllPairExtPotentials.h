#ifndef __PAIR_EXT_POTENTIALS__H__
#define __PAIR_EXT_POTENTIALS__H__

// need to include hoomd_config and PotentialPair here
#include "hoomd/hoomd_config.h"
#include "hoomd/PotentialPair.h"

// include all of the evaluators that the plugin contains
#include "EvaluatorPairLJ2.h"
#include "EvaluatorPairSoft.h"
#include "EvaluatorPairCoulomb.h"
#include "EvaluatorPairSGauss.h"
#include "EvaluatorPairSYukawa.h"
#include "EvaluatorPairLoweThermo.h"
#include "PotentialPairLoweThermo.h"

#ifdef ENABLE_CUDA
// PotentialPairGPU is the class that performs the pair computations on the GPU
#include "hoomd/PotentialPairGPU.h"
#include "PotentialPairLoweThermoGPU.h"
#include "PotentialPairLoweThermoGPU.cuh"
#include "AllDriverPotentialPairExtGPU.cuh"
#endif

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

//! Pair potential force compute for LJ forces
typedef PotentialPair<EvaluatorPairLJ2> PotentialPairLJ2;
typedef PotentialPair<EvaluatorPairSoft> PotentialPairSoft;
typedef PotentialPair<EvaluatorPairCoulomb> PotentialPairCoulomb;
typedef PotentialPair<EvaluatorPairSGauss> PotentialPairSGauss;
typedef PotentialPair<EvaluatorPairSYukawa> PotentialPairSYukawa;
//! Pair potential force compute for lowe conservative forces
typedef PotentialPair<EvaluatorPairLoweThermo> PotentialPairLowe;
//! Pair potential force compute for lowe thermostat and conservative forces
typedef PotentialPairLoweThermo<EvaluatorPairLoweThermo> PotentialPairLoweThermoLowe;

#ifdef ENABLE_CUDA
//! Pair potential force compute for LJ forces on the GPU
typedef PotentialPairGPU< EvaluatorPairLJ2, gpu_compute_lj2_forces > PotentialPairLJ2GPU;
typedef PotentialPairGPU< EvaluatorPairSoft, gpu_compute_soft_forces > PotentialPairSoftGPU;
typedef PotentialPairGPU< EvaluatorPairCoulomb, gpu_compute_coulomb_forces > PotentialPairCoulombGPU;
typedef PotentialPairGPU< EvaluatorPairSGauss, gpu_compute_sgauss_forces > PotentialPairSGaussGPU;
typedef PotentialPairGPU< EvaluatorPairSYukawa, gpu_compute_syukawa_forces > PotentialPairSYukawaGPU;
//! Pair potential force compute for lowe conservative forces on the GPU
typedef PotentialPairGPU<EvaluatorPairLoweThermo, gpu_compute_lowethermo_forces > PotentialPairLoweGPU;
//! Pair potential force compute for lowe thermostat and conservative forces on the GPU
typedef PotentialPairLoweThermoGPU<EvaluatorPairLoweThermo, gpu_compute_lowethermolowe_forces > PotentialPairLoweThermoLoweGPU;
#endif

#endif // __PAIR_EXT_POTENTIALS_H__

