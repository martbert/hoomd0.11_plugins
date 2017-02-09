/*
Highly Optimized Object-oriented Many-particle Dynamics -- Blue Edition
(HOOMD-blue) Open Source Software License Copyright 2008, 2009 Ames Laboratory
Iowa State University and The Regents of the University of Michigan All rights
reserved.

HOOMD-blue may contain modifications ("Contributions") provided, and to which
copyright is held, by various Contributors who have granted The Regents of the
University of Michigan the right to modify and/or distribute such Contributions.

Redistribution and use of HOOMD-blue, in source and binary forms, with or
without modification, are permitted, provided that the following conditions are
met:

* Redistributions of source code must retain the above copyright notice, this
list of conditions, and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice, this
list of conditions, and the following disclaimer in the documentation and/or
other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of HOOMD-blue's
contributors may be used to endorse or promote products derived from this
software without specific prior written permission.

Disclaimer

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER AND CONTRIBUTORS ``AS IS''
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND/OR
ANY WARRANTIES THAT THIS SOFTWARE IS FREE OF INFRINGEMENT ARE DISCLAIMED.

IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

// $Id: EvaluatorPairCoulomb.h 2862 2010-03-12 19:16:16Z martbert $
// Maintainer: martbert

#include <hoomd/hoomd_config.h>
#include <hoomd/HOOMDMath.h>

#ifndef __PAIR_EVALUATOR_COULOMB_H__
#define __PAIR_EVALUATOR_COULOMB_H__

#ifndef NVCC
#include <string>
#endif

/*! \file EvaluatorPairCoulomb.h
    \brief Defines the pair evaluator class for Soft-sphere potentials
*/

// need to declare these class methods with __device__ qualifiers when building in nvcc
//! DEVICE is __host__ __device__ when included in nvcc and blank when included into the host compiler
#ifdef NVCC
#define DEVICE __device__
#else
#define DEVICE
#endif

// call different optimized sqrt functions on the host / device
//! RSQRT is rsqrtf when included in nvcc and 1.0 / sqrt(x) when included into the host compiler
#ifdef NVCC
#define RSQRT(x) rsqrtf( (x) )
#else
#define RSQRT(x) Scalar(1.0) / sqrt( (x) )
#endif

//! Class for evaluating the Soft-sphere pair potential
/*! <b>General Overview</b>

    See EvaluatorPairLJ
    
    <b>Coulomb specifics</b>
    
    EvaluatorPairCoulomb evaluates the function:
    \f[ V_{\mathrm{C}}(r) = f\frac{q_iq_j}{ \epsilon_rr_{ij}}\f]
    
    The Coulomb potential does not need diameter but does need charge. Two parameters are specified and stored in a Scalar2. 
    \a f is placed in \a params.x, \a \epsilon_r is in \a params.y.
*/
class EvaluatorPairCoulomb
    {
    public:
        //! Define the parameter type used by this pair potential evaluator
        typedef Scalar4 param_type;

        //! Constructs the pair potential evaluator
        /*! \param _rsq Squared distance beteen the particles
            \param _rcutsq Sqauared distance at which the potential goes to 0
            \param _params Per type pair parameters of this potential
        */
        DEVICE EvaluatorPairCoulomb(Scalar _rsq, Scalar _rcutsq, const param_type& _params)
            : rsq(_rsq), rcutsq(_rcutsq), fer(_params.x), ron(_params.y), shiftA(_params.z), shiftB(_params.w)
            {
            }
        
        //! Soft doesn't use diameter
        DEVICE static bool needsDiameter() { return false; }
        //! Accept the optional diameter values
        /*! \param di Diameter of particle i
            \param dj Diameter of particle j
        */
        DEVICE void setDiameter(Scalar di, Scalar dj) { }

        //! Soft doesn't use charge
        DEVICE static bool needsCharge() { return true; }
        //! Accept the optional diameter values
        /*! \param qi Charge of particle i
            \param qj Charge of particle j
        */
        DEVICE void setCharge(Scalar qi, Scalar qj) 
		  {
		  qiqj = qi * qj;
		  }
        
        //! Evaluate the force and energy
        /*! \param force_divr Output parameter to write the computed force divided by r.
            \param pair_eng Output parameter to write the computed pair energy
            \param energy_shift If true, the potential must be shifted so that V(r) is continuous at the cutoff
            \note There is no need to check if rsq < rcutsq in this method. Cutoff tests are performed 
                  in PotentialPair.
            
            \return True if they are evaluated or false if they are not because we are beyond the cuttoff
        */
        DEVICE bool evalForceAndEnergy(Scalar& force_divr, Scalar& pair_eng, bool energy_shift)
            {
            if (rsq < rcutsq && qiqj != 0)
                {
				Scalar rinv = RSQRT(rsq);
                Scalar r2inv = Scalar(1.0) / rsq;
				Scalar r2on = ron*ron;
                
				pair_eng = fer * qiqj * rinv;
                force_divr = pair_eng * r2inv;

                if (energy_shift && rsq >= r2on)
                    {
					Scalar rcutinv = RSQRT(rcutsq);
					Scalar old_pair_eng = pair_eng;
					Scalar old_force_divr = force_divr;
					Scalar rcut = Scalar(1.0) / RSQRT(rcutsq);
					Scalar factor0 = rcut - ron;
					Scalar factor2 = (Scalar(1.0) / rinv - ron);
					
					force_divr = old_force_divr + factor2 * factor2 * qiqj * ( shiftA + factor2 * shiftB ) * rinv;
					
                    // Shift energy
					Scalar pair_eng_cut = fer * qiqj * rcutinv;
					Scalar C = pair_eng_cut - factor0 * factor0 * factor0 * qiqj * (shiftA / Scalar(3.0) + shiftB * factor0 / Scalar(4.0));
					
					pair_eng = old_pair_eng - factor2 * factor2 * factor2 * qiqj * (shiftA / Scalar(3.0) + shiftB * factor2 / Scalar(4.0)) - C;
                    }
                return true;
                }
            else
                return false;
            }
        
        #ifndef NVCC
        //! Get the name of this potential
        /*! \returns The potential name. Must be short and all lowercase, as this is the name energies will be logged as
            via analyze.log.
        */
        static std::string getName()
            {
            return std::string("coulomb");
            }
        #endif

    protected:
        Scalar rsq;     //!< Stored rsq from the constructor
        Scalar rcutsq;  //!< Stored rcutsq from the constructor
        Scalar fer;		//!< f over er parameter extracted from the params passed to the constructor
        Scalar ron;		//!< ron parameter extracted from the params passed to the constructor
		Scalar shiftA;		//!< shiftA parameter extracted from the params passed to the constructor
		Scalar shiftB;		//!< shiftB parameter extracted from the params passed to the constructor
		Scalar qiqj;
    };


#endif // __PAIR_EVALUATOR_COULOMB_H__

