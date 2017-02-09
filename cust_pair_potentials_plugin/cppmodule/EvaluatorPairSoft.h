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

// $Id: EvaluatorPairSoft.h 2862 2010-03-12 19:16:16Z martbert $
// Maintainer: martbert

#include <hoomd/hoomd_config.h>
#include <hoomd/HOOMDMath.h>

#ifndef __PAIR_EVALUATOR_SOFT_H__
#define __PAIR_EVALUATOR_SOFT_H__

#ifndef NVCC
#include <string>
#endif

/*! \file EvaluatorPairSoft.h
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
#define POW(x,y) powf((x),(y))
#else
#define RSQRT(x) Scalar(1.0) / sqrt( (x) )
#define POW(x,y) pow((x),(y))
#endif

//! Class for evaluating the Soft-sphere pair potential
/*! <b>General Overview</b>

    See EvaluatorPairLJ
    
    <b>Soft-sphere specifics</b>
    
    EvaluatorPairSoft evaluates the function:
    \f[ V_{\mathrm{soft}}(r) = a \left(r-r_{off} \right)^{-n} }{r} \f]
    
    The Soft-sphere potential does not need diameter or charge. Three parameters are specified and stored in a Scalar3. 
    \a a is placed in \a params.x, \a n is in \a params.y, and \a r_{off} is in \a params.z .
*/
class EvaluatorPairSoft
    {
    public:
        //! Define the parameter type used by this pair potential evaluator
        typedef Scalar3 param_type;

        //! Constructs the pair potential evaluator
        /*! \param _rsq Squared distance beteen the particles
            \param _rcutsq Sqauared distance at which the potential goes to 0
            \param _params Per type pair parameters of this potential
        */
        DEVICE EvaluatorPairSoft(Scalar _rsq, Scalar _rcutsq, const param_type& _params)
            : rsq(_rsq), rcutsq(_rcutsq), a(_params.x), n(_params.y), roff(_params.z)
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
        DEVICE static bool needsCharge() { return false; }
        //! Accept the optional diameter values
        /*! \param qi Charge of particle i
            \param qj Charge of particle j
        */
        DEVICE void setCharge(Scalar qi, Scalar qj) { }
        
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
            // precompute some quantities
            Scalar rinv = RSQRT(rsq);
            Scalar r = Scalar(1.0) / rinv;
			Scalar rcutinv = RSQRT(rcutsq);
			Scalar rcut = Scalar(1.0) / rcutinv;
			
            if (r < (rcut+roff) && a != 0)
                {
                Scalar dr = r-roff;
                
                Scalar pow_val = a * Scalar(1.0) / POW(dr,n);
                
                force_divr = n * pow_val * rinv / dr;
                pair_eng = pow_val;

                if (energy_shift)
                    {
                    pair_eng -= a * Scalar(1.0) / POW(rcut,n);
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
            return std::string("soft");
            }
        #endif

    protected:
        Scalar rsq;     //!< Stored rsq from the constructor
        Scalar rcutsq;  //!< Stored rcutsq from the constructor
        Scalar a;		//!< a parameter extracted from the params passed to the constructor
        Scalar n;		//!< n parameter extracted from the params passed to the constructor
		Scalar roff;	//!< roff parameter extracted from the params passed to the constructor
    };


#endif // __PAIR_EVALUATOR_SOFT_H__

