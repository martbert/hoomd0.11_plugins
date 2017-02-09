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

// $Id: TempBerendsenUpdater.h martbert $
// Maintainer: martbert

/*! \file TempBerendsenUpdater.h
    \brief Declares an updater that applies appropriate BC at given walls
*/

#include <boost/shared_ptr.hpp>

// First, hoomd.h should be included
#include <hoomd/hoomd.h>

#ifndef __TEMPBERENDSENUPDATER_H__
#define __TEMPBERENDSENUPDATER_H__

//! Updates particle velocities to set a temperature
/*! This updater computes the current temperature of the system and then scales the velocities in order to set the 
    temperature.

    \ingroup updaters
*/
class TempBerendsenUpdater : public Updater
    {
    public:
        //! Constructor
        TempBerendsenUpdater(boost::shared_ptr<SystemDefinition> sysdef,
                             boost::shared_ptr<ParticleGroup> group,
							 boost::shared_ptr<ComputeThermo> thermo,
							 Scalar tau,
							 boost::shared_ptr<Variant> T,
							 Scalar deltaT);
        virtual ~TempBerendsenUpdater() {};
		
		//! Update the temperature
        //! \param T New temperature to set
        virtual void setT(boost::shared_ptr<Variant> T)
		{
            m_T = T;
		}
		
        //! Update the tau value
        //! \param tau New time constant to set
        virtual void setTau(Scalar tau)
		{
            m_tau = tau;
		}
		
        //! Take one timestep forward
        virtual void update(unsigned int timestep);
                
    protected:
        boost::shared_ptr<ParticleGroup> m_group;  //!< Group of particles
        const boost::shared_ptr<ComputeThermo> m_thermo; //!< compute for thermodynamic quantities
        Scalar m_tau;                    //!< time constant for Berendsen thermostat
        boost::shared_ptr<Variant> m_T;    //!< set temperature
		Scalar m_deltaT;					//!< time step
    };

//! Export the TempBerendsenUpdater to python
void export_TempBerendsenUpdater();

#ifdef ENABLE_CUDA

//! A GPU accelerated version of the updater
/*! This updater simply sets all of the particle's velocities to 0 (on the GPU) when update() is called.
*/
class TempBerendsenUpdaterGPU : public TempBerendsenUpdater
    {
    public:
         //! Constructor
        TempBerendsenUpdaterGPU(boost::shared_ptr<SystemDefinition> sysdef,
								boost::shared_ptr<ParticleGroup> group,
								boost::shared_ptr<ComputeThermo> thermo,
								Scalar tau,
								boost::shared_ptr<Variant> T,
								Scalar deltaT);
        virtual ~TempBerendsenUpdaterGPU() {};
		
        //! Take one timestep forward
        virtual void update(unsigned int timestep);
		
	protected:
        unsigned int m_block_size; //!< Block size to launch on the GPU
    };

//! Export the ExampleUpdaterGPU class to python
void export_TempBerendsenUpdaterGPU();

#endif // ENABLE_CUDA


#endif

