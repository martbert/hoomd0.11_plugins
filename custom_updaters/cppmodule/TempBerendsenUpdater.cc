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

// $Id: TempBerendsenUpdater.cc martbert $
// Maintainer: martbert

/*! \file TempBerendsenUpdater.cc
    \brief Declares an updater that applies appropriate BC at given walls
*/

#ifdef WIN32
#pragma warning( push )
#pragma warning( disable : 4103 4244 )
#endif

#include <boost/python.hpp>
using namespace boost::python;

// we need to include boost.bind for GPUWorker execution
#include <boost/bind.hpp>
using namespace boost;

#include "TempBerendsenUpdater.h"
#ifdef ENABLE_CUDA
#include "TempBerendsenUpdater.cuh"
#endif

#include <iostream>
#include <math.h>
#include <stdexcept>

using namespace std;

/*! \param sysdef System to set temperature on
    \param thermo ComputeThermo to compute the temperature with
    \param tset Temperature set point
*/
TempBerendsenUpdater::TempBerendsenUpdater(boost::shared_ptr<SystemDefinition> sysdef,
										   boost::shared_ptr<ParticleGroup> group,
										   boost::shared_ptr<ComputeThermo> thermo,
										   Scalar tau,
										   boost::shared_ptr<Variant> T,
										   Scalar deltaT)
        : Updater(sysdef), m_group(group), m_thermo(thermo), m_tau(tau), m_T(T), m_deltaT(deltaT)
    {
    assert(m_pdata);
    }


/*! Perform the proper velocity rescaling
    \param timestep Current time step of the simulation
*/
void TempBerendsenUpdater::update(unsigned int timestep)
{
	unsigned int group_size = m_group->getNumMembers();
	if (group_size == 0)
		return;
		
	if (m_prof) m_prof->push("TempBerendsenUpdater");
	
	// compute the current thermodynamic properties and get the temperature
    m_thermo->compute(timestep);
    Scalar curr_T = m_thermo->getTemperature();
	
    // compute the value of lambda for the current timestep
    Scalar lambda = sqrt(Scalar(1.0) + m_deltaT / m_tau * (m_T->getValue(timestep) / curr_T - Scalar(1.0)));
	
    // access the particle data for writing on the CPU
    assert(m_pdata);
    ArrayHandle<Scalar4> h_vel(m_pdata->getVelocities(), access_location::host, access_mode::readwrite);
    
	for (unsigned int group_idx = 0; group_idx < group_size; group_idx++)
	{
        unsigned int j = m_group->getMemberIndex(group_idx);
		
        h_vel.data[j].x *= lambda;
        h_vel.data[j].y *= lambda;
        h_vel.data[j].z *= lambda;
	}
    
    // scale all the rigid body com velocities and angular momenta
	boost::shared_ptr<RigidData> rigid_data = m_sysdef->getRigidData();
	unsigned int n_bodies = rigid_data->getNumBodies();
	if (n_bodies > 0)
	{
		ArrayHandle<Scalar4> h_body_vel(rigid_data->getVel(), access_location::host, access_mode::readwrite);
		ArrayHandle<Scalar4> h_body_angmom(rigid_data->getAngMom(), access_location::host, access_mode::readwrite);
		
		for (unsigned int body = 0; body < n_bodies; body++)
			{
			h_body_vel.data[body].x *= lambda;
			h_body_vel.data[body].y *= lambda;
			h_body_vel.data[body].z *= lambda;

			h_body_angmom.data[body].x *= lambda;
			h_body_angmom.data[body].y *= lambda;
			h_body_angmom.data[body].z *= lambda;
			}
		// ensure that the particle velocities are up to date
		rigid_data->setRV(false);
	}
        
    if (m_prof) m_prof->pop();
}

void export_TempBerendsenUpdater()
    {
    class_<TempBerendsenUpdater, boost::shared_ptr<TempBerendsenUpdater>, bases<Updater>, boost::noncopyable>
    ("TempBerendsenUpdater", init< boost::shared_ptr<SystemDefinition>,
							 boost::shared_ptr<ParticleGroup>,
							 boost::shared_ptr<ComputeThermo>,
							 Scalar,
							 boost::shared_ptr<Variant>,
							 Scalar >());
    }

#ifdef ENABLE_CUDA
//! A GPU accelerated version of the updater

TempBerendsenUpdaterGPU::TempBerendsenUpdaterGPU(boost::shared_ptr<SystemDefinition> sysdef,
												 boost::shared_ptr<ParticleGroup> group,
												 boost::shared_ptr<ComputeThermo> thermo,
												 Scalar tau,
												 boost::shared_ptr<Variant> T,
												 Scalar deltaT)
        : TempBerendsenUpdater(sysdef,group,thermo,tau,T,deltaT)
    {
		m_block_size = 256;
    }

void TempBerendsenUpdaterGPU::update(unsigned int timestep)
{
	unsigned int group_size = m_group->getNumMembers();
	if (group_size == 0)
		return;	
	
    if (m_prof) m_prof->push("TempBerendsenUpdater");
    
	// compute the current thermodynamic quantities and get the temperature
    m_thermo->compute(timestep);
    Scalar curr_T = m_thermo->getTemperature();
	
    // compute the value of lambda for the current timestep
    Scalar lambda = sqrt(Scalar(1.0) + m_deltaT / m_tau * (m_T->getValue(timestep) / curr_T - Scalar(1.0)));
	
    // access the particle data arrays for writing on the GPU
    ArrayHandle<Scalar4> d_vel(m_pdata->getVelocities(), access_location::device, access_mode::readwrite);
	
	ArrayHandle< unsigned int > d_index_array(m_group->getIndexArray(), access_location::device, access_mode::read);
	
    // call the kernel defined in TempBerendsenUpdater.cu
    gpu_temp_berendsen(d_vel.data,
					   d_index_array.data,
					   group_size,
					   m_block_size,
					   lambda);
	
	boost::shared_ptr<RigidData> rigid_data = m_sysdef->getRigidData();
	unsigned int n_bodies = rigid_data->getNumBodies();
	if (n_bodies > 0)
	{
		ArrayHandle<Scalar4> d_body_vel(rigid_data->getVel(), access_location::device, access_mode::readwrite);
		ArrayHandle<Scalar4> d_body_angmom(rigid_data->getAngMom(), access_location::device, access_mode::readwrite);
		
		//call the kernel defined in TempBerendsenUpdater.cu
		gpu_temp_berendsen_rigid(d_body_vel.data,
								 d_body_angmom.data,
								 n_bodies,
								 m_block_size,
								 lambda);
		
		// ensure that the particle velocities are up to date
		rigid_data->setRV(false);
	}
    
    // check for error codes from the GPU if error checking is enabled
    if (exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();
    
    if (m_prof) m_prof->pop();
    }

void export_TempBerendsenUpdaterGPU()
    {
    class_<TempBerendsenUpdaterGPU, boost::shared_ptr<TempBerendsenUpdaterGPU>, bases<TempBerendsenUpdater>, boost::noncopyable>
		("TempBerendsenUpdaterGPU", init< boost::shared_ptr<SystemDefinition>,
									 boost::shared_ptr<ParticleGroup>,
									 boost::shared_ptr<ComputeThermo>,
									 Scalar,
									 boost::shared_ptr<Variant>,
									 Scalar >());
    }

#endif // ENABLE_CUDA

#ifdef WIN32
#pragma warning( pop )
#endif

