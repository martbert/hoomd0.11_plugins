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

// $Id: PressBerendsenUpdater.cc martbert $
// Maintainer: martbert

/*! \file PressBerendsenUpdater.cc
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

#include "PressBerendsenUpdater.h"
#include <hoomd/RigidData.h>
#ifdef ENABLE_CUDA
#include "PressBerendsenUpdater.cuh"
#endif

#include <iostream>
#include <math.h>
#include <stdexcept>

using namespace std;

/*! \param sysdef System to set pressure on
    \param thermo ComputeThermo to compute the pressure with
*/
PressBerendsenUpdater::PressBerendsenUpdater(boost::shared_ptr<SystemDefinition> sysdef,
											 boost::shared_ptr<ParticleGroup> group,
											 boost::shared_ptr<ComputeThermo> thermo,
											 Scalar tau,
											 boost::shared_ptr<Variant> P,
											 Scalar bk,
											 Scalar deltaT,
											 integrationMode mode)
        : Updater(sysdef), m_group(group), m_thermo(thermo), m_tau(tau), m_P(P), m_bk(bk),m_deltaT(deltaT), m_mode(mode), m_state_initialized(false)
    {
	m_curr_P_diag = make_scalar3(0.0,0.0,0.0);
	//cout << "The current pressure diagonal has been initialized."<<endl;
    }


/*! Perform the proper velocity rescaling
    \param timestep Current time step of the simulation
*/
void PressBerendsenUpdater::update(unsigned int timestep)
{
	unsigned int group_size = m_group->getNumMembers();
	if (group_size == 0)
		return;
		
	if (m_prof) m_prof->push("PressBerendsenUpdater");
	
	//! compute the current pressure tensor
	m_thermo->compute(timestep);
	
	// compute pressure tensor for next half time step
	PressureTensor P;
	P = m_thermo->getPressureTensor();
	
	// If for some reason the pressure is not valid, assume internal pressure = external pressure
	if ( isnan(P.xx) || isnan(P.xy) || isnan(P.xz) || isnan(P.yy) || isnan(P.yz) || isnan(P.zz) )
	{
		Scalar extP = m_P->getValue(timestep);
		m_curr_P_diag = make_scalar3(extP,extP,extP);
	}
	else
	{
		// store diagonal elements of pressure tensor
		m_curr_P_diag = make_scalar3(P.xx,P.yy,P.zz);
	}
			
	// obtain box lengths	
    BoxDim curBox = m_pdata->getBox();
	
	// obtain current target pressure
	Scalar extP = m_P->getValue(timestep);
	
	// Calculate rescaling factors given the update mode
	Scalar mux, muy, muz = Scalar(0.0);
	if (m_mode == isotropic)
	{
		Scalar p_xyz = (m_curr_P_diag.x+m_curr_P_diag.y+m_curr_P_diag.z)/Scalar(3.0);
		mux = pow(Scalar(1.0) - m_deltaT / m_tau * (extP - p_xyz) * m_bk,Scalar(1.0)/Scalar(3.0));
		muy = mux;
		muz = mux;
	} else if (m_mode == anisotropic)
	{
		mux = pow(Scalar(1.0) - m_deltaT / m_tau * (extP - m_curr_P_diag.x) * m_bk,Scalar(1.0)/Scalar(3.0));
		muy = pow(Scalar(1.0) - m_deltaT / m_tau * (extP - m_curr_P_diag.y) * m_bk,Scalar(1.0)/Scalar(3.0));
		muz = pow(Scalar(1.0) - m_deltaT / m_tau * (extP - m_curr_P_diag.z) * m_bk,Scalar(1.0)/Scalar(3.0));
	} else if (m_mode == semi_isotropic)
	{
		Scalar p_yz = (m_curr_P_diag.y+m_curr_P_diag.z)/Scalar(2.0);
		mux = pow(Scalar(1.0) - m_deltaT / m_tau * (extP - m_curr_P_diag.x) * m_bk,Scalar(1.0)/Scalar(3.0));
		muy = pow(Scalar(1.0) - m_deltaT / m_tau * (extP - p_yz) * m_bk,Scalar(1.0)/Scalar(3.0));
		muz = muy;
	} else if (m_mode == one_dimensional)
	{
		Scalar p_xyz = (m_curr_P_diag.x+m_curr_P_diag.y+m_curr_P_diag.z)/Scalar(3.0);
		mux = Scalar(1.0) - m_deltaT / m_tau * (extP - p_xyz) * m_bk;
		muy = Scalar(1.0);
		muz = Scalar(1.0);
	}
	
	// first, compute the new box size
	Scalar3 curL = curBox.getL();
	Scalar3 newL;
    newL.x = mux * (curL.x);
    newL.y = muy * (curL.y);
    newL.z = muz * (curL.z);
	
	// Assign new dimensions to box
	BoxDim newBox = curBox;
    newBox.setL(newL);
	
	// move the particles to be inside the new box
	ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::readwrite);
	
	for (unsigned int i = 0; i < m_pdata->getN(); i++)
	{
		// intentionally scale both rigid body and free particles, this may waste a few cycles but it enables
		// the debug inBox checks to be left as is (otherwise, setRV cannot fixup rigid body positions without
		// failing the check)
		h_pos.data[i].x = mux * h_pos.data[i].x;
		h_pos.data[i].y = muy * h_pos.data[i].y;
		h_pos.data[i].z = muz * h_pos.data[i].z;
	}
	
	// also rescale rigid body COMs
	boost::shared_ptr<RigidData> rigid_data = m_sysdef->getRigidData();
	unsigned int n_bodies = rigid_data->getNumBodies();
	if (n_bodies > 0)
	{
		ArrayHandle<Scalar4> com_handle(rigid_data->getCOM(), access_location::host, access_mode::readwrite);
		
		for (unsigned int body = 0; body < n_bodies; body++)
		{
			com_handle.data[body].x = mux * com_handle.data[body].x;
			com_handle.data[body].y = muy * com_handle.data[body].y;
			com_handle.data[body].z = muz * com_handle.data[body].z;
		}
	}
        
    if (m_prof) m_prof->pop();
	
	// set the new box
	m_pdata->setGlobalBoxL(newBox.getL());
	
	// update the body particle positions to reflect the new rigid body positions
	if (n_bodies > 0)
		rigid_data->setRV(true);
}

void export_PressBerendsenUpdater()
    {    
    scope in_press_berendsen = class_<PressBerendsenUpdater, boost::shared_ptr<PressBerendsenUpdater>, bases<Updater>, boost::noncopyable>
    ("PressBerendsenUpdater", init< boost::shared_ptr<SystemDefinition>,
							 boost::shared_ptr<ParticleGroup>,
							 boost::shared_ptr<ComputeThermo>,
							 Scalar,
							 boost::shared_ptr<Variant>,
							 Scalar,
							 Scalar,
							 PressBerendsenUpdater::integrationMode >())
		.def("setMode", &PressBerendsenUpdater::setMode)
        .def("setTau", &PressBerendsenUpdater::setTau)
        .def("setP", &PressBerendsenUpdater::setP);
		
		enum_<PressBerendsenUpdater::integrationMode>("integrationMode")
		.value("isotropic", PressBerendsenUpdater::isotropic)
		.value("anisotropic", PressBerendsenUpdater::anisotropic)
		.value("semi_isotropic", PressBerendsenUpdater::semi_isotropic)
		.value("one_dimensional", PressBerendsenUpdater::one_dimensional);
    }

#ifdef ENABLE_CUDA
//! A GPU accelerated version of the updater

PressBerendsenUpdaterGPU::PressBerendsenUpdaterGPU(boost::shared_ptr<SystemDefinition> sysdef,
												   boost::shared_ptr<ParticleGroup> group,
												   boost::shared_ptr<ComputeThermo> thermo,
												   Scalar tau,
												   boost::shared_ptr<Variant> P,
												   Scalar bk,
												   Scalar deltaT,
												   integrationMode mode)
: PressBerendsenUpdater(sysdef,group,thermo,tau,P,bk,deltaT,mode)
    {
		m_block_size = 256;
		//cout<<"The block size is: "<<m_block_size<<endl;
		//cout<<"The mode of integration is: "<<m_mode<<endl;
    }

void PressBerendsenUpdaterGPU::update(unsigned int timestep)
{
	//cout << "Updater has been called" <<endl;
	unsigned int group_size = m_group->getNumMembers();
	if (group_size == 0)
		return;	
	
    if (m_prof) m_prof->push("PressBerendsenUpdater");
	
	//! compute the current pressure tensor
	m_thermo->compute(timestep);
	
	// compute pressure tensor for next half time step
	PressureTensor P;
	P = m_thermo->getPressureTensor();
	
	// If for some reason the pressure is not valid, assume internal pressure = external pressure
	if ( isnan(P.xx) || isnan(P.xy) || isnan(P.xz) || isnan(P.yy) || isnan(P.yz) || isnan(P.zz) )
	{
		Scalar extP = m_P->getValue(timestep);
		m_curr_P_diag = make_scalar3(extP,extP,extP);
	}
	else
	{
		// store diagonal elements of pressure tensor
		m_curr_P_diag = make_scalar3(P.xx,P.yy,P.zz);
	}
		
	// obtain box lengths	
    BoxDim curBox = m_pdata->getBox();
	
	// obtain current target pressure
	Scalar extP = m_P->getValue(timestep);
	
	// Calculate rescaling factors given the update mode
	Scalar mux, muy, muz = Scalar(0.0);
	if (m_mode == isotropic)
	{
		Scalar p_xyz = (m_curr_P_diag.x+m_curr_P_diag.y+m_curr_P_diag.z)/Scalar(3.0);
		mux = pow(Scalar(1.0) - m_deltaT / m_tau * (extP - p_xyz) * m_bk,Scalar(1.0)/Scalar(3.0));
		muy = mux;
		muz = mux;
	} else if (m_mode == anisotropic)
	{
		mux = pow(Scalar(1.0) - m_deltaT / m_tau * (extP - m_curr_P_diag.x) * m_bk,Scalar(1.0)/Scalar(3.0));
		muy = pow(Scalar(1.0) - m_deltaT / m_tau * (extP - m_curr_P_diag.y) * m_bk,Scalar(1.0)/Scalar(3.0));
		muz = pow(Scalar(1.0) - m_deltaT / m_tau * (extP - m_curr_P_diag.z) * m_bk,Scalar(1.0)/Scalar(3.0));
	} else if (m_mode == semi_isotropic)
	{
		Scalar p_yz = (m_curr_P_diag.y+m_curr_P_diag.z)/Scalar(2.0);
		mux = pow(Scalar(1.0) - m_deltaT / m_tau * (extP - m_curr_P_diag.x) * m_bk,Scalar(1.0)/Scalar(3.0));
		muy = pow(Scalar(1.0) - m_deltaT / m_tau * (extP - p_yz) * m_bk,Scalar(1.0)/Scalar(3.0));
		muz = muy;
		//cout<< "Pressure is: "<<m_curr_P_diag.x<<" and "<<p_yz<<endl;
	} else if (m_mode == one_dimensional)
	{
		Scalar p_xyz = (m_curr_P_diag.x+m_curr_P_diag.y+m_curr_P_diag.z)/Scalar(3.0);
		mux = Scalar(1.0) - m_deltaT / m_tau * (extP - p_xyz) * m_bk;
		muy = Scalar(1.0);
		muz = Scalar(1.0);
	}
	
	// first, compute the new box size
    Scalar3 curL = curBox.getL();
	Scalar3 newL;
    newL.x = mux * (curL.x);
    newL.y = muy * (curL.y);
    newL.z = muz * (curL.z);
	
	// Assign new dimensions to box
	BoxDim newBox = curBox;
    newBox.setL(newL);
	
    // access the particle data arrays for writing on the GPU
	ArrayHandle<Scalar4> d_pos(m_pdata->getPositions(), access_location::device, access_mode::readwrite);
	ArrayHandle< unsigned int > d_index_array(m_group->getIndexArray(), access_location::device, access_mode::read);
	
	//cout << "Calculations are all done "<<mux<<" "<<muy<<" "<<muz<<endl;
	
    // call the kernel defined in PressBerendsenUpdater.cu
    gpu_press_berendsen(d_pos.data,
					   d_index_array.data,
					   group_size,
					   m_block_size,
					   mux,
					   muy,
					   muz,
					   m_deltaT);
	
	//cout << "Particles are back in the box" <<endl;
	
	// also rescale rigid body COMs
	boost::shared_ptr<RigidData> rigid_data = m_sysdef->getRigidData();
	unsigned int n_bodies = rigid_data->getNumBodies();
	if (n_bodies > 0)
	{
		ArrayHandle<Scalar4> d_com_handle(rigid_data->getCOM(), access_location::device, access_mode::readwrite);
    	gpu_press_berendsen_rigid(d_com_handle.data,
								  n_bodies,
								  m_block_size,
								  mux,
								  muy,
								  muz,
								  m_deltaT);
	}
	
	// set the new box
	m_pdata->setGlobalBoxL(newBox.getL());
	
	// update the body particle positions to reflect the new rigid body positions
	if (n_bodies > 0)
		rigid_data->setRV(true);
	
    // check for error codes from the GPU if error checking is enabled
    if (exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();
    
    if (m_prof) m_prof->pop();

}

void export_PressBerendsenUpdaterGPU()
    {
    class_<PressBerendsenUpdaterGPU, boost::shared_ptr<PressBerendsenUpdaterGPU>, bases<PressBerendsenUpdater>, boost::noncopyable>
		("PressBerendsenUpdaterGPU", init< boost::shared_ptr<SystemDefinition>,
									 boost::shared_ptr<ParticleGroup>,
									 boost::shared_ptr<ComputeThermo>,
									 Scalar,
									 boost::shared_ptr<Variant>,
									 Scalar,
									 Scalar,
									 PressBerendsenUpdaterGPU::integrationMode >());
    }

#endif // ENABLE_CUDA

#ifdef WIN32
#pragma warning( pop )
#endif

