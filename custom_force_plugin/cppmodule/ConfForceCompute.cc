/*
Highly Optimized Object-oriented Many-particle Dynamics -- Blue Edition
(HOOMD-blue) Open Source Software License Copyright 2008, 2009 Ames Laboratory
Iowa State University and The Regents of the University of Michigan All rights
reserved.
*/

// $Id: ConfForceCompute.cc martbert $
// Maintainer: martbert

#include <boost/python.hpp>
using namespace boost::python;

// we need to include boost.bind for GPUWorker execution
#include <boost/bind.hpp>
using namespace boost;

#include "ConfForceCompute.h"
#ifdef ENABLE_CUDA
#include "ConfForceCompute.cuh"
#endif

using namespace std;

/*! \file ConfForceCompute.cc
    \brief Contains code for the ConfForceCompute class
*/

/*! \param sysdef SystemDefinition containing the ParticleData to compute forces on
    \param k strength of harmonic confinement
    \param roff offset of the harmonic potential
*/
ConfForceCompute::ConfForceCompute(boost::shared_ptr<SystemDefinition> sysdef, 
									boost::shared_ptr<ParticleGroup> group,   
									Scalar k,
									Scalar roff,
									int dflag)
        : ForceCompute(sysdef), m_k(k), m_roff(roff), m_group(group), m_dflag(dflag)
    {

    }

/*! Set force
 \param timestep Current timestep
 */
void ConfForceCompute::setParams(Scalar k, Scalar roff, int dflag)
{
	m_k = k;
	m_roff = roff;
	m_dflag = dflag;
}

/*! Confinement acting on members of a given group
    \param timestep Current timestep
*/
void ConfForceCompute::computeForces(unsigned int timestep)
{	
	unsigned int group_size = m_group->getNumMembers();
    if (group_size == 0)
        return;
        	
	// Access particle data		
	ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::read);
	ArrayHandle<Scalar4> h_force(m_force,access_location::host, access_mode::overwrite);
	
	// Need to start from zero force, potential, virial
	const unsigned int size = (unsigned int)m_pdata->getN();
	// Zero data for force calculation.
	memset((void*)h_force.data,0,sizeof(Scalar4)*size);
	
	 // for each of the particles in the group
    for (unsigned int group_idx = 0; group_idx < group_size; group_idx++)
    {
        // get the current particle properties
        unsigned int j = m_group->getMemberIndex(group_idx);
        // Get position
        if (m_dflag == 1)
        {
			Scalar y = h_pos.data[j].y;
			Scalar z = h_pos.data[j].z;
		
			// Calculate distance to center of the box
			Scalar r = sqrt(y*y+z*z);
			
			// Add force if within offset radius
			if (r < m_roff)
			{
				Scalar forcedivr = - m_k * (r - m_roff) / r;
				h_force.data[j].x += Scalar(0.0);
				h_force.data[j].y += forcedivr * y;
				h_force.data[j].z += forcedivr * z;
				h_force.data[j].w += Scalar(0.0);
			}
		}
		else if (m_dflag == 2)
		{
			Scalar x = h_pos.data[j].x;
			Scalar z = h_pos.data[j].z;
		
			// Calculate distance to center of the box
			Scalar r = sqrt(x*x+z*z);
			
			// Add force if within offset radius
			if (r < m_roff)
			{
				Scalar forcedivr = - m_k * (r - m_roff) / r;
				h_force.data[j].y += Scalar(0.0);
				h_force.data[j].x += forcedivr * x;
				h_force.data[j].z += forcedivr * z;
				h_force.data[j].w += Scalar(0.0);
			}
		}
		else if (m_dflag == 3)
		{
			Scalar x = h_pos.data[j].x;
			Scalar y = h_pos.data[j].y;
		
			// Calculate distance to center of the box
			Scalar r = sqrt(x*x+y*y);
			
			// Add force if within offset radius
			if (r < m_roff)
			{
				Scalar forcedivr = - m_k * (r - m_roff) / r;
				h_force.data[j].z += Scalar(0.0);
				h_force.data[j].x += forcedivr * x;
				h_force.data[j].y += forcedivr * y;
				h_force.data[j].w += Scalar(0.0);
			}
		}
	}
}


void export_ConfForceCompute()
    {
    class_< ConfForceCompute, boost::shared_ptr<ConfForceCompute>, bases<ForceCompute>, boost::noncopyable >
    ("ConfForceCompute", init< boost::shared_ptr<SystemDefinition>, 
    					 boost::shared_ptr<ParticleGroup>,   
									Scalar,
									Scalar,
									int >())
    .def("setParams", &ConfForceCompute::setParams)
    ;
    }

// ********************************
// here follows the code for ExampleUpdater on the GPU

#ifdef ENABLE_CUDA

ConfForceComputeGPU::ConfForceComputeGPU(boost::shared_ptr<SystemDefinition> sysdef, 
										boost::shared_ptr<ParticleGroup> group,   
										Scalar k,
										Scalar roff,
										int dflag)
: ConfForceCompute(sysdef, group, k, roff, dflag), m_block_size(128)
{
    // only one GPU is currently supported
    if (!exec_conf->isCUDAEnabled())
	{
        cerr << endl 
		<< "***Error! Creating a ConfForceComputeGPU with no GPU in the execution configuration" 
		<< endl << endl;
        throw std::runtime_error("Error initializing ConfForceComputeGPU");
	}
}
 
/*! Internal method for computing the forces on the GPU.
 \post The force data on the GPU is written with the calculated forces
 
 \param timestep Current time step of the simulation
 
 Calls gpu_compute_conf_forces to do the dirty work.
 */
void ConfForceComputeGPU::computeForces(unsigned int timestep)
{
    unsigned int group_size = m_group->getNumMembers();
    if (group_size == 0)
        return;
    
    assert(m_pdata);
    
    // access the particle data arrays
    const GPUArray< unsigned int >& group_members = m_group->getIndexArray();
    ArrayHandle<unsigned int> d_group_members(group_members, access_location::device, access_mode::read);

    ArrayHandle<Scalar4> d_pos(m_pdata->getPositions(), access_location::device, access_mode::read);
    ArrayHandle<Scalar4> d_force(m_force,access_location::device,access_mode::overwrite);

    // run the kernel in parallel on all GPUs
    gpu_compute_conf_forces(d_force.data,
                                         d_group_members.data,
                                         m_group->getNumMembers(),
                                         m_pdata->getN(),
                                         d_pos.data,
                                         m_k,
                                         m_roff,
                                         m_dflag,
                                         m_block_size);
    
    if (exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();
}

void export_ConfForceComputeGPU()
{
    class_<ConfForceComputeGPU, boost::shared_ptr<ConfForceComputeGPU>, bases<ConfForceCompute>, boost::noncopyable >
    ("ConfForceComputeGPU", init< boost::shared_ptr<SystemDefinition>, 
    						boost::shared_ptr<ParticleGroup>,   
							Scalar,
							Scalar,
							int >())
    ;
}

#endif
