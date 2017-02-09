/*
Highly Optimized Object-oriented Many-particle Dynamics -- Blue Edition
(HOOMD-blue) Open Source Software License Copyright 2008, 2009 Ames Laboratory
Iowa State University and The Regents of the University of Michigan All rights
reserved.
*/

// $Id: CuboidForceCompute.cc 2843 2010-03-09 17:01:45Z martbert $
// $URL: http://codeblue.umich.edu/hoomd-blue/svn/trunk/libhoomd/computes/CuboidForceCompute.cc $
// Maintainer: martbert

#include <boost/python.hpp>
using namespace boost::python;

// we need to include boost.bind for GPUWorker execution
#include <boost/bind.hpp>
using namespace boost;

#include "CuboidForceCompute.h"
#ifdef ENABLE_CUDA
#include "CuboidForceCompute.cuh"
#endif

using namespace std;

/*! \file CuboidForceCompute.cc
    \brief Contains code for the CuboidForceCompute class
*/

/*! \param sysdef SystemDefinition containing the ParticleData to compute forces on
    \param f external force
    \param min mins of the cuboid
    \param max maxs of the cuboid
*/
CuboidForceCompute::CuboidForceCompute(boost::shared_ptr<SystemDefinition> sysdef, Scalar3 f, Scalar3 min, Scalar3 max)
        : ForceCompute(sysdef), m_f(f), m_min(min), m_max(max)
    {

    }

/*! Set force
 \param timestep Current timestep
 */
void CuboidForceCompute::setParams(Scalar3 f, Scalar3 min, Scalar3 max)
{
	m_f = f;
	m_min = min;
	m_max = max;
}

/*! Compute constant forces on particles in a cuboid
    \param timestep Current timestep
*/
void CuboidForceCompute::computeForces(unsigned int timestep)
	{	
		
		assert(m_pdata);
		ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::read);
		ArrayHandle<Scalar4> h_force(m_force,access_location::host, access_mode::overwrite);
		
		// Need to start from zero force, potential, virial
		const unsigned int size = (unsigned int)m_pdata->getN();
		// Zero data for force calculation.
		memset((void*)h_force.data,0,sizeof(Scalar4)*m_force.getNumElements());
		
		// Loop through data
		for (unsigned int i = 0; i < size; i++)
		{
			// Get position
			Scalar x = h_pos.data[i].x;
			Scalar y = h_pos.data[i].y;
			Scalar z = h_pos.data[i].z;
			
			// Verify that the particle is in the cuboid
			if (x >= m_min.x && x < m_max.x && y >= m_min.y && y < m_max.y && z >= m_min.z && z < m_max.z)
			{
				h_force.data[i].x += m_f.x;
				h_force.data[i].y += m_f.y;
				h_force.data[i].z += m_f.z;
				h_force.data[i].w += 0;
			}
		}
    }


void export_CuboidForceCompute()
    {
    class_< CuboidForceCompute, boost::shared_ptr<CuboidForceCompute>, bases<ForceCompute>, boost::noncopyable >
    ("CuboidForceCompute", init< boost::shared_ptr<SystemDefinition>, Scalar3, Scalar3, Scalar3 >())
    .def("setParams", &CuboidForceCompute::setParams)
    ;
    }

// ********************************
// here follows the code for ExampleUpdater on the GPU

#ifdef ENABLE_CUDA

CuboidForceComputeGPU::CuboidForceComputeGPU(boost::shared_ptr<SystemDefinition> sysdef, Scalar3 f, Scalar3 min, Scalar3 max)
: CuboidForceCompute(sysdef, f, min, max), m_block_size(256)
{
    // only one GPU is currently supported
    if (!exec_conf->isCUDAEnabled())
	{
        cerr << endl 
		<< "***Error! Creating a CuboidForceComputeGPU with no GPU in the execution configuration" 
		<< endl << endl;
        throw std::runtime_error("Error initializing CuboidForceComputeGPU");
	}
}
 
/*! Internal method for computing the forces on the GPU.
 \post The force data on the GPU is written with the calculated forces
 
 \param timestep Current time step of the simulation
 
 Calls gpu_compute_cuboid_forces to do the dirty work.
 */
void CuboidForceComputeGPU::computeForces(unsigned int timestep)
{
    // start the profile
    if (m_prof) m_prof->push(exec_conf, "Cuboid");
    
    ArrayHandle<Scalar4> d_pos(m_pdata->getPositions(), access_location::device, access_mode::read);
    ArrayHandle<Scalar4> d_force(m_force,access_location::device,access_mode::overwrite);
	
	// run the kernel
    gpu_compute_cuboid_forces(d_force.data,
							  d_pos.data,
							  m_pdata->getN(),
							  m_f,
							  m_min,
							  m_max,
							  m_block_size);
    if (exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();
}

void export_CuboidForceComputeGPU()
{
    class_<CuboidForceComputeGPU, boost::shared_ptr<CuboidForceComputeGPU>, bases<CuboidForceCompute>, boost::noncopyable >
    ("CuboidForceComputeGPU", init< boost::shared_ptr<SystemDefinition>, Scalar3, Scalar3, Scalar3 >())
    ;
}

#endif
