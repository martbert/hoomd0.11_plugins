/*
Highly Optimized Object-oriented Many-particle Dynamics -- Blue Edition
(HOOMD-blue) Open Source Software License Copyright 2008, 2009 Ames Laboratory
Iowa State University and The Regents of the University of Michigan All rights
reserved.
*/

// $Id: FreeDrainForceCompute.h 2011-11 martbert $
// Maintainer: martbert

#include <boost/python.hpp>
using namespace boost::python;

// we need to include boost.bind for GPUWorker execution
#include <boost/bind.hpp>
using namespace boost;

#include "FreeDrainForceCompute.h"
#ifdef ENABLE_CUDA
#include "FreeDrainForceCompute.cuh"
#endif

using namespace std;

/*! \file FreeDrainForceCompute.cc
    \brief Contains code for the FreeDrainForceCompute class
*/

/*! \param sysdef SystemDefinition containing the ParticleData to compute forces on
    \param Ex,Ey,Ez external electrical Field
    \param lD Debye length
    \param group2 generates EOF around group1
*/
FreeDrainForceCompute::FreeDrainForceCompute(boost::shared_ptr<SystemDefinition> sysdef, 
											 boost::shared_ptr<ParticleGroup> group1, 
											 boost::shared_ptr<ParticleGroup> group2, 
											 Scalar Ex, 
											 Scalar Ey, 
											 Scalar Ez, 
											 Scalar lD,
											 Scalar bj,
											 Scalar qt,
											 Scalar cut)
        : ForceCompute(sysdef), m_group1(group1), m_group2(group2)
    {
        // Set parameters
        setParams(Ex,Ey,Ez,lD,bj,qt,cut);

        // Save an original copy of the charges on the solute
        unsigned int numMembersGroup1 = m_group1->getNumMembers();
        GPUArray<Scalar> org_charge(numMembersGroup1, exec_conf);
        m_org_charge.swap(org_charge);

        // Set original charges
        ArrayHandle<Scalar> h_charge(m_pdata->getCharges(), access_location::host, access_mode::read);
        ArrayHandle<Scalar> h_org_charge(m_org_charge, access_location::host, access_mode::overwrite);
        for (unsigned int idx = 0; idx < numMembersGroup1; idx++)
        {
            unsigned int tag_idx = m_group1->getMemberTag(idx);
            h_org_charge.data[idx] = h_charge.data[tag_idx];
        }
    }

/*! Set force
 \param timestep Current timestep
 */
void FreeDrainForceCompute::setParams(Scalar Ex, Scalar Ey, Scalar Ez, Scalar lD, Scalar bj, Scalar qt, Scalar cut)
{
	m_Ex = Ex;
	m_Ey = Ey;
	m_Ez = Ez;
	m_lD = lD;
	m_lD2 = lD*lD;
	m_bj = bj;
	m_qt = qt;
	m_cut = cut;
	m_cut2 = cut*cut;
	//cout<<"Bjerrum length = "<<m_bj<<"; Debye length = "<<m_lD<<endl;
	
	// Set m_type
	//ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::read);
	//ArrayHandle<unsigned int> h_rtag(m_pdata->getRTags(), access_location::host, access_mode::read);
	//unsigned int i0 = 0;
	//m_type1 = __scalar_as_int(h_pos.data[h_rtag.data[m_group1->getMemberTag(i0)]].w);
	//m_type2 = __scalar_as_int(h_pos.data[h_rtag.data[m_group2->getMemberTag(i0)]].w);
	//cout<<"Types initialized: "<<m_type1<<","<<m_type2<<endl;
}

void FreeDrainForceCompute::computeForces(unsigned int timestep)
{	
	// access the particle data arrays
	ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::read);
	ArrayHandle<Scalar> h_charge(m_pdata->getCharges(), access_location::host, access_mode::read);
	ArrayHandle<unsigned int> h_rtag(m_pdata->getRTags(), access_location::host, access_mode::read);
	
	ArrayHandle<Scalar4> h_force(m_force,access_location::host, access_mode::overwrite);
	ArrayHandle<Scalar> h_virial(m_virial,access_location::host, access_mode::overwrite);

	// Zero data for force calculation.
	memset((void*)h_force.data,0,sizeof(Scalar4)*m_force.getNumElements());
	memset((void*)h_virial.data,0,sizeof(Scalar)*m_virial.getNumElements());
	
	 // get a local copy of the simulation box too
    const BoxDim& box = m_pdata->getBox();
	
	// Step 1 - Loop through group1:
	// - Compute for that will have to be normalized in step 2
	// - Calculate the number of particles of group2 that are going to be affected
	unsigned int numMembersGroup1 = m_group1->getNumMembers();
	unsigned int numMembersGroup2 = m_group2->getNumMembers();
	Scalar3 totalForceGroup1 = make_scalar3(0.0,0.0,0.0);
	Scalar3 totalForceGroup2 = make_scalar3(0.0,0.0,0.0);
	for (unsigned int i = 0; i < numMembersGroup1; i++)
        {
        // get the tag for the current group member from the group
        unsigned int tagi = m_group1->getMemberTag(i);
        // identify the index of the current particle tag
        unsigned int idxi = h_rtag.data[tagi];
		// Get charge
		Scalar qi = h_charge.data[idxi];
		// Add force to particle idx
		h_force.data[idxi].x  += qi*m_Ex;
		h_force.data[idxi].y += qi*m_Ey;
		h_force.data[idxi].z += qi*m_Ez;
		h_force.data[idxi].w = 0.0;
		// Update the total force
		totalForceGroup1.x += qi*m_Ex;
		totalForceGroup1.y += qi*m_Ey;
		totalForceGroup1.z += qi*m_Ez;
		
		// loop over all members of group 2
        for (unsigned int j = 0; j < numMembersGroup2; j++)
		{
			 // get the tag for the current group member from the group
       		 unsigned int tagj = m_group2->getMemberTag(j);
        	// identify the index of the current particle tag
       		 unsigned int idxj = h_rtag.data[tagj];
			
			// Calculate distance between idx and j
       		Scalar3 dr;
			dr.x = h_pos.data[idxi].x - h_pos.data[idxj].x;
			dr.y = h_pos.data[idxi].y - h_pos.data[idxj].y;
			dr.z = h_pos.data[idxi].z - h_pos.data[idxj].z;
			
			// apply minimum image conventions to all 3 vectors
        	dr = box.minImage(dr);
			
			Scalar rsq = dr.x*dr.x + dr.y*dr.y + dr.z*dr.z;
			Scalar r =  sqrt(rsq);
			// Add force to particle j if it is within cutoff
			if (rsq < m_cut2)
			{
				Scalar qj = h_charge.data[j]*exp(-r/m_lD)/r;					
				h_force.data[idxj].x  += qj*m_Ex;
				h_force.data[idxj].y += qj*m_Ey;
				h_force.data[idxj].z += qj*m_Ez;
				h_force.data[idxj].w = 0.0;
				
				// Total force on group 2
				totalForceGroup2.x += qj*m_Ex;
				totalForceGroup2.y += qj*m_Ey;
				totalForceGroup2.z += qj*m_Ez;
			}
		}
		
		// loop over all members of group 1
		// add interparticle force on group1
        for (unsigned int j = i+1; j < numMembersGroup1; j++)
		{
			// get the tag for the current group member from the group
       		 unsigned int tagj = m_group1->getMemberTag(j);
        	// identify the index of the current particle tag
       		 unsigned int idxj = h_rtag.data[tagj];
			
			// Calculate distance between idx and j
       		Scalar3 dr;
			dr.x = h_pos.data[idxi].x - h_pos.data[idxj].x;
			dr.y = h_pos.data[idxi].y - h_pos.data[idxj].y;
			dr.z = h_pos.data[idxi].z - h_pos.data[idxj].z;
			
			// if the vector crosses the box, pull it back
			// (FLOPS: 9 (worst case: first branch is missed, the 2nd is taken and the add is done))
        	dr = box.minImage(dr);
			
			Scalar rsq = dr.x*dr.x + dr.y*dr.y + dr.z*dr.z;
			Scalar r =  sqrt(rsq);
			// Add force to particle j if it is within cutoff
			if (rsq < m_cut2)
			{
				Scalar energy = qi * h_charge.data[j] * m_bj * exp(-r / m_lD) / r;
				Scalar forcedivr = energy / r / r / m_lD * (r + m_lD);
				//cout<<"Energy = "<<energy<<"; Force div r = "<<forcedivr<<endl;
				h_force.data[idxi].x += forcedivr*dr.x;
				h_force.data[idxi].y += forcedivr*dr.y;
				h_force.data[idxi].z += forcedivr*dr.z;
				h_force.data[idxi].w += Scalar(0.5)*energy;
				h_force.data[idxj].x -= forcedivr*dr.x;
				h_force.data[idxj].y -= forcedivr*dr.y;
				h_force.data[idxj].z -= forcedivr*dr.z;
				h_force.data[idxj].w += Scalar(0.5)*energy;
			}
		}		
	}
        
	//Step 2 - Normalize the force on group2
	Scalar3 factor;
	if (totalForceGroup2.x > Scalar(0.0))
		factor.x = totalForceGroup1.x/totalForceGroup2.x;
	else
		factor.x = Scalar(1.0);
	if (totalForceGroup2.y > Scalar(0.0))
		factor.y = totalForceGroup1.y/totalForceGroup2.y;
	else
		factor.y = Scalar(1.0);
	if (totalForceGroup2.z > Scalar(0.0))
		factor.z = totalForceGroup1.z/totalForceGroup2.z;
	else
		factor.z = Scalar(1.0);	
	for (unsigned int i = 0; i < numMembersGroup2; i++)
    {
        // get the tag for the current group member from the group
        unsigned int tag = m_group2->getMemberTag(i);
        // identify the index of the current particle tag
        unsigned int idx = h_rtag.data[tag];
		
		h_force.data[idx].x *= factor.x;
		h_force.data[idx].y *= factor.y;
		h_force.data[idx].z *= factor.z;
    }
            
}


void export_FreeDrainForceCompute()
    {
    class_< FreeDrainForceCompute, boost::shared_ptr<FreeDrainForceCompute>, bases<ForceCompute>, boost::noncopyable >
    ("FreeDrainForceCompute", init< boost::shared_ptr<SystemDefinition>, 
    								boost::shared_ptr<ParticleGroup>,
    								boost::shared_ptr<ParticleGroup>,
    								Scalar, Scalar, Scalar, Scalar, Scalar, Scalar, Scalar >())
    .def("setParams", &FreeDrainForceCompute::setParams)
    ;
    }

// ********************************
// here follows the code for ExampleUpdater on the GPU

#ifdef ENABLE_CUDA

FreeDrainForceComputeGPU::FreeDrainForceComputeGPU(boost::shared_ptr<SystemDefinition> sysdef,
												   boost::shared_ptr<ParticleGroup> group1, 
												   boost::shared_ptr<ParticleGroup> group2, 
												   Scalar Ex, 
												   Scalar Ey, 
												   Scalar Ez, 
												   Scalar lD,
												   Scalar bj,
												   Scalar qt,
												   Scalar cut)
: FreeDrainForceCompute(sysdef, group1, group2, Ex, Ey, Ez, lD, bj, qt, cut), m_block_size(256)
{
    // only one GPU is currently supported
    if (!exec_conf->isCUDAEnabled())
	{
        cerr << endl 
		<< "***Error! Creating a FreeDrainForceComputeGPU with no GPU in the execution configuration" 
		<< endl << endl;
        throw std::runtime_error("Error initializing FreeDrainForceComputeGPU");
	}
}
 
/*! Internal method for computing the forces on the GPU.
 \post The force data on the GPU is written with the calculated forces
 
 \param timestep Current time step of the simulation
 
 Calls gpu_compute_freedrain_forces to do the dirty work.
 */

void FreeDrainForceComputeGPU::computeForces(unsigned int timestep)
{   
    // start the profile
    if (m_prof) m_prof->push(exec_conf, "FreeDrain");
    
    // Call the kernel
    ArrayHandle<Scalar4> d_pos(m_pdata->getPositions(), access_location::device, access_mode::read);
    ArrayHandle<Scalar> d_charge(m_pdata->getCharges(), access_location::device, access_mode::overwrite);
    ArrayHandle<Scalar> d_org_charge(m_org_charge, access_location::device, access_mode::read);
    BoxDim box = m_pdata->getBox();
    
    ArrayHandle<Scalar4> d_force(m_force,access_location::device,access_mode::overwrite);
    ArrayHandle<Scalar> d_virial(m_virial,access_location::device,access_mode::overwrite);
	
	const GPUArray< unsigned int >& group_members1 = m_group1->getIndexArray();
    ArrayHandle<unsigned int> d_group_members1(group_members1, access_location::device, access_mode::read);
    const GPUArray< unsigned int >& group_members2 = m_group2->getIndexArray();
    ArrayHandle<unsigned int> d_group_members2(group_members2, access_location::device, access_mode::read);
	
	// run the kernel
    // cout << "Entering force compute" << endl;
    gpu_compute_freedrain_forces(d_force.data,
								 d_virial.data,
								 m_pdata->getN(),
								 d_pos.data,
								 d_charge.data,
                                 d_org_charge.data,
								 box,
								 d_group_members1.data,
								 m_group1->getNumMembers(),
								 d_group_members2.data,
								 m_group2->getNumMembers(),
								 m_Ex,
								 m_Ey,
								 m_Ez,
								 m_lD,
								 m_bj,
								 m_qt,
								 m_cut2,
								 m_block_size);
    
    if (exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();
}

void export_FreeDrainForceComputeGPU()
{
    class_<FreeDrainForceComputeGPU, boost::shared_ptr<FreeDrainForceComputeGPU>, bases<FreeDrainForceCompute>, boost::noncopyable >
    ("FreeDrainForceComputeGPU", init< boost::shared_ptr<SystemDefinition>, 
    								   boost::shared_ptr<ParticleGroup>, 
    								   boost::shared_ptr<ParticleGroup>, 
    								   Scalar, Scalar, Scalar, Scalar, Scalar, Scalar, Scalar >())
    ;
}

#endif
