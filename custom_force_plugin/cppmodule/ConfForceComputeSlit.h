/*
Highly Optimized Object-oriented Many-particle Dynamics -- Blue Edition
(HOOMD-blue) Open Source Software License Copyright 2008, 2009 Ames Laboratory
Iowa State University and The Regents of the University of Michigan All rights
reserved.
*/

// $Id: ConfForceComputeSlit.h 2012 martbert $
// Maintainer: martbert

#include <hoomd/hoomd_config.h>

// First, hoomd.h should be included
#include <hoomd/hoomd.h>

#include <boost/shared_ptr.hpp>

/*! \file ConfForceComputeSlit.h
    \brief Class to add a position dependent confinement force in the yz plane to a given group in the system
*/

#ifndef _CONFFORCECOMPUTESLIT_H_
#define _CONFFORCECOMPUTESLIT_H_

//! Adds a confinement force to each particle of a given group
/*! \ingroup computes
*/
class ConfForceComputeSlit : public ForceCompute
    {
    public:
        //! Constructs the compute
        ConfForceComputeSlit(boost::shared_ptr<SystemDefinition> sysdef, 
        				boost::shared_ptr<ParticleGroup> group,   
        				Scalar k,
        				Scalar roff,
        				unsigned int dflag);
		
		//void setParams(group,epsilon,sigma,roff,rcut);
		void setParams(boost::shared_ptr<ParticleGroup> group,Scalar k,Scalar roff,unsigned int dflag);
		
    protected:
		
		//Scalar m_epsilon;	//!< LJ epsilon parameter to mimic LJ wall
		//Scalar m_sigma;         //!< LJ sigma parameter
		//Scalar m_roff;		//!< Offset radius where the potential is suppose to act
		//Scalar m_rcut;		//!< Cutting radius to keep the interaction short range
		Scalar m_k;
		Scalar m_roff;
		unsigned int m_dflag;
		
		boost::shared_ptr<ParticleGroup> m_group; //!< Group on which the potential is acting
        
	//! Actually compute the forces
        virtual void computeForces(unsigned int timestep);
    };

//! Exports the ConstForceComputeClass to python
void export_ConfForceComputeSlit();

#ifdef ENABLE_CUDA

//! A GPU accelerated version of CuboidForceCompute

//! Implements the cuboid force calculation on the GPU
/*! CuboidForceComputeGPU implements the same calculations as CuboidForceCompute,
 but executing on the GPU.
 
 The parameters are all stored in float3
 
 The GPU computation is implemented in cuboidforce_kernel.cu
 
 \ingroup computes
 */
class ConfForceComputeSlitGPU : public ConfForceComputeSlit
    {
    public:
        //! Constructs the compute
        ConfForceComputeSlitGPU(boost::shared_ptr<SystemDefinition> sysdef,
        					  boost::shared_ptr<ParticleGroup> group, 
        					  Scalar k,
        					  Scalar roff,
        					  unsigned int dflag);
        
    protected:
        unsigned int m_block_size;                   //!< Block size to run calculation on
        		
        //! Actually compute the forces
        virtual void computeForces(unsigned int timestep);
    };

//! Export the BondForceComputeGPU class to python
void export_ConfForceComputeSlitGPU();

#endif //CUDA stuff

#endif // CuboidForceCompute

