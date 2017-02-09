/*
Highly Optimized Object-oriented Many-particle Dynamics -- Blue Edition
(HOOMD-blue) Open Source Software License Copyright 2008, 2009 Ames Laboratory
Iowa State University and The Regents of the University of Michigan All rights
reserved.
*/

// $Id: FreeDrainForceCompute.h 2011-11 martbert $
// Maintainer: martbert

#include <hoomd/hoomd_config.h>

// First, hoomd.h should be included
#include <hoomd/hoomd.h>

#include <boost/shared_ptr.hpp>

/*! \file FreeDrainForceCompute.h
    \brief Class to implement the procedure detailed in Duong-Hong et al., Electrophoresis 29 (2008)
	\brief where the free-draining nature of DNA is reproduced by generating EOF around the polymer
*/

#ifndef _FREEDRAINFORCECOMPUTE_H_
#define _FREEDRAINFORCECOMPUTE_H_

//! Adds a constant force to each particle of group2 found within a cutoff distance lD of group1
/*! \ingroup computes
*/
class FreeDrainForceCompute : public ForceCompute
    {
    public:
        //! Constructs the compute
        FreeDrainForceCompute(boost::shared_ptr<SystemDefinition> sysdef, 
							  boost::shared_ptr<ParticleGroup> group1, 
							  boost::shared_ptr<ParticleGroup> group2, 
							  Scalar Ex, 
							  Scalar Ey, 
							  Scalar Ez, 
							  Scalar lD,
							  Scalar bj,
                              Scalar qt,
							  Scalar cut);
		
		void setParams(Scalar Ex, Scalar Ey, Scalar Ez, Scalar lD, Scalar bj, Scalar qt, Scalar cut);
		
    protected:
		
		Scalar m_Ex; //!< Field in the x direction
        Scalar m_Ey; //!< Field in the y direction
        Scalar m_Ez; //!< Field in the z direction
		Scalar m_lD; //!< Debye length
		Scalar m_lD2;//!< Debye length squared
		Scalar m_bj; //!< Bjerrum length
        Scalar m_qt; //!< Condensation criterion
		Scalar m_cut;//!< Cutoff for the force
		Scalar m_cut2;//!< Cutoff for the force squared
		
        // Array containing the original charges
        GPUArray<Scalar> m_org_charge;
        
		//! Group of particles around which the EOF flow is generated and are affected by the field
        boost::shared_ptr<ParticleGroup> m_group1;
        //! Group of particles that generate the EOF (thin Debye layer approximation)
        boost::shared_ptr<ParticleGroup> m_group2;
		//! Types of the particles involved
		//unsigned int m_type1;
		//unsigned int m_type2;
		
        //! Actually compute the forces
        virtual void computeForces(unsigned int timestep);
    };

//! Exports the FreeDrainForceComputeClass to python
void export_FreeDrainForceCompute();

#ifdef ENABLE_CUDA

//! A GPU accelerated version of FreeDrainForceCompute

//! Implements the EOF generation on the GPU
/*! FreeDrainForceComputeGPU implements the same calculations as FreeDrainForceCompute,
 but executing on the GPU.
 
 The GPU computation is implemented in freedrainforce_kernel.cu
 
 \ingroup computes
 */
class FreeDrainForceComputeGPU : public FreeDrainForceCompute
    {
    public:
        //! Constructs the compute
        FreeDrainForceComputeGPU(boost::shared_ptr<SystemDefinition> sysdef,
								 boost::shared_ptr<ParticleGroup> group1, 
								 boost::shared_ptr<ParticleGroup> group2, 
								 Scalar Ex, 
								 Scalar Ey, 
								 Scalar Ez, 
								 Scalar lD,
								 Scalar bj,
                                 Scalar qt,
								 Scalar cut);
        
    protected:
        unsigned int m_block_size;                   //!< Block size to run calculation on
        		
        //! Actually compute the forces
        virtual void computeForces(unsigned int timestep);
    };

//! Export the FreeDrainForceComputeGPU class to python
void export_FreeDrainForceComputeGPU();

#endif //CUDA stuff

#endif // FreeDrainForceCompute

