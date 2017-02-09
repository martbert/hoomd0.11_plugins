/*
Highly Optimized Object-oriented Many-particle Dynamics -- Blue Edition
(HOOMD-blue) Open Source Software License Copyright 2008, 2009 Ames Laboratory
Iowa State University and The Regents of the University of Michigan All rights
reserved.
*/

// $Id: ConstForceCompute.h 2843 2010-03-09 17:01:45Z martbert $
// $URL: http://codeblue.umich.edu/hoomd-blue/svn/trunk/libhoomd/computes/ConstForceCompute.h $
// Maintainer: martbert

#include <hoomd/hoomd_config.h>

// First, hoomd.h should be included
#include <hoomd/hoomd.h>

#include <boost/shared_ptr.hpp>

/*! \file CuboidForceCompute.h
    \brief Class to add constant force to particles in a cuboid
*/

#ifndef _CUBOIDFORCECOMPUTE_H_
#define _CUBOIDFORCECOMPUTE_H_

//! Adds a constant force to each particle in a cuboid
/*! \ingroup computes
*/
class CuboidForceCompute : public ForceCompute
    {
    public:
        //! Constructs the compute
        CuboidForceCompute(boost::shared_ptr<SystemDefinition> sysdef, Scalar3 f, Scalar3 min, Scalar3 max);
		
		void setParams(Scalar3 f, Scalar3 min, Scalar3 max);
		
    protected:
		
		Scalar3 m_f;	//!< Const force
		Scalar3 m_min;	//!< Minimum
		Scalar3 m_max;	//!< Maximum
		
        //! Actually compute the forces
        virtual void computeForces(unsigned int timestep);
    };

//! Exports the ConstForceComputeClass to python
void export_CuboidForceCompute();

#ifdef ENABLE_CUDA

//! A GPU accelerated version of CuboidForceCompute

//! Implements the cuboid force calculation on the GPU
/*! CuboidForceComputeGPU implements the same calculations as CuboidForceCompute,
 but executing on the GPU.
 
 The parameters are all stored in float3
 
 The GPU computation is implemented in cuboidforce_kernel.cu
 
 \ingroup computes
 */
class CuboidForceComputeGPU : public CuboidForceCompute
    {
    public:
        //! Constructs the compute
        CuboidForceComputeGPU(boost::shared_ptr<SystemDefinition> sysdef, Scalar3 f, Scalar3 min, Scalar3 max);
        
    protected:
        unsigned int m_block_size;                   //!< Block size to run calculation on
        		
        //! Actually compute the forces
        virtual void computeForces(unsigned int timestep);
    };

//! Export the BondForceComputeGPU class to python
void export_CuboidForceComputeGPU();

#endif //CUDA stuff

#endif // CuboidForceCompute

