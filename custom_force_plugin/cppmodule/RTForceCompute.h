/*
Highly Optimized Object-oriented Many-particle Dynamics -- Blue Edition
(HOOMD-blue) Open Source Software License Copyright 2008, 2014 Ames Laboratory
Iowa State University and The Regents of the University of Michigan All rights
reserved.
*/

// $Id: RTForceCompute.h martbert $
// Maintainer: martbert

#include <hoomd/hoomd_config.h>

// First, hoomd.h should be included
#include <hoomd/hoomd.h>

#include <boost/shared_ptr.hpp>

/*! \file RTForceCompute.h
*/

#ifndef _RTFORCECOMPUTE_H_
#define _RTFORCECOMPUTE_H_

/*! \ingroup computes
*/
class RTForceCompute : public ForceCompute
    {
    public:
        //! Constructs the compute
        RTForceCompute(boost::shared_ptr<SystemDefinition> sysdef);
		
		
		void setParams(unsigned int type, Scalar RtoT, Scalar TtoR, Scalar Frun, Scalar Ftumble);
		
    protected:		
        // Array containing the parameters of the swimmers per type index
        // x: RtoT, y: TtoR, z: Frun, w: Ftumble
        GPUArray<Scalar4> m_params;

        // Array containing the states of the swimmers
        GPUArray<int> m_state;

        boost::shared_ptr<BondData> m_bond_data;    //!< Bond data to use in computing RT forces

	//! Actually compute the forces
        virtual void computeForces(unsigned int timestep);
    };

//! Exports the RTForceComputeClass to python
void export_RTForceCompute();

#ifdef ENABLE_CUDA

//! A GPU accelerated version of RTForceCompute
class RTForceComputeGPU : public RTForceCompute
    {
    public:
        //! Constructs the compute
        RTForceComputeGPU(boost::shared_ptr<SystemDefinition> sysdef);
        
    protected:
        unsigned int m_block_size; //!< Block size to run calculation on
        		
        //! Actually compute the forces
        virtual void computeForces(unsigned int timestep);
    };

//! Export the BondForceComputeGPU class to python
void export_RTForceComputeGPU();

#endif

#endif

