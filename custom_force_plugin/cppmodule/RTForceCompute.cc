/*
Highly Optimized Object-oriented Many-particle Dynamics -- Blue Edition
(HOOMD-blue) Open Source Software License Copyright 2008, 2014 Ames Laboratory
Iowa State University and The Regents of the University of Michigan All rights
reserved.
*/

// $Id: RTForceCompute.cc martbert $
// Maintainer: martbert

#include <boost/python.hpp>
using namespace boost::python;

// we need to include boost.bind for GPUWorker execution
#include <boost/bind.hpp>
using namespace boost;

// import saru random number generator
#include <hoomd/saruprng.h>

#include "RTForceCompute.h"
#ifdef ENABLE_CUDA
#include "RTForceCompute.cuh"
#endif

using namespace std;

/*! \file RTForceCompute.cc
    \brief Contains code for the RTForceCompute class

 	Only use this to simulate run and tumble in 2D at the moment. Could potentially be extended to 3D.
*/

RTForceCompute::RTForceCompute(boost::shared_ptr<SystemDefinition> sysdef)
        : ForceCompute(sysdef)
    {
    	// access the bond data for later use
        m_bond_data = m_sysdef->getBondData();

        // Number of bonds
    	const unsigned int nbonds = (unsigned int)m_bond_data->getNumBonds();

        // Initialize state array
        GPUArray<int> state(nbonds, exec_conf);
        m_state.swap(state);
        
        // Set all swimmers to run state == 1
        ArrayHandle<int> h_state(m_state, access_location::host, access_mode::overwrite);
        for (unsigned int pair_idx = 0; pair_idx < nbonds; pair_idx++)
        {
            h_state.data[pair_idx] = 0;
        }

        // Initialize parameter array
        GPUArray<Scalar4> params(m_bond_data->getNBondTypes(), exec_conf);
        m_params.swap(params);
        //cout<<"Arrays initialized"<<endl;
    }

/*! Set force
 \param timestep Current timestep
 */
void RTForceCompute::setParams(unsigned int type, Scalar RtoT, Scalar TtoR, Scalar Frun, Scalar Ftumble)
{
	ArrayHandle<Scalar4> h_params(m_params, access_location::host, access_mode::readwrite);
    h_params.data[type].x = RtoT;
    h_params.data[type].y = TtoR;
	h_params.data[type].z = Frun;
	h_params.data[type].w = Ftumble;
    //cout<<h_params.data[type].x<< " "<<h_params.data[type].y<< " "<<h_params.data[type].z<< " "<<h_params.data[type].w<<endl;
}

/*! Compute forces acting on members of a given group
    \param timestep Current timestep
*/
void RTForceCompute::computeForces(unsigned int timestep)
{	    	
	int debug_flag = 0;

    // Access particle data		
	ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_rtag(m_pdata->getRTags(), access_location::host, access_mode::read);
	ArrayHandle<Scalar4> h_force(m_force,access_location::host, access_mode::overwrite);
	
    // Access swimmers data
    ArrayHandle<int> h_state(m_state, access_location::host, access_mode::readwrite);

    // Parameters array
    ArrayHandle<Scalar4> h_params(m_params, access_location::host, access_mode::read);

	const BoxDim& box = m_pdata->getBox();

	// Need to start from zero force, potential, virial
	const unsigned int size = (unsigned int)m_pdata->getN();
	// Zero data for force calculation.
	memset((void*)h_force.data,0,sizeof(Scalar4)*size);
	
     // for each of the bonds
    const unsigned int nbonds = (unsigned int)m_bond_data->getNumBonds();
    for (unsigned int bid = 0; bid < nbonds; bid++)
    {
        // lookup the tag of each of the particles participating in the bond
        const Bond& bond = m_bond_data->getBond(bid);
        assert(bond.a < m_pdata->getN());
        assert(bond.b < m_pdata->getN());

        // transform a and b into indicies into the particle data arrays
        // (MEM TRANSFER: 4 integers)
        unsigned int i = h_rtag.data[bond.a];
        unsigned int j = h_rtag.data[bond.b];
        assert(i < m_pdata->getN());
        assert(j < m_pdata->getN());

        // Get swimmer's paramaters
        Scalar4 params = h_params.data[bond.type];

        // Get swimmer's state
        int current_state = h_state.data[bid];
        // Generate a random number
        Saru s(i, j, timestep); // 3 dimensional seeding
        Scalar ran = s.f(0,1);
        cout <<i<<" "<<j<<" " << ran << " "<<current_state<< endl;
        // If it's running see if changes to tumbling
        // And vice versa
        // if (debug_flag)
        // {
        //     cout << "Current state before = " << current_state <<endl;
        //     cout << "Random number generated = " << ran <<endl;
        // }
        if (current_state == 0 && ran < 0.5*params.x)
            current_state = 1;
        else if (current_state == 0 && ran < params.x)
            current_state = -1;
        else if ((current_state == 1 || current_state == -1) && ran < params.y)
            current_state = 0;
        else
            ;
        h_state.data[bid] = current_state;
        // if (debug_flag)
        //     cout << "Current state after = " << current_state <<endl;
            
        // Get positions
        Scalar3 pi = make_scalar3(h_pos.data[i].x, h_pos.data[i].y, h_pos.data[i].z);
        Scalar3 pj = make_scalar3(h_pos.data[j].x, h_pos.data[j].y, h_pos.data[j].z);

        // Initialize empty force
        Scalar3 f = make_scalar3(Scalar(0.0),Scalar(0.0),Scalar(0.0));

        // Calculate dr
        Scalar3 dr = pi - pj;

        // Calculate min image
        dr = box.minImage(dr);

        // Calculate rsq
        Scalar rsq = dot(dr,dr);
        Scalar r = sqrt(rsq);

        if (debug_flag)
            cout << "State is "<< current_state << " at " << timestep <<endl;

        // If running apply run force else torque the damn thing
        if (current_state == 0)
        {
            // Run: apply force on both i and j in direction of dr
            f.x = dr.x / r * params.z;
            f.y = dr.y / r * params.z;
            f.z = dr.z / r * params.z;
            h_force.data[i].x += f.x;
            h_force.data[i].y += f.y;
            h_force.data[i].z += f.z;
            h_force.data[j].x += f.x;
            h_force.data[j].y += f.y;
            h_force.data[j].z += f.z;
            if (debug_flag)
            {
                cout << "(Dx,Fx) = (" << dr.x <<","<<f.x<<")" << " => (" << pi.x <<","<<pj.x<<")" << endl;
                cout << "(Dy,Fy) = (" << dr.y <<","<<f.y<<")" << " => (" << pi.y <<","<<pj.y<<")" << endl;
            }

        }
        else
        {
            // Tumble: apply force on both i and j perpendicular to dr
            if (m_twoDflag == 1)
            {
                Scalar3 dt = make_scalar3(current_state*dr.y,-current_state*dr.x,0.0);
                Scalar tsq = dot(dt,dt);
                Scalar t = sqrt(tsq);
                f.x = dt.x / t * params.w;
                f.y = dt.y / t * params.w;
                h_force.data[i].x += f.x;
                h_force.data[i].y += f.y;
                h_force.data[j].x += -f.x;
                h_force.data[j].y += -f.y;
            } else {
                // Generate random vector in 3D
                Scalar u1 = s.f(-1,1);
                Scalar u2 = s.f(-1,1);
                Scalar u3 = s.f(-1,1);
                Scalar cross = make_scalar3(u2*dr.,-current_state*dr.x,0.0);
            }
        }
    }
}


void export_RTForceCompute()
    {
    class_< RTForceCompute, boost::shared_ptr<RTForceCompute>, bases<ForceCompute>, boost::noncopyable >
    ("RTForceCompute", init< boost::shared_ptr<SystemDefinition> 
    					  >())
    .def("setParams", &RTForceCompute::setParams)
    ;
    }

// ********************************
// here follows the code for ExampleUpdater on the GPU

#ifdef ENABLE_CUDA

RTForceComputeGPU::RTForceComputeGPU(boost::shared_ptr<SystemDefinition> sysdef)
: RTForceCompute(sysdef), m_block_size(128)
{
    // only one GPU is currently supported
    if (!exec_conf->isCUDAEnabled())
	{
        cerr << endl 
		<< "***Error! Creating a RTForceComputeGPU with no GPU in the execution configuration" 
		<< endl << endl;
        throw std::runtime_error("Error initializing RTForceComputeGPU");
	}
}
 
/*! Internal method for computing the forces on the GPU.
 \post The force data on the GPU is written with the calculated forces
 
 \param timestep Current time step of the simulation
 
 Calls gpu_compute_conf_forces to do the dirty work.
 */
void RTForceComputeGPU::computeForces(unsigned int timestep)
{   
    assert(m_pdata);
    
    // access the particle data arrays
    ArrayHandle<Scalar4> d_pos(m_pdata->getPositions(), access_location::device, access_mode::read);
    ArrayHandle<unsigned int> d_rtag(m_pdata->getRTags(), access_location::device, access_mode::read);

    ArrayHandle<Scalar4> d_force(m_force,access_location::device,access_mode::overwrite);

    // Access the bond table for reading
    ArrayHandle<uint2> d_bond_table(m_bond_data->getBondTable(), access_location::device, access_mode::read);
    ArrayHandle<unsigned int> d_bond_types(m_bond_data->getBondTypes(), access_location::device, access_mode::read);
    const unsigned int nbonds = (unsigned int)m_bond_data->getNumBonds();

    // Parameters
    ArrayHandle<Scalar4> d_params(m_params, access_location::device, access_mode::read);

    // Access swimmers data
    ArrayHandle<int> d_state(m_state, access_location::device, access_mode::readwrite);

    const BoxDim& box = m_pdata->getBox();

    // run the kernel in parallel on all GPUs
    gpu_compute_rt_forces(d_force.data,
                          m_pdata->getN(),
                          d_pos.data,
                          d_rtag.data,
                          box,
                          d_bond_table.data,
                          d_bond_types.data,
                          nbonds,
                          d_params.data,
                          d_state.data,
                          timestep,
                          m_block_size);
    
    if (exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();
}

void export_RTForceComputeGPU()
{
    class_<RTForceComputeGPU, boost::shared_ptr<RTForceComputeGPU>, bases<RTForceCompute>, boost::noncopyable >
    ("RTForceComputeGPU", init< boost::shared_ptr<SystemDefinition>
    >())
    ;
}

#endif
