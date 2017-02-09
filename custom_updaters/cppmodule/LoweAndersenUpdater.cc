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

// $Id: LoweAndersenUpdater.cc martbert $
// Maintainer: martbert

/*! \file LoweAndersenUpdater.cc
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

#include "hoomd/saruprng.h"
#include "LoweAndersenUpdater.h"

#include <iostream>
#include <math.h>
#include <stdexcept>

using namespace std;

/*! \param sysdef System to set pressure on
    \param thermo ComputeThermo to compute the pressure with
*/
LoweAndersenUpdater::LoweAndersenUpdater(boost::shared_ptr<SystemDefinition> sysdef,
										 boost::shared_ptr<NeighborList> nlist,
											 Scalar T,
											 Scalar gdt,
											 Scalar rcut,
											 unsigned int seed)
        : Updater(sysdef), m_nlist(nlist), m_T(T), m_gdt(gdt), m_rcut(rcut), m_seed(seed)
    {
    	m_rcutsq = m_rcut*m_rcut;
    }


/*! Perform the proper velocity rescaling
    \param timestep Current time step of the simulation
*/
void LoweAndersenUpdater::update(unsigned int timestep)
{
		
	if (m_prof) m_prof->push("LoweAndersenUpdater");
			
	// obtain box lengths	
    BoxDim box = m_pdata->getBox();
	
	// start by updating the neighborlist
    m_nlist->compute(timestep);

    // access the neighbor list, particle data, and system box
    ArrayHandle<unsigned int> h_n_neigh(m_nlist->getNNeighArray(), access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_nlist(m_nlist->getNListArray(), access_location::host, access_mode::read);
    Index2D nli = m_nlist->getNListIndexer();

    ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::read);
    ArrayHandle<Scalar4> h_vel(m_pdata->getVelocities(), access_location::host, access_mode::readwrite);
	
	for (unsigned int i = 0; i < m_pdata->getN(); i++)
	{
		// access the particle's position, velocity, and type (MEM TRANSFER: 7 scalars)
        Scalar3 pi = make_scalar3(h_pos.data[i].x, h_pos.data[i].y, h_pos.data[i].z);
        Scalar3 vi = make_scalar3(h_vel.data[i].x, h_vel.data[i].y, h_vel.data[i].z);

        // loop over all of the neighbors of this particle
        const unsigned int size = (unsigned int)h_n_neigh.data[i];
        for (unsigned int k = 0; k < size; k++)
        {
            // access the index of this neighbor (MEM TRANSFER: 1 scalar)
            unsigned int j = h_nlist.data[nli(i, k)];

            if (j > i)
            {
            	// calculate dr_ji (MEM TRANSFER: 3 scalars / FLOPS: 3)
                Scalar3 pj = make_scalar3(h_pos.data[j].x, h_pos.data[j].y, h_pos.data[j].z);
                Scalar3 dx = pi - pj;
    
                // calculate dv_ji (MEM TRANSFER: 3 scalars / FLOPS: 3)
                Scalar3 vj = make_scalar3(h_vel.data[j].x, h_vel.data[j].y, h_vel.data[j].z);
                Scalar3 dv = vi - vj;
    
                // apply periodic boundary conditions
                dx = box.minImage(dx);
    
                // calculate r_ij squared (FLOPS: 5)
                Scalar rsq = dot(dx, dx);
    			if (rsq < m_rcutsq)
	            {
		            Scalar r = sqrt(rsq);

		            //calculate the drag term r \dot v
		            Scalar rdotv = dot(dx, dv);

		            Saru saru( i , j , m_seed + timestep );

		            // Generate a first random number
		            Scalar r1 = saru.f( 0, 1) ;
		            
		            // If the random number is lower or than gamma sample vpar
		            if ( r1 < m_gdt)
		            {
		                // Generate two other random numbers
		                Scalar r2 = saru.f( 0, 1) ;
		                Scalar r3 = saru.f( 0, 1) ;
		                
		                // Calculate normal distributed variable
		                Scalar n = sqrt(-2.0*log(r2)) * cos(2.0*3.1416*r3);
		                
		                // Velocity sampled
		                Scalar vpar = sqrt(2.0*m_T) * n;

		                // Delta ij
		                Scalar dij = 0.5*(vpar - rdotv / r) / r;
		                // cout << "Dx = "<<dx.x << " " << dx.y << " " << dx.z <<endl;
		                // cout <<"v1 = " <<vi.x << " " << vi.y << " " << vi.z <<endl;
		                // cout <<"v2 = "<< vj.x << " " << vj.y << " " << vj.z <<endl;
		                // cout <<"Delta = "<<dij*dx.x << " " << dij*dx.y << " " << dij*dx.z <<endl;
						h_vel.data[i].x += dij*dx.x;
						h_vel.data[i].y += dij*dx.y;
						h_vel.data[i].z += dij*dx.z;
						h_vel.data[j].x -= dij*dx.x;
						h_vel.data[j].y -= dij*dx.y;
						h_vel.data[j].z -= dij*dx.z;
		            }
		        }
            }
		}
	}
        
    if (m_prof) m_prof->pop();
}

void export_LoweAndersenUpdater()
    {    
    class_<LoweAndersenUpdater, boost::shared_ptr<LoweAndersenUpdater>, bases<Updater>, boost::noncopyable>
    ("LoweAndersenUpdater", init< boost::shared_ptr<SystemDefinition>,
    							  boost::shared_ptr<NeighborList>,
								 Scalar,
								 Scalar,
								 Scalar,
								 unsigned int >());
    }

#ifdef WIN32
#pragma warning( pop )
#endif

