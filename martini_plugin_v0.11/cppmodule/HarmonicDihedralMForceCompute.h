/*
Highly Optimized Object-oriented Many-particle Dynamics -- Blue Edition
(HOOMD-blue) Open Source Software License Copyright 2008-2011 Ames Laboratory
Iowa State University and The Regents of the University of Michigan All rights
reserved.

HOOMD-blue may contain modifications ("Contributions") provided, and to which
copyright is held, by various Contributors who have granted The Regents of the
University of Michigan the right to modify and/or distribute such Contributions.

You may redistribute, use, and create derivate works of HOOMD-blue, in source
and binary forms, provided you abide by the following conditions:

* Redistributions of source code must retain the above copyright notice, this
list of conditions, and the following disclaimer both in the code and
prominently in any materials provided with the distribution.

* Redistributions in binary form must reproduce the above copyright notice, this
list of conditions, and the following disclaimer in the documentation and/or
other materials provided with the distribution.

* All publications and presentations based on HOOMD-blue, including any reports
or published results obtained, in whole or in part, with HOOMD-blue, will
acknowledge its use according to the terms posted at the time of submission on:
http://codeblue.umich.edu/hoomd-blue/citations.html

* Any electronic documents citing HOOMD-Blue will link to the HOOMD-Blue website:
http://codeblue.umich.edu/hoomd-blue/

* Apart from the above required attributions, neither the name of the copyright
holder nor the names of HOOMD-blue's contributors may be used to endorse or
promote products derived from this software without specific prior written
permission.

Disclaimer

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER AND CONTRIBUTORS ``AS IS'' AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND/OR ANY
WARRANTIES THAT THIS SOFTWARE IS FREE OF INFRINGEMENT ARE DISCLAIMED.

IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

// Maintainer: dnlebard

#include <boost/shared_ptr.hpp>

#include <hoomd/hoomd_config.h>
#include <hoomd/hoomd.h>

#include <vector>

/*! \file HarmonicDihedralMForceCompute.h
    \brief Declares a class for computing harmonic dihedrals
*/

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#ifndef __HARMONICDIHEDRALMFORCECOMPUTE_H__
#define __HARMONICDIHEDRALMFORCECOMPUTE_H__

//! Computes harmonic dihedral forces on each particle
/*! Harmonic dihedral forces are computed on every particle in the simulation.

    The dihedrals which forces are computed on are accessed from ParticleData::getDihedralData
    \ingroup computes
*/
class HarmonicDihedralMForceCompute : public ForceCompute
    {
    public:
        //! Constructs the compute
        HarmonicDihedralMForceCompute(boost::shared_ptr<SystemDefinition> sysdef);
        
        //! Destructor
        virtual ~HarmonicDihedralMForceCompute();
        
        //! Set the parameters
        virtual void setParams(unsigned int type, Scalar K, Scalar phi0, unsigned int multiplicity);
        
        //! Returns a list of log quantities this compute calculates
        virtual std::vector< std::string > getProvidedLogQuantities();
        
        //! Calculates the requested log value and returns it
        virtual Scalar getLogValue(const std::string& quantity, unsigned int timestep);
        
    protected:
        Scalar *m_K;     //!< K parameter for multiple dihedral tyes
        Scalar *m_cos_phi0;  //!< Cosine of equilibrium angle
        Scalar *m_sin_phi0;  //!< Sine of equilibrium angle
        Scalar *m_multi; //!< multiplicity parameter for multiple dihedral types
        
        boost::shared_ptr<DihedralData> m_dihedral_data;    //!< Dihedral data to use in computing dihedrals
        
        //! Actually compute the forces
        virtual void computeForces(unsigned int timestep);
    };

//! Exports the DihedralForceCompute class to python
void export_HarmonicDihedralMForceCompute();

#endif

