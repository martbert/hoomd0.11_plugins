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

// $Id: DCDDumpWriter.h 3048 2010-05-12 13:56:35Z joaander $
// $URL: http://codeblue.umich.edu/hoomd-blue/svn/trunk/libhoomd/analyzers/DCDDumpWriter.h $
// Maintainer: joaander

#ifndef __STRESSPERATOMDUMPWRITER_H__
#define __STRESSPERATOMDUMPWRITER_H__

#include <hoomd/hoomd_config.h>

#include <string>
#include <boost/shared_ptr.hpp>
#include <fstream>
#include <hoomd/Analyzer.h>
#include <hoomd/ForceCompute.h>
#include <hoomd/ParticleGroup.h>

/*! \file DCDVELDumpWriter.h
    \brief Declares the DCDVELDumpWriter class
*/

// The DCD Dump writer is based on code from the molfile plugin to VMD
// and is use under the following license

// University of Illinois Open Source License
// Copyright 2003 Theoretical and Computational Biophysics Group,
// All rights reserved.

// Developed by:       Theoretical and Computational Biophysics Group
//             University of Illinois at Urbana-Champaign
//            http://www.ks.uiuc.edu/

//! Analyzer for writing out DCD dump files
/*! DCDVELDumpWriter writes out the current velocity of all particles to a DCD file
    every time analyze() is called. 
	
    On the first call to analyze() \a fname is created with a dcd header. If the file already exists,
    it is overwritten.

    Due to a limitation in the DCD format, the time step period between calls to
    analyze() \b must be specified up front. If analyze() detects that this period is
    not being maintained, it will print a warning but continue.
    \ingroup analyzers
*/
class StressPerAtomDumpWriter : public Analyzer
    {
    public:
        //! Construct the writer
        StressPerAtomDumpWriter(boost::shared_ptr<SystemDefinition> sysdef,
                      const std::string &fname_diag,
					  const std::string &fname_off,
                      unsigned int period,
                      boost::shared_ptr<ParticleGroup> group,
                      bool overwrite=false,
					  bool offdiag=false);
        
        //! Destructor
        ~StressPerAtomDumpWriter();
        
		//! Get needed pdata flags
        /*! StressPerAtomDumpWriter needs the diagonal components of the virial tensor, so the full_virial flag is set
        */
        virtual PDataFlags getRequestedPDataFlags()
            {
            PDataFlags flags;

            flags[pdata_flag::pressure_tensor] = 1;

            return flags;
            }
		
        //! Write out the data for the current timestep
        void analyze(unsigned int timestep);
        
    private:
        std::string m_fname_diag;                //!< The file name we are writing to
		std::string m_fname_off;                //!< The file name we are writing to
		boost::shared_ptr<ParticleGroup> m_group;
        unsigned int m_start_timestep;      //!< First time step written to the file
        unsigned int m_period;              //!< Time step period bewteen writes
        unsigned int m_num_frames_written;  //!< Count the number of frames written to the file
        unsigned int m_last_written_step;   //!< Last timestep written in a a file we are appending to
        bool m_appending;                   //!< True if this instance is appending to an existing DCD file
		bool m_offdiag;
        
        float *m_staging_buffer;            //!< Buffer for staging particle stress in tag order
        
        // helper functions
        
        //! Initalizes the file header
        void write_file_header(std::fstream &file);
        //! Writes the frame header
        void write_frame_header(std::fstream &file);
        //! Writes the particle velocities for a frame
        void write_frame_data_diag(std::fstream &file_diag);
		void write_frame_data_off(std::fstream &file_off);
        //! Updates the file header
        void write_updated_header(std::fstream &file, unsigned int timestep);
    };

//! Exports the StressPerAtomDumpWriter class to python
void export_StressPerAtomDumpWriter();

#endif

