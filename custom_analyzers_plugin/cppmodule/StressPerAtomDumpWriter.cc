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

// $Id: StressPerAtomDumpWriter.cc 3048 2010-05-12 13:56:35Z $
// Maintainer: martbert

/*! \file StressPerAtomDumpWriter.cc
    \brief Defines the StressPerAtomDumpWriter class and related helper functions
*/

#ifdef WIN32
#pragma warning( push )
#pragma warning( disable : 4244 )
#endif

#include <stdexcept>

#include "StressPerAtomDumpWriter.h"
#include "time.h"

#include <boost/python.hpp>
#include <boost/filesystem/operations.hpp>
#include <boost/filesystem/convenience.hpp>
using namespace boost::filesystem;
using namespace boost::python;
using namespace std;

//! File position of NFILE in DCD header
#define NFILE_POS 8L
//! File position of NSTEP in DCD header
#define NSTEP_POS 20L

//! simple helper function to write an integer
/*! \param file file to write to
    \param val integer to write
*/
static void write_int(fstream &file, unsigned int val)
    {
    file.write((char *)&val, sizeof(unsigned int));
    }

//! simple helper function to read in integer
/*! \param file file to read from
    \returns integer read
*/
static unsigned int read_int(fstream &file)
    {
    unsigned int val;
    file.read((char *)&val, sizeof(unsigned int));
    return val;
    }

/*! Constructs the StressPerAtomDumpWriter. After construction, settings are set. No file operations are
    attempted until analyze() is called.

    \param sysdef SystemDefinition containing the ParticleData to dump
    \param fname File name to write DCD data to
    \param period Period which analyze() is going to be called at
    \param group Group of particles to include in the output
    \param overwrite If false, existing files will be appended to. If true, existing files will be overwritten.

    \note You must call analyze() with the same period specified in the constructor or
    the time step inforamtion in the file will be invalid. analyze() will print a warning
    if it is called out of sequence.
*/
StressPerAtomDumpWriter::StressPerAtomDumpWriter(boost::shared_ptr<SystemDefinition> sysdef,
                             const std::string &fname_diag,
							 const std::string &fname_off,
                             unsigned int period,
                             boost::shared_ptr<ParticleGroup> group,
                             bool overwrite,
							 bool offdiag)
    : Analyzer(sysdef), m_fname_diag(fname_diag), m_fname_off(fname_off), m_start_timestep(0),
    	m_period(period), m_group(group), m_num_frames_written(0), m_last_written_step(0), m_appending(false), m_offdiag(offdiag)
    {
    // handle appending to an existing file if it is requested
    if (!overwrite && (exists(fname_diag) || exists(fname_off)))
        {
        cout << "Notice: Appending to existing DCD file \"" << fname_diag << " and " << fname_off << "\"" << endl;
        
        // open the file and get data from the header
        fstream file_diag;
        file_diag.open(m_fname_diag.c_str(), ios::ate | ios::in | ios::out | ios::binary);
        file_diag.seekp(NFILE_POS);
        
        m_num_frames_written = read_int(file_diag);
        m_start_timestep = read_int(file_diag);
        unsigned int file_period = read_int(file_diag);
        
        // warn the user if we are now dumping at a different period
        if (file_period != m_period)
            cout << "***Warning! DCDDumpWriter is appending to a file that has period " << file_period << " that is not the same as the requested period of " << m_period << endl;
            
        m_last_written_step = read_int(file_diag);
        
        // check for errors
        if (!file_diag.good())
            {
            cerr << endl << "***Error! Error reading DCD header data" << endl << endl;
            throw runtime_error("Error appending to DCD file");
            }
		
		if (m_offdiag)
			{ 
			fstream file_off;
			file_off.open(m_fname_off.c_str(), ios::ate | ios::in | ios::out | ios::binary);
			file_off.seekp(NFILE_POS);
			
			// check for errors
			if (!file_off.good())
				{
				cerr << endl << "***Error! Error reading DCD header data" << endl << endl;
				throw runtime_error("Error appending to DCD file");
				}
			
			}
            
        m_appending = true;
        }
    
    m_staging_buffer = new float[m_pdata->getNGlobal()];
    }

StressPerAtomDumpWriter::~StressPerAtomDumpWriter()
    {
    delete[] m_staging_buffer;
    }

/*! \param timestep Current time step of the simulation
    The very first call to analyze() will result in the creation (or overwriting) of the
    file fname and the writing of the current timestep snapshot. After that, each call to analyze
    will add a new snapshot to the end of the file.
*/
void StressPerAtomDumpWriter::analyze(unsigned int timestep)
    {
    // the file object
    fstream file_diag;
	fstream file_off;
    
    // initialize the file on the first frame written
    if (m_num_frames_written == 0)
        {
        // open the file and truncate it
        file_diag.open(m_fname_diag.c_str(), ios::trunc | ios::out | ios::binary);
        
        // write the file header
        m_start_timestep = timestep;
        write_file_header(file_diag);
		
		if (m_offdiag)
			{
			file_off.open(m_fname_off.c_str(), ios::trunc | ios::out | ios::binary);
			write_file_header(file_off);
			}
        }
    else
        {
        if (m_appending && timestep <= m_last_written_step)
            {
            cout << "***Warning! DCDDumpWriter is not writing output at timestep " << timestep << " because the file reports that it already has data up to step " << m_last_written_step << endl;
            return;
            }
            
        // open the file and move the file pointer to the end
        file_diag.open(m_fname_diag.c_str(), ios::ate | ios::in | ios::out | ios::binary);
		
		if (m_offdiag)
			file_off.open(m_fname_off.c_str(), ios::ate | ios::in | ios::out | ios::binary);
        
        // verify the period on subsequent frames
        if ( (timestep - m_start_timestep) % m_period != 0)
            cout << "***Warning! DCDDumpWriter is writing time step " << timestep << " which is not specified in the period of the DCD file: " << m_start_timestep << " + i * " << m_period << endl;
        }
         
    // write the data for the current time step
    write_frame_header(file_diag);
    write_frame_data_diag(file_diag);
    
    // update the header with the number of frames written
    m_num_frames_written++;
    write_updated_header(file_diag, timestep);
    file_diag.close();
	
	if (m_offdiag)
		{
		write_frame_header(file_off);
		write_frame_data_off(file_off);
		write_updated_header(file_off, timestep);
		file_off.close();
		}
    }

/*! \param file File to write to
    Writes the initial DCD header to the beginning of the file. This must be
    called on a newly created (or truncated file).
*/
void StressPerAtomDumpWriter::write_file_header(std::fstream &file)
    {
    // the first 4 bytes in the file must be 84
    write_int(file, 84);
    
    // the next 4 bytes in the file must be "CORD"
    char cord_data[] = "CORD";
    file.write(cord_data, 4);
    write_int(file, 0);      // Number of frames in file, none written yet
    write_int(file, m_start_timestep); // Starting timestep
    write_int(file, m_period);  // Timesteps between frames written to the file
    write_int(file, 0);      // Number of timesteps in simulation
    write_int(file, 0);
    write_int(file, 0);
    write_int(file, 0);
    write_int(file, 0);
    write_int(file, 0);
    write_int(file, 0);         // timestep (unused)
    write_int(file, 1);         // include unit cell
    write_int(file, 0);
    write_int(file, 0);
    write_int(file, 0);
    write_int(file, 0);
    write_int(file, 0);
    write_int(file, 0);
    write_int(file, 0);
    write_int(file, 0);
    write_int(file, 24);    // Pretend to be CHARMM version 24
    write_int(file, 84);
    write_int(file, 164);
    write_int(file, 2);
    
    char title_string[81];
    memset(title_string, 0, 81);
    char remarks[] = "Created by HOOMD";
    strncpy(title_string, remarks, 80);
    title_string[79] = '\0';
    file.write(title_string, 80);
    
    char time_str[81];
    memset(time_str, 0, 81);
    time_t cur_time = time(NULL);
    tm *tmbuf=localtime(&cur_time);
    strftime(time_str, 80, "REMARKS Created  %d %B, %Y at %H:%M", tmbuf);
    file.write(time_str, 80);
    
    write_int(file, 164);
    write_int(file, 4);
    unsigned int nparticles = m_group->getNumMembers();
    write_int(file, nparticles);
    write_int(file, 4);
    
    // check for errors
    if (!file.good())
        {
        cerr << endl << "***Error! Error writing DCD header" << endl << endl;
        throw runtime_error("Error writing DCD file");
        }
    }

/*! \param file File to write to
    Writes the header that precedes each snapshot in the file. This header
    includes information on the box size of the simulation.
*/
void StressPerAtomDumpWriter::write_frame_header(std::fstream &file)
    {
    double unitcell[6];
    BoxDim box = m_pdata->getBox();
    // set box dimensions
    Scalar3 L = box.getL();
    unitcell[0] = L.x;
    unitcell[2] = L.y;
    unitcell[5] = L.z;
    // box angles are 90 degrees
    unitcell[1] = 0.0f;
    unitcell[3] = 0.0f;
    unitcell[4] = 0.0f;
    
    write_int(file, 48);
    file.write((char *)unitcell, 48);
    write_int(file, 48);

    // check for errors
    if (!file.good())
        {
        cerr << endl << "***Error! Error writing DCD frame header" << endl << endl;
        throw runtime_error("Error writing DCD file");
        }
    }

/*! \param file File to write to
*/
void StressPerAtomDumpWriter::write_frame_data_diag(std::fstream &file)
    {
    // we need to unsort the positions and write in tag order
    assert(m_staging_buffer);

    unsigned int nparticles = m_group->getNumMembers();
    
    // access the net virial
    const GPUArray< Scalar >& net_virial = m_pdata->getNetVirial();
    ArrayHandle<Scalar> h_net_virial(net_virial, access_location::host, access_mode::read);
    unsigned int virial_pitch = net_virial.getPitch();

    // access velocities
    ArrayHandle<Scalar4> h_vel(m_pdata->getVelocities(), access_location::host, access_mode::readwrite);

   // prepare sxx for writing, looping in tag order
    for (unsigned int group_idx = 0; group_idx < nparticles; group_idx++)
        {
        unsigned int j = m_group->getMemberTag(group_idx);
        m_staging_buffer[group_idx] = (float)(h_vel.data[j].w*h_vel.data[j].x*h_vel.data[j].x + h_net_virial.data[j+0*virial_pitch]);
        }
    // write x coords
    write_int(file, nparticles * sizeof(float));
    file.write((char *)m_staging_buffer, nparticles * sizeof(float));
    write_int(file, nparticles * sizeof(float));
    
    // prepare syy for writing
    for (unsigned int group_idx = 0; group_idx < nparticles; group_idx++)
        {
        unsigned int j = m_group->getMemberTag(group_idx);
        m_staging_buffer[group_idx] = (float)(h_vel.data[j].w*h_vel.data[j].y*h_vel.data[j].y +h_net_virial.data[j+3*virial_pitch]);
        }
    // write y coords
    write_int(file, nparticles * sizeof(float));
    file.write((char *)m_staging_buffer, nparticles * sizeof(float));
    write_int(file, nparticles * sizeof(float));
    
    // prepare szz for writing
    for (unsigned int group_idx = 0; group_idx < nparticles; group_idx++)
        {
        unsigned int j = m_group->getMemberTag(group_idx);
        m_staging_buffer[group_idx] = (float)(h_vel.data[j].w*h_vel.data[j].z*h_vel.data[j].z +h_net_virial.data[j+5*virial_pitch]);
        }
    // write z coords
    write_int(file, nparticles * sizeof(float));
    file.write((char *)m_staging_buffer, nparticles * sizeof(float));
    write_int(file, nparticles * sizeof(float));
    
    // check for errors
    if (!file.good())
        {
        cerr << endl << "***Error! Error writing DCD frame data" << endl << endl;
        throw runtime_error("Error writing DCD file");
        }
    }
	
void StressPerAtomDumpWriter::write_frame_data_off(std::fstream &file)
    {
    // we need to unsort the positions and write in tag order
    assert(m_staging_buffer);
	
    unsigned int nparticles = m_group->getNumMembers();

	// access the net virial
    const GPUArray< Scalar >& net_virial = m_pdata->getNetVirial();
    ArrayHandle<Scalar> h_net_virial(net_virial, access_location::host, access_mode::read);
    unsigned int virial_pitch = net_virial.getPitch();

    // access velocities
    ArrayHandle<Scalar4> h_vel(m_pdata->getVelocities(), access_location::host, access_mode::readwrite);

   // prepare sxy for writing, looping in tag order
    for (unsigned int group_idx = 0; group_idx < nparticles; group_idx++)
        {
        unsigned int j = m_group->getMemberTag(group_idx);
        m_staging_buffer[group_idx] = (float)(h_vel.data[j].w*h_vel.data[j].x*h_vel.data[j].y + h_net_virial.data[j+1*virial_pitch]);
        }
    // write x coords
    write_int(file, nparticles * sizeof(float));
    file.write((char *)m_staging_buffer, nparticles * sizeof(float));
    write_int(file, nparticles * sizeof(float));
    
    // prepare sxz for writing
    for (unsigned int group_idx = 0; group_idx < nparticles; group_idx++)
        {
        unsigned int j = m_group->getMemberTag(group_idx);
        m_staging_buffer[group_idx] = (float)(h_vel.data[j].w*h_vel.data[j].x*h_vel.data[j].z + h_net_virial.data[j+2*virial_pitch]);
        }
    // write y coords
    write_int(file, nparticles * sizeof(float));
    file.write((char *)m_staging_buffer, nparticles * sizeof(float));
    write_int(file, nparticles * sizeof(float));
    
    // prepare syz for writing
    for (unsigned int group_idx = 0; group_idx < nparticles; group_idx++)
        {
        unsigned int j = m_group->getMemberTag(group_idx);
        m_staging_buffer[group_idx] = (float)(h_vel.data[j].w*h_vel.data[j].y*h_vel.data[j].z + h_net_virial.data[j+4*virial_pitch]);
        }
    // write z coords
    write_int(file, nparticles * sizeof(float));
    file.write((char *)m_staging_buffer, nparticles * sizeof(float));
    write_int(file, nparticles * sizeof(float));
    
    // check for errors
    if (!file.good())
        {
        cerr << endl << "***Error! Error writing DCD frame data" << endl << endl;
        throw runtime_error("Error writing DCD file");
        }
    }

/*! \param file File to write to
    \param timestep Current time step of the simulation

    Updates the pointers in the main file header to reflect the current number of frames
    written and the last time step written.
*/
void StressPerAtomDumpWriter::write_updated_header(std::fstream &file, unsigned int timestep)
    {
    file.seekp(NFILE_POS);
    write_int(file, m_num_frames_written);
    
    file.seekp(NSTEP_POS);
    write_int(file, timestep);
    }

void export_StressPerAtomDumpWriter()
    {
    class_<StressPerAtomDumpWriter, boost::shared_ptr<StressPerAtomDumpWriter>, bases<Analyzer>, boost::noncopyable>
    ("StressPerAtomDumpWriter", init< boost::shared_ptr<SystemDefinition>, std::string, std::string, unsigned int, boost::shared_ptr<ParticleGroup>, bool, bool>());
    }

#ifdef WIN32
#pragma warning( pop )
#endif

