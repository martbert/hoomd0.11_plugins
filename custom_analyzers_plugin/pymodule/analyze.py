# -*- coding: iso-8859-1 -*-
#Highly Optimized Object-oriented Many-particle Dynamics -- Blue Edition
#(HOOMD-blue) Open Source Software License Copyright 2008, 2009 Ames Laboratory
#Iowa State University and The Regents of the University of Michigan All rights
#reserved.

#HOOMD-blue may contain modifications ("Contributions") provided, and to which
#copyright is held, by various Contributors who have granted The Regents of the
#University of Michigan the right to modify and/or distribute such Contributions.

#Redistribution and use of HOOMD-blue, in source and binary forms, with or
#without modification, are permitted, provided that the following conditions are
#met:

#* Redistributions of source code must retain the above copyright notice, this
#list of conditions, and the following disclaimer.

#* Redistributions in binary form must reproduce the above copyright notice, this
#list of conditions, and the following disclaimer in the documentation and/or
#other materials provided with the distribution.

#* Neither the name of the copyright holder nor the names of HOOMD-blue's
#contributors may be used to endorse or promote products derived from this
#software without specific prior written permission.

#Disclaimer

#THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER AND CONTRIBUTORS ``AS IS''
#AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND/OR
#ANY WARRANTIES THAT THIS SOFTWARE IS FREE OF INFRINGEMENT ARE DISCLAIMED.

#IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
#INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
#BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
#DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
#LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
#OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
#ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# $Id: DCDVELDump.py martbert $
# Maintainer: martbert / All Developers are free to add commands for new features

import _custom_analyzers_plugin

import hoomd;
from hoomd_script import globals;
from hoomd_script import analyze;
from hoomd_script import sys;
from hoomd_script import util;
from hoomd_script import group as hs_group;

class dcdvel(analyze._analyzer):
    ## Initialize the dcd writer
    #
    # \param filename File name to write
    # \param period Number of time steps between file dumps
    # \param group Particle group to output to the dcd file. If left as None, all particles will be written
    # \param overwrite When False, (the default) an existing DCD file will be appended to. When True, an existing DCD
    #        file \a filename will be overwritten.
    # \param wrap When True, (the default) wrapped particle coordinates are written. When False, particles will be
    #        unwrapped into their current box image before writing to the dcd file.
    # 
    # \b Examples:
    # \code
    # dump.dcd(filename="trajectory.dcd", period=1000)
    # dcd = dump.dcd(filename"data/dump.dcd", period=1000)
    # \endcode
    #
    # \warning 
    # When you use dump.dcd to append to an existing dcd file
    # - The period must be the same or the time data in the file will not be consistent.
    # - dump.dcd will not write out data at time steps that already are present in the dcd file to maintain a
    #   consistent timeline
    #
    # \a period can be a function: see \ref variable_period_docs for details
    def __init__(self, filename, period, group=None, overwrite=False):
        util.print_status_line();
        
        # initialize base class
        analyze._analyzer.__init__(self);
        
        # create the c++ mirror class
        reported_period = period;
        if type(period) != type(1):
            reported_period = 1000;
            
        if group is None:
            util._disable_status_lines = True;
            group = hs_group.all();
            util._disable_status_lines = False;
            
        self.cpp_analyzer = _custom_analyzers_plugin.DCDVELDumpWriter(globals.system_definition, filename, int(reported_period), group.cpp_group, overwrite);
        self.setupAnalyzer(period);
    
    def enable(self):
        util.print_status_line();
        
        if self.enabled == False:
            print >> sys.stderr, "\n***Error! you cannot re-enable DCD output after it has been disabled\n";
            raise RuntimeError('Error enabling updater');
    
    def set_period(self, period):
        util.print_status_line();
        
        print >> sys.stderr, "\n***Error! you cannot change the period of a dcd dump writer\n";
        raise RuntimeError('Error changing updater period');

class dcdq(analyze._analyzer):
    ## Initialize the dcd writer
    #
    # \param filename File name to write
    # \param period Number of time steps between file dumps
    # \param group Particle group to output to the dcd file. If left as None, all particles will be written
    # \param overwrite When False, (the default) an existing DCD file will be appended to. When True, an existing DCD
    #        file \a filename will be overwritten.
    # \param wrap When True, (the default) wrapped particle coordinates are written. When False, particles will be
    #        unwrapped into their current box image before writing to the dcd file.
    # 
    # \b Examples:
    # \code
    # dump.dcd(filename="trajectory.dcd", period=1000)
    # dcd = dump.dcd(filename"data/dump.dcd", period=1000)
    # \endcode
    #
    # \warning 
    # When you use dump.dcd to append to an existing dcd file
    # - The period must be the same or the time data in the file will not be consistent.
    # - dump.dcd will not write out data at time steps that already are present in the dcd file to maintain a
    #   consistent timeline
    #
    # \a period can be a function: see \ref variable_period_docs for details
    def __init__(self, filename, period, group=None, overwrite=False):
        util.print_status_line();
        
        # initialize base class
        analyze._analyzer.__init__(self);
        
        # create the c++ mirror class
        reported_period = period;
        if type(period) != type(1):
            reported_period = 1000;
            
        if group is None:
            util._disable_status_lines = True;
            group = hs_group.all();
            util._disable_status_lines = False;
            
        self.cpp_analyzer = _custom_analyzers_plugin.DCDQDumpWriter(globals.system_definition, filename, int(reported_period), group.cpp_group, overwrite);
        self.setupAnalyzer(period);
    
    def enable(self):
        util.print_status_line();
        
        if self.enabled == False:
            print >> sys.stderr, "\n***Error! you cannot re-enable DCD output after it has been disabled\n";
            raise RuntimeError('Error enabling updater');
    
    def set_period(self, period):
        util.print_status_line();
        
        print >> sys.stderr, "\n***Error! you cannot change the period of a dcd dump writer\n";
        raise RuntimeError('Error changing updater period');

class stressPerAtom(analyze._analyzer):
    ## Initialize the dcd writer
    #
    # \param filename Base file name to write
    # \param period Number of time steps between file dumps
    # \param group Particle group to output to the dcd file. If left as None, all particles will be written
    # \param overwrite When False, (the default) an existing DCD file will be appended to. When True, an existing DCD
    #        file \a filename will be overwritten.
    # \param wrap When True, (the default) wrapped particle coordinates are written. When False, particles will be
    #        unwrapped into their current box image before writing to the dcd file.
    # 
    # \b Examples:
    # \code
    # dump.dcd(filename="trajectory.dcd", period=1000)
    # dcd = dump.dcd(filename"data/dump.dcd", period=1000)
    # \endcode
    #
    # \warning 
    # When you use dump.dcd to append to an existing dcd file
    # - The period must be the same or the time data in the file will not be consistent.
    # - dump.dcd will not write out data at time steps that already are present in the dcd file to maintain a
    #   consistent timeline
    #
    # \a period can be a function: see \ref variable_period_docs for details
    def __init__(self, filename,period, group=None, overwrite=False, offdiag=False):
        util.print_status_line();
        
        # initialize base class
        analyze._analyzer.__init__(self);

        # Find the dcd extension and construct new file names
        end = filename.find('.dcd')
        file_diag = filename[:end]+'.diag.dcd'
        file_off = filename[:end]+'.off.dcd'
        
        # create the c++ mirror class
        reported_period = period;
        if type(period) != type(1):
            reported_period = 1000;
            
        if group is None:
            util._disable_status_lines = True;
            group = hs_group.all();
            util._disable_status_lines = False;
            
        self.cpp_analyzer = _custom_analyzers_plugin.StressPerAtomDumpWriter(globals.system_definition, file_diag, file_off, int(reported_period), group.cpp_group, overwrite, offdiag);
        self.setupAnalyzer(period);
        
    def enable(self):
        util.print_status_line();
        
        if self.enabled == False:
            print >> sys.stderr, "\n***Error! you cannot re-enable DCD output after it has been disabled\n";
            raise RuntimeError('Error enabling updater');
    
    def set_period(self, period):
        util.print_status_line();
        
        print >> sys.stderr, "\n***Error! you cannot change the period of a dcd dump writer\n";
        raise RuntimeError('Error changing updater period');
