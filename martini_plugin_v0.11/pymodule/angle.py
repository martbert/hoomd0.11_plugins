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

import _martini_plugin

from hoomd_script import force;
from hoomd_script import globals;
from hoomd_script import hoomd;
from hoomd_script import util;
from hoomd_script import tune;

import math;
import sys;

## Cosine %angle force
#
# The command bond.anglecos2 specifies a %cosine angle squared potential energy between every triplet of particles
# with an angle specified between them.
#
# \f[ V(\theta) = \frac{1}{2} k \left( \cos\theta - \cos\theta_0 \right)^2 \f]
# where \f$ \theta \f$ is the angle defined by the triplet
#
# Coefficients:
# - \f$ \theta_0 \f$ - rest %angle (in radians)
# - \f$ k \f$ - %force constant (in units of energy/radians^2)
#
# Coefficients \f$ k \f$ and \f$ \theta_0 \f$ must be set for each type of %angle in the simulation using
# set_coeff().
#
#
class cos2(force._force):
    ## Specify the %cosine2 %angle %force for MARTINI scheme
    #
    # \b Example:
    # \code
    # cos2 = martini_plugin.angle.cos2()
    # \endcode
    def __init__(self):
        util.print_status_line();
        # check that some angles are defined
        if globals.system_definition.getAngleData().getNumAngles() == 0:
            print >> sys.stderr, "\n***Error! No angles are defined.\n";
            raise RuntimeError("Error creating angle forces");
        
        # initialize the base class
        force._force.__init__(self);
        
        # append compute to stress array
        #globals.stress.append(self);
        
        # create the c++ mirror class
        if not globals.exec_conf.isCUDAEnabled():
            self.cpp_force = _martini_plugin.CosineAngleForceCompute(globals.system_definition);
        else:
            self.cpp_force = _martini_plugin.CosineAngleForceComputeGPU(globals.system_definition);
            self.cpp_force.setBlockSize(tune._get_optimal_block_size('angle.harmonic'));

        globals.system.addCompute(self.cpp_force, self.force_name);
        
        # variable for tracking which angle type coefficients have been set
        self.angle_types_set = [];
    
    ## Sets the %cosine %angle coefficients for a particular %angle type
    #
    # \param angle_type Angle type to set coefficients for
    # \param k Coefficient \f$ k \f$ (in units of energy/radians^2)
    # \param t0 Coefficient \f$ \theta_0 \f$ (in radians)
    #
    # Using set_coeff() requires that the specified %angle %force has been saved in a variable. i.e.
    # \code
    # cosine = martini_plugin.angle.cos2()
    # \endcode
    #
    # \b Examples:
    # \code
    # cosine.set_coeff('polymer', k=3.0, t0=0.7851)
    # cosine.set_coeff('backbone', k=100.0, t0=1.0)
    # \endcode
    #
    # The coefficients for every %angle type in the simulation must be set 
    # before the run() can be started.
    def set_coeff(self, angle_type, k, t0):
        util.print_status_line();

        cos0 = math.cos(t0)
		 
        # set the parameters for the appropriate type
        self.cpp_force.setParams(globals.system_definition.getAngleData().getTypeByName(angle_type), k, cos0);
        
        # track which particle types we have set
        if not angle_type in self.angle_types_set:
            self.angle_types_set.append(angle_type);
        
    def update_coeffs(self):
        # get a list of all angle types in the simulation
        ntypes = globals.system_definition.getAngleData().getNAngleTypes();
        type_list = [];
        for i in xrange(0,ntypes):
            type_list.append(globals.system_definition.getAngleData().getNameByType(i));
            
        # check to see if all particle types have been set
        for cur_type in type_list:
            if not cur_type in self.angle_types_set:
                print >> sys.stderr, "\n***Error:", cur_type, " coefficients missing in angle.harmonic\n";
                raise RuntimeError("Error updating coefficients");