# -*- coding: iso-8859-1 -*-

# this simple python interface just actiavates the c++ CuboidForceCompute from cppmodule
# Check out any of the python code in lib/hoomd-python-module/hoomd_script for moreexamples

# First, we need to import the C++ module. It has the same name as this module (plugin_template) but with an underscore
# in front
import _freedrain_force_plugin

# Next, since we are extending an updater, we need to bring in the base class updater and some other parts from 
# hoomd_script
from hoomd_script.force import _force
from hoomd_script import util
from hoomd_script import globals
from hoomd_script import init
from hoomd_script import sys
from hoomd_script import pair
import hoomd

# Adds a constant force to particles in a plugin, gets updated all the time
class force(_force):
	
    def __init__(self,group1=None,group2=None,Ex=None,Ey=None,Ez=None,lD=None,bj=None,qt=1000.0,cut=None):
        util.print_status_line();
	
		# initialize the base class
        _force.__init__(self);
		
		# update the neighbor list
        #neighbor_list = pair._update_global_nlist(0.0)
        #neighbor_list.subscribe(lambda: self.log*0.0)
		# initialize the reflected c++ class
        if not globals.exec_conf.isCUDAEnabled():
            self.cpp_force = _freedrain_force_plugin.FreeDrainForceCompute(globals.system_definition,group1.cpp_group,group2.cpp_group,Ex,Ey,Ez,lD,bj,qt,cut);
        else:
            self.cpp_force = _freedrain_force_plugin.FreeDrainForceComputeGPU(globals.system_definition,group1.cpp_group,group2.cpp_group,Ex,Ey,Ez,lD,bj,qt,cut);
        #self.cpp_force = _freedrain_force_plugin.FreeDrainForceCompute(globals.system_definition,neighbor_list.cpp_nlist,group1.cpp_group,group2.cpp_group,Ex,Ey,Ez,lD,bj,cut);
		
        globals.system.addCompute(self.cpp_force, self.force_name);
		
    def set_params(self, Ex,Ey,Ez,lD,bj,qt,cut):
        self.check_initialization();
        self.cpp_force.setParams(Ex,Ey,Ez,lD,bj,qt,cut);

    def update_coeffs(self):
        pass
