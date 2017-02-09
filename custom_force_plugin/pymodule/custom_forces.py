# -*- coding: iso-8859-1 -*-

# this simple python interface just actiavates the c++ CuboidForceCompute from cppmodule
# Check out any of the python code in lib/hoomd-python-module/hoomd_script for moreexamples

# First, we need to import the C++ module. It has the same name as this module (plugin_template) but with an underscore
# in front
import _custom_force_plugin

# Next, since we are extending an updater, we need to bring in the base class updater and some other parts from 
# hoomd_script
from hoomd_script.force import _force
from hoomd_script import util
from hoomd_script import globals
from hoomd_script import init
from hoomd_script import sys
import hoomd

# Adds a constant force to particles in a plugin, gets updated all the time
class cuboid_force(_force):
	
	def __init__(self,fx=None,fy=None,fz=None,xmin=None, xmax=None, ymin=None, ymax=None, zmin=None, zmax=None):
		util.print_status_line();
	
		# initialize the base class
		_force.__init__(self);
	
		# handle the optional arguments
		box = globals.system_definition.getParticleData().getBox();
		if xmin is None:
			xmin = box.xlo - 0.5;
		if xmax is None:
			xmax = box.xhi + 0.5;
		if ymin is None:
			ymin = box.ylo - 0.5;
		if ymax is None:
			ymax = box.yhi + 0.5;
		if zmin is None:
			zmin = box.zlo - 0.5;
		if zmax is None:
			zmax = box.zhi + 0.5;
			
		f = hoomd.make_scalar3(fx,fy,fz);
		min = hoomd.make_scalar3(xmin,ymin,zmin);
		max = hoomd.make_scalar3(xmax,ymax,zmax);
				
		# initialize the reflected c++ class
		if not globals.exec_conf.isCUDAEnabled():
			self.cpp_force = _cuboid_force_plugin.CuboidForceCompute(globals.system_definition,f,min,max);
		else:
			self.cpp_force = _cuboid_force_plugin.CuboidForceComputeGPU(globals.system_definition,f,min,max);
		
		globals.system.addCompute(self.cpp_force, self.force_name);
		
	def set_params(self, fx,fy,fz,xmin, xmax, ymin, ymax, zmin, zmax):
		self.check_initialization();
		
		f = hoomd.make_scalar3(fx,fy,fz);
		min = hoomd.make_scalar3(xmin,ymin,zmin);
		max = hoomd.make_scalar3(xmax,ymax,zmax);
		
		self.cpp_force.setParams(f,min,max);

	def update_coeffs(self):
		pass

class harm_conf(_force):
	
	def __init__(self,group=None,k=None,roff=None,axis='x'):
		util.print_status_line();
	
		# initialize the base class
		_force.__init__(self);
		
		if axis == 'x':
			dflag = 1
		elif axis == 'y':
			dflag = 2
		elif axis == 'z':
			dflag = 3
		
		# initialize the reflected c++ class
		if not globals.exec_conf.isCUDAEnabled():
			self.cpp_force = _custom_force_plugin.ConfForceCompute(globals.system_definition, group.cpp_group,k,roff,dflag);
		else:
			self.cpp_force = _custom_force_plugin.ConfForceComputeGPU(globals.system_definition,group.cpp_group,k,roff,dflag);
		
		globals.system.addCompute(self.cpp_force, self.force_name);
		
	def set_params(self,k,roff,axis):
		self.check_initialization();
		if axis == 'x':
			dflag = 1
		elif axis == 'y':
			dflag = 2
		elif axis == 'z':
			dflag = 3
		self.cpp_force.setParams(k,roff,dflag);

	def update_coeffs(self):
		pass

class harm_slit(_force):
	
	def __init__(self,group=None,k=None,roff=None,normal='x'):
		util.print_status_line();
	
		# initialize the base class
		_force.__init__(self);
		
		if normal == 'x':
			dflag = 1
		elif normal == 'y':
			dflag = 2
		elif normal == 'z':
			dflag = 3
		
		# initialize the reflected c++ class
		if not globals.exec_conf.isCUDAEnabled():
			self.cpp_force = _custom_force_plugin.ConfForceComputeSlit(globals.system_definition, group.cpp_group,k,roff,dflag);
		else:
			self.cpp_force = _custom_force_plugin.ConfForceComputeSlitGPU(globals.system_definition,group.cpp_group,k,roff,dflag);
		
		globals.system.addCompute(self.cpp_force, self.force_name);
		
	def set_params(self,k,roff,normal):
		self.check_initialization();
		if normal == 'x':
			dflag = 1
		elif normal == 'y':
			dflag = 2
		elif normal == 'z':
			dflag = 3
		self.cpp_force.setParams(k,roff,dflag);

	def update_coeffs(self):
		pass

class runtumble(_force):
	
	def __init__(self):
		util.print_status_line();
		
		# check that some bonds are defined
		if globals.system_definition.getBondData().getNumBonds() == 0:
			globals.msg.error("No bonds are defined.\n");
			raise RuntimeError("Error creating bond forces");

		# initialize the base class
		_force.__init__(self);
		
		# initialize the reflected c++ class
		if not globals.exec_conf.isCUDAEnabled():
			self.cpp_force = _custom_force_plugin.RTForceCompute(globals.system_definition);
		else:
			self.cpp_force = _custom_force_plugin.RTForceComputeGPU(globals.system_definition);
		
		globals.system.addCompute(self.cpp_force, self.force_name);
		
	def set_params(self,type=None,RtoT=None,TtoR=None,Frun=None,Ftumble=None):
		self.check_initialization();
		self.cpp_force.setParams(globals.system_definition.getBondData().getTypeByName(type),RtoT,TtoR,Frun,Ftumble);

	def update_coeffs(self):
		pass