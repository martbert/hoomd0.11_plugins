# -*- coding: iso-8859-1 -*-

import _custom_pair_potentials

# Next, since we are extending an pair potential, we need to bring in the base class and some other parts from 
# hoomd_script
from hoomd_script import pair
from hoomd_script import util
from hoomd_script import globals
from hoomd_script import tune
from hoomd_script import variant
import hoomd
import math

class lj2(pair.pair):
    ## Specify the Lennard-Jones %pair %force
    #
    # \param r_cut Default cutoff radius (in distance units)
    # \param name Name of the force instance 
    #
    # \b Example:
    # \code
    # lj = pair.lj(r_cut=3.0)
    # lj.pair_coeff.set('A', 'A', epsilon=1.0, sigma=1.0)
    # lj.pair_coeff.set('A', 'B', epsilon=2.0, sigma=1.0, alpha=0.5, r_cut=3.0, r_on=2.0);
    # lj.pair_coeff.set('B', 'B', epsilon=1.0, sigma=1.0, r_cut=2**(1.0/6.0), r_on=2.0);
    # \endcode
    #
    # \note %Pair coefficients for all type pairs in the simulation must be
    # set before it can be started with run()
	def __init__(self, r_cut, name=None):
		util.print_status_line();
		
		self.r_cut = r_cut;
		# tell the base class how we operate

		# initialize the base class
		pair.pair.__init__(self, r_cut, name);

		# update the neighbor list
		neighbor_list = pair._update_global_nlist(r_cut);
		neighbor_list.subscribe(lambda: self.log*self.get_max_rcut())

		# create the c++ mirror class
		if not globals.exec_conf.isCUDAEnabled():
			self.cpp_force = _custom_pair_potentials.PotentialPairLJ2(globals.system_definition, neighbor_list.cpp_nlist, self.name);
			self.cpp_class = _custom_pair_potentials.PotentialPairLJ2;
		else:
			neighbor_list.cpp_nlist.setStorageMode(hoomd.NeighborList.storageMode.full);
			self.cpp_force = _custom_pair_potentials.PotentialPairLJ2GPU(globals.system_definition, neighbor_list.cpp_nlist, self.name);
			self.cpp_class = _custom_pair_potentials.PotentialPairLJ2GPU;
			self.cpp_force.setBlockSize(tune._get_optimal_block_size('pair.lj'));
	
		globals.system.addCompute(self.cpp_force, self.force_name);

		# setup the coefficent options
		self.required_coeffs = ['epsilon', 'sigma','ron',];
		#self.pair_coeff.set_default_coeff('ron', r_cut);

	def process_coeff(self, coeff):
		epsilon = coeff['epsilon'];
		sigma = coeff['sigma'];
		ron = coeff['ron'];
		ron2 = ron*ron;
		A12 = (13*ron-16*self.r_cut)/math.pow(self.r_cut, 14.0)/math.pow(self.r_cut-ron,2.0);
		B12 = (15*self.r_cut-13*ron)/math.pow(self.r_cut, 14.0)/math.pow(self.r_cut-ron,3.0);
		A6 = (7*ron-10*self.r_cut)/math.pow(self.r_cut, 8.0)/math.pow(self.r_cut-ron,2.0);
		B6 = (9*self.r_cut-7*ron)/math.pow(self.r_cut, 8.0)/math.pow(self.r_cut-ron,3.0);
		
		lj1 = 4.0 * epsilon * math.pow(sigma, 12.0);
		lj2 = 4.0 * epsilon * math.pow(sigma, 6.0);
		
		shiftA = (12.0*lj1*A12-6.0*lj2*A6);
		shiftB = (12.0*lj1*B12-6.0*lj2*B6);
		
		return hoomd.make_scalar4(lj1, lj2, shiftA, shiftB);

class soft(pair.pair):
    ## Specify the Soft-sphere %pair %force
    #
    # \param r_cut Default cutoff radius
    # \param name Name of the force instance 
    #
    # \b Example:
    # \code
    # soft = pair.soft(r_cut=3.0)
    # soft.pair_coeff.set('A', 'A', es=1.0, n=1.0, roff=1.0)
    # \endcode
    #
    # \note %Pair coefficients for all type pairs in the simulation must be
    # set before it can be started with run()
	def __init__(self, r_cut, name=None):
		util.print_status_line();

		# tell the base class how we operate

		# initialize the base class
		pair.pair.__init__(self, r_cut, name);

		# update the neighbor list
		neighbor_list = pair._update_global_nlist(r_cut);
		neighbor_list.subscribe(lambda: self.log*self.get_max_rcut())

		# create the c++ mirror class
		if not globals.exec_conf.isCUDAEnabled():
			self.cpp_force = _custom_pair_potentials.PotentialPairSoft(globals.system_definition, neighbor_list.cpp_nlist, self.name);
			self.cpp_class = _custom_pair_potentials.PotentialPairSoft;
		else:
			neighbor_list.cpp_nlist.setStorageMode(hoomd.NeighborList.storageMode.full);
			self.cpp_force = _custom_pair_potentials.PotentialPairSoftGPU(globals.system_definition, neighbor_list.cpp_nlist, self.name);
			self.cpp_class = _custom_pair_potentials.PotentialPairSoftGPU;
			self.cpp_force.setBlockSize(128);

		globals.system.addCompute(self.cpp_force, self.force_name);

		# setup the coefficent options
		self.required_coeffs = ['es', 'n', 'roff'];
		self.pair_coeff.set_default_coeff('roff', 0.0);

	def process_coeff(self, coeff):
		es = coeff['es'];
		n = coeff['n'];
		roff = coeff['roff'];

		return hoomd.make_scalar3(es, n, roff);

class coulomb(pair.pair):
	def __init__(self, r_cut, name=None):
		util.print_status_line();
		
		self.r_cut = r_cut;
		# tell the base class how we operate

		# initialize the base class
		pair.pair.__init__(self, r_cut, name);

		# update the neighbor list
		neighbor_list = pair._update_global_nlist(r_cut);
		neighbor_list.subscribe(lambda: self.log*self.get_max_rcut())

		# create the c++ mirror class
		if not globals.exec_conf.isCUDAEnabled():
			self.cpp_force = _custom_pair_potentials.PotentialPairCoulomb(globals.system_definition, neighbor_list.cpp_nlist, self.name);
			self.cpp_class = _custom_pair_potentials.PotentialPairCoulomb;
		else:
			neighbor_list.cpp_nlist.setStorageMode(hoomd.NeighborList.storageMode.full);
			self.cpp_force = _custom_pair_potentials.PotentialPairCoulombGPU(globals.system_definition, neighbor_list.cpp_nlist, self.name);
			self.cpp_class = _custom_pair_potentials.PotentialPairCoulombGPU;
			self.cpp_force.setBlockSize(128);

		globals.system.addCompute(self.cpp_force, self.force_name);

		# setup the coefficent options
		self.required_coeffs = ['f', 'er', 'ron'];

	def process_coeff(self, coeff):
		er = coeff['er'];
		f = coeff['f'];
		ron = coeff['ron'];
		
		fer = f/er;
		shiftA = fer*(2*ron-5*self.r_cut)/math.pow(self.r_cut, 3.0)/math.pow(self.r_cut-ron,2.0);
		shiftB = fer*(4*self.r_cut-2*ron)/math.pow(self.r_cut, 3.0)/math.pow(self.r_cut-ron,3.0);

		return hoomd.make_scalar4(fer, ron, shiftA, shiftB);

class sgauss(pair.pair):
	def __init__(self, r_cut, d_max=None, name=None):
		util.print_status_line();

		# tell the base class how we operate

		# initialize the base class
		pair.pair.__init__(self, r_cut, name);

		# update the neighbor list
		if d_max is None :
			sysdef = globals.system_definition;
			d_max = max([x.diameter for x in data.particle_data(sysdef.getParticleData())])
			globals.msg.notice(2, "Notice: sgauss set d_max=" + str(d_max) + "\n");
		                
		neighbor_list = pair._update_global_nlist(r_cut);
		neighbor_list.subscribe(lambda: self.log*self.get_max_rcut());
		neighbor_list.cpp_nlist.setMaximumDiameter(d_max);

		# create the c++ mirror class
		if not globals.exec_conf.isCUDAEnabled():
			self.cpp_force = _custom_pair_potentials.PotentialPairSGauss(globals.system_definition, neighbor_list.cpp_nlist, self.name);
			self.cpp_class = _custom_pair_potentials.PotentialPairSGauss;
		else:
			neighbor_list.cpp_nlist.setStorageMode(hoomd.NeighborList.storageMode.full);
			self.cpp_force = _custom_pair_potentials.PotentialPairSGaussGPU(globals.system_definition, neighbor_list.cpp_nlist, self.name);
			self.cpp_class = _custom_pair_potentials.PotentialPairSGaussGPU;
			self.cpp_force.setBlockSize(128);

		globals.system.addCompute(self.cpp_force, self.force_name);

		# setup the coefficent options
		self.required_coeffs = ['epsilon', 'sigma'];

	def process_coeff(self, coeff):
		epsilon = coeff['epsilon'];
		sigma = coeff['sigma'];

		return hoomd.make_scalar2(epsilon,sigma);

class syukawa(pair.pair):
	def __init__(self, r_cut, d_max=None, name=None):
		util.print_status_line();

		# tell the base class how we operate

		# initialize the base class
		pair.pair.__init__(self, r_cut, name);

		# update the neighbor list
		if d_max is None :
			sysdef = globals.system_definition;
			d_max = max([x.diameter for x in data.particle_data(sysdef.getParticleData())])
			globals.msg.notice(2, "Notice: sgauss set d_max=" + str(d_max) + "\n");
		                
		neighbor_list = pair._update_global_nlist(r_cut);
		neighbor_list.subscribe(lambda: self.log*self.get_max_rcut());
		neighbor_list.cpp_nlist.setMaximumDiameter(d_max);

		# create the c++ mirror class
		if not globals.exec_conf.isCUDAEnabled():
			self.cpp_force = _custom_pair_potentials.PotentialPairSYukawa(globals.system_definition, neighbor_list.cpp_nlist, self.name);
			self.cpp_class = _custom_pair_potentials.PotentialPairSYukawa;
		else:
			neighbor_list.cpp_nlist.setStorageMode(hoomd.NeighborList.storageMode.full);
			self.cpp_force = _custom_pair_potentials.PotentialPairSYukawaGPU(globals.system_definition, neighbor_list.cpp_nlist, self.name);
			self.cpp_class = _custom_pair_potentials.PotentialPairSYukawaGPU;
			self.cpp_force.setBlockSize(128);

		globals.system.addCompute(self.cpp_force, self.force_name);

		# setup the coefficent options
		self.required_coeffs = ['epsilon', 'kappa'];

	def process_coeff(self, coeff):
		epsilon = coeff['epsilon'];
		kappa = coeff['kappa'];

		return hoomd.make_scalar2(epsilon,kappa);

class lowe(pair.pair):
	def __init__(self, r_cut, T, seed=1, name=None):
		util.print_status_line();

		# tell the base class how we operate

		# initialize the base class
		pair.pair.__init__(self, r_cut, name);

		# update the neighbor list
		neighbor_list = pair._update_global_nlist(r_cut);
		neighbor_list.subscribe(lambda: self.log*self.get_max_rcut())

		# create the c++ mirror class
		if globals.exec_conf.isCUDAEnabled():
			self.cpp_force = _custom_pair_potentials.PotentialPairLoweThermoLowe(globals.system_definition, neighbor_list.cpp_nlist, self.name);
			self.cpp_class = _custom_pair_potentials.PotentialPairLoweThermoLowe;
		else:
			neighbor_list.cpp_nlist.setStorageMode(hoomd.NeighborList.storageMode.full);
			self.cpp_force = _custom_pair_potentials.PotentialPairLoweThermoLoweGPU(globals.system_definition, neighbor_list.cpp_nlist, self.name);
			self.cpp_class = _custom_pair_potentials.PotentialPairLoweThermoLoweGPU;
			self.cpp_force.setBlockSize(tune._get_optimal_block_size('pair.dpd'));
		        
		globals.system.addCompute(self.cpp_force, self.force_name);

		# setup the coefficent options
		self.required_coeffs = ['A', 'gamma'];

		# set the seed for dpd thermostat
		self.cpp_force.setSeed(seed);

		# set the temperature
		# setup the variant inputs
		T = variant._setup_variant_input(T);
		self.cpp_force.setT(T.cpp_variant);  
	        

	## Changes parameters
	# \param T Temperature (if set) (in energy units)
	#
	# To change the parameters of an existing pair force, you must save it in a variable when it is
	# specified, like so:
	# \code
	# dpd = pair.dpd(r_cut = 1.0)
	# \endcode
	#
	# \b Examples:
	# \code
	# dpd.set_params(T=2.0)
	# \endcode
	def set_params(self, T=None):
		util.print_status_line();
		self.check_initialization();

		# change the parameters
		if T is not None:
			# setup the variant inputs
			T = variant._setup_variant_input(T);
			self.cpp_force.setT(T.cpp_variant);
	    
	def process_coeff(self, coeff):
		a = coeff['A'];
		gamma = coeff['gamma'];
		return hoomd.make_scalar2(a, gamma);

class lowe_conservative(pair.pair):
	def __init__(self, r_cut, name=None):
		util.print_status_line();

		# tell the base class how we operate

		# initialize the base class
		pair.pair.__init__(self, r_cut, name);

		# update the neighbor list
		neighbor_list = pair._update_global_nlist(r_cut);
		neighbor_list.subscribe(lambda: self.log*self.get_max_rcut())

		# create the c++ mirror class
		if not globals.exec_conf.isCUDAEnabled():
			self.cpp_force = _custom_pair_potentials.PotentialPairLowe(globals.system_definition, neighbor_list.cpp_nlist, self.name);
			self.cpp_class = _custom_pair_potentials.PotentialPairLowe;
		else:
			neighbor_list.cpp_nlist.setStorageMode(hoomd.NeighborList.storageMode.full);
			self.cpp_force = _custom_pair_potentials.PotentialPairLoweGPU(globals.system_definition, neighbor_list.cpp_nlist, self.name);
			self.cpp_class = _custom_pair_potentials.PotentialPairLoweGPU;
			self.cpp_force.setBlockSize(tune._get_optimal_block_size('pair.dpd_conservative'));
			self.cpp_force.setBlockSize(64);
		        
		globals.system.addCompute(self.cpp_force, self.force_name);

		# setup the coefficent options
		self.required_coeffs = ['A'];

	    
	def process_coeff(self, coeff):
		a = coeff['A'];
		gamma = 0;
		return hoomd.make_scalar2(a, gamma);     

	## Not implemented for dpd_conservative
	# 
	def set_params(self, coeff):
		raise RuntimeError('Not implemented for DPD Conservative');
		return;