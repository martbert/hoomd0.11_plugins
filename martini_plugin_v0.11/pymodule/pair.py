# -*- coding: iso-8859-1 -*-

import _martini_plugin

# Next, since we are extending a pair potential, we need to bring in the base class and some other parts from 
# hoomd_script
from hoomd_script import pair
from hoomd_script import util
from hoomd_script import globals
from hoomd_script import tune
import hoomd
import math

class lj_martini(pair.pair):
    ## Specify the Lennard-Jones %pair %force for the MARTINI scheme
    #
    # \param r_cut Cutoff radius that defaults to 1.2 (in distance units)
    # \param name Name of the force instance 
    #
    # \b Example:
    # \code
    # martini_params = martini_plugin.utils.martini_nonbonded_params(fname='martini_v2.0.itp')
    # sig,eps = martini_params.nb_params[('Q0','Qa')]
    # lj = martini_plugin.pair.lj_martini()
    # lj.set_params(mode='shift')
    # lj.pair_coeff.set('Q0', 'Qa', epsilon=eps, sigma=sig)
    # \endcode
    #
    # \note %Pair coefficients for all type pairs in the simulation must be
    # set before it can be started with run()
	def __init__(self, r_cut=1.2, name=None):
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
			self.cpp_force = _martini_plugin.PotentialPairLJM(globals.system_definition, neighbor_list.cpp_nlist, self.name);
			self.cpp_class = _martini_plugin.PotentialPairLJM;
		else:
			neighbor_list.cpp_nlist.setStorageMode(hoomd.NeighborList.storageMode.full);
			self.cpp_force = _martini_plugin.PotentialPairLJMGPU(globals.system_definition, neighbor_list.cpp_nlist, self.name);
			self.cpp_class = _martini_plugin.PotentialPairLJMGPU;
			self.cpp_force.setBlockSize(tune._get_optimal_block_size('pair.lj'));
	
		globals.system.addCompute(self.cpp_force, self.force_name);

		# setup the coefficent options
		self.required_coeffs = ['epsilon', 'sigma',];

	def process_coeff(self, coeff):
		epsilon = coeff['epsilon'];
		sigma = coeff['sigma'];
		ron = 0.9
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

class coulomb_martini(pair.pair):
    ## Specify the Coulomb %pair %force for the MARTINI scheme
    #
    # \param r_cut Default cutoff radius
    # \param name Name of the force instance 
    #
    # \b Example:
    # \code
    # f = 138.935485
    # er = 15.0
    # coul = pair.coulomb_martini()
    # coul.set_params(mode='shift')
    # coul.pair_coeff.set('Q0', 'Qa', f=f,er=er,ron=0.0)
    # \endcode
    #
    # \note %Pair coefficients for all type pairs in the simulation must be
    # set before it can be started with run()
	def __init__(self, r_cut=1.2, name=None):
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
			self.cpp_force = _martini_plugin.PotentialPairCoulombM(globals.system_definition, neighbor_list.cpp_nlist, self.name);
			self.cpp_class = _martini_plugin.PotentialPairCoulombM;
		else:
			neighbor_list.cpp_nlist.setStorageMode(hoomd.NeighborList.storageMode.full);
			self.cpp_force = _martini_plugin.PotentialPairCoulombMGPU(globals.system_definition, neighbor_list.cpp_nlist, self.name);
			self.cpp_class = _martini_plugin.PotentialPairCoulombMGPU;
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