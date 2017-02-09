# -*- coding: iso-8859-1 -*-

import _custom_pair_potentials

# Next, since we are extending an pair potential, we need to bring in the base class and some other parts from 
# hoomd_script
from hoomd_script import pair
from hoomd_script import bond
from hoomd_script import util
from hoomd_script import globals
from hoomd_script import tune
import hoomd
import math

class fenenolj(bond._bond):
    ## Specify the %fene %bond %force
    #
    # \param name Name of the bond instance
    #
    # \b Example:
    # \code
    # fene = bond.fene()
    # fene.bond_coeff.set('polymer', k=30.0, r0=1.5, sigma=1.0, epsilon= 2.0)
    # fene.bond_coeff.set('backbone', k=100.0, r0=1.0, sigma=1.0, epsilon= 2.0)
    # \endcode
    def __init__(self, name=None):
        util.print_status_line();
        
        # check that some bonds are defined
        if globals.system_definition.getBondData().getNumBonds() == 0:
            globals.msg.error("No bonds are defined.\n");
            raise RuntimeError("Error creating bond forces");
        
        # initialize the base class
        bond._bond.__init__(self, name);
        
        # create the c++ mirror class
        if not globals.exec_conf.isCUDAEnabled():
            self.cpp_force = _custom_pair_potentials.PotentialBondFENENOLJ(globals.system_definition,self.name);
        else:
            self.cpp_force = _custom_pair_potentials.PotentialBondFENENOLJGPU(globals.system_definition,self.name);
            self.cpp_force.setBlockSize(tune._get_optimal_block_size('bond.fene'));

        globals.system.addCompute(self.cpp_force, self.force_name);

        # setup the coefficient options
        self.required_coeffs = ['k','r0','roff'];

    ## Set parameters for %fene %bond %force (\b deprecated)
    # \param type bond type
    # \param coeffs named bond coefficients
    def set_coeff(self, type, **coeffs):
        globals.msg.warning("Syntax bond.fene.set_coeff deprecated.\n");
        self.bond_coeff.set(type, **coeffs)

    def process_coeff(self, coeff):
        k = coeff['k'];
        r0 = coeff['r0'];
        roff = coeff['roff'];
        return hoomd.make_scalar3(k, r0, roff);
