# -*- coding: iso-8859-1 -*-

# this python interface actiavates the c++ WallBcUpdater from cppmodule

# First, we need to import the C++ module. It has the same name as this module (plugin_template) but with an underscore
# in front
import _custom_updaters

# Next, since we are extending an updater, we need to bring in the base class updater and some other parts from 
# hoomd_script
from hoomd_script.update import _updater
from hoomd_script import util
from hoomd_script import pair
from hoomd_script import globals
from hoomd_script import compute
from hoomd_script import sys
from hoomd_script import variant;
import hoomd

## Updates the temperature following berendsen's scheme
#
class temp_berendsen(_updater):
    ## Initialize the updater
    #
	# \param group Group to which the Berendsen thermostat will be applied.
    # \param T Temperature of thermostat. (in energy units)
    # \param tau Time constant of thermostat. (in time units)
	# \param dt Time step
	# \param period Period at which the updater is evaluated (by default 1)
    #
    def __init__(self, group, T, tau, dt, period=1):
        util.print_status_line();
    
        # initialize base class
        _updater.__init__(self);
        
		# setup the variant inputs
        T = variant._setup_variant_input(T);
		
        # create the compute thermo
        thermo = compute._get_unique_thermo(group = group);
		
		# Get deltaT
        deltaT = dt;

        # initialize the reflected c++ class
        if not globals.exec_conf.isCUDAEnabled():
            self.cpp_updater = _custom_updaters.TempBerendsenUpdater(globals.system_definition,group.cpp_group,thermo.cpp_compute,tau,T.cpp_variant,deltaT);
        else:
            self.cpp_updater = _custom_updaters.TempBerendsenUpdaterGPU(globals.system_definition,group.cpp_group,thermo.cpp_compute,tau,T.cpp_variant,deltaT);

        self.setupUpdater(period);

## Updates the pressure following berendsen's scheme
#
class press_berendsen(_updater):
    ## Initialize the updater
    #
	# \param group Group to which the Berendsen barostat will be applied.
    # \param P Target pressure. (in energy units)
    # \param tau Time constant of barostat. (in time units)
    # \param bk Bulk compressibility
	# \param mode Updater mode:
	#
	# - isotropic: all three dimensions are scaled the same way
	# - anisotropic: all three dimensions are scaled independently
	# - semi_isotropic: the normal x and in plane yz dimensions are scaled independently
	#
	# \param dt Integration timestep
	# \param period Period at which the updater is evaluated (by default 1)
    #
    def __init__(self, group, P, tau, bk, mode, dt, period=1):
        util.print_status_line();
    
        # initialize base class
        _updater.__init__(self);
        
		# setup the variant inputs
        P = variant._setup_variant_input(P);
		
        # create the compute thermo
        thermo_all = compute._get_unique_thermo(group=globals.group_all);
        
		# Get deltaT
        deltaT = dt;

        # initialize the reflected c++ class
        if not globals.exec_conf.isCUDAEnabled():
            if (mode == "isotropic"):
                cpp_mode = _custom_updaters.PressBerendsenUpdater.integrationMode.isotropic;
            elif (mode == "anisotropic"):
                cpp_mode = _custom_updaters.PressBerendsenUpdater.integrationMode.anisotropic;
            elif (mode == "semi-isotropic"):
                cpp_mode = _custom_updaters.PressBerendsenUpdater.integrationMode.semi_isotropic;
            elif (mode == "uniaxial"):
                cpp_mode = _custom_updaters.PressBerendsenUpdater.integrationMode.one_dimensional;
            else:
                print >> sys.stderr, "\n***Error! Invalid mode\n";
                raise RuntimeError("Error changing parameters in update.press_berendsen");
            self.cpp_updater = _custom_updaters.PressBerendsenUpdater(globals.system_definition, group.cpp_group, thermo_all.cpp_compute, tau, P.cpp_variant, bk, deltaT, cpp_mode);
        else:
            if (mode == "isotropic"):
                cpp_mode = _custom_updaters.PressBerendsenUpdaterGPU.integrationMode.isotropic;
            elif (mode == "anisotropic"):
                cpp_mode = _custom_updaters.PressBerendsenUpdaterGPU.integrationMode.anisotropic;
            elif (mode == "semi-isotropic"):
                cpp_mode = _custom_updaters.PressBerendsenUpdaterGPU.integrationMode.semi_isotropic;
            elif (mode == "uniaxial"):
                cpp_mode = _custom_updaters.PressBerendsenUpdaterGPU.integrationMode.one_dimensional;
            else:
                print >> sys.stderr, "\n***Error! Invalid mode\n";
                raise RuntimeError("Error changing parameters in update.press_berendsen");
            self.cpp_updater = _custom_updaters.PressBerendsenUpdaterGPU(globals.system_definition, group.cpp_group, thermo_all.cpp_compute, tau, P.cpp_variant, bk, deltaT, cpp_mode);
        self.setupUpdater(period);

class lowe_andersen(_updater):
    ## Initialize the updater
    #
    # \param Temperature
    # \param Gamma
    # \param Rcut
    # \param Seed
    #
    def __init__(self, T=1.0, gdt=0.1, rcut=1.0, seed=1, period=1):
        util.print_status_line();

        # initialize base class
        _updater.__init__(self);
        
        # setup the variant inputs
        #T = variant._setup_variant_input(T);

        # update the neighbor list
        neighbor_list = pair._update_global_nlist(rcut);
        neighbor_list.subscribe(lambda: rcut)

        # initialize the reflected c++ class
        self.cpp_updater = _custom_updaters.LoweAndersenUpdater(globals.system_definition,neighbor_list.cpp_nlist,T,gdt,rcut,int(seed));
        #if not globals.exec_conf.isCUDAEnabled():
        #    self.cpp_updater = _custom_updaters.LoweAndersenUpdater(globals.system_definition,group.cpp_group,thermo.cpp_compute,tau,T.cpp_variant,deltaT);
        #else:
        #    self.cpp_updater = _custom_updaters.LoweAndersenUpdaterGPU(globals.system_definition,group.cpp_group,thermo.cpp_compute,tau,T.cpp_variant,deltaT);

        self.setupUpdater(period);