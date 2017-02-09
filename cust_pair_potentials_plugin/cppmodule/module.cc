// Include the defined classes that are to be exported to python
#include "AllPairExtPotentials.h"
#include "AllBondExtPotentials.h"

// Include boost.python to do the exporting
#include <boost/python.hpp>
using namespace boost::python;

// specify the python module. Note that the name must expliclty match the PROJECT() name provided in CMakeLists
// (with an underscore in front)
BOOST_PYTHON_MODULE(_custom_pair_potentials)
    {
    export_PotentialPair<PotentialPairLJ2>("PotentialPairLJ2");
	export_PotentialPair<PotentialPairSoft>("PotentialPairSoft");
	export_PotentialPair<PotentialPairCoulomb>("PotentialPairCoulomb");
    export_PotentialPair<PotentialPairSGauss>("PotentialPairSGauss");
    export_PotentialPair<PotentialPairSYukawa>("PotentialPairSYukawa");
    export_PotentialPair<PotentialPairLowe> ("PotentialPairLowe");
    export_PotentialPairLoweThermo<PotentialPairLoweThermoLowe, PotentialPairLowe>("PotentialPairLoweThermoLowe");   

    export_PotentialBond<PotentialBondFENENOLJ>("PotentialBondFENENOLJ");
    
    #ifdef ENABLE_CUDA
    export_PotentialPairGPU<PotentialPairLJ2GPU, PotentialPairLJ2>("PotentialPairLJ2GPU");
	export_PotentialPairGPU<PotentialPairSoftGPU, PotentialPairSoft>("PotentialPairSoftGPU");
	export_PotentialPairGPU<PotentialPairCoulombGPU, PotentialPairCoulomb>("PotentialPairCoulombGPU");
    export_PotentialPairGPU<PotentialPairSGaussGPU, PotentialPairSGauss>("PotentialPairSGaussGPU");
    export_PotentialPairGPU<PotentialPairSYukawaGPU, PotentialPairSYukawa>("PotentialPairSYukawaGPU");
    export_PotentialPair<PotentialPairLoweGPU> ("PotentialPairLoweGPU");
    export_PotentialPairLoweThermoGPU<PotentialPairLoweThermoLoweGPU, PotentialPairLoweThermoLowe >("PotentialPairLoweThermoLoweGPU");    
    
    export_PotentialBondGPU<PotentialBondFENENOLJGPU, PotentialBondFENENOLJ>("PotentialBondFENENOLJGPU");
    #endif
    }

