// Include the defined classes that are to be exported to python
#include "AllPairExtPotentials.h"
#include "CosineAngleForceCompute.h"
#include "CosineAngleForceComputeGPU.h"
#include "HarmonicDihedralMForceCompute.h"
#include "HarmonicDihedralMForceComputeGPU.h"

// Include boost.python to do the exporting
#include <boost/python.hpp>
using namespace boost::python;

// specify the python module. Note that the name must expliclty match the PROJECT() name provided in CMakeLists
// (with an underscore in front)
BOOST_PYTHON_MODULE(_martini_plugin)
    {
    export_PotentialPair<PotentialPairLJM>("PotentialPairLJM");
	export_PotentialPair<PotentialPairCoulombM>("PotentialPairCoulombM");
	export_CosineAngleForceCompute();
    export_HarmonicDihedralMForceCompute();
    
    #ifdef ENABLE_CUDA
    export_PotentialPairGPU<PotentialPairLJMGPU, PotentialPairLJM>("PotentialPairLJMGPU");
	export_PotentialPairGPU<PotentialPairCoulombMGPU, PotentialPairCoulombM>("PotentialPairCoulombMGPU");
	export_CosineAngleForceComputeGPU();
    export_HarmonicDihedralMForceComputeGPU();
    #endif
    }

