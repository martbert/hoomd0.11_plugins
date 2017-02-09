// Include the defined classes that are to be exported to python
#include "CuboidForceCompute.h"
#include "ConfForceCompute.h"
#include "ConfForceComputeSlit.h"
#include "RTForceCompute.h"

// Include boost.python to do the exporting
#include <boost/python.hpp>
using namespace boost::python;

// specify the python module. Note that the name must expliclty match the PROJECT() name provided in CMakeLists
// (with an underscore in front)
BOOST_PYTHON_MODULE(_custom_force_plugin)
    {
    export_ConfForceCompute();
    export_ConfForceComputeSlit();
    export_CuboidForceCompute();
    export_RTForceCompute();
    
    #ifdef ENABLE_CUDA
    export_ConfForceComputeGPU();
    export_ConfForceComputeSlitGPU();
    export_CuboidForceComputeGPU();
    export_RTForceComputeGPU();
    #endif
    }

