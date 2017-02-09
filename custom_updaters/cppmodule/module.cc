// Include the defined classes that are to be exported to python
#include "TempBerendsenUpdater.h"
#include "PressBerendsenUpdater.h"
#include "LoweAndersenUpdater.h"

// Include boost.python to do the exporting
#include <boost/python.hpp>
using namespace boost::python;

// specify the python module. Note that the name must expliclty match the PROJECT() name provided in CMakeLists
// (with an underscore in front)
BOOST_PYTHON_MODULE(_custom_updaters)
    {
    export_TempBerendsenUpdater();
    export_PressBerendsenUpdater();
    export_LoweAndersenUpdater();
    
    #ifdef ENABLE_CUDA
    export_TempBerendsenUpdaterGPU();
    export_PressBerendsenUpdaterGPU();
    #endif
    }

