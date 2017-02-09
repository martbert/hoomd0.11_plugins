// Include the defined classes that are to be exported to python
#include "DCDVELDumpWriter.h"
#include "DCDQDumpWriter.h"
#include "StressPerAtomDumpWriter.h"
//#include "StressProfileAnalyzer.h"

// Include boost.python to do the exporting
#include <boost/python.hpp>
using namespace boost::python;

// specify the python module. Note that the name must expliclty match the PROJECT() name provided in CMakeLists
// (with an underscore in front)
BOOST_PYTHON_MODULE(_custom_analyzers_plugin)
    {
    export_DCDVELDumpWriter();
    export_DCDQDumpWriter();
	export_StressPerAtomDumpWriter();
	//export_StressProfileAnalyzer();
    }

