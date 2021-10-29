//#include "planck_utils.h"
#include "matrix_utils_omp.h"

using namespace boost::python;

namespace p = boost::python;
namespace np = boost::python::numpy;

BOOST_PYTHON_MODULE(gmca)
{
	np::initialize();

    class_< MATRIX_OMP >("MATRIX_OMP",init<int, int, int, int, int, int, int, double, double, bool, int>())
		.def("GMCA_Basic", &MATRIX_OMP::GMCA_Basic)
		.def("AMCA_Basic", &MATRIX_OMP::AMCA_Basic)
		.def("GMCA_Batches_omp", &MATRIX_OMP::GMCA_Batches_omp)
		.def("GMCA_OneIter_omp", &MATRIX_OMP::GMCA_OneIteration_Batches_omp)
		.def("GMCA_OneIter_omp_TEMP", &MATRIX_OMP::GMCA_OneIteration_Batches_omp_TEMP)
		.def("bGMCA_OneIter_omp", &MATRIX_OMP::bGMCA_OneIteration_Batches_omp)
		.def("bGMCA_OneIter_RandBlock_omp", &MATRIX_OMP::bGMCA_OneIteration_RandBlock_Batches_omp)
		.def("AMCA_Batches_omp", &MATRIX_OMP::AMCA_Batches_omp)
		.def("PALM_Basic_OMPs", &MATRIX_OMP::PALM_Basic_OMPs)
		.def("GMCA_GetS_Batches_omp", &MATRIX_OMP::GMCA_GetS_Batches_omp)
		.def("PALM_Basic_Batches", &MATRIX_OMP::PALM_Basic_Batches)
		.def("PALM_Basic_Basic", &MATRIX_OMP::PALM_Basic_Basic)
		.def("bGMCA_Residual_Batches_omp", &MATRIX_OMP::bGMCA_Residual_Batches_omp)
		.def("bGMCA_UpdateS_Batches_omp", &MATRIX_OMP::bGMCA_UpdateS_Batches_omp)
		.def("bGMCA_UpdateAS_Batches_omp", &MATRIX_OMP::bGMCA_UpdateAS_Batches_omp)
		.def("PowerMethod_Batches", &MATRIX_OMP::PowerMethod_Batches)
		.def("CorrectPerm_Batches", &MATRIX_OMP::CorrectPerm_Batches_omp)
		.def("UpdateS_Batches", &MATRIX_OMP::UpdateS_Batches_omp)
		.def("UpdateA_Batches", &MATRIX_OMP::UpdateA_Batches_omp)
    .def("GMCA_Batches", &MATRIX_OMP::GMCA_Batches);
}
