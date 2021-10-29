#include "ManiMR.h"
#include "starlet2d.h"

using namespace boost::python;
namespace p = boost::python;
namespace np = boost::python::numpy;

BOOST_PYTHON_MODULE(ManiMR)
{
	np::initialize();

	class_< ManiMR >("ManiMR", init<int, int, int,int,int,int >())
	  .def("backward2d_omp", &ManiMR::backward2d_omp)
		.def("backward1d", &ManiMR::backward1d)
		.def("forward2d_ref_omp", &ManiMR::forward2d_ref_omp)
		.def("forward2d_omp", &ManiMR::forward2d_omp)
		.def("forward1d_ref", &ManiMR::forward1d_ref)
		.def("forward1d", &ManiMR::forward1d)
		.def("logSn", &ManiMR::LogSn)
		.def("expSn", &ManiMR::ExpSn);

	class_< Starlet2D >("Starlet2D", init<int, int, int,int,int >())
	  .def("forward", &Starlet2D::transform_numpy);
//	  .def("forward_omp", &Starlet2D::transform_omp_numpy)
//	  .def("forward1d_omp", &Starlet2D::transform1d_omp_numpy)
//	  .def("backward", &Starlet2D::reconstruct_numpy)
//	  .def("adjoint1d",&Starlet2D:: adjoint1d_omp_numpy)
//	  .def("backward_omp", &Starlet2D::reconstruct_omp_numpy);
}
