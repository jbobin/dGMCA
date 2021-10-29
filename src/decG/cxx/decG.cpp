#include "decG_utils.h"

using namespace boost::python;

namespace p = boost::python;
namespace np = boost::python::numpy;

BOOST_PYTHON_MODULE(decG)
{
	np::initialize();

	class_< PLK >("PLK",init<>())
	.def("UpdateA",&PLK::UpdateA_numpy)
	.def("UpdateA_Cpx",&PLK::UpdateA_Cpx_numpy)
	.def("UpdateA_batches",&PLK::UpdateA_batches_numpy)
	.def("GradientA_batches",&PLK::GradientA_numpy)
	.def("UpdateA_Cpx_batches",&PLK::UpdateA_Cpx_batches_numpy)
	.def("GradientS",&PLK::GradientS_numpy)
	.def("UpdateS_Cpx",&PLK::UpdateS_Cpx_numpy)
	.def("CG",&PLK::Left_CG_numpy)
  .def("applyH_Pinv",&PLK::applyHt_PInv_numpy);

}
