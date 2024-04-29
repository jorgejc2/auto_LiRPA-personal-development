#include <iostream>

#include <boost/python.hpp>
#include <boost/python/numpy.hpp>
#include <Eigen>
#include <vector>

#include <sdlp.hpp>

namespace p = boost::python;
namespace np = boost::python::numpy;

template<int declared_d>
np::ndarray wrap_solve_lp(np::ndarray const &c, np::ndarray const &A, np::ndarray const &b) {
    // Make sure the ranks are fine
	if (c.get_nd() != 1 || A.get_nd() != 2 || b.get_nd() != 1) {
        PyErr_SetString(PyExc_TypeError, "Incorrect input ranks");
        p::throw_error_already_set();
    }

    // Make sure the dimesions are fine
    int d = c.shape(0);
    int n = b.shape(0);
    if (A.shape(0) != n || A.shape(1) != d) {
        PyErr_SetString(PyExc_TypeError, "Incorrect input dimensions");
        p::throw_error_already_set();
    }
    if (d != declared_d) {
        PyErr_SetString(PyExc_TypeError, "Problem dim is not consistent with the function invoked");
        p::throw_error_already_set();
    }

    // Make sure we get doubles
    if (c.get_dtype() != np::dtype::get_builtin<double>()) {
        PyErr_SetString(PyExc_TypeError, "Incorrect c data type");
        p::throw_error_already_set();
    }
    if (A.get_dtype() != np::dtype::get_builtin<double>()) {
        PyErr_SetString(PyExc_TypeError, "Incorrect A data type");
        p::throw_error_already_set();
    }
    if (b.get_dtype() != np::dtype::get_builtin<double>()) {
        PyErr_SetString(PyExc_TypeError, "Incorrect b data type");
        p::throw_error_already_set();
    }

	double *p = reinterpret_cast<double *>(c.get_data());
    Eigen::Map< const Eigen::MatrixXd > c_map(p, declared_d, 1);

    p = reinterpret_cast<double *>(A.get_data());
	Eigen::Map< const Eigen::Matrix<double, -1, -1, Eigen::RowMajor> > A_map(p, n, d);

    p = reinterpret_cast<double *>(b.get_data());
	Eigen::Map< const Eigen::MatrixXd > b_map(p, n, 1);

    Eigen::Matrix<double, declared_d, 1> x_matrix;
    sdlp::linprog<declared_d>(c_map, A_map, b_map, x_matrix);
        
    // Turning the output into a numpy array
    np::dtype dt = np::dtype::get_builtin<double>();
    p::tuple shape = p::make_tuple(d); // It has shape (d,)
    p::tuple stride = p::make_tuple(sizeof(double)); // 1D array, so its just size of double
    np::ndarray result = np::from_data(x_matrix.data(), dt, shape, stride, p::object());
    return result.copy();
}


BOOST_PYTHON_MODULE(py_sdlp)
{
    Py_Initialize();
    np::initialize();

    p::def("solve_lp_2d", wrap_solve_lp<2>);
    p::def("solve_lp_3d", wrap_solve_lp<3>);
}
