#include <pybind11/pybind11.h>
#include <pybind11/stl.h> 
#include <pybind11/numpy.h>
#include "Math.hpp"            
#include "GonioSensor.hpp"
#include "Microfacet.hpp"
#include "modules/virtualgoniolib/src/NDFs/PiecewiseLinear.cpp"

namespace py = pybind11;


PYBIND11_MODULE(ndf_py, m) {
    m.doc() = "Python bindings for generating dataset of machine learning";
    
    //Vec2f
    py::class_<Vec2f>(m, "Vec2f")
        .def(py::init<float, float>(), "Constructor for a 2D vector")
        .def_readwrite("x", &Vec2f::x)
        .def_readwrite("y", &Vec2f::y);

    //Vec3f
    py::class_<Vec3f>(m, "Vec3f")
        .def(py::init<float, float, float>(), "Constructor for a 3D vector")
        .def_readwrite("x", &Vec3f::x)
        .def_readwrite("y", &Vec3f::y)
        .def_readwrite("z", &Vec3f::z);

    //NDF
    py::class_<PiecewiseLinearNDF>(m, "PiecewiseLinearNDF")
        //constructor needs Dx and Dy
        .def(py::init<const std::vector<float>&, const std::vector<float>&>(),
            "Constructor taking two lists of ControlPoints for Dx and Dy")
        //G1
        .def("G1", &PiecewiseLinearNDF::G1, "Calculates the Smith G1 shadowing-masking function",
            py::arg("w"), py::arg("wh"));
}