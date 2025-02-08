# PX4-ECL Python Bindings
## Links to the original library
- [PX4-ECL GitHub Repo](https://github.com/PX4/PX4-ECL)
- [EKF Documentation and Tuning Guide](https://docs.px4.io/v1.12/en/advanced_config/tuning_the_ecl_ekf.html)

## Usage
MacOS, Ubuntu, and Windows wheels for Python 3.10 should be available in the `dist` folder. Download and pip install these.
To build on your own machine, run ```python3 -m build --wheel``` then pip install the .whl file. 
As of now, the Python bindings do not have any documentation. In general, please refer the the original documentation provided by PixHawk.

### Numpy and the Matrix library
All matrix types (including vectors) are equivalent to numpy arrays. Below is an example of converting between the two types:
```
>>> import ecl
>>> import numpy as np
>>> n = np.array([[0,10,20]], dtype = np.float32)
>>> v = ecl.Vector3f(n.transpose())
>>> nv = np.array(v)
```
Note (1) that it is necessary to specify 'dtype = np.float32' and (2) that the numpy array passed to Vector3f must be two-dimensional, even though it represents a vector. In this case, it also needs to be transposed to comply with Vector3f stride expectations. Incorrect array dimensions will break your code without raising any errors. This can get confusing, so make sure to test your logic with simple cases.

If you are indexing a 2-dimensional dataframe, you can preserve the dimensionality of indexed vectors using 'np.newaxis':
```
>>> data = np.array([[1,2,3],[4,5,6],[7,8,9]], dtype = np.float32)
>>> n = data[0,np.newaxis]
>>> v = ecl.Vector3f(n.transpose())
>>> nv = np.array(v)
```
## Working on this library in VSCode on a windows machine! 

First install Windows Subsystem for Linux + some distribution (I use ubuntu). Use the command 'wsl --install' in Windows PowerShell.
Then use 'wsl' to open your Linux terminal. 
Navigate to the project you want to work on in VSCode and use command 'code .' or 'code [path to project home directory]'.
You now have access to your project in a nice IDE!

make sure to change google_test to v1.11 (https://github.com/google/googletest/archive/refs/tags/release-1.11.0.zip) in test/CmakeLists.txt.in if you want to use 'make test'

## Miserable process of interfacing w/ Python D,:
### motivations
C++ is too hard for dumbass engineers, Python makes data handling, graphing very easy. Also, C++ is not very portable if you use non-standard C++14 extensions like this library does. By building wheels for MacOS, Linux, and Windows that are pip-installable, I hopefully save a few people from headaches dealing with CMake, compilers, and arcane build systems.

### getting started
```
> pip install pybind11
> git clone https://github.com/pybind/pybind11.git into the main directory
```
or
```
> git submodule update --init --recursive
```

Add the following lines to CMakeLists.txt after subdirectories:
```
add_subdirectory(pybind11)
pybind11_add_module(ecl src/binder.cpp)
target_link_libraries(ecl PRIVATE ecl_EKF ecl_geo ecl_geo_lookup ecl_airdata)
```
In order to make changes to PX4-matrix code, remove matrix ExternalProject in CMakeLists, instead git clone https://github.com/PX4/Matrix.git in main directory, then add these lines where ExternalProject used to be:
```
file(GLOB MATRIX_SRC
  Matrix/matrix/*.hpp
)
add_library(matrix INTERFACE ${MATRIX_SRC})
add_dependencies(prebuild_targets matrix)
include_directories(Matrix)
```
In `src/binder.cpp`: 
* include pybind11 and all headers (e.g. `EKF/ekf.h`, `matrix/math.hpp`)
```
namespace py = pybind11;
PYBIND11_MODULE(ecl, m) {
  py::class_<Ekf>(m, "Ekf")
    .def(blah); // add bindings for all functions in Ekf.h and EstimatorInterface.h
}
```
note: '&Ekf::setEkfGlobalOriginAltitude' breaks the module because it is not implemented

### Getting Matrix library to work with numpy and python buffer
Only the casting functionalities of Matrix types need to be exposed to Python. ALL other operations are either done in Numpy or internally by Ekf(), etc.

I referred to the pybind11 docs (https://pybind11.readthedocs.io/en/stable/advanced/pycpp/numpy.html) for interfacing with buffers. This needed to be done for all Matrix types, so I used a template to avoid nightmarish repetitiveness:
```
template<typename MType, typename Type>
void bindMatrix(py::module &m, const std::string &name) {
    py::class_<MType>(m, name.c_str(), py::buffer_protocol())

        // Matrix constructor that takes a py::buffer as an argument, COPIES data to the new MType object

        .def(py::init([](py::buffer b) {
            py::buffer_info info = b.request();
            // these checks are important, casting pointers is pretty dangerous, can cause headaches
            if (info.format != py::format_descriptor<Type>::format())
                throw std::runtime_error("Incompatible array format!" + py::format_descriptor<Type>::format());
            if (info.ndim != 2)
                throw std::runtime_error("Incompatible buffer dimension!");
            
            // this assumes row-major matrix representation
            ssize_t stride_n = info.strides[0] / (py::ssize_t)sizeof(Type);
            ssize_t stride_m = info.strides[1] / (py::ssize_t)sizeof(Type);

            auto ptr = 
                static_cast<Type *>(info.ptr);
            return MType(ptr, stride_n, stride_m);
        })) 

        // This tells Python how to convert a matrix into a buffer type (e.g. numpy array)
        // see docs for more info: https://pybind11.readthedocs.io/en/stable/advanced/pycpp/numpy.html

        .def_buffer([](MType &matrix) -> py::buffer_info {
            auto info = getBufferInfo(matrix);
            return py::buffer_info(
                info.ptr,        // Pointer to buffer
                info.item_size,  // Size of one scalar
                info.format,     // Python struct-style format descriptor
                2,               // Number of dimensions
                info.shape,      // Buffer dimensions
                info.strides     // Strides (in bytes) for each axis
            );
        });
}
```
In order for this code to work, some changes in the matrix class were necessary
In `Matrix.hpp`, make data pointer accessible by adding a getter:
```
Type *data() {
  return &_data[0][0]; // not sure if this works for slice, might not start at 0,0
}
```
For all relevant matrix-derived classes, I needed to write constructors that take a buffer pointer as well as stride information from two axes. Note that all of the derived classes inherit their constructors from matrix:
```
explicit Matrix(const Type data_[M*N], ssize_t stride_m, ssize_t stride_n)
{
  for (size_t i = 0; i < M; i++) {
    for (size_t j = 0; j < N; j++) {
      _data[i][j] = data_[stride_m*i + stride_n*j];
    }
  }
}
```
(not important) Added missing pointer constructor to `Euler.hpp`:
``` 
explicit Euler(const Type data_[3]) :
    Vector<Type, 3>(data_)
{
}
```
* NOTE: we will never have a templated flass available to the Python interface! So have templated 'class-declarer' functions for vector, matrix, maybe slice and scalar (unfortunately I dont think we can reuse bindings between these because some of them extend each other, not just typedefs). Then make a binder macro for each of these that will make it easy to bind random types like Quat which is a Vector<float, 4>

### Handling pass by reference

See "Limitations involving reference arguments" in the PyBind11 FAQ page (https://pybind11.readthedocs.io/en/stable/faq.html)
Pass/return by reference is not possible with PyBind. To get around this, just wrap the function with a lambda and return the modified values, return a tuple (std::make_tuple()) if multiple reference arguments.
Note that when accessing state/members of an object, must explicitly pass self as first parameter like you would in Python.
The convention I have followed for returned values is as follows:
- the first value in the tuple is the actual return value of the C++ function
- all proceeding values are the reference arguments in their original order (arrays are returned as vectors, which work in Python automatically thanks to the pybind11/stl.h header)
Note that these bindings create some additional overhead

To test in Python, just run Python3 in same directory as the generated .so file and import ecl. Note that ekf is a submodule of ecl, should be called w/ 'ecl.Ekf'.

make sure ecl_EKF target has 'PROPERTIES POSITION_INDEPENDENT_CODE ON' in `EKF/CMakeLists.txt`.

TODO [done]: get access to the EstimatorInterface interface. decent solution in the docs: https://pybind11.readthedocs.io/en/stable/advanced/classes.html#binding-protected-member-functions. Still had to move destructor to public, no solution in docs (not ideal, also not a big deal)
TODO: overall finish binding public interface to python
TODO: get versions of literally everything being used here (gcc, cmake, all libraries, etc)

To build C++ Library and shared objects file:
```
> mkdir build
> cd build
> cmake ..
> make [target]
```
To build platform-specific Python wheels that can be installed with pip, manually run `wheels.yml` and download the artifact.
