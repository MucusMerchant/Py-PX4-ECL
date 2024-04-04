# PX4-ECL
## Links to the original library
- [EKF Documentation and Tuning Guide](https://docs.px4.io/master/en/advanced_config/tuning_the_ecl_ekf.html)
- [PX4-ECL GitHub Repo](https://github.com/PX4/PX4-ECL)

## Working on this library in VSCode on a windows machine! 

First install Windows Subsystem for Linux + some distribution (I use ubuntu). Use the command 'wsl --install' in Windows PowerShell.
Then use 'wsl' to open your Linux terminal. 
Navigate to the project you want to work on in VSCode and use command 'code .' or 'code [path to project home directory]'.
You now have access to your project in a nice IDE!

make sure to change google_test to v1.11 (https://github.com/google/googletest/archive/refs/tags/release-1.11.0.zip) in test/CmakeLists.txt.in if you want to use 'make test'

## Miserable process of interfacing w/ Python D,:
### motivations
C++ is too hard for dumbass engineers, Python makes data handling, graphing very easy
documenting this in case me or another poor soul has to do this again

## getting started
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
In src/binder.cpp (full file will be posted somewhere): 
* include pybind11 and all headers (e.g. EKF/ekf.h, matrix/math.hpp)
```
namespace py = pybind11;
PYBIND11_MODULE(ecl, m) {
  py::class_<Ekf>(m, "Ekf")
    .def(blah); // add bindings for all functions in Ekf.h and EstimatorInterface.h
}
```
note: &Ekf::setEkfGlobalOriginAltitude' breaks the module because it is not implemented

### Worst part: Getting Matrix library to work with numpy and python buffer
note, we need to pass numpy arrays/matrixes with dtype=np.float32 or we get runtime error thrown at us
Also, only the casting functionality is exposed to Python (this simplifies this step a little). ALL other operations are either done in Numpy or internally by Ekf(), etc.

I referred to the pybind11 docs (https://pybind11.readthedocs.io/en/stable/advanced/pycpp/numpy.html) for interfacing with buffers. This needed to be done for all Matrix types, but I used a template to avoid nightmarish repetitiveness:
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
In Matrix.hpp, make data pointer accessible by adding a getter:
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
(not important) Added pointer constructor to Euler:
``` 
explicit Euler(const Type data_[3]) :
    Vector<Type, 3>(data_)
{
}
```
* NOTE: we will never have a templated flass available to the Python interface! So have templated 'class-declarer' functions for vector, matrix, maybe slice and scalar (unfortunately I dont think we can reuse bindings between these because some of them extend each other, not just typedefs). Then make a binder macro for each of these that will make it easy to bind random types like Quat which is a Vector<float, 4>

To test in Python, just run Python3 in same directory as the generated .so file and import ecl. Note that ekf is a submodule of ecl, should be called w/ 'ecl.Ekf'.

make sure ecl_EKF target has 'PROPERTIES POSITION_INDEPENDENT_CODE ON' in EKF/CMakeLists.txt.

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

To build Python wheel (to be stored in './dist') using CMake:
```
> pip install wheel
> python3 setup.py bdist_wheel
```

**Very lightweight Estimation & Control Library.**

[![DOI](https://zenodo.org/badge/22634/PX4/PX4-ECL.svg)](https://zenodo.org/badge/latestdoi/22634/PX4/PX4-ECL)

This library solves the estimation & control problems of a number of robots and drones. It accepts GPS, vision and inertial sensor inputs. It is extremely lightweight and efficient and yet has the rugged field-proven performance.

The library is BSD 3-clause licensed.

## EKF Documentation

  * [EKF Documentation and Tuning Guide](https://docs.px4.io/master/en/advanced_config/tuning_the_ecl_ekf.html)

## Building EKF

### Prerequisites:

  * Matrix: A lightweight, BSD-licensed matrix math library: https://github.com/px4/matrix - it is automatically included as submodule.


By following the steps mentioned below you can create a static library which can be included in projects:

```
make
// OR
mkdir build/
cd build/
cmake ..
make
```

## Testing ECL
By following the steps you can run the unit tests

```
make test
```
### Change Indicator / Unit Tests
Change indication is the concept of running the EKF on different data-sets and compare the state of the EKF to a previous version. If a contributor makes a functional change that is run during the change_indication tests, this will produce a different output of the EKF's state. As the tests are run in CI, this checks if a contributor forgot to run the checks themselves and add the [new EKF's state outputs](https://github.com/PX4/ecl/blob/master/test/change_indication/iris_gps.csv) to the pull request.

The unit tests include a check to see if the pull request results in a difference to the [output data csv file](https://github.com/PX4/ecl/blob/master/test/change_indication/iris_gps.csv) when replaying the [sensor data csv file](https://github.com/PX4/ecl/blob/master/test/replay_data/iris_gps.csv). If a pull request results in an expected difference, then it is important that the output reference file be re-generated and included as part of the pull request. A non-functional pull request should not result in changes to this file, however the default test case does not exercise all sensor types so this test passing is a necessary, but not sufficient requirement for a non-functional pull request.

The functionality that supports this test consists of:
* Python scripts that extract sensor data from ulog files and writes them to a sensor data csv file. The default [sensor data csv file](https://github.com/PX4/ecl/blob/master/test/replay_data/iris_gps.csv) used by the unit test was generated from a ulog created from an iris SITL flight.
* A [script file](https://github.com/PX4/ecl/blob/master/test/test_EKF_withReplayData.cpp) using functionality provided by  the [sensor simulator](https://github.com/PX4/ecl/blob/master/test/sensor_simulator/sensor_simulator.cpp), that loads sensor data from the [sensor data csv file](https://github.com/PX4/ecl/blob/master/test/replay_data/iris_gps.csv) , replays the EKF with it and logs the EKF's state and covariance data to the [output data csv file](https://github.com/PX4/ecl/blob/master/test/replay_data/iris_gps.csv).
* CI action that checks if the logs of the test running with replay data is changing. This helps to see if there are functional changes.

#### How to run the Change Indicator test during development on your own logs:

* create sensor_data.csv file from ulog file 'cd test/sensor_simulator/
python3 createSensorDataFile.py <path/to/ulog> ../replay_data/<descriptive_name>.csv'
* Setup the test file to use the EKF with the created sensor data by copy&paste an existing test case in [test/test_EKF_withReplayData.cpp](https://github.com/PX4/ecl/blob/master/test/test_EKF_withReplayData.cpp) and adapt the paths to load the right sensor data and write it to the right place, eg
_sensor_simulator.loadSensorDataFromFile("../../../test/replay_data/<descriptive_name>.csv");
_ekf_logger.setFilePath("../../../test/change_indication/<descriptive_name>.csv");
* You can feed the EKF with the data in the csv file, by running '_sensor_simulator.runReplaySeconds(duration_in_seconds)'. Be aware that replay sensor data will only be available when the corresponding sensor simulation are running. By default only imu, baro and mag sensor simulators are running. You can start a sensor simulation by calling _sensor_simulator._<sensor>.start(). Be also aware that you still have to setup the EKF yourself. This includes setting the bit mask (fusion_mode in common.h) according to what you intend to fuse.
* In between _sensor_simulator.runReplaySeconds(duration_in_seconds) calls, write the state and covariances to the change_indication file by including a _ekf_logger.writeStateToFile(); line.
* Run the EKF with your data and all the other tests by running 'make test' from the ecl directory. The [default output data csv file](https://github.com/PX4/ecl/blob/master/test/change_indication/iris_gps.csv) changes can then be included in the PR if differences are causing the CI test to fail.

#### Known Issues
If compiler versions other than GCC 7.5 are used to generate the output data file, then is is possible that the file will cause CI failures due to small numerical differences to file generated by the CI test.
