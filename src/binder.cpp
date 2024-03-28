#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <EKF/ekf.h>
#include <geo/geo.h>
#include <geo_lookup/geo_mag_declination.h>
#include <airdata/WindEstimator.hpp>
#include <matrix/math.hpp>

namespace py = pybind11;
/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
| Trampoline and publicist class for access to EstimatorInterface         |
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/

// Trampoline class exposes virtual methods of EstimatorInterface
// https://pybind11.readthedocs.io/en/stable/advanced/classes.html 

class PyEstimatorInterface : EstimatorInterface {
    public:

        using EstimatorInterface::EstimatorInterface;
        virtual ~PyEstimatorInterface() = default;
        
        bool collect_gps(const gps_message &gps) override {
            PYBIND11_OVERRIDE_PURE(
                bool,
                EstimatorInterface,
                collect_gps,
                gps
            );
        }

        bool init(uint64_t timestamp) override {
            PYBIND11_OVERRIDE_PURE(
                bool,
                EstimatorInterface,
                init,
                timestamp
            );
        }

        float compensateBaroForDynamicPressure(const float baro_alt_uncompensated) const override {
            PYBIND11_OVERRIDE_PURE(
                float,
                EstimatorInterface,
                compensateBaroForDynamicPressure,
                baro_alt_uncompensated
            );
        }
};

// Publicist class makes protected methods public
class PublicEstimatorInterface : EstimatorInterface {
    public:
        ~PublicEstimatorInterface() = default;
        using EstimatorInterface::collect_gps;
        using EstimatorInterface::init;
        using EstimatorInterface::compensateBaroForDynamicPressure;
};

/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
| Templated helpers for linking PX4-Matrix types to Python buffer protocol|
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/

// Structure containing necessary information to make a call to py::buffer_info, template takes matrix parameters
// All classes derived from Matrix are handled here implicitly

template<typename Type, size_t M, size_t N>
struct BufferInfo {
    Type *ptr;
    size_t item_size;
    std::string format;
    std::vector<size_t> shape;
    std::vector<size_t> strides;
};

// Helper function to get buffer information for a Matrix object

template<typename Type, size_t M, size_t N>
BufferInfo<Type, M, N> getBufferInfo(matrix::Matrix<Type, M, N> &matrix) {
    BufferInfo<Type, M, N> info;
    info.ptr = matrix.data();
    info.item_size = sizeof(Type);
    info.format = py::format_descriptor<Type>::format();
    info.shape = {M, N};
    info.strides = {N * sizeof(Type), sizeof(Type)};
    return info;
}

// Template parameter MType expects matrix::[derived class]<Type, [other template parameters]>
// Template parameter Type must be the same as the Type specified in MType
// example usage: bindMatrix<matrix::Vector3<float, 3>, float>(...)

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

// Bindings! No documentation is provided here, please refer to the PX4-ECL documentation
// https://docs.px4.io/main/en/advanced_config/tuning_the_ecl_ekf.html

PYBIND11_MODULE(ecl, m) {

/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
| Matrix bindings needed for Ekf (all types referenced in common.h)       |
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/

    bindMatrix<matrix::SquareMatrix<float,3>,float>(m, "Matrix3f");
    bindMatrix<matrix::Vector3<float>,float>(m, "Vector3f");
    bindMatrix<matrix::Vector2<float>,float>(m, "Vector2f");
    bindMatrix<matrix::Quaternion<float>,float>(m, "Quatf");
    bindMatrix<matrix::Dcm<float>,float>(m, "Dcmf");
    bindMatrix<matrix::AxisAngle<float>,float>(m, "AxisAnglef");
    bindMatrix<matrix::Euler<float>,float>(m, "Eulerf");

/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
| Struct bindings for sensor input and filter tuning                      |
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/

    py::class_<gps_message>(m, "gps_message")
        .def(py::init<>())
        .def_readwrite("time_usec", &gps_message::time_usec)
        .def_readwrite("lat", &gps_message::lat)
        .def_readwrite("lon", &gps_message::lon)
        .def_readwrite("alt", &gps_message::alt)
        .def_readwrite("yaw", &gps_message::yaw)
        .def_readwrite("yaw_offset", &gps_message::yaw_offset)
        .def_readwrite("fix_type", &gps_message::fix_type)
        .def_readwrite("eph", &gps_message::eph)
        .def_readwrite("epv", &gps_message::epv)
        .def_readwrite("sacc", &gps_message::sacc)
        .def_readwrite("vel_m_s", &gps_message::vel_m_s)
        .def_readwrite("vel_ned", &gps_message::vel_ned)
        .def_readwrite("vel_ned_valid", &gps_message::vel_ned_valid)
        .def_readwrite("nsats", &gps_message::nsats)
        .def_readwrite("pdop", &gps_message::pdop);

    py::class_<outputSample>(m, "outputSample")
        .def(py::init<>())
        .def_readwrite("time_us", &outputSample::time_us)
        .def_readwrite("quat_nominal", &outputSample::quat_nominal)
        .def_readwrite("vel", &outputSample::vel)
        .def_readwrite("pos", &outputSample::pos);

    py::class_<outputVert>(m, "outputVert")
        .def(py::init<>())
        .def_readwrite("time_us", &outputVert::time_us)
        .def_readwrite("vert_vel", &outputVert::vert_vel)
        .def_readwrite("vert_vel_integ", &outputVert::vert_vel_integ)
        .def_readwrite("dt", &outputVert::dt);

    py::class_<imuSample>(m, "imuSample")
        .def(py::init<>())
        .def_readwrite("time_us", &imuSample::time_us)
        .def_readwrite("delta_ang", &imuSample::delta_ang)
        .def_readwrite("delta_vel", &imuSample::delta_vel)
        .def_readwrite("delta_ang_dt", &imuSample::delta_ang_dt)
        .def_readwrite("delta_vel_dt", &imuSample::delta_vel_dt)
        //.def_readwrite("delta_vel_clipping", &imuSample::delta_vel_clipping)
        ;

    py::class_<gpsSample>(m, "gpsSample")
        .def(py::init<>())
        .def_readwrite("time_us", &gpsSample::time_us)
        .def_readwrite("pos", &gpsSample::pos)
        .def_readwrite("hgt", &gpsSample::hgt)
        .def_readwrite("vel", &gpsSample::vel)
        .def_readwrite("yaw", &gpsSample::yaw)
        .def_readwrite("hacc", &gpsSample::hacc)
        .def_readwrite("vacc", &gpsSample::vacc)
        .def_readwrite("sacc", &gpsSample::sacc);

    py::class_<magSample>(m, "magSample")
        .def(py::init<>())
        .def_readwrite("time_us", &magSample::time_us)
        .def_readwrite("mag", &magSample::mag);

    py::class_<baroSample>(m, "baroSample")
        .def(py::init<>())
        .def_readwrite("time_us", &baroSample::time_us)
        .def_readwrite("hgt", &baroSample::hgt);

    py::class_<rangeSample>(m, "rangeSample")
        .def(py::init<>())
        .def_readwrite("time_us", &rangeSample::time_us)
        .def_readwrite("rng", &rangeSample::rng)
        .def_readwrite("quality", &rangeSample::quality);

    py::class_<airspeedSample>(m, "airspeedSample")
        .def(py::init<>())
        .def_readwrite("time_us", &airspeedSample::time_us)
        .def_readwrite("true_airspeed", &airspeedSample::true_airspeed)
        .def_readwrite("eas2tas", &airspeedSample::eas2tas);

    py::class_<flowSample>(m, "flowSample")
        .def(py::init<>())
        .def_readwrite("time_us", &flowSample::time_us)
        .def_readwrite("flow_xy_rad", &flowSample::flow_xy_rad)
        .def_readwrite("gyro_xyz", &flowSample::gyro_xyz)
        .def_readwrite("dt", &flowSample::dt)
        .def_readwrite("quality", &flowSample::quality);

    py::class_<extVisionSample>(m, "extVisionSample")
        .def(py::init<>())
        .def_readwrite("time_us", &extVisionSample::time_us)
        .def_readwrite("pos", &extVisionSample::pos)
        .def_readwrite("vel", &extVisionSample::vel)
        .def_readwrite("quat", &extVisionSample::quat)
        .def_readwrite("posVar", &extVisionSample::posVar)
        .def_readwrite("velCov", &extVisionSample::velCov)
        .def_readwrite("angVar", &extVisionSample::angVar)
        .def_readwrite("vel_frame", &extVisionSample::vel_frame);

    // parameter struct, accessed (by reference) with getParamHandle(), can be directly modified to tune the filter
    py::class_<parameters>(m, "parameters")
        .def(py::init<>())
        .def_readwrite("fusion_mode", &parameters::fusion_mode)
        .def_readwrite("vdist_sensor_type", &parameters::vdist_sensor_type)
        .def_readwrite("terrain_fusion_mode", &parameters::terrain_fusion_mode)
        .def_readwrite("sensor_interval_min_ms", &parameters::sensor_interval_min_ms)
        .def_readwrite("mag_delay_ms", &parameters::mag_delay_ms)
        .def_readwrite("baro_delay_ms", &parameters::baro_delay_ms)
        .def_readwrite("gps_delay_ms", &parameters::gps_delay_ms)
        .def_readwrite("airspeed_delay_ms", &parameters::airspeed_delay_ms)
        .def_readwrite("flow_delay_ms", &parameters::flow_delay_ms)
        .def_readwrite("range_delay_ms", &parameters::range_delay_ms)
        .def_readwrite("ev_delay_ms", &parameters::ev_delay_ms)
        .def_readwrite("auxvel_delay_ms", &parameters::auxvel_delay_ms)
        .def_readwrite("gyro_noise", &parameters::gyro_noise)
        .def_readwrite("accel_noise", &parameters::accel_noise)
        .def_readwrite("gyro_bias_p_noise", &parameters::gyro_bias_p_noise)
        .def_readwrite("accel_bias_p_noise", &parameters::accel_bias_p_noise)
        .def_readwrite("mage_p_noise", &parameters::mage_p_noise)
        .def_readwrite("magb_p_noise", &parameters::magb_p_noise)
        .def_readwrite("wind_vel_p_noise", &parameters::wind_vel_p_noise)
        .def_readonly("wind_vel_p_noise_scaler", &parameters::wind_vel_p_noise_scaler)
        .def_readwrite("terrain_p_noise", &parameters::terrain_p_noise)
        .def_readwrite("terrain_gradient", &parameters::terrain_gradient)
        .def_readonly("terrain_timeout", &parameters::terrain_timeout)
        .def_readwrite("switch_on_gyro_bias", &parameters::switch_on_gyro_bias)
        .def_readwrite("switch_on_accel_bias", &parameters::switch_on_accel_bias)
        .def_readwrite("initial_tilt_err", &parameters::initial_tilt_err)
        .def_readonly("initial_wind_uncertainty", &parameters::initial_wind_uncertainty)
        .def_readwrite("gps_vel_noise", &parameters::gps_vel_noise)
        .def_readwrite("gps_pos_noise", &parameters::gps_pos_noise)
        .def_readwrite("pos_noaid_noise", &parameters::pos_noaid_noise)
        .def_readwrite("baro_noise", &parameters::baro_noise)
        .def_readwrite("baro_innov_gate", &parameters::baro_innov_gate)
        .def_readwrite("gps_pos_innov_gate", &parameters::gps_pos_innov_gate)
        .def_readwrite("gps_vel_innov_gate", &parameters::gps_vel_innov_gate)
        .def_readwrite("gnd_effect_deadzone", &parameters::gnd_effect_deadzone)
        .def_readwrite("gnd_effect_max_hgt", &parameters::gnd_effect_max_hgt)
        .def_readwrite("mag_heading_noise", &parameters::mag_heading_noise)
        .def_readwrite("mag_noise", &parameters::mag_noise)
        .def_readwrite("mag_declination_deg", &parameters::mag_declination_deg)
        .def_readwrite("heading_innov_gate", &parameters::heading_innov_gate)
        .def_readwrite("mag_innov_gate", &parameters::mag_innov_gate)
        .def_readwrite("mag_declination_source", &parameters::mag_declination_source)
        .def_readwrite("mag_fusion_type", &parameters::mag_fusion_type)
        .def_readwrite("mag_acc_gate", &parameters::mag_acc_gate)
        .def_readwrite("mag_yaw_rate_gate", &parameters::mag_yaw_rate_gate)
        .def_readonly("quat_max_variance", &parameters::quat_max_variance)
        .def_readwrite("tas_innov_gate", &parameters::tas_innov_gate)
        .def_readwrite("eas_noise", &parameters::eas_noise)
        .def_readwrite("arsp_thr", &parameters::arsp_thr)
        .def_readwrite("beta_innov_gate", &parameters::beta_innov_gate)
        .def_readwrite("beta_noise", &parameters::beta_noise)
        .def_readonly("beta_avg_ft_us", &parameters::beta_avg_ft_us)
        .def_readwrite("range_noise", &parameters::range_noise)
        .def_readwrite("range_innov_gate", &parameters::range_innov_gate)
        .def_readwrite("rng_gnd_clearance", &parameters::rng_gnd_clearance)
        .def_readwrite("rng_sens_pitch", &parameters::rng_sens_pitch)
        .def_readwrite("range_noise_scaler", &parameters::range_noise_scaler)
        .def_readonly("vehicle_variance_scaler", &parameters::vehicle_variance_scaler)
        .def_readwrite("max_hagl_for_range_aid", &parameters::max_hagl_for_range_aid)
        .def_readwrite("max_vel_for_range_aid", &parameters::max_vel_for_range_aid)
        .def_readwrite("range_aid", &parameters::range_aid)
        .def_readwrite("range_aid_innov_gate", &parameters::range_aid_innov_gate)
        .def_readwrite("range_valid_quality_s", &parameters::range_valid_quality_s)
        .def_readwrite("range_cos_max_tilt", &parameters::range_cos_max_tilt)
        .def_readwrite("ev_vel_innov_gate", &parameters::ev_vel_innov_gate)
        .def_readwrite("ev_pos_innov_gate", &parameters::ev_pos_innov_gate)
        .def_readwrite("flow_noise", &parameters::flow_noise)
        .def_readwrite("flow_noise_qual_min", &parameters::flow_noise_qual_min)
        .def_readwrite("flow_qual_min", &parameters::flow_qual_min)
        .def_readwrite("flow_innov_gate", &parameters::flow_innov_gate)
        .def_readwrite("gps_check_mask", &parameters::gps_check_mask)
        .def_readwrite("req_hacc", &parameters::req_hacc)
        .def_readwrite("req_vacc", &parameters::req_vacc)
        .def_readwrite("req_sacc", &parameters::req_sacc)
        .def_readwrite("req_nsats", &parameters::req_nsats)
        .def_readwrite("req_pdop", &parameters::req_pdop)
        .def_readwrite("req_hdrift", &parameters::req_hdrift)
        .def_readwrite("req_vdrift", &parameters::req_vdrift)
        .def_readwrite("imu_pos_body", &parameters::imu_pos_body)
        .def_readwrite("gps_pos_body", &parameters::gps_pos_body)
        .def_readwrite("rng_pos_body", &parameters::rng_pos_body)
        .def_readwrite("flow_pos_body", &parameters::flow_pos_body)
        .def_readwrite("ev_pos_body", &parameters::ev_pos_body)
        .def_readwrite("vel_Tau", &parameters::vel_Tau)
        .def_readwrite("pos_Tau", &parameters::pos_Tau)
        .def_readwrite("acc_bias_lim", &parameters::acc_bias_lim)
        .def_readwrite("acc_bias_learn_acc_lim", &parameters::acc_bias_learn_acc_lim)
        .def_readwrite("acc_bias_learn_gyr_lim", &parameters::acc_bias_learn_gyr_lim)
        .def_readwrite("acc_bias_learn_tc", &parameters::acc_bias_learn_tc)
        .def_readonly("reset_timeout_max", &parameters::reset_timeout_max)
        .def_readonly("no_aid_timeout_max", &parameters::no_aid_timeout_max)
        .def_readwrite("valid_timeout_max", &parameters::valid_timeout_max)
        .def_readwrite("static_pressure_coef_xp", &parameters::static_pressure_coef_xp)
        .def_readwrite("static_pressure_coef_xn", &parameters::static_pressure_coef_xn)
        .def_readwrite("static_pressure_coef_yp", &parameters::static_pressure_coef_yp)
        .def_readwrite("static_pressure_coef_yn", &parameters::static_pressure_coef_yn)
        .def_readwrite("static_pressure_coef_z", &parameters::static_pressure_coef_z)
        .def_readwrite("max_correction_airspeed", &parameters::max_correction_airspeed)
        .def_readwrite("drag_noise", &parameters::drag_noise)
        .def_readwrite("bcoef_x", &parameters::bcoef_x)
        .def_readwrite("bcoef_y", &parameters::bcoef_y)
        .def_readwrite("mcoef", &parameters::mcoef)
        .def_readonly("vert_innov_test_lim", &parameters::vert_innov_test_lim)
        .def_readonly("bad_acc_reset_delay_us", &parameters::bad_acc_reset_delay_us)
        .def_readwrite("is_moving_scaler", &parameters::is_moving_scaler)
        .def_readwrite("synthesize_mag_z", &parameters::synthesize_mag_z)
        .def_readwrite("check_mag_strength", &parameters::check_mag_strength)
        .def_readwrite("EKFGSF_tas_default", &parameters::EKFGSF_tas_default)
        .def_readonly("EKFGSF_reset_delay", &parameters::EKFGSF_reset_delay)
        .def_readonly("EKFGSF_yaw_err_max", &parameters::EKFGSF_yaw_err_max)
        .def_readonly("EKFGSF_reset_count_limit", &parameters::EKFGSF_reset_count_limit)
        ;
    
    // DELETE: Internal state sample, this actually should not be in the public interface
    /*
    py::class_<stateSample>(m, "stateSample")
        .def(py::init<>())
        .def_readwrite("quat_nominal", &stateSample::quat_nominal)
        .def_readwrite("vel", &stateSample::vel)
        .def_readwrite("pos", &stateSample::pos)
        .def_readwrite("delta_ang_bias", &stateSample::delta_ang_bias)
        .def_readwrite("delta_vel_bias", &stateSample::delta_vel_bias)
        .def_readwrite("mag_I", &stateSample::mag_I)
        .def_readwrite("mag_B", &stateSample::mag_B)
        .def_readwrite("wind_vel", &stateSample::wind_vel)
        ;
    */
/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
| Bindings for Ekf and EstimatorInterface                                 |
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/

    py::class_<EstimatorInterface>(m, "EstimatorInterface");
    
    py::class_<Ekf, EstimatorInterface>(m, "Ekf")
        // Public functions from EstimatorInterface
        .def(py::init<>())
        .def("hi", &Ekf::hi)
        .def("getParamHandle", &Ekf::getParamHandle, py::return_value_policy::reference)
        .def("setIMUData", &Ekf::setIMUData)
        .def("print_status", &Ekf::print_status)

        // Functions from ekf.h
        .def("init", &Ekf::init)
        .def("update", &Ekf::update)
        .def("getGpsVelPosInnov", &Ekf::getGpsVelPosInnov)
        .def("getGpsVelPosInnovVar", &Ekf::getGpsVelPosInnovVar)
        .def("getEvVelPosInnov", &Ekf::getEvVelPosInnov)
        .def("getEvVelPosInnovVar", &Ekf::getEvVelPosInnovVar)
        .def("getBaroHgtInnov", &Ekf::getBaroHgtInnov)
        .def("getBaroHgtInnovVar", &Ekf::getBaroHgtInnovVar)
        .def("getBaroHgtInnovRatio", &Ekf::getBaroHgtInnovRatio)
        .def("getRngHgtInnov", &Ekf::getRngHgtInnov)
        .def("getRngHgtInnovVar", &Ekf::getRngHgtInnovVar)
        .def("getRngHgtInnovRatio", &Ekf::getRngHgtInnovRatio)
        .def("getAuxVelInnov", &Ekf::getAuxVelInnov)
        .def("getAuxVelInnovVar", &Ekf::getAuxVelInnovVar)
        .def("getAuxVelInnovRatio", &Ekf::getAuxVelInnovRatio)
        .def("getFlowInnov", &Ekf::getFlowInnov)
        .def("getFlowInnovVar", &Ekf::getFlowInnovVar)
        .def("getFlowInnovRatio", &Ekf::getFlowInnovRatio)
        .def("getFlowVelBody", &Ekf::getFlowVelBody, py::return_value_policy::reference)
        .def("getFlowVelNE", &Ekf::getFlowVelNE, py::return_value_policy::reference)
        .def("getFlowCompensated", &Ekf::getFlowCompensated, py::return_value_policy::reference)
        .def("getFlowUncompensated", &Ekf::getFlowUncompensated, py::return_value_policy::reference)
        .def("getFlowGyro", &Ekf::getFlowGyro, py::return_value_policy::reference)
        .def("getHeadingInnov", &Ekf::getHeadingInnov)
        .def("getHeadingInnovVar", &Ekf::getHeadingInnovVar)
        .def("getHeadingInnovRatio", &Ekf::getHeadingInnovRatio)
        .def("getMagInnov", &Ekf::getMagInnov)
        .def("getMagInnovVar", &Ekf::getMagInnovVar)
        .def("getMagInnovRatio", &Ekf::getMagInnovRatio)
        .def("getDragInnov", &Ekf::getDragInnov)
        .def("getDragInnovVar", &Ekf::getDragInnovVar)
        .def("getDragInnovRatio", &Ekf::getDragInnovRatio)
        .def("getAirspeedInnov", &Ekf::getAirspeedInnov)
        .def("getAirspeedInnovVar", &Ekf::getAirspeedInnovVar)
        .def("getAirspeedInnovRatio", &Ekf::getAirspeedInnovRatio)
        .def("getBetaInnov", &Ekf::getBetaInnov)
        .def("getBetaInnovVar", &Ekf::getBetaInnovVar)
        .def("getBetaInnovRatio", &Ekf::getBetaInnovRatio)
        .def("getHaglInnov", &Ekf::getHaglInnov)
        .def("getHaglInnovVar", &Ekf::getHaglInnovVar)
        .def("getHaglInnovRatio", &Ekf::getHaglInnovRatio)
        .def("getStateAtFusionHorizonAsVector", &Ekf::getStateAtFusionHorizonAsVector)
        .def("getWindVelocity", &Ekf::getWindVelocity, py::return_value_policy::reference)
        .def("getWindVelocityVariance", &Ekf::getWindVelocityVariance)
        .def("get_true_airspeed", &Ekf::get_true_airspeed)
        .def("covariances", &Ekf::covariances, py::return_value_policy::reference)
        .def("covariances_diagonal", &Ekf::covariances_diagonal)
        .def("orientation_covariances", &Ekf::orientation_covariances, py::return_value_policy::reference)
        .def("velocity_covariances", &Ekf::velocity_covariances, py::return_value_policy::reference)
        .def("position_covariances", &Ekf::position_covariances, py::return_value_policy::reference)
        .def("collect_gps", &Ekf::collect_gps)
        .def("getEkfGlobalOrigin", &Ekf::getEkfGlobalOrigin)
        .def("setEkfGlobalOrigin", &Ekf::setEkfGlobalOrigin)
        .def("getEkfGlobalOriginAltitude", &Ekf::getEkfGlobalOriginAltitude)
        //.def("setEkfGlobalOriginAltitude", &Ekf::setEkfGlobalOriginAltitude)
        .def("get_ekf_gpos_accuracy", &Ekf::get_ekf_gpos_accuracy)
        .def("get_ekf_lpos_accuracy", &Ekf::get_ekf_lpos_accuracy)
        .def("get_ekf_vel_accuracy", &Ekf::get_ekf_vel_accuracy)
        .def("get_ekf_ctrl_limits", &Ekf::get_ekf_ctrl_limits)
        .def("resetImuBias", &Ekf::resetImuBias)
        .def("resetGyroBias", &Ekf::resetGyroBias)
        .def("resetAccelBias", &Ekf::resetAccelBias)
        .def("resetMagBias", &Ekf::resetMagBias)
        .def("getVelocityVariance", &Ekf::getVelocityVariance)
        .def("getPositionVariance", &Ekf::getPositionVariance)
        .def("getOutputTrackingError", &Ekf::getOutputTrackingError, py::return_value_policy::reference)
        .def("get_gps_drift_metrics", &Ekf::get_gps_drift_metrics)
        .def("global_position_is_valid", &Ekf::global_position_is_valid)
        .def("local_position_is_valid", &Ekf::local_position_is_valid)
        .def("isTerrainEstimateValid", &Ekf::isTerrainEstimateValid)
        .def("getTerrainEstimateSensorBitfield", &Ekf::getTerrainEstimateSensorBitfield)
        .def("getTerrainVertPos", &Ekf::getTerrainVertPos)
        .def("getTerrainVertPosResetCounter", &Ekf::getTerrainVertPosResetCounter)
        .def("get_terrain_var", &Ekf::get_terrain_var)
        .def("getGyroBias", &Ekf::getGyroBias)
        .def("getAccelBias", &Ekf::getAccelBias)
        .def("getMagBias", &Ekf::getMagBias, py::return_value_policy::reference)
        .def("getGyroBiasVariance", &Ekf::getGyroBiasVariance)
        .def("getAccelBiasVariance", &Ekf::getAccelBiasVariance)
        .def("getMagBiasVariance", &Ekf::getMagBiasVariance, py::return_value_policy::reference)
        .def("get_gps_check_status", &Ekf::get_gps_check_status)
        .def("state_reset_status", &Ekf::state_reset_status, py::return_value_policy::reference)
        .def("get_posD_reset", &Ekf::get_posD_reset)
        .def("get_velD_reset", &Ekf::get_velD_reset)
        .def("get_posNE_reset", &Ekf::get_posNE_reset)
        .def("get_velNE_reset", &Ekf::get_velNE_reset)
        .def("get_quat_reset", &Ekf::get_quat_reset)
        .def("get_innovation_test_status", &Ekf::get_innovation_test_status)
        .def("get_ekf_soln_status", &Ekf::get_ekf_soln_status)
        .def("getVisionAlignmentQuaternion", &Ekf::getVisionAlignmentQuaternion)
        .def("calculate_quaternion", &Ekf::calculate_quaternion)
        .def("set_min_required_gps_health_time", &Ekf::set_min_required_gps_health_time)
        .def("getDataEKFGSF", &Ekf::getDataEKFGSF)
        ;
}
