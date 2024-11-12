#ifndef STEREOCAMERAINFO_HPP
#define STEREOCAMERAINFO_HPP

#include <utility>
#include <ostream>

#include <opencv2/opencv.hpp>

namespace stereo_vision {
    /// Contains all information from the stereo camera calibration which is needed to rectify images afterward.
    struct StereoCameraInfo {
        /// Image size used for camera calibration.
        cv::Size const image_size;
        /// Camera intrinsic matrix for the left camera.
        /// See `cv::stereoCalibrate` for more details.
        cv::Mat const camera_matrix_left;
        /// Camera intrinsic matrix for the right camera.
        /// See `cv::stereoCalibrate` for more details.
        cv::Mat const camera_matrix_right;
        /// distortion coefficients for the left camera.
        /// See `cv::stereoCalibrate` for more details.
        cv::Mat const dist_coeffs_left;
        /// distortion coefficients for the right camera.
        /// See `cv::stereoCalibrate` for more details.
        cv::Mat const dist_coeffs_right;
        /// 3x3 Rotation matrix from first camera coordinate system to second camera coordinate system.
        /// See `cv::stereoCalibrate` for more details.
        cv::Mat const R;
        /// 3x1 Translation vector from first camera coordinate system to second camera coordinate system.
        /// See `cv::stereoCalibrate` for more details.
        cv::Mat const T;
        /// 3x3 Essential matrix.
        /// See `cv::stereoCalibrate` for more details.
        cv::Mat const E;
        /// 3x3 Fundametal matrix.
        /// See `cv::stereoCalibrate` for more details.
        cv::Mat const F;
        /// The reprojection error calculated by the calibration. Not needed for rectification but informs about the quality of the calibration.
        /// See `cv::stereoRectify` for more details.
        double reprojection_error;
    };


    // These write and read functions must be defined for the serialization in FileStorage to work
    // Oddly enough they must be in the same namespace as `StereoCameraInfo` and not outside the namespace in
    // order to be found ¯\_(ツ)_/¯.
    static void write(cv::FileStorage& fs, std::string const&, StereoCameraInfo const& x) {
        fs << "{"
           << "image_size" << x.image_size
           << "camera_matrix_left" << x.camera_matrix_left
           << "camera_matrix_right" << x.camera_matrix_right
           << "dist_coeffs_left" << x.dist_coeffs_left
           << "dist_coeffs_right" << x.dist_coeffs_right
           << "R" << x.R
           << "T" << x.T
           << "E" << x.E
           << "F" << x.F
           << "reprojection_error" << x.reprojection_error
           << "}";
    }
}

inline std::ostream& operator<< (std::ostream& os, stereo_vision::StereoCameraInfo const& stereo_camera_info) {
    return os << "Stereo camera calibration data:" << std::endl
              << "  reprojection error = " << stereo_camera_info.reprojection_error << std::endl
              << "  image size = " << stereo_camera_info.image_size << std::endl
              << "  R = " << stereo_camera_info.R << std::endl
              << "  T = " << stereo_camera_info.T << std::endl
              << "  E = " << stereo_camera_info.E << std::endl
              << "  F = " << stereo_camera_info.F << std::endl
              << std::endl
              << "  left camera:" << std::endl
              << "    camera_matrix = " << stereo_camera_info.camera_matrix_left << std::endl
              << "    dist_coeffs = " << stereo_camera_info.dist_coeffs_left << std::endl
              << "  right camera:" << std::endl
              << "    camera_matrix = " << stereo_camera_info.camera_matrix_right << std::endl
              << "    dist_coeffs = " << stereo_camera_info.dist_coeffs_right << std::endl
    ;
}

#endif //STEREOCAMERAINFO_HPP
