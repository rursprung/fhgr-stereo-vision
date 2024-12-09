#ifndef STEREOCAMERAINFO_HPP
#define STEREOCAMERAINFO_HPP

#include <ostream>
#include <stdexcept>
#include <filesystem>

#include <opencv2/opencv.hpp>

namespace stereo_vision {
  /// Version of the format of `StereoCameraInfo`. Stored in the config file to validate that it matches when loading it.
  constexpr uint8_t kStereoCameraInfoVersion = 1;

  /// Contains all information from the stereo camera calibration which is needed to rectify images afterward.
  struct StereoCameraInfo {
    /// Image size used for camera calibration.
    cv::Size image_size;
    /// Camera intrinsic matrix for the left camera.
    /// See `cv::stereoCalibrate` for more details.
    cv::Mat camera_matrix_left;
    /// Camera intrinsic matrix for the right camera.
    /// See `cv::stereoCalibrate` for more details.
    cv::Mat camera_matrix_right;
    /// distortion coefficients for the left camera.
    /// See `cv::stereoCalibrate` for more details.
    cv::Mat dist_coeffs_left;
    /// distortion coefficients for the right camera.
    /// See `cv::stereoCalibrate` for more details.
    cv::Mat dist_coeffs_right;
    /// 3x3 Rotation matrix from first camera coordinate system to second camera coordinate system.
    /// See `cv::stereoCalibrate` for more details.
    cv::Mat R;
    /// 3x1 Translation vector from first camera coordinate system to second camera coordinate system.
    /// See `cv::stereoCalibrate` for more details.
    cv::Mat T;
    /// 3x3 Essential matrix.
    /// See `cv::stereoCalibrate` for more details.
    cv::Mat E;
    /// 3x3 Fundametal matrix.
    /// See `cv::stereoCalibrate` for more details.
    cv::Mat F;
    /// Reprojection matrix
    cv::Mat Q = cv::Mat::zeros(4, 4, CV_64F);
    /// The reprojection error calculated by the calibration. Not needed for rectification but informs about the quality
    /// of the calibration. See `cv::stereoRectify` for more details.
    double reprojection_error;
  };


  // These write and read functions must be defined for the serialization in FileStorage to work
  // Oddly enough they must be in the same namespace as `StereoCameraInfo` and not outside the namespace in
  // order to be found ¯\_(ツ)_/¯.
  static void write(cv::FileStorage& fs, std::string const&, StereoCameraInfo const& x) {
    fs << "{"
       << "file_version" << kStereoCameraInfoVersion
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

  static void read(cv::FileNode const& node, StereoCameraInfo& x, StereoCameraInfo const& default_value = {}) {
    if (node.empty()) {
      throw std::runtime_error("missing camera info in config file!");
    }

    uint8_t version;
    node["file_version"] >> version;
    if (version != kStereoCameraInfoVersion) {
      throw std::runtime_error(std::format("invalid config file version! Expected {} but found {}! Please re-generate the calibration file using the matching calibration software version.", kStereoCameraInfoVersion, version));
    }

    node["image_size"] >> x.image_size;
    node["camera_matrix_left"] >> x.camera_matrix_left;
    node["camera_matrix_right"] >> x.camera_matrix_right;
    node["dist_coeffs_left"] >> x.dist_coeffs_left;
    node["dist_coeffs_right"] >> x.dist_coeffs_right;
    node["R"] >> x.R;
    node["T"] >> x.T;
    node["E"] >> x.E;
    node["F"] >> x.F;
    node["reprojection_error"] >> x.reprojection_error;
  }

  [[nodiscard]]
  inline auto LoadStereoCameraInfo(std::filesystem::path const& path) -> StereoCameraInfo {
    cv::FileStorage config_file{path.string(), cv::FileStorage::READ};
    StereoCameraInfo stereo_camera_info{};
    config_file["calibration"] >> stereo_camera_info;
    return stereo_camera_info;
  }
} // namespace stereo_vision

inline std::ostream &operator<<(std::ostream &os, stereo_vision::StereoCameraInfo const &stereo_camera_info) {
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
            << "    dist_coeffs = " << stereo_camera_info.dist_coeffs_right << std::endl;
}

#endif // STEREOCAMERAINFO_HPP
