#include <algorithm>
#include <stereo-vision-lib/lib.hpp>

namespace stereo_vision {

  auto AnalysisError::ToString() const -> std::string {
    switch (this->value_) {
      case kInvalidImage:
        return "invalid image!";
      case kNotYetImplemented:
        return "not yet implemented!";
      default:
        throw std::runtime_error("unknown AnalysisError type!");
    }
  }

  AnalysisError::operator std::string() const {
    return this->ToString();
  }

  std::ostream &operator<<(std::ostream &o, AnalysisResult const &analysis_result) {
    // TODO: add output when adding data
    return o;
  }

  StereoVision::StereoVision(Settings const settings) : settings_(std::move(settings)) {
    cv::Mat R_left, R_right, P_left, P_right, Q;

    cv::stereoRectify(
        this->settings_.stereo_camera_info.camera_matrix_left, this->settings_.stereo_camera_info.dist_coeffs_left,
        this->settings_.stereo_camera_info.camera_matrix_right, this->settings_.stereo_camera_info.dist_coeffs_right,
        this->settings_.stereo_camera_info.image_size, this->settings_.stereo_camera_info.R,
        this->settings_.stereo_camera_info.T, R_left, R_right, P_left, P_right, Q, cv::CALIB_ZERO_DISPARITY, 0,
        this->settings_.stereo_camera_info.image_size);

    cv::initUndistortRectifyMap(this->settings_.stereo_camera_info.camera_matrix_left,
                                this->settings_.stereo_camera_info.dist_coeffs_left, R_left, P_left,
                                this->settings_.stereo_camera_info.image_size, CV_16SC2,
                                this->undistort_rectify_map_left.first, this->undistort_rectify_map_left.second);
    cv::initUndistortRectifyMap(this->settings_.stereo_camera_info.camera_matrix_right,
                                this->settings_.stereo_camera_info.dist_coeffs_right, R_right, P_right,
                                this->settings_.stereo_camera_info.image_size, CV_16SC2,
                                this->undistort_rectify_map_right.first, this->undistort_rectify_map_right.second);
  }

  auto StereoVision::AnalyzeAndAnnotateImage(cv::Mat const &left_image, cv::Mat const &right_image) const -> std::expected<AnalysisResult, AnalysisError> {
    if (left_image.data == nullptr || right_image.data == nullptr) {
      return std::unexpected{AnalysisError::kInvalidImage};
    }

    return std::unexpected{AnalysisError::kNotYetImplemented};
  }

  auto StereoVision::RectifyImages(cv::Mat const &left_image, cv::Mat const &right_image) const -> std::tuple<cv::Mat, cv::Mat> {
    cv::Mat left_image_rectified, right_image_rectified;
    cv::remap(left_image, left_image_rectified, this->undistort_rectify_map_left.first,
              this->undistort_rectify_map_left.second, cv::INTER_LINEAR);
    cv::remap(right_image, right_image_rectified, this->undistort_rectify_map_right.first,
              this->undistort_rectify_map_right.second, cv::INTER_LINEAR);
    return {left_image_rectified, right_image_rectified};
  }
} // namespace stereo_vision
