#include <algorithm>
#include <stereo-vision-lib/lib.hpp>

namespace stereo_vision {

  auto AnalysisError::ToString() const -> std::string {
    switch (this->value_) {
      case kInvalidImage:
        return "invalid image!";
      default:
        throw std::runtime_error("unknown AnalysisError type!");
    }
  }

  AnalysisError::operator std::string() const {
    return this->ToString();
  }

  std::ostream &operator<<(std::ostream& o, AnalysisResult const& analysis_result) {
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

  auto StereoVision::AnalyzeAndAnnotateImage(cv::Mat const& left_image, cv::Mat const& right_image) const -> std::expected<AnalysisResult, AnalysisError> {
    if (left_image.empty() || right_image.empty()) {
      return std::unexpected{AnalysisError::kInvalidImage};
    }

    auto const left_image_rescaled = this->rescaleImage(this->settings_.stereo_camera_info.image_size.width, left_image);
    auto const right_image_rescaled = this->rescaleImage(this->settings_.stereo_camera_info.image_size.width, right_image);

    auto const& [left_image_rectified, right_image_rectified] = this->RectifyImages(left_image_rescaled, right_image_rescaled);
    auto disparity = this-> CalculateDisparityMap(left_image_rectified, right_image_rectified);
    auto distance = this-> CalculateDistanceMap(disparity);

    return {{
      .left_image = left_image_rectified,
      .right_image = right_image_rectified,
      .disparity_map = disparity
    }};
  }

  auto StereoVision::rescaleImage(auto width, auto const& image) const -> cv::Mat {
    cv::Mat out;
    if (image.size().width > width) {
        auto const new_width = width;
        ///new width while maintaining the aspect ratio
        float aspect_ratio = static_cast<float>(image.size().height) / image.size().width;
        int new_height = static_cast<int>(new_width * aspect_ratio);

        cv::resize(image, out, cv::Size(new_width, new_height), 0, 0, cv::INTER_LINEAR);
    } else {
        out = image;
    }
    return out;
}

  auto StereoVision::RectifyImages(cv::Mat const& left_image, cv::Mat const& right_image) const -> std::tuple<cv::Mat, cv::Mat> {
    cv::Mat left_image_rectified, right_image_rectified;
    cv::remap(left_image, left_image_rectified, this->undistort_rectify_map_left.first,
              this->undistort_rectify_map_left.second, cv::INTER_LINEAR);
    cv::remap(right_image, right_image_rectified, this->undistort_rectify_map_right.first,
              this->undistort_rectify_map_right.second, cv::INTER_LINEAR);
    return {left_image_rectified, right_image_rectified};
  }

  auto StereoVision::CalculateDisparityMap(cv::Mat const& left_image_rectified, cv::Mat const& right_image_rectified) const -> cv::Mat {
    cv::Mat left_image_filtered;
    cv::Mat right_image_filtered;
    //cv::medianBlur(left_image_rectified, left_image_filtered, 5);
    //cv::medianBlur(right_image_rectified, right_image_filtered, 5);
    //auto kernelGauss = cv::Size(5, 5);
    //cv::GaussianBlur(left_image_rectified, left_image_filtered, kernelGauss, 1.4);
    //cv::GaussianBlur(right_image_rectified, right_image_filtered, kernelGauss, 1.4);
    cv::bilateralFilter(left_image_rectified, left_image_filtered, 9, 75, 75);
    cv::bilateralFilter(right_image_rectified, right_image_filtered, 9, 75, 75);

    cv::Mat left_gray, right_gray;
    cv::cvtColor(left_image_filtered, left_gray, cv::COLOR_BGR2GRAY);
    cv::cvtColor(right_image_filtered, right_gray, cv::COLOR_BGR2GRAY);

    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(2.0, cv::Size(8, 8));
    clahe->apply(left_gray, left_gray);
    clahe->apply(right_gray, right_gray);

    cv::Mat sharpening_kernel = (cv::Mat_<float>(3, 3) << 0, -1, 0, -1, 5, -1, 0, -1, 0);
    cv::filter2D(left_gray, left_gray, left_gray.depth(), sharpening_kernel);
    cv::filter2D(right_gray, right_gray, right_gray.depth(), sharpening_kernel);

    cv::imshow("filtered Image Left", left_gray);
    cv::imshow("filtered Image right", right_gray);

    cv::Ptr<cv::StereoSGBM> stereo = cv::StereoSGBM::create();
    stereo->setBlockSize(5);
    stereo->setMinDisparity(0);
    stereo->setNumDisparities(12 * 16);
    stereo->setSpeckleRange(32);
    stereo->setSpeckleWindowSize(150);
    stereo->setUniquenessRatio(15);
    stereo->setPreFilterCap(15);
    stereo->setDisp12MaxDiff(1);

    cv::Mat disparity;
    stereo->compute(left_gray, right_gray, disparity);

    disparity.setTo(cv::Scalar(0), disparity == -1);
    //cv::filterSpeckles(disparity, 0, 50, 8);

    cv::Mat kernelMorph = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
    cv::morphologyEx(disparity, disparity, cv::MORPH_CLOSE, kernelMorph);

    cv::Mat disparity_normalized;
    cv::normalize(disparity, disparity_normalized, 0, 255, cv::NORM_MINMAX, CV_8U);
    cv::imshow("Disparity Map", disparity_normalized);

    return disparity;
  }

  auto StereoVision::CalculateDistanceMap(cv::Mat const& disparity) const -> cv::Mat {
      cv::Mat depth(disparity.size(), CV_32F);
      float focal_length = this->settings_.stereo_camera_info.camera_matrix_left.at<double>(0, 0);   // in pixel
      float baseline = this->settings_.stereo_camera_info.T.at<double>(0, 0);   // in millimeters

      for (int y = 0; y < disparity.rows; ++y) {
          for (int x = 0; x < disparity.cols; ++x) {
              auto disparityValue = disparity.at<int16_t>(y, x);
              depth.at<float>(y, x) = (disparityValue > 0) ? (focal_length * baseline) / disparityValue: 0.0f;
          }
      }

      std::cout << depth << std::endl;

      return depth;
  }

} // namespace stereo_vision
