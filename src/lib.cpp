#include <algorithm>
#include <ranges>
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
    auto const vec = analysis_result.points_3d.second - analysis_result.points_3d.first;
    return o << "Distance between the two selected points: " << cv::norm(vec) << " mm" << std::endl
             << "Distance of first point to camera:        " << analysis_result.points_3d.first[2] << " mm" << std::endl
             << "Distance of second point to camera:       " << analysis_result.points_3d.second[2] << " mm" << std::endl;
  }

  StereoVision::StereoVision(Settings const settings) : settings_(std::move(settings)) {
    cv::Mat R_left, R_right, P_left, P_right;

    cv::stereoRectify(
        this->settings_.stereo_camera_info.camera_matrix_left, this->settings_.stereo_camera_info.dist_coeffs_left,
        this->settings_.stereo_camera_info.camera_matrix_right, this->settings_.stereo_camera_info.dist_coeffs_right,
        this->settings_.stereo_camera_info.image_size, this->settings_.stereo_camera_info.R,
        this->settings_.stereo_camera_info.T, R_left, R_right, P_left, P_right, this->Q_, cv::CALIB_ZERO_DISPARITY, 0,
        this->settings_.stereo_camera_info.image_size);

    cv::initUndistortRectifyMap(this->settings_.stereo_camera_info.camera_matrix_left,
                                this->settings_.stereo_camera_info.dist_coeffs_left, R_left, P_left,
                                this->settings_.stereo_camera_info.image_size, CV_16SC2,
                                this->undistort_rectify_map_left_.first, this->undistort_rectify_map_left_.second);
    cv::initUndistortRectifyMap(this->settings_.stereo_camera_info.camera_matrix_right,
                                this->settings_.stereo_camera_info.dist_coeffs_right, R_right, P_right,
                                this->settings_.stereo_camera_info.image_size, CV_16SC2,
                                this->undistort_rectify_map_right_.first, this->undistort_rectify_map_right_.second);
  }

  auto StereoVision::RescaleAndRectifyImages(cv::Mat const& left_image, cv::Mat const& right_image) const -> std::pair<cv::Mat, cv::Mat> {
    return this->RectifyImages(this->RescaleImage(left_image), this->RescaleImage(right_image));
  }

  auto StereoVision::AnalyzeAndAnnotateImage(cv::Mat const& left_image, cv::Mat const& right_image, std::pair<cv::Point, cv::Point> const& search_points) const -> std::expected<AnalysisResult, AnalysisError> {
    if (left_image.empty() || right_image.empty()) {
      return std::unexpected{AnalysisError::kInvalidImage};
    }

    auto const& [left_image_rectified, right_image_rectified] = this->RescaleAndRectifyImages(left_image, right_image);

    auto const disparity = this->CalculateDisparityMapAtSpecificPoints({search_points.first, search_points.second}, left_image_rectified, right_image_rectified);
    auto const& search_3d_points = this->Reproject2DPointsTo3D(search_points, disparity);

    return {{
      .left_image = left_image_rectified,
      .right_image = right_image_rectified,
      .points_3d = search_3d_points,
    }};
  }

  auto StereoVision::RectifyImages(cv::Mat const& left_image, cv::Mat const& right_image) const -> std::pair<cv::Mat, cv::Mat> {
    cv::Mat left_image_rectified, right_image_rectified;
    cv::remap(left_image, left_image_rectified, this->undistort_rectify_map_left_.first,
              this->undistort_rectify_map_left_.second, cv::INTER_LINEAR);
    cv::remap(right_image, right_image_rectified, this->undistort_rectify_map_right_.first,
              this->undistort_rectify_map_right_.second, cv::INTER_LINEAR);
    return {left_image_rectified, right_image_rectified};
  }

  auto StereoVision::RescaleImage(auto const& image) const -> cv::Mat {
    if (image.size().width == this->settings_.stereo_camera_info.image_size.width) {
      return image;
    }

    cv::Mat out;
    cv::resize(image, out, {this->settings_.stereo_camera_info.image_size.width, this->settings_.stereo_camera_info.image_size.height}, 0, 0, cv::INTER_LINEAR);
    return out;
  }

  auto StereoVision::CalculateDisparityMapAtSpecificPoints(
      std::vector<cv::Point> const &searchPoints,
      cv::Mat const &left_image_rectified,
      cv::Mat const &right_image_rectified) const -> cv::Mat {
    // define the left search roi
    auto const patch_size = (left_image_rectified.size().width / 20) & ~1; // see https://stackoverflow.com/a/4360378 for ~1
    cv::Mat disparity = cv::Mat::zeros(left_image_rectified.size(), CV_16S);

    for (auto searchPoint : searchPoints) {
      cv::Rect const roi_left{searchPoint.x - patch_size / 2,
                              searchPoint.y - patch_size / 2, patch_size,
                              patch_size};
      auto const patch = left_image_rectified(roi_left);

      // create a search area for the right side
      cv::Rect roi_right{0, searchPoint.y - patch_size / 2,
                         searchPoint.x + patch_size / 2, patch_size};
      roi_right = roi_right & cv::Rect{0, 0, right_image_rectified.rows,
                                       right_image_rectified.cols};

      // search for the associated area
      auto const search_area = right_image_rectified(roi_right);
      cv::Mat result;
      cv::matchTemplate(search_area, patch, result, cv::TM_CCOEFF_NORMED);

      // find the position of the best match
      double min_val, max_val;
      cv::Point min_loc, max_loc;
      cv::minMaxLoc(result, &min_val, &max_val, &min_loc, &max_loc);
      cv::Point found_point_right{roi_right.x + max_loc.x + patch_size / 2,
                                  roi_right.y + max_loc.y + patch_size / 2};

      disparity.at<int16_t>(searchPoint) =
          static_cast<int16_t>(searchPoint.x - found_point_right.x);

      // paint ROIs into images to be shown
      cv::rectangle(left_image_rectified, roi_left, {0, 255, 0}, 2);
      cv::Rect roi_found(found_point_right.x - patch_size / 2,
                         found_point_right.y - patch_size / 2, patch_size,
                         patch_size);
      cv::rectangle(right_image_rectified, roi_found, {0, 255, 0}, 2);
    }

    return disparity;
  }

  auto StereoVision::Reproject2DPointsTo3D(std::pair<cv::Point, cv::Point> const& points2D, cv::Mat const& disparity) const -> std::pair<cv::Vec3f, cv::Vec3f> {
    cv::Mat points3D;
    cv::reprojectImageTo3D(disparity, points3D, this->Q_, true);
    return {points3D.at<cv::Vec3f>(points2D.first), points3D.at<cv::Vec3f>(points2D.second)};
  }

} // namespace stereo_vision
