#include <algorithm>
#include <ranges>
#include <stereo-vision-lib/lib.hpp>

namespace stereo_vision {

  auto AnalysisError::ToString() const -> std::string {
    switch (this->value_) {
      case kInvalidImage:
        return "invalid image!";
      case kInvalidSettings:
        return "Invalid settings!";
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
        this->settings_.stereo_camera_info.T, R_left, R_right, P_left, P_right, this->Q_, cv::CALIB_ZERO_DISPARITY, 1,
        this->settings_.stereo_camera_info.image_size, &this->valid_roi_left, &this->valid_roi_right);

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

    cv::Mat disparity;
    switch (this->settings_.algorithm) {
    case Settings::Algorithm::kMatchTemplate:
      disparity = this->CalculateDisparityMapAtSpecificPoints({search_points.first, search_points.second}, left_image_rectified, right_image_rectified);;
      break;
    case Settings::Algorithm::kORB:
      disparity = this->CalculateDisparityUsingFeatureExtractionAtSpecificPoints({search_points.first, search_points.second}, left_image_rectified, right_image_rectified);
      break;
    case Settings::Algorithm::kSGBM:
      disparity = this->CalculateDisparityMapUsingSGBM(left_image_rectified, right_image_rectified);
      break;
    default:
      std::cerr << "unknown algorithm: " << static_cast<int>(this->settings_.algorithm) << std::endl;
      return std::unexpected{AnalysisError::kInvalidSettings};
    }
    auto const& search_3d_points = this->Reproject2DPointsTo3D(search_points, disparity);

    std::optional<cv::Mat> depth_map = this->settings_.algorithm == Settings::Algorithm::kSGBM ? std::make_optional(this->CalculateDepthMapSimple(disparity)) : std::nullopt;

    return {{
      .left_image = left_image_rectified,
      .right_image = right_image_rectified,
      .points_3d = search_3d_points,
      .depth_map = depth_map,
    }};
  }

  auto StereoVision::RectifyImages(cv::Mat const& left_image, cv::Mat const& right_image) const -> std::pair<cv::Mat, cv::Mat> {
    cv::Mat left_image_rectified, right_image_rectified;
    cv::remap(left_image, left_image_rectified, this->undistort_rectify_map_left_.first,
              this->undistort_rectify_map_left_.second, cv::INTER_LINEAR);
    cv::remap(right_image, right_image_rectified, this->undistort_rectify_map_right_.first,
              this->undistort_rectify_map_right_.second, cv::INTER_LINEAR);

    cv::rectangle(left_image_rectified, this->valid_roi_left, {255, 255, 255}, 3);
    cv::rectangle(right_image_rectified, this->valid_roi_right, {255, 255, 255}, 3);

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
      std::vector<cv::Point> const &search_points,
      cv::Mat const &left_image_rectified,
      cv::Mat const &right_image_rectified) const -> cv::Mat {
    // define the left search roi
    auto const patch_size = (left_image_rectified.size().width / 20) & ~1; // see https://stackoverflow.com/a/4360378 for ~1
    cv::Mat disparity = cv::Mat::zeros(left_image_rectified.size(), CV_16S);

    for (auto searchPoint : search_points) {
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

  auto StereoVision::CalculateDisparityUsingFeatureExtractionAtSpecificPoints(std::vector<cv::Point> const& search_points, cv::Mat const& left_image_rectified, cv::Mat const& right_image_rectified) const -> cv::Mat {
    auto orb = cv::ORB::create();

    std::vector<cv::KeyPoint> kp_left, kp_right;
    cv::Mat descriptors_left, descriptors_right;
    orb->detectAndCompute(left_image_rectified, cv::noArray(), kp_left, descriptors_left);
    orb->detectAndCompute(right_image_rectified, cv::noArray(), kp_right, descriptors_right);

    cv::BFMatcher matcher{cv::NORM_HAMMING, true};

    std::vector<cv::DMatch> matches;
    matcher.match(descriptors_left, descriptors_right, matches);

    std::sort(matches.begin(), matches.end(), [](const cv::DMatch& a, const cv::DMatch& b) {
      return a.distance < b.distance;
    });

    auto const delta_y_threshold = 3.0f; // Maximum difference in y coordinate
    std::vector<cv::DMatch> horizontal_matches;

    for (auto const& match : matches) {
      auto const& ptLeft = kp_left[match.queryIdx].pt;
      auto const& ptRight = kp_right[match.trainIdx].pt;

      // Check if the y-coordinates are horizontal
      if (std::abs(ptLeft.y - ptRight.y) <= delta_y_threshold) {
        horizontal_matches.push_back(match);
      }
    }

    if (horizontal_matches.empty()) {
      std::cerr << "No valid horizontal match." << std::endl;
      return {};
    }

    std::vector<cv::DMatch> best_matches;
    std::vector<cv::KeyPoint> key_points_left, key_points_right;
    cv::Mat disparity = cv::Mat::zeros(left_image_rectified.size(), CV_16S);

    for(auto const& searchPoint: search_points) {
      auto min_distance = std::numeric_limits<double>::max();
      cv::DMatch best_match;

      for (const auto &match: horizontal_matches) {
        const cv::Point2f &point_left = kp_left[match.queryIdx].pt;
        auto distance = cv::norm(point_left - cv::Point2f(searchPoint));

        if (distance < min_distance) {
          min_distance = distance;
          best_match = match;
        }
      }

      key_points_left.emplace_back(kp_left[best_match.queryIdx].pt, 1.0f);
      key_points_right.emplace_back(kp_right[best_match.trainIdx].pt, 1.0f);
      best_matches.emplace_back(static_cast<int>(key_points_left.size() - 1), static_cast<int>(key_points_right.size() - 1), best_match.distance);

      disparity.at<int16_t>(searchPoint) = static_cast<int16_t>(kp_left[best_match.queryIdx].pt.x - kp_right[best_match.trainIdx].pt.x);
    }

    if (this->settings_.show_debug_info) {
      cv::Mat img_best_match;
      cv::drawMatches(left_image_rectified, key_points_left, right_image_rectified, key_points_right,
                      best_matches, img_best_match);
      cv::imshow("Best Match to the given Point", img_best_match);
    }

    return disparity;
  }

  auto StereoVision::CalculateDisparityMapUsingSGBM(cv::Mat const& left_image_rectified, cv::Mat const& right_image_rectified) const -> cv::Mat {
    cv::Mat left_image_filtered;
    cv::Mat right_image_filtered;
    cv::Size kernel_gauss{11, 11};
    cv::GaussianBlur(left_image_rectified, left_image_filtered, kernel_gauss, 1.4);
    cv::GaussianBlur(right_image_rectified, right_image_filtered, kernel_gauss, 1.4);

    cv::Mat left_gray, right_gray;
    cv::cvtColor(left_image_filtered, left_gray, cv::COLOR_BGR2GRAY);
    cv::cvtColor(right_image_filtered, right_gray, cv::COLOR_BGR2GRAY);

    // optionally we can also use CLAHE here to re-sharpen the images:
    //cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(1.5, cv::Size(8, 8));
    //clahe->apply(left_gray, left_gray);
    //clahe->apply(right_gray, right_gray);

    cv::Mat const sharpening_kernel = (cv::Mat_<float>(3, 3) << 0, -1, 0, -1, 5, -1, 0, -1, 0);
    cv::filter2D(left_gray, left_gray, left_gray.depth(), sharpening_kernel);
    cv::filter2D(right_gray, right_gray, right_gray.depth(), sharpening_kernel);

    auto stereo = cv::StereoSGBM::create();
    stereo->setBlockSize(7);
    stereo->setMinDisparity(0);
    stereo->setNumDisparities(14 * 16 - stereo->getMinDisparity());
    stereo->setSpeckleRange(2);
    stereo->setSpeckleWindowSize(0);
    stereo->setUniquenessRatio(15);
    stereo->setPreFilterCap(32);
    stereo->setDisp12MaxDiff(1);
    stereo->setP1(8 * stereo->getBlockSize() * stereo->getBlockSize());
    stereo->setP2(32 * stereo->getBlockSize() * stereo->getBlockSize());
    stereo->setMode(cv::StereoSGBM::MODE_HH4);

    cv::Mat disparity;
    stereo->compute(left_gray, right_gray, disparity);

    disparity.setTo(cv::Scalar(0), disparity == -1);
    cv::filterSpeckles(disparity, 0, 150, 32);

    auto const kernel_morph = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
    cv::morphologyEx(disparity, disparity, cv::MORPH_CLOSE, kernel_morph);

    if (this->settings_.show_debug_info) {
      cv::Mat disparity_normalized;
      cv::normalize(disparity, disparity_normalized, 0, 255, cv::NORM_MINMAX, CV_8U);
      cv::imshow("Disparity Map", disparity_normalized);
    }

    return disparity;
  }

  auto StereoVision::CalculateDepthMapSimple(cv::Mat const& disparity) const -> cv::Mat {
    // the "cleaner" solution would be to use cv::reprojectImageTo3D(disparity, points3D, this->Q_, true);

    cv::Mat depth{disparity.size(), CV_32F};
    auto baseline = static_cast<float>(std::abs(this->settings_.stereo_camera_info.T.at<double>(0, 0))); // in millimeters
    auto focal_length = static_cast<float>(this->settings_.stereo_camera_info.camera_matrix_left.at<double>(0, 0)); // in pixel

    for (int y = 0; y < disparity.rows; ++y) {
      for (int x = 0; x < disparity.cols; ++x) {
        auto disparity_value = disparity.at<int16_t>(y, x);
        depth.at<float>(y, x) = (disparity_value > 0) ? (baseline * focal_length) / static_cast<float>(disparity_value): 0.0f;
      }
    }

    return depth;
  }

} // namespace stereo_vision
