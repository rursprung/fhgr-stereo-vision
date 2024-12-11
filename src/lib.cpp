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
    cv::Mat R_left, R_right, P_left, P_right;

    cv::stereoRectify(
        this->settings_.stereo_camera_info.camera_matrix_left, this->settings_.stereo_camera_info.dist_coeffs_left,
        this->settings_.stereo_camera_info.camera_matrix_right, this->settings_.stereo_camera_info.dist_coeffs_right,
        this->settings_.stereo_camera_info.image_size, this->settings_.stereo_camera_info.R,
        this->settings_.stereo_camera_info.T, R_left, R_right, P_left, P_right, this->settings_.stereo_camera_info.Q, cv::CALIB_ZERO_DISPARITY, 0,
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
    std::vector<cv::Point> searchPoints = {cv::Point(330, 350), cv::Point(430, 350)};
    //auto disparity = this-> CalculateDisparityMap(left_image_rectified, right_image_rectified);
    auto disparity = this-> FindAssociatedMatch(searchPoints, left_image_rectified, right_image_rectified);
    std::cout << "Disparity P1: " << disparity.at<int16_t>(searchPoints[0]) << std::endl;
    std::cout << "Disparity P2: " << disparity.at<int16_t>(searchPoints[1]) << std::endl;
    auto disparityORB = this-> FeatureExtraction(searchPoints, left_image_rectified, right_image_rectified);
    std::cout << "Disparity with ORB P1: " << disparityORB.at<int16_t>(searchPoints[0]) << std::endl;
    std::cout << "Disparity with ORB P2: " << disparityORB.at<int16_t>(searchPoints[1]) << std::endl;
    auto distance = this->CalculateDepthMapSimple(disparity);
    std::cout << "depth map P1: " << distance.at<float>(searchPoints[0]) << std::endl;
    std::cout << "depth map P2: " << distance.at<float>(searchPoints[1]) << std::endl;
    auto distanceToPoint = this-> PointToPointDistance(searchPoints[0], searchPoints[1], disparity);

    return {{
      .left_image = left_image_rectified,
      .right_image = right_image_rectified,
      //.disparity_map = disparity,
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
    auto kernelGauss = cv::Size(11, 11);
    cv::GaussianBlur(left_image_rectified, left_image_filtered, kernelGauss, 1.4);
    cv::GaussianBlur(right_image_rectified, right_image_filtered, kernelGauss, 1.4);
    //cv::bilateralFilter(left_image_rectified, left_image_filtered, 9, 75, 75);
    //cv::bilateralFilter(right_image_rectified, right_image_filtered, 9, 75, 75);

    cv::Mat left_gray, right_gray;
    cv::cvtColor(left_image_filtered, left_gray, cv::COLOR_BGR2GRAY);
    cv::cvtColor(right_image_filtered, right_gray, cv::COLOR_BGR2GRAY);

    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(1.5, cv::Size(8, 8));
    //clahe->apply(left_gray, left_gray);
    //clahe->apply(right_gray, right_gray);

    cv::Mat sharpening_kernel = (cv::Mat_<float>(3, 3) << 0, -1, 0, -1, 5, -1, 0, -1, 0);
    cv::filter2D(left_gray, left_gray, left_gray.depth(), sharpening_kernel);
    cv::filter2D(right_gray, right_gray, right_gray.depth(), sharpening_kernel);

    cv::imshow("filtered Image Left", left_gray);
    cv::imshow("filtered Image right", right_gray);

    cv::Ptr<cv::StereoSGBM> stereo = cv::StereoSGBM::create();
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

    cv::Mat kernelMorph = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
    cv::morphologyEx(disparity, disparity, cv::MORPH_CLOSE, kernelMorph);

    cv::Mat disparity_normalized;
    cv::normalize(disparity, disparity_normalized, 0, 255, cv::NORM_MINMAX, CV_8U);
    cv::imshow("Disparity Map", disparity_normalized);

    return disparity;
  }

  auto StereoVision::FindAssociatedMatch(std::vector<cv::Point> const& searchPoints, cv::Mat const& left_image_rectified, cv::Mat const& right_image_rectified) const -> cv::Mat {
      // define the left search roi
      int patch_size = 22;
      cv::Mat disparity = cv::Mat::zeros(left_image_rectified.size(), CV_16S);

      for(auto searchPoint: searchPoints) {
          cv::Rect roi_left(searchPoint.x - patch_size / 2, searchPoint.y - patch_size / 2, patch_size, patch_size);
          cv::Mat patch = left_image_rectified(roi_left);

          // create a search area for the right side
          cv::Rect roi_right(0, searchPoint.y - patch_size / 2, searchPoint.x + patch_size / 2, patch_size);
          roi_right = roi_right & cv::Rect(0, 0, right_image_rectified.rows, right_image_rectified.cols);

          // search for the associated area
          cv::Mat search_area = right_image_rectified(roi_right);
          cv::Mat result;
          cv::matchTemplate(search_area, patch, result, cv::TM_CCOEFF_NORMED);

          // Find the position of the best match
          double min_val, max_val;
          cv::Point min_loc, max_loc;
          cv::minMaxLoc(result, &min_val, &max_val, &min_loc, &max_loc);
          cv::Point foundedPoint(roi_right.x + max_loc.x + patch_size / 2, roi_right.y + max_loc.y + patch_size / 2);

          disparity.at<int16_t>(searchPoint) = searchPoint.x - foundedPoint.x;

          cv::rectangle(left_image_rectified, roi_left, cv::Scalar(0, 255, 0), 2);
          //cv::rectangle(right_image_rectified, roi_right, cv::Scalar(0, 255, 0), 2);
          cv::Rect roi_founded(foundedPoint.x - patch_size / 2, foundedPoint.y - patch_size / 2, patch_size, patch_size);
          cv::rectangle(right_image_rectified, roi_founded, cv::Scalar(0, 255, 0), 2);
      }

      return disparity;
  }

  auto StereoVision::FeatureExtraction(std::vector<cv::Point> const& searchPoints, cv::Mat const& left_image_rectified, cv::Mat const& right_image_rectified) const -> cv::Mat {
      cv::Ptr<cv::ORB> orb = cv::ORB::create();

      std::vector<cv::KeyPoint> kpLeft, kpRight;
      cv::Mat desLeft, desRight;
      orb->detectAndCompute(left_image_rectified, cv::noArray(), kpLeft, desLeft);
      orb->detectAndCompute(right_image_rectified, cv::noArray(), kpRight, desRight);

      cv::BFMatcher matcher(cv::NORM_HAMMING, true);

      std::vector<cv::DMatch> matches;
      matcher.match(desLeft, desRight, matches);

      std::sort(matches.begin(), matches.end(), [](const cv::DMatch& a, const cv::DMatch& b) {
          return a.distance < b.distance;
      });

      const float deltaYThreshold = 3.0f; // Maximum difference in y coordinate
      std::vector<cv::DMatch> horizontalMatches;

      for (const auto& match : matches) {
          const cv::Point2f& ptLeft = kpLeft[match.queryIdx].pt;
          const cv::Point2f& ptRight = kpRight[match.trainIdx].pt;

          // Check if the y-coordinates are horizontal
          if (std::abs(ptLeft.y - ptRight.y) <= deltaYThreshold) {
              horizontalMatches.push_back(match);
          }
      }

      if (horizontalMatches.empty()) {
          std::cerr << "No valid horizontal match." << std::endl;
      }

      cv::Mat imgBestMatch;
      std::vector<cv::DMatch> bestMatches;
      std::vector<cv::KeyPoint> KeyPointsLeft;
      std::vector<cv::KeyPoint> KeyPointsRight;
      cv::Mat disparity = cv::Mat::zeros(left_image_rectified.size(), CV_16S);

      for(auto searchPoint: searchPoints) {
          auto minDistance = std::numeric_limits<double>::max();
          cv::DMatch bestMatch;

          for (const auto &match: horizontalMatches) {
              const cv::Point2f &ptLeft = kpLeft[match.queryIdx].pt;
              auto distance = cv::norm(ptLeft - cv::Point2f(searchPoint));

              if (distance < minDistance) {
                  minDistance = distance;
                  bestMatch = match;
              }
          }

          KeyPointsLeft.push_back(cv::KeyPoint(kpLeft[bestMatch.queryIdx].pt, 1.0f));
          KeyPointsRight.push_back(cv::KeyPoint(kpRight[bestMatch.trainIdx].pt, 1.0f));
          cv::DMatch Match(KeyPointsLeft.size() - 1, KeyPointsRight.size() - 1, bestMatch.distance);
          bestMatches.push_back(Match);

          disparity.at<int16_t>(searchPoint) = static_cast<int16_t>(kpLeft[bestMatch.queryIdx].pt.x - kpRight[bestMatch.trainIdx].pt.x);
      }

      cv::drawMatches(left_image_rectified, KeyPointsLeft, right_image_rectified, KeyPointsRight,
                      bestMatches, imgBestMatch);
      cv::imshow("Best Match to the given Point", imgBestMatch);

      return disparity;
  }

  auto StereoVision::CalculateDepthMapSimple(cv::Mat const& disparity) const -> cv::Mat {
      cv::Mat depth(disparity.size(), CV_32F);
      auto baseline = static_cast<float>(std::abs(this->settings_.stereo_camera_info.T.at<double>(0, 0)));   // in millimeters
      auto focal_length = static_cast<float>(this->settings_.stereo_camera_info.camera_matrix_left.at<double>(0, 0));   // in pixel
      std::cout << "Baseline: " << baseline << std::endl;
      std::cout << "Focal Length: " << focal_length << std::endl;

      for (int y = 0; y < disparity.rows; ++y) {
          for (int x = 0; x < disparity.cols; ++x) {
              auto disparityValue = disparity.at<int16_t>(y, x);
              depth.at<float>(y, x) = (disparityValue > 0) ? (baseline * focal_length) / static_cast<float>(disparityValue): 0.0f;
          }
      }

      return depth;
  }

  auto StereoVision::PointToPointDistance(cv::Point const& firstPoint, cv::Point const& secondPoint, cv::Mat const& disparity) const -> float {
      cv::Mat points3D;
      cv::reprojectImageTo3D(disparity, points3D, this->settings_.stereo_camera_info.Q, true);

      cv::Vec3f point1 = points3D.at<cv::Vec3f>(firstPoint);
      cv::Vec3f point2 = points3D.at<cv::Vec3f>(secondPoint.y, secondPoint.x);

      auto distance = static_cast<float>(std::sqrt(std::pow(point2[0] - point1[0], 2)
                                + std::pow(point2[1] - point1[1], 2)
                                + std::pow(point2[2] - point1[2], 2)));

      std::cout << "3D Point 1: (" << point1[0] << ", " << point1[1] << ", " << point1[2] << ")" << std::endl;
      std::cout << "3D Point 2: (" << point2[0] << ", " << point2[1] << ", " << point2[2] << ")" << std::endl;
      std::cout << "Distance between the two points: " << distance << " mm" << std::endl;

      return distance;
  }

} // namespace stereo_vision
