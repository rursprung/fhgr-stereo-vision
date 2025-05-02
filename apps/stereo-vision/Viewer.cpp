#include "Viewer.hpp"

namespace stereo_vision {

  void Viewer::MouseCallback(int event, int x, int y, int flags, void* param) {
    static_cast<Viewer*>(param)->MouseCallback(event, x, y, flags);
  }

  void Viewer::MouseCallback(int const event, int const x, int const y, int const flags) {
    if (event != cv::EVENT_LBUTTONDOWN) {
      return;
    }

    this->selected_points_.emplace_back(x, y);
  }

  auto Viewer::SelectPoints(auto const& images) -> auto {
    std::cout << "Please select two points by clicking into the left image. The distance between the two will be calculated afterwards." << std::endl;

    cv::imshow("left", images.first);
    cv::imshow("right", images.second);
    this->selected_points_.clear();
    cv::setMouseCallback("left", MouseCallback, this);
    while (this->selected_points_.size() < 2) {
      if (!this->selected_points_.empty()) {
        cv::circle(images.first, this->selected_points_.front(), images.first.size().width / 128, {0, 255, 0}, -1);
        cv::imshow("left", images.first);
      }
      cv::waitKey(5);
    }

    return std::pair{this->selected_points_.front(), this->selected_points_.back()};
  }

  void Viewer::ProcessImagePair(cv::Mat const& left_image, cv::Mat const& right_image) {
    auto const& images = this->stereo_vis_.RescaleAndRectifyImages(left_image, right_image);
    auto const search_points = this->SelectPoints(images);

    auto const& result = this->stereo_vis_.AnalyzeAndAnnotateImage(left_image, right_image, search_points);

    if (!result) {
      std::cerr << "Failed to analyze the images: " << result.error().ToString() << std::endl;
      cv::destroyAllWindows();
      return;
    }

    std::cout << "Analysis result:" << std::endl;
    std::cout << *result << std::endl;

    if (this->stereo_vis_.settings().algorithm != StereoVision::Settings::Algorithm::kMatchTemplate) {
      cv::circle(result->left_image, this->selected_points_.front(), images.first.size().width / 128, {0, 255, 0}, -1);
      cv::circle(result->left_image, this->selected_points_.back(), images.first.size().width / 128, {0, 255, 0}, -1);
    }

    cv::imshow("left", result->left_image);
    cv::imshow("right", result->right_image);
    if (result->depth_map) {
      cv::imshow("depth map", *result->depth_map);
    }
    cv::waitKey();
  }

  void Viewer::DisplayOnlyImagePair(cv::Mat const& left_image, cv::Mat const& right_image) const {
    auto const& [left_image_rectified, right_image_rectified] = this->stereo_vis_.RescaleAndRectifyImages(left_image, right_image);
    cv::imshow("left", left_image_rectified);
    cv::imshow("right", right_image_rectified);
  }

  void Viewer::SaveImagePair(cv::Mat const& left_image,
                             cv::Mat const& right_image,
                             std::filesystem::path const& root_path) {
    auto const& [left_image_rectified, right_image_rectified] = this->stereo_vis_.RescaleAndRectifyImages(left_image, right_image);
    auto const filename = std::format("{:04d}.jpg", this->saved_frame_id);
    std::cout << "saving images " << filename << " to " << root_path.string() << std::endl;
    cv::imwrite((root_path / "left" / filename).string(), left_image_rectified);
    cv::imwrite((root_path / "right" / filename).string(), right_image_rectified);
    ++this->saved_frame_id;
  }

  } // stereo_vision
