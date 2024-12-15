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

  void Viewer::ProcessImagePair(StereoVision const& stereo_vis, cv::Mat const& left_image, cv::Mat const& right_image) {
    auto const& images = stereo_vis.RescaleAndRectifyImages(left_image, right_image);
    auto const search_points = this->SelectPoints(images);

    auto const& result = stereo_vis.AnalyzeAndAnnotateImage(left_image, right_image, search_points);

    if (!result) {
      std::cerr << "Failed to analyze the images: " << result.error().ToString() << std::endl;
      cv::destroyAllWindows();
      return;
    }

    std::cout << "Analysis result:" << std::endl;
    std::cout << *result << std::endl;

    cv::imshow("left", result->left_image);
    cv::imshow("right", result->right_image);
    cv::waitKey();
  }

} // stereo_vision
