#include "Viewer.hpp"

namespace stereo_vision {

  void Viewer::ProcessImagePair(StereoVision const& stereo_vis, cv::Mat const& left_image, cv::Mat const& right_image) const {
    std::pair<cv::Point, cv::Point> const search_points = {{330, 350}, {430, 350}}; // TODO: make this user-selectable
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
