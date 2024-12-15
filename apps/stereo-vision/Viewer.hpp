#ifndef VIEWER_HPP
#define VIEWER_HPP

#include <opencv2/opencv.hpp>

#include <stereo-vision-lib/lib.hpp>
#include <utility>

namespace stereo_vision {

  class Viewer {
  public:
    explicit Viewer(StereoVision  stereo_vis) : stereo_vis_(std::move(stereo_vis)) {};

    /**
     * Process a single pair of stereo images and show the result, both in the GUI and printed to the console.
     *
     * @param left_image image for the left camera. must have been taken at the same time as the right image.
     * @param right_image image for the rigt camera. must have been taken at the same time as the left image.
     */
    void ProcessImagePair(cv::Mat const& left_image, cv::Mat const& right_image);

    /**
     * Display a single pair of stereo images without processing them.
     *
     * @param left_image image for the left camera. must have been taken at the same time as the right image.
     * @param right_image image for the rigt camera. must have been taken at the same time as the left image.
     */
    void DisplayOnlyImagePair(cv::Mat const& left_image, cv::Mat const& right_image) const;

  private:
    stereo_vision::StereoVision const stereo_vis_;
    static void MouseCallback(int event, int x, int y, int flags, void* param);
    void MouseCallback(int event, int x, int y, int flags);
    auto SelectPoints(auto const& images) -> auto;

    std::vector<cv::Point> selected_points_;
  };

} // stereo_vision

#endif //VIEWER_HPP
