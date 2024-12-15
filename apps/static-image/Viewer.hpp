#ifndef VIEWER_HPP
#define VIEWER_HPP

#include <opencv2/opencv.hpp>

#include <stereo-vision-lib/lib.hpp>

namespace stereo_vision {

class Viewer {
public:
  /**
   * Process a single pair of stereo images and show the result, both in the GUI and printed to the console.
   *
   * @param stereo_vis stereo vision algorithm used to process the images.
   * @param left_image image for the left camera. must have been taken at the same time as the right image.
   * @param right_image image for the rigt camera. must have been taken at the same time as the left image.
   */
    void ProcessImagePair(StereoVision const& stereo_vis, cv::Mat const& left_image, cv::Mat const& right_image) const;
  };

} // stereo_vision

#endif //VIEWER_HPP
