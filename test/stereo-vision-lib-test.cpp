#include <gtest/gtest.h>

#include <stereo-vision-lib/lib.hpp>

TEST(GeneralStereoVisionTestSuite, FailOnNonExistingImage) {
    stereo_vision::StereoVision const analyzer{{
      .stereo_camera_info = stereo_vision::LoadStereoCameraInfo({"resources/calibration/lighthouse-simulation/calibration.yml"}),
    }};
    auto const image = cv::imread("non-existent-image.jpg");
    auto const result = analyzer.AnalyzeAndAnnotateImage(image, image);
    ASSERT_FALSE(result);
    ASSERT_EQ(stereo_vision::AnalysisError::kInvalidImage, result.error());
}

TEST(GeneralStereoVisionTestSuite, RectifyImages) {
  stereo_vision::StereoVision const analyzer{{
    .stereo_camera_info = stereo_vision::LoadStereoCameraInfo({"resources/calibration/lighthouse-simulation/calibration.yml"}),
  }};
  auto const left_image = cv::imread("resources/calibration/lighthouse-simulation/left/000.png");
  auto const right_image = cv::imread("resources/calibration/lighthouse-simulation/right/000.png");
  auto const result = analyzer.AnalyzeAndAnnotateImage(left_image, right_image);
  ASSERT_TRUE(result);
  cv::imwrite("left_correct.png", left_image);
  cv::imwrite("right_correct.png", right_image);
}
