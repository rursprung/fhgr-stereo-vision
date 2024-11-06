#include <gtest/gtest.h>

#include <stereo-vision-lib/lib.hpp>

TEST(GeneralStereoVisionTestSuite, FailOnNonExistingImage) {
    stereo_vision::StereoVision const analyzer{{
    }};
    auto const image = cv::imread("non-existent-image.jpg");
    auto const result = analyzer.AnalyzeAndAnnotateImage(image, image);
    ASSERT_FALSE(result);
    ASSERT_EQ(stereo_vision::AnalysisError::kInvalidImage, result.error());
}
