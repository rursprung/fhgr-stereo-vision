#ifndef STEREO_VISION_LIB_HPP
#define STEREO_VISION_LIB_HPP

#include <expected>

#include <opencv2/opencv.hpp>

#include "StereoCameraInfo.hpp"

namespace stereo_vision {

  /**
   * All errors which may occur during analysis of the image.
   *
   * Note that this is a class wrapping an enum (instead of an `enum class`) to be able to provide methods on the
   * values.
   */
  class AnalysisError {
  public:
    /**
     * Implementation Detail. Use `AnalysisError` to access the enum constants and interact with them.
     *
     * Ensure that you add any value listed here also to `AnalysisError::ToString`!
     */
    enum Value {
      /// The provided image is invalid (e.g. empty / no data).
      kInvalidImage,
    };

    AnalysisError() = default;
    constexpr AnalysisError(Value const value) : value_(value) {}
    constexpr explicit operator Value() const { return value_; }
    bool operator==(Value const value) const { return this->value_ == value; }
    explicit operator bool() const = delete;

    [[nodiscard]] auto ToString() const -> std::string;

    explicit operator std::string() const;

  private:
    Value value_;
  };

  /**
   * The analysis results for processed stereo images.
   */
  struct AnalysisResult {
    /// The rectified image from the left camera.
    cv::Mat const left_image;
    /// The rectified image from the right camera.
    cv::Mat const right_image;
  };

  /// Print detailed information about the result to an output stream.
  std::ostream &operator<<(std::ostream& o, AnalysisResult const& analysis_result);

  class StereoVision {
  public:
    struct Settings {
      /// The calibration data previously generated using the calibration application.
      /// Must match the camera used in this run!
      StereoCameraInfo const stereo_camera_info;
    };

    explicit StereoVision(Settings settings);

    /**
     * Analyse an image for the presence of bananas and their properties.
     *
     * @param left_image the left image of the stereo cameras, taken at the same time as `right_image`.
     * @param right_image the right image of the stereo cameras, taken at the same time as `left_image`.
     * @return the result of the analysis and the annotated image, see the description of {@link AnalysisResult} for
     * more details.
     */
    [[nodiscard]]
    auto AnalyzeAndAnnotateImage(cv::Mat const& left_image, cv::Mat const& right_image) const -> std::expected<AnalysisResult, AnalysisError>;

  private:
    /// All externally configurable settings used by the analyzer.
    Settings const settings_;
    /// Undistortion and rectification transformation map for the left camera.
    /// See `cv::initUndistortRectifyMap` for more details.
    std::pair<cv::Mat, cv::Mat> undistort_rectify_map_left;
    /// Undistortion and rectification transformation map for the right camera.
    /// See `cv::initUndistortRectifyMap` for more details.
    std::pair<cv::Mat, cv::Mat>  undistort_rectify_map_right;

    /**
     * Rectify a stereo image pair.
     *
     * @param left_image The unrectified left image.
     * @param right_image The unrectified right image.
     * @return A tuple with the rectified image pair. The first value is the left image, the second the right.
     */
    [[nodiscard]]
    auto RectifyImages(cv::Mat const& left_image, cv::Mat const& right_image) const -> std::tuple<cv::Mat, cv::Mat>;
  };

} // namespace stereo_vision

#endif // STEREO_VISION_LIB_HPP
