#ifndef STEREO_VISION_LIB_HPP
#define STEREO_VISION_LIB_HPP

#include <expected>
#include <vector>
#include <utility>
#include <optional>

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
      /// Some configuration in the settings is wrong.
      kInvalidSettings,
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
    /// The 3D reprojected points of the original input points.
    std::pair<cv::Vec3f, cv::Vec3f> const points_3d;
    /// The depth map where each pixel contains the physical distance (in mm) of that point from the camera.
    std::optional<cv::Mat> const depth_map;
  };

  /// Print detailed information about the result to an output stream.
  std::ostream &operator<<(std::ostream& o, AnalysisResult const& analysis_result);

  class StereoVision {
  public:
    struct Settings {
      enum class Algorithm {
        kMatchTemplate,
        kORB,
        kSGBM,
      };

      /// The calibration data previously generated using the calibration application.
      /// Must match the camera used in this run!
      StereoCameraInfo const stereo_camera_info;

      /// The algorithm to use to calculate the distances.
      Algorithm const algorithm = Algorithm::kMatchTemplate;

      /// Show debug information
      bool const show_debug_info = false;
    };

    explicit StereoVision(Settings settings);

    /**
     * Rectify a stereo image pair.
     *
     * @param left_image The unrectified left image.
     * @param right_image The unrectified right image.
     * @return A pair with the rectified image pair. The first value is the left image, the second the right.
     */
    [[nodiscard]]
    auto RescaleAndRectifyImages(cv::Mat const& left_image, cv::Mat const& right_image) const -> std::pair<cv::Mat, cv::Mat>;

    /**
     * Analyse the images for the measurement of the depicted object.
     *
     * @param left_image the left image of the stereo cameras, taken at the same time as `right_image`.
     * @param right_image the right image of the stereo cameras, taken at the same time as `left_image`.
     * @param search_points the points between which the distance should be calculated.
     * @return the result of the analysis and the annotated image, see the description of {@link AnalysisResult} for
     * more details.
     */
    [[nodiscard]]
    auto AnalyzeAndAnnotateImage(cv::Mat const& left_image, cv::Mat const& right_image, std::pair<cv::Point, cv::Point> const& search_points) const -> std::expected<AnalysisResult, AnalysisError>;

    /// Read the settings used by the analyzer.
    [[nodiscard]]
    Settings settings() const { return this->settings_; }

  private:
    /// All externally configurable settings used by the analyzer.
    Settings const settings_;
    /// Undistortion and rectification transformation map for the left camera.
    /// See `cv::initUndistortRectifyMap` for more details.
    std::pair<cv::Mat, cv::Mat> undistort_rectify_map_left_;
    /// Undistortion and rectification transformation map for the right camera.
    /// See `cv::initUndistortRectifyMap` for more details.
    std::pair<cv::Mat, cv::Mat>  undistort_rectify_map_right_;
    /// Disparity-to-depth mapping matrix (see `cv::stereoRectify` for further details).
    cv::Mat Q_;

    cv::Rect valid_roi_left, valid_roi_right;

    /**
     * Rectify a stereo image pair.
     *
     * @param left_image The unrectified left image.
     * @param right_image The unrectified right image.
     * @return A tuple with the rectified image pair. The first value is the left image, the second the right.
     */
    [[nodiscard]]
    auto RectifyImages(cv::Mat const& left_image, cv::Mat const& right_image) const -> std::pair<cv::Mat, cv::Mat>;

    /**
    * Scales the incoming image to the same width as the calibration images.
    * @param image to rescale
    * @return the rescaled image
    */
    [[nodiscard]]
    auto RescaleImage(auto const& image) const -> cv::Mat;

    /**
    * Finds the given patch in the right image and calculates the disparity for each point.
    *
    * @param search_points Points to be found in the right image.
    * @param left_image_rectified the left rectified image. this will be modified: the ROI will be painted into it.
    * @param right_image_rectified the right rectified image. this will be modified: the ROI will be painted into it.
    * @return the disparity map for the given points. all other points will be set to zero in the disparity map.
    */
    [[nodiscard]]
    auto CalculateDisparityMapAtSpecificPoints(std::vector<cv::Point> const& search_points, cv::Mat const& left_image_rectified, cv::Mat const& right_image_rectified) const -> cv::Mat;

    /**
     * @param points2D the 2D points which should be reprojected to 3D based on
     * the disparity map.
     * @param disparity the disparity map used for the reprojection.
     * @return the reprojected 3D points.
     */
    [[nodiscard]]
    auto Reproject2DPointsTo3D(std::pair<cv::Point, cv::Point> const& points2D, cv::Mat const& disparity) const -> std::pair<cv::Vec3f, cv::Vec3f>;

    /**
    * Finds the same features in the left and right image using ORB and calculates the disparity between them.
    *
    * @param search_points Points to search features around them.
    * @param left_image_rectified the left rectified image.
    * @param right_image_rectified the right rectified image.
    * @return the disparity map for the given points. all other points will be set to zero in the disparity map.
    */
    [[nodiscard]]
    auto CalculateDisparityUsingFeatureExtractionAtSpecificPoints(std::vector<cv::Point> const& search_points, cv::Mat const& left_image_rectified, cv::Mat const& right_image_rectified) const -> cv::Mat;

    /**
    * Calculates a disparity map from a stereo image pair.
    *
    * @param left_image_rectified the left rectified image.
    * @param right_image_rectified the right rectified image.
    * @return the disparity map for the full image.
    */
    [[nodiscard]]
    auto CalculateDisparityMapUsingSGBM(cv::Mat const& left_image_rectified, cv::Mat const& right_image_rectified) const -> cv::Mat;

    /**
    * Calculates the real distance in front of the camera for each point in the disparity map.
    * This is done using manual calculation rather than relying on OpenCV.
    *
    * @param disparity the disparity map.
    * @return the depth map with each pixel storing the distance from the camera at that position.
    */
    [[nodiscard]]
    auto CalculateDepthMapSimple(cv::Mat const& disparity) const -> cv::Mat;
  };

} // namespace stereo_vision

#endif // STEREO_VISION_LIB_HPP
