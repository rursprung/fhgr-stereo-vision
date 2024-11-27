#ifndef CALIBRATIONRUN_HPP
#define CALIBRATIONRUN_HPP

#include <filesystem>
#include <vector>
#include <ranges>
#include <optional>

#include <opencv2/opencv.hpp>

#include <stereo-vision-lib/StereoCameraInfo.hpp>

namespace stereo_vision::calibration {

  /// The maximum acceptable reprojection error. Anything above that will trigger a warning before saving the calibration result.
  double constexpr kMaxReprojectionError = 0.5f;

  /// Contains the data needed for the calibration.
  struct BoardInformation {
    /// The size of the board - for ChArUco boards this is the amount of squares in x & y direction.
    cv::Size board_size;
    /// The side length of one square **in mm**.
    float square_length;
    /// The side length of one marker **in mm**.
    float marker_length;
    /// The dictionary used to generate the markers.
    cv::aruco::PredefinedDictionaryType aruco_marker_dictionary;
    /// Whether the board is using the legacy pattern (starting with a white square in the top-left corner).
    /// See `cv::aruco::CharucoBoard::setLegacyPattern` for more details.
    bool legacy_pattern;
  };

  /// Small wrapper to read `BoardInformation` from the specified config file.
  BoardInformation LoadBoardInformationFromConfigFile(std::filesystem::path const& config_file_path);

  void OptionallyStoreCalibrationResult(StereoCameraInfo const& stereo_camera_info, std::filesystem::path const& calibration_result_file_path);

  std::ostream &operator<<(std::ostream &os, BoardInformation const& bi);

  /// API defined by OpenCV for `cv::FileStorage` interaction.
  /// See: https://docs.opencv.org/4.x/dd/d74/tutorial_file_input_output_with_xml_yml.html
  static void read(cv::FileNode const& node, BoardInformation& x, const BoardInformation& default_value = {});

  class CalibrationRun {
  public:
    struct Config {
      /// Should progress be reported on the console?
      bool report_progress = true;
      /// Should images used for calibration be shown on screen (with the found board & markers drawn into it)?
      bool show_images = false;
      /// Only relevant if `report_progress` is `true`: how long should be the delay between two images?
      /// If set to 0 it will wait indefinitely for a key press!
      uint8_t wait_time = 1;
    };

    /**
     * Instantiate a new calibration run but does not yet start it.
     *
     * @param config Defines the behaviour of the calibration run.
     * @param board_information Contains information on the board used for the calibration images. If this does not match the board used the calibration will fail!
     */
    explicit CalibrationRun(Config config, BoardInformation board_information);

    /**
     * Run a calibration based on images stored in a folder.
     * @param image_folder_path A containing two sub-folders: `left` and `right`, each of which contains the images for the respective camera. The images for left/right must be sorted the same way, otherwise the calibration will fail!
     * @return the calibration result.
     */
    [[nodiscard]]
    auto RunCalibration(std::filesystem::path const& image_folder_path) -> StereoCameraInfo;

    [[nodiscard]]
    auto RunCalibration(std::filesystem::path const& folder_path, cv::VideoCapture& cap_left, cv::VideoCapture& cap_right) -> StereoCameraInfo;

    /**
     * Run a calibration based on a range of images.
     * @param images_left  A range of images for the left camera. Must be sorted the same way as the images for the right camera, otherwise the calibration will fail!
     * @param images_right A range of images for the right camera. Must be sorted the same way as the images for the left camera, otherwise the calibration will fail!
     * @return the calibration result.
     */
    [[nodiscard]]
    auto RunCalibration(std::ranges::range auto const& images_left, std::ranges::range auto const& images_right) -> StereoCameraInfo;
  private:
    /// Defines the behaviour of the calibration run.
    Config const config_;
    /// Contains information on the board used for the calibration images.
    BoardInformation const board_information_;
    /// Board used for calibration.
    cv::aruco::CharucoBoard board_;
    /// Board detector used for calibration.
    cv::aruco::CharucoDetector board_detector_;

    /// The size of the images. Must be the same for all images. Set based on the first image being processed.
    cv::Size image_size_;

    std::vector<std::vector<cv::Point3f>> object_points_left_, object_points_right_;
    std::vector<std::vector<cv::Point2f>> image_points_left_, image_points_right_;

    /**
     * Find the ChArUco board in an image and calculate the object points and image points.
     * @param image The image to be processed. Must already be rectified.
     * @param window_title Title of the window in case the `show_images` config option is enabled (unused otherwise).
     * @return A tuple containing the object points and image points discovered in the image or `std::nullopt` if no
     *         ChArUco board was found or it was found incomplete (this is not considered an error but the image must be ignored).
     */
    [[nodiscard]]
    auto ProcessImage(cv::Mat const& image, std::string const& window_title) -> std::optional<std::tuple<std::vector<cv::Point3f>, std::vector<cv::Point2f>>>;

    void ProcessImagePair(cv::Mat const& image_left, cv::Mat const& image_right);

    [[nodiscard]]
    auto CalculateCalibration() const -> StereoCameraInfo;
  };

}


#endif //CALIBRATIONRUN_HPP
