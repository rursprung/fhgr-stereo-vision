#include <iostream>
#include <filesystem>
#include <format>
#include <vector>
#include <ranges>

#include <opencv2/opencv.hpp>
#include <opencv2/core/utils/logger.hpp>

#include <gcc-bug-117560-workaround.hpp>
#include <stereo-vision-lib/StereoCameraInfo.hpp>

/// Temporary helper to help with re-scaling for easy display.
void ShowImage(auto const& title, auto const& image) {
  cv::Mat out;
  if (image.size().width > 800) {
    auto const scale = 800.0f / image.size().width;
    cv::resize(image, out, {}, scale, scale);
  } else {
    out = image;
  }
  cv::imshow(title, out);
}

[[nodiscard]]
auto GetDirectoryPathFromArgs(int const argc, char const *const argv[]) -> std::filesystem::path {
  if (argc != 2) {
    throw std::runtime_error(std::format("expected 1 argument but got {}!", argc - 1));
  }

  std::filesystem::path const path{argv[1]};
  if (!std::filesystem::exists(path)) {
    throw std::runtime_error(std::format("specified path does not exist: {}", path.string()));
  }
  if (!std::filesystem::is_directory(path)) {
    throw std::runtime_error(std::format("specified path is not a directory: {}", path.string()));
  }

  return path;
}

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

/// API defined by OpenCV for `cv::FileStorage` interaction.
/// See: https://docs.opencv.org/4.x/dd/d74/tutorial_file_input_output_with_xml_yml.html
static void read(cv::FileNode const& node, BoardInformation& x, const BoardInformation& default_value = {}){
  if(node.empty()) {
    x = default_value;
  } else {
    node["board_size"] >> x.board_size;
    node["square_length"] >> x.square_length;
    node["marker_length"] >> x.marker_length;
    node["aruco_marker_dictionary"] >> x.aruco_marker_dictionary;
    node["legacy_pattern"] >> x.legacy_pattern;
  }
}

/// Small wrapper to read `BoardInformation` from the specified config file.
BoardInformation LoadBoardInformationFromConfigFile(std::filesystem::path const& config_file_path) {
  BoardInformation board_information;
  cv::FileStorage config_file{config_file_path.string(), cv::FileStorage::READ};
  config_file["board"] >> board_information;
  return board_information;
}

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
  auto RunCalibration(std::filesystem::path const& image_folder_path) -> stereo_vision::StereoCameraInfo;

  /**
   * Run a calibration based on a range of images.
   * @param images_left  A range of images for the left camera. Must be sorted the same way as the images for the right camera, otherwise the calibration will fail!
   * @param images_right A range of images for the right camera. Must be sorted the same way as the images for the left camera, otherwise the calibration will fail!
   * @return the calibration result.
   */
  [[nodiscard]]
  auto RunCalibration(std::ranges::range auto const& images_left, std::ranges::range auto const& images_right) -> stereo_vision::StereoCameraInfo;
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

  void ProcessImage(cv::Mat const& image, std::vector<std::vector<cv::Point3f>>& all_object_points, std::vector<std::vector<cv::Point2f>>& all_image_points, std::string const& window_title);
};

CalibrationRun::CalibrationRun(Config const config, BoardInformation const board_information):
      config_(config),
      board_information_(board_information),
      board_({board_information.board_size, board_information.square_length, board_information.marker_length, cv::aruco::getPredefinedDictionary(board_information.aruco_marker_dictionary)}),
      board_detector_({board_}) {
  this->board_.setLegacyPattern(board_information.legacy_pattern);
};

void CalibrationRun::ProcessImage(cv::Mat const& image, std::vector<std::vector<cv::Point3f>>& all_object_points, std::vector<std::vector<cv::Point2f>>& all_image_points, std::string const& window_title) {
  auto const& board = this->board_detector_.getBoard();

    if (this->config_.report_progress) {
      std::cout << ".";
    }
    if (this->image_size_.empty()) {
      this->image_size_ = image.size();
    } else {
      if (this->image_size_ != image.size()) {
        std::cerr << "the image does not match the size of the first image! all images must have the same size!" << std::endl;
        return;
      }
    }

    std::vector<int> marker_ids;
    std::vector<int> current_charuco_ids;
    std::vector<std::vector<cv::Point2f>> marker_corners;
    std::vector<cv::Point2f> current_charuco_corners;
    std::vector<cv::Point3f> current_object_points;
    std::vector<cv::Point2f> current_image_points;
    this->board_detector_.detectBoard(image, current_charuco_corners, current_charuco_ids, marker_corners, marker_ids);

    if (this->config_.show_images) {
      cv::aruco::drawDetectedMarkers(image, marker_corners, marker_ids);
      cv::aruco::drawDetectedCornersCharuco(image, current_charuco_corners, current_charuco_ids);
      ShowImage(window_title, image);
      cv::waitKey(this->config_.wait_time);
    }

    if(marker_ids.empty() || current_charuco_ids.empty() || current_charuco_corners.size() < 3) {
      std::cerr << "WARNING: image does not contain a recognizable ChArUco board!" << (marker_ids.empty() ? "" : " but markers were found => legacy mode might have to be enabled)!") << std::endl;
      return;
    }

    board.matchImagePoints(current_charuco_corners, current_charuco_ids, current_object_points, current_image_points);
    all_image_points.push_back(current_image_points);
    all_object_points.push_back(current_object_points);
}

auto CalibrationRun::RunCalibration(std::filesystem::path const& image_folder_path) -> stereo_vision::StereoCameraInfo {
  auto const folder_left = image_folder_path / "left";
  auto const folder_right = image_folder_path / "right";
  if (!std::filesystem::exists(folder_left) || !std::filesystem::exists(folder_right)) {
    throw std::runtime_error("the image folder must contain a folder called 'left' and one called 'right'!");
  }

  auto const directory_entry_to_image = [](std::filesystem::directory_entry const& e) -> cv::Mat {
    return cv::imread(e.path().string(), cv::IMREAD_COLOR);
  };

  auto images_left = std::filesystem::directory_iterator{folder_left} | std::views::transform(directory_entry_to_image);
  auto images_right = std::filesystem::directory_iterator{folder_right} | std::views::transform(directory_entry_to_image);

  return this->RunCalibration(images_left, images_right);
};

auto CalibrationRun::RunCalibration(std::ranges::range auto const& images_left, std::ranges::range auto const& images_right) -> stereo_vision::StereoCameraInfo {
  std::vector<std::vector<cv::Point3f>> object_points_left, object_points_right;
  std::vector<std::vector<cv::Point2f>> image_points_left, image_points_right;

  if (this->config_.report_progress) {
    std::cout << "Processing images";
  }

  for (auto const& [image_left, image_right] : std::views::zip(images_left, images_right)) {
    this->ProcessImage(image_left, object_points_left, image_points_left, "ongoing calibration, left");
    this->ProcessImage(image_right, object_points_right, image_points_right, "ongoing calibration, right");

    if (this->config_.report_progress) {
      std::cout << ".";
    }
  }

  if (this->config_.report_progress) {
    std::cout << " done" << std::endl;
    std::cout << "Calculating...";
  }

  auto const flags = cv::CALIB_FIX_K3 | cv::CALIB_FIX_K4 | cv::CALIB_FIX_K5 | cv::CALIB_FIX_K6 | cv::CALIB_ZERO_TANGENT_DIST;
  cv::Mat camera_matrix_left, camera_matrix_right, dist_coeffs_left, dist_coeffs_right, R, T, E, F, per_view_errors;
  auto const reprojection_error = cv::stereoCalibrate(object_points_left, image_points_left, image_points_right, camera_matrix_left, dist_coeffs_left, camera_matrix_right, dist_coeffs_right, this->image_size_, R, T, E, F, per_view_errors, flags);

  if (this->config_.report_progress) {
    std::cout << " done" << std::endl;
  }

  return {
    .image_size = this->image_size_,
    .camera_matrix_left = camera_matrix_left,
    .camera_matrix_right = camera_matrix_right,
    .dist_coeffs_left = dist_coeffs_left,
    .dist_coeffs_right = dist_coeffs_right,
    .R = R,
    .T = T,
    .E = E,
    .F = F,
    .reprojection_error = reprojection_error,
  };
}

bool ReadYesNoFromConsole() {
  char c;
  while (true) {
    std::cin >> c;
    std::cin.clear();
    if (c == 'y' || c == 'Y') {
      return true;
    }
    if (c == 'n' || c == 'N') {
      return false;
    }
    std::cerr << "expected 'Y' or 'N', please input again: ";
  }
}

void OptionallyStoreCalibrationResult(stereo_vision::StereoCameraInfo const& stereo_camera_info, std::filesystem::path const& calibration_result_file_path) {
  if (std::filesystem::exists(calibration_result_file_path)) {
    std::cout << "A file for the calibration result already exists in the target location (" << calibration_result_file_path.string() << ")! Do you want to overwrite this file (if not, the result will not be stored)? [Y/N] " << std::endl;
    if (!ReadYesNoFromConsole()) {
      std::cout << "The calibration result will not be stored." << std::endl;
      return;
    }
  }
  cv::FileStorage calibration_result_file{calibration_result_file_path.string(), cv::FileStorage::WRITE};
  calibration_result_file << "calibration" << stereo_camera_info;
  std::cout << "Calibration result written to " << calibration_result_file_path.string() << std::endl;
}


int main(int const argc, char const * const argv[]) {
  cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_WARNING); // stop OpenCV log spamming

  try {
    auto const path = GetDirectoryPathFromArgs(argc, argv);

    CalibrationRun::Config config{
      .show_images = true,
      .wait_time = 1,
    };
    CalibrationRun calibration_run{config, LoadBoardInformationFromConfigFile(path / "config.yml")};

    auto const stereo_camera_info = calibration_run.RunCalibration(path);
    std::cout << "Calibration result:" << std::endl << stereo_camera_info << std::endl;

    OptionallyStoreCalibrationResult(stereo_camera_info, path / "calibration.yml");

    auto const calibration_result_file_path = path / "calibration.yml";
  } catch (std::exception const& ex) {
    std::cerr << ex.what() << std::endl;
    std::cerr << "Usage: " << argv[0] << " [image_folder_path]" << std::endl;
    return 1;
  }

  return 0;
}
