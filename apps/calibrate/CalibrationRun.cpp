#include <gcc-bug-117560-workaround.hpp>

#include "CalibrationRun.hpp"

namespace stereo_vision::calibration {

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

  std::ostream &operator<<(std::ostream &os, BoardInformation const& bi) {
    return os << "board configuration:" << std::endl
              << "  board_size = " << bi.board_size << std::endl
              << "  square_length = " << bi.square_length << std::endl
              << "  marker_length = " << bi.marker_length << std::endl
              << "  aruco_marker_dictionary = " << bi.aruco_marker_dictionary << std::endl
              << "  legacy_pattern = " << bi.legacy_pattern << std::endl;
  }


  /// API defined by OpenCV for `cv::FileStorage` interaction.
  /// See: https://docs.opencv.org/4.x/dd/d74/tutorial_file_input_output_with_xml_yml.html
  static void read(cv::FileNode const& node, BoardInformation& x, const BoardInformation& default_value) {
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


  CalibrationRun::CalibrationRun(Config const config, BoardInformation const board_information):
        config_(config),
        board_information_(board_information),
        board_({board_information.board_size, board_information.square_length, board_information.marker_length, cv::aruco::getPredefinedDictionary(board_information.aruco_marker_dictionary)}),
        board_detector_({board_}) {
    this->board_.setLegacyPattern(board_information.legacy_pattern);
  };

  auto CalibrationRun::ProcessImage(cv::Mat const& image, std::string const& window_title) -> std::optional<std::tuple<std::vector<cv::Point3f>, std::vector<cv::Point2f>>> {
    auto const& board = this->board_detector_.getBoard();

    if (this->config_.report_progress) {
      std::cout << ".";
    }
    if (this->image_size_.empty()) {
      this->image_size_ = image.size();
    }
    if (this->image_size_ != image.size() || this->image_size_.empty()) {
      std::cerr << "the image does not match the size of the first image! all images must have the same size!" << std::endl;
      return std::nullopt;
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
      if (marker_ids.empty() && !this->board_information_.legacy_pattern) {
        std::cerr << "WARNING: image does not contain a recognizable ChArUco board, but markers were found => legacy mode might have to be enabled)!" << std::endl;
      } else {
        std::cerr << "image does not contain a recognizable ChArUco board" << std::endl;
      }
      return std::nullopt;
    }

    board.matchImagePoints(current_charuco_corners, current_charuco_ids, current_object_points, current_image_points);

    if (current_image_points.size() != current_object_points.size()) {
      std::cerr << "current_image_points.size() = " << current_image_points.size() << " does not match current_object_points.size() = " << current_object_points.size() << "!" << std::endl;
      return std::nullopt;
    }
    return {{current_object_points, current_image_points}};
  }

  void CalibrationRun::ProcessImagePair(cv::Mat const& image_left, cv::Mat const& image_right) {
    auto const result_left = this->ProcessImage(image_left, "ongoing calibration, left");
    auto const result_right = this->ProcessImage(image_right, "ongoing calibration, right");

    if (result_left && result_right) {
      auto const& [opl, ipl] = *result_left;
      auto const& [opr, ipr] = *result_right;
      if (opl.size() != opr.size()) {
        std::cerr << "left & right didn't match the same amount of points (" << opl.size() << " vs. " << opr.size() << ") => skipping" << std::endl;
        return;
      }
      this->object_points_left_.push_back(opl);
      this->image_points_left_.push_back(ipl);
      this->object_points_right_.push_back(opr);
      this->image_points_right_.push_back(ipr);
    }

    if (this->config_.report_progress) {
      std::cout << ".";
    }
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

  auto CalibrationRun::RunCalibration(std::filesystem::path const& folder_path, cv::VideoCapture& cap_left, cv::VideoCapture& cap_right) -> StereoCameraInfo {
    size_t i = 0;
    while (true) {
      cv::Mat image_left, image_right;
      cap_left >> image_left;
      cap_right >> image_right;
      auto const key = cv::pollKey();
      if (key == 'q') {
        break;
      }
      if (key != ' ') {
        ShowImage("ongoing calibration, left", image_left);
        ShowImage("ongoing calibration, right", image_right);
        continue;
      }

      auto path_left = folder_path / "left" / std::format("{:0>3d}.jpg", i);
      cv::imwrite(path_left.string(), image_left);
      auto path_right = folder_path / "right" / std::format("{:0>3d}.jpg", i);
      cv::imwrite(path_right.string(), image_right);
      ++i;

      this->ProcessImagePair(image_left, image_right);
    }

    return this->CalculateCalibration();
  }

  auto CalibrationRun::CalculateCalibration() const -> StereoCameraInfo {
    if (this->config_.report_progress) {
      std::cout << "Calculating using " << this->object_points_left_.size() << " usable images...";
    }

    auto const flags = cv::CALIB_FIX_K3 | cv::CALIB_FIX_K4 | cv::CALIB_FIX_K5 | cv::CALIB_FIX_K6 | cv::CALIB_ZERO_TANGENT_DIST;
    cv::Mat camera_matrix_left, camera_matrix_right, dist_coeffs_left, dist_coeffs_right, R, T, E, F, per_view_errors;
    auto const reprojection_error = cv::stereoCalibrate(this->object_points_left_, this->image_points_left_, this->image_points_right_, camera_matrix_left, dist_coeffs_left, camera_matrix_right, dist_coeffs_right, this->image_size_, R, T, E, F, per_view_errors, flags);

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

  auto CalibrationRun::RunCalibration(std::ranges::range auto const& images_left, std::ranges::range auto const& images_right) -> StereoCameraInfo {
    if (this->config_.report_progress) {
      std::cout << "Processing images";
    }

    for (auto const& [image_left, image_right] : std::views::zip(images_left, images_right)) {
      this->ProcessImagePair(image_left, image_right);
    }

    if (this->config_.report_progress) {
      std::cout << " done" << std::endl;
    }

    return this->CalculateCalibration();
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

  void OptionallyStoreCalibrationResult(StereoCameraInfo const& stereo_camera_info, std::filesystem::path const& calibration_result_file_path) {
    if (stereo_camera_info.reprojection_error > kMaxReprojectionError) {
      std::cout << std::format("Warning: the reprojection error ({:.3f}) exceeds the recommended upper limit of {:.3f}!", stereo_camera_info.reprojection_error, kMaxReprojectionError)
                << " This implies that the calibration result is not satisfactory and will most likely not result in a usable rectification! Do you want to safe this result anyway? (If not, the result will not be stored) [Y/N] " << std::endl;
    }
    if (std::filesystem::exists(calibration_result_file_path)) {
      std::cout << "A file for the calibration result already exists in the target location (" << calibration_result_file_path.string() << ")! Do you want to overwrite this file? If not, the result will not be stored) [Y/N] " << std::endl;
      if (!ReadYesNoFromConsole()) {
        std::cout << "The calibration result will not be stored." << std::endl;
        return;
      }
    }
    cv::FileStorage calibration_result_file{calibration_result_file_path.string(), cv::FileStorage::WRITE};
    calibration_result_file << "calibration" << stereo_camera_info;
    std::cout << "Calibration result written to " << calibration_result_file_path.string() << std::endl;
  }

}
