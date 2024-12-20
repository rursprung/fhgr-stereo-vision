#include <filesystem>
#include <format>
#include <iostream>
#include <stdexcept>
#include <ranges>

#include <opencv2/opencv.hpp>
#include <opencv2/core/utils/logger.hpp>

#include <stereo-vision-lib/lib.hpp>
#include "Viewer.hpp"

[[nodiscard]]
auto GetConfigPathAndCamerasFromArgs(int const argc, char const *const argv[]) -> std::tuple<std::filesystem::path, int, int> {
  if (argc != 4) {
    throw std::runtime_error(std::format("expected 3 argument but got {}!", argc - 1));
  }

  std::filesystem::path const path{argv[1]};
  if (!std::filesystem::exists(path)) {
    throw std::runtime_error(std::format("specified path does not exist: {}", path.string()));
  }
  if (std::filesystem::is_directory(path)) {
    throw std::runtime_error(std::format("specified path is not a file: {}", path.string()));
  }

  auto const cap1 = std::stoi(argv[2]);
  auto const cap2 = std::stoi(argv[3]);

  return {path, cap1, cap2};
}

auto NewStereoVision(std::filesystem::path const& calibration_file_path) -> stereo_vision::StereoVision {
  return stereo_vision::StereoVision{
           {
             .stereo_camera_info = stereo_vision::LoadStereoCameraInfo(calibration_file_path),
             .algorithm = stereo_vision::StereoVision::Settings::Algorithm::kMatchTemplate,
             .show_debug_info = true,
           }
         };
}

/**
 * Show a continuous live stream from two cameras. Upon pressing any key the current frames
 * are used to analyse them. Pressing another key will continue the livestream afterward.
 *
 * @param viewer the image viewer to be used.
 * @param cap_left the left video capture device.
 * @param cap_right the right video capture device.
 */
void ProcessLiveStream(stereo_vision::Viewer& viewer, cv::VideoCapture& cap_left, cv::VideoCapture& cap_right) {
  std::cout << "Press 'q' to quit, ' ' (space) to freeze a frame and process it" << std::endl;

  while (true) {
    cv::Mat image_left, image_right;
    cap_left >> image_left;
    cap_right >> image_right;
    auto const key = cv::pollKey() & 0xFF;
    switch (key) {
    case 'q':
      return;
    case ' ':
      viewer.ProcessImagePair(image_left, image_right);
      break;
    default:
      viewer.DisplayOnlyImagePair(image_left, image_right);
    }
  }
}

int main(int const argc, char const * const argv[]) {
  cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_WARNING); // stop OpenCV log spamming

  try {
    auto const [calibration_file_path, cap1_id, cap2_id] = GetConfigPathAndCamerasFromArgs(argc, argv);

    stereo_vision::StereoVision stereo_vis{
             {
               .stereo_camera_info = stereo_vision::LoadStereoCameraInfo(calibration_file_path),
               .algorithm = stereo_vision::StereoVision::Settings::Algorithm::kMatchTemplate,
               .show_debug_info = true,
             }
    };
    stereo_vision::Viewer viewer{stereo_vis};
    cv::VideoCapture cap_left(cap1_id), cap_right(cap2_id);

    ProcessLiveStream(viewer, cap_left, cap_right);

  } catch (std::exception const& ex) {
    std::cerr << ex.what() << std::endl;
    std::cerr << "Usage: " << argv[0] << " [calibration_file_path] [left_camera_id] [right_camera_id]" << std::endl;
    return 1;
  }

  return 0;
}
