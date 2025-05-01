#include <opencv2/core/utils/logger.hpp>
#include <variant>

#include "calibration.hpp"

[[nodiscard]]
auto GetIntOrString(std::string const& s) -> std::variant<int, std::string> {
  auto const is_number = [] (std::string const& s) -> auto { return !s.empty() && std::ranges::all_of(s, ::isdigit); };

  if (is_number(s)) {
    return std::stoi(s);
  } else {
    return s;
  }
}

[[nodiscard]]
auto VideoCaptureFromIdOrString(std::variant<int, std::string> const& id) -> cv::VideoCapture {
  if (std::holds_alternative<int>(id)) {
    return cv::VideoCapture(std::get<int>(id));
  } else {
    return cv::VideoCapture(std::get<std::string>(id));
  }
}

[[nodiscard]]
auto GetDirectoryPathAndCamerasFromArgs(int const argc, char const *const argv[]) -> std::tuple<std::filesystem::path, std::variant<int, std::string>, std::variant<int, std::string>> {
  if (argc != 4) {
    throw std::runtime_error(std::format("expected 3 argument but got {}!", argc - 1));
  }

  std::filesystem::path const path{argv[1]};
  if (!std::filesystem::exists(path)) {
    throw std::runtime_error(std::format("specified path does not exist: {}", path.string()));
  }
  if (!std::filesystem::is_directory(path)) {
    throw std::runtime_error(std::format("specified path is not a directory: {}", path.string()));
  }

  return {path, GetIntOrString(argv[2]), GetIntOrString(argv[3])};
}

int main(int const argc, char const * const argv[]) {
  cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_WARNING); // stop OpenCV log spamming

  try {
    auto const [path, cap1_id, cap2_id] = GetDirectoryPathAndCamerasFromArgs(argc, argv);

    auto cap1 = VideoCaptureFromIdOrString(cap1_id);
    auto cap2 = VideoCaptureFromIdOrString(cap2_id);

    if (!cap1.isOpened() || !cap2.isOpened()) {
      throw std::runtime_error("Failed to open camera capture!");
    }

    stereo_vision::calibration::CalibrationRun::Config config{
      .show_images = true,
      .wait_time = 1,
    };
    auto const board_information = stereo_vision::calibration::BoardInformation::LoadFromConfigFile(path / "config.yml");
    std::cout << board_information;

    stereo_vision::calibration::CalibrationRun calibration_run{config, board_information};

    auto const stereo_camera_info = calibration_run.RunCalibration(path, cap1, cap2);
    std::cout << "Calibration result:" << std::endl << stereo_camera_info << std::endl;

    stereo_vision::calibration::OptionallyStoreCalibrationResult(stereo_camera_info, path / "calibration.yml");

    auto const calibration_result_file_path = path / "calibration.yml";
  } catch (std::exception const& ex) {
    std::cerr << ex.what() << std::endl;
    std::cerr << "Usage: " << argv[0] << " [image_folder_path] [left_camera_id|left_camera_url] [right_camera_id|left_camera_url]" << std::endl;
    return 1;
  }

  return 0;
}
