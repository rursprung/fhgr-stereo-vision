#include <filesystem>
#include <format>
#include <iostream>
#include <stdexcept>
#include <variant>
#include <string_view>
#include <expected>
#include <ranges>

#include <opencv2/opencv.hpp>
#include <opencv2/core/utils/logger.hpp>

#include <gcc-bug-117560-workaround.hpp>
#include <stereo-vision-lib/lib.hpp>

/// Input parameters if a folder was chosen.
struct FolderPath {
  /// A path to a folder containing two sub-folders `left` and `right` and optionally a `calibration.yml`.
  /// The sub-folders must contain the stereo images in matching order.
  std::filesystem::path const folder_path;
  /// The calibration file. Either explicitly specified on the command line or implicitly assumed to be `calibration.yml`.
  /// Must have been generated using the calibration tool before for the same camera setup as used to take the pictures.
  std::filesystem::path const calibration_file_path;
};

struct ImagePath {
  /// The path to the left image of the stereo pair.
  std::filesystem::path const left_path;
  /// The path to the right image of the stereo pair.
  std::filesystem::path const right_path;
  /// The calibration file.
  /// Must have been generated using the calibration tool before for the same camera setup as used to take the pictures.
  std::filesystem::path const calibration_file_path;
};

/// Possible input options.
using path_t = std::variant<FolderPath, ImagePath>;

/// Needed helper for `std::visit`, see https://en.cppreference.com/w/cpp/utility/variant/visit (taken from there)
template<class... Ts>
struct overloaded : Ts... { using Ts::operator()...; };

/**
 * Check if the specified path exists and either is or isn't a folder. Throws an exception if the path did not meet
 * expectations!
 *
 * @param path_str The path to be validated.
 * @param expects_folder whether the path should be a folder or not.
 * @return the path, in case it is valid.
 * @throws std::runtime_error if the path either doesn't exist or is/isn't a folder (based on `expects_folder`).
 */
[[nodiscard]]
auto ValidatePath(std::string_view const& path_str, bool const expects_folder) -> std::filesystem::path {
  std::filesystem::path path{path_str};
  if (!std::filesystem::exists(path)) {
    throw std::runtime_error(std::format("specified path does not exist: {}", path.string()));
  }
  if (std::filesystem::is_directory(path) != expects_folder) {
    throw std::runtime_error(std::format("specified path is {}a directory: {}", (expects_folder ? "not " : ""), path.string()));
  }
  return path;
}

/**
 * Extracts the paths specified on the command line, either for a run with a single pair of images or for a run with a
 * folder.
 *
 * @param argc argument count, from `main` invocation.
 * @param argv argument values, from `main` invocation.
 * @return the paths specified on the command line.
 * @throws std::runtime_error if any of the arguments does not meet expectations.
 */
[[nodiscard]]
auto GetPathsFromArgs(int const argc, char const *const argv[]) -> path_t {
  switch (argc) {
    case 4:
      return ImagePath{ValidatePath(argv[1], false), ValidatePath(argv[2], false), ValidatePath(argv[3], false)};
    case 3:
      return FolderPath{ValidatePath(argv[1], true), ValidatePath(argv[2], false)};
    case 2: {
      auto const path = ValidatePath(argv[1], true);
      return FolderPath{path, path / "calibration.yml"};
    }
    default:
      throw std::runtime_error(std::format("expected two, three or four arguments but got {}!", argc - 1));
  }
}

/**
 * Process a single pair of stereo images and show the result, both in the GUI and printed to the console.
 *
 * @param stereo_vis stereo vision algorithm used to process the images.
 * @param left_image image for the left camera. must have been taken at the same time as the right image.
 * @param right_image image for the right camera. must have been taken at the same time as the left image.
 */
void ProcessImagePair(stereo_vision::StereoVision const& stereo_vis, cv::Mat const& left_image, cv::Mat const& right_image) {
  auto const& result = stereo_vis.AnalyzeAndAnnotateImage(left_image, right_image);

  if (!result) {
    std::cerr << "Failed to analyze the images: " << result.error().ToString() << std::endl;
    cv::destroyAllWindows();
    return;
  }

  std::cout << "Analysis result:" << std::endl;
  std::cout << *result << std::endl;

  // Perform HOG detection on the left and right images
  stereo_vision::HOGObjDetect(result->left_image);
  stereo_vision::HOGObjDetect(result->right_image);

  // Display the rectified images
  cv::imshow("left", result->left_image);
  cv::imshow("right", result->right_image);
  cv::waitKey();
}

/**
 * Process a set of images from a folder. One image pair at a time will be processed and the program will wait for a
 * keypress before showing the next.
 *
 * @param folder_path Path to a folder containing `left` and `right` subfolders. Also contains the path to the
 * calibration file.
 */
void ProcessFolderPath(FolderPath const& folder_path) {
  stereo_vision::StereoVision const stereo_vis{{stereo_vision::LoadStereoCameraInfo(folder_path.calibration_file_path)}};
  auto const folder_left = folder_path.folder_path / "left";
  auto const folder_right = folder_path.folder_path / "right";
  if (!std::filesystem::exists(folder_left) || !std::filesystem::exists(folder_right)) {
    throw std::runtime_error("the image folder must contain a folder called 'left' and one called 'right'!");
  }

  auto const directory_entry_to_image = [](std::filesystem::directory_entry const& e) -> cv::Mat {
    return cv::imread(e.path().string(), cv::IMREAD_COLOR);
  };

  auto images_left = std::filesystem::directory_iterator{folder_left} | std::views::transform(directory_entry_to_image);
  auto images_right = std::filesystem::directory_iterator{folder_right} | std::views::transform(directory_entry_to_image);

  for (auto const& [image_left, image_right] : std::views::zip(images_left, images_right)) {
    ProcessImagePair(stereo_vis, image_left, image_right);
  }
}

/**
 * Process a single pair of stereo images and show the result, both in the GUI and printed to the console.
 *
 * @param image_path Path to a pair of stereo images. Also contains the path to the calibration file.
 */
void ProcessImagePath(ImagePath const& image_path) {
  stereo_vision::StereoVision const stereo_vis{{stereo_vision::LoadStereoCameraInfo(image_path.calibration_file_path)}};
  auto const left_image = cv::imread(image_path.left_path.string(), cv::IMREAD_COLOR);
  auto const right_image = cv::imread(image_path.right_path.string(), cv::IMREAD_COLOR);
  ProcessImagePair(stereo_vis, left_image, right_image);
}

int main(int const argc, char const * const argv[]) {
  cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_WARNING); // stop OpenCV log spamming

  try {
    auto const input_path = GetPathsFromArgs(argc, argv);

    std::visit(overloaded{
      [&](FolderPath const& arg) { ProcessFolderPath(arg); },
      [&](ImagePath const& arg) { ProcessImagePath(arg); },
    }, input_path);
  } catch (std::exception const& ex) {
    std::cerr << ex.what() << std::endl;
    std::cerr << "Usage:" << std::endl;
    std::cerr << "    " << argv[0] << " [image_folder_path]" << std::endl;
    std::cerr << " or " << argv[0] << " [image_folder_path] [calibration_file_path]" << std::endl;
    std::cerr << " or " << argv[0] << " [left_image_path] [right_image_path] [calibration_file_path]" << std::endl;
    return 1;
  }
  
  return 0;
}
