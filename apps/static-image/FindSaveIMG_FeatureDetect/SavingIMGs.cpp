#include <filesystem>
#include <fstream>
#include <format>
#include <opencv2/opencv.hpp>
#include <nlohmann/json.hpp>
#include <stereo-vision-lib/lib.hpp>

using json = nlohmann::json; // Using JSON library

/**
 * Save an image to a specified file.
 * 
 * @param filename The file name to save the image.
 * @param image The image to save.
 */
void SaveImage(const std::string& filename, const cv::Mat& image) {
    if (cv::imwrite(filename, image)) {
        std::cout << "Saved image: " << filename << std::endl;
    } else {
        std::cerr << "Failed to save image: " << filename << std::endl;
    }
}

/**
 * Save calibration and board parameters to a JSON file.
 * 
 * @param filename The file name to save the parameters.
 * @param stereo_camera_info The stereo camera calibration info.
 */
void SaveParametersToJson(const std::string& filename, const stereo_vision::StereoCameraInfo& stereo_camera_info) {
    json params;

    // Save camera parameters
    params["camera_matrix_left"] = stereo_camera_info.camera_matrix_left;
    params["camera_matrix_right"] = stereo_camera_info.camera_matrix_right;
    params["dist_coeffs_left"] = stereo_camera_info.dist_coeffs_left;
    params["dist_coeffs_right"] = stereo_camera_info.dist_coeffs_right;
    params["R"] = stereo_camera_info.R;
    params["T"] = stereo_camera_info.T;

    // Write JSON to file
    std::ofstream json_file(filename);
    if (json_file) {
        json_file << std::setw(4) << params << std::endl;
        std::cout << "Saved parameters to: " << filename << std::endl;
    } else {
        std::cerr << "Failed to save parameters to: " << filename << std::endl;
    }
}

/**
 * Capture and save stereo images and calibration parameters.
 * 
 * @param folder_path Path to save captured data.
 * @param stereo_camera_info Stereo camera calibration info.
 * @param left_image Left stereo image.
 * @param right_image Right stereo image.
 * @param frame_index Current frame index.
 */
void CaptureAndSaveStereoData(
    const std::filesystem::path& folder_path,
    const stereo_vision::StereoCameraInfo& stereo_camera_info,
    const cv::Mat& left_image,
    const cv::Mat& right_image,
    int frame_index) {

    // Ensure the data directory exists
    if (!std::filesystem::exists(folder_path)) {
        std::filesystem::create_directory(folder_path);
    }

    // Save images
    auto left_image_path = folder_path / std::format("left_image_{}.png", frame_index);
    auto right_image_path = folder_path / std::format("right_image_{}.png", frame_index);
    SaveImage(left_image_path.string(), left_image);
    SaveImage(right_image_path.string(), right_image);

    // Save calibration parameters
    auto json_file_path = folder_path / std::format("calibration_{}.json", frame_index);
    SaveParametersToJson(json_file_path.string(), stereo_camera_info);
}

