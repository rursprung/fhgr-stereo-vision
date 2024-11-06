#include <algorithm>
#include <stereo-vision-lib/lib.hpp>

namespace stereo_vision {

    auto AnalysisError::ToString() const -> std::string {
        switch(this->value_) {
            case kInvalidImage:
                return "invalid image!";
            case kNotYetImplemented:
                return "not yet implemented!";
            default:
                throw std::runtime_error("unknown AnalysisError type!");
        }
    }

    AnalysisError::operator std::string() const {
        return this->ToString();
    }

    std::ostream& operator << (std::ostream& o, AnalysisResult const& analysis_result) {
        // TODO: add output when adding data
        return o;
    }

    StereoVision::StereoVision(Settings const settings) : settings_(std::move(settings)) {
    }

    auto StereoVision::AnalyzeAndAnnotateImage(cv::Mat const& left_image, cv::Mat const& right_image) const -> std::expected<AnalysisResult, AnalysisError> {
        if (left_image.data == nullptr || right_image.data == nullptr) {
            return std::unexpected{AnalysisError::kInvalidImage};
        }

        return std::unexpected{AnalysisError::kNotYetImplemented};
    }

}
