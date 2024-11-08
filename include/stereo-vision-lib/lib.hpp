#ifndef STEREO_VISION_LIB_HPP
#define STEREO_VISION_LIB_HPP

#include <expected>

#include <opencv2/opencv.hpp>

namespace stereo_vision {

    /**
     * All errors which may occur during analysis of the image.
     *
     * Note that this is a class wrapping an enum (instead of an `enum class`) to be able to provide methods on the values.
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
            /// The functionality is not yet implemented (will be removed the moment we start implementing this!)
            kNotYetImplemented,
        };

        AnalysisError() = default;
        constexpr AnalysisError(Value const value) : value_(value) { }
        constexpr explicit operator Value() const { return value_; }
        bool operator==(Value const value) const { return this->value_ == value; }
        explicit operator bool() const = delete;

        [[nodiscard]]
        auto ToString() const -> std::string;

        explicit operator std::string() const;

    private:
        Value value_;
    };

    /**
     * The analysis results for processed stereo images.
     */
    struct AnalysisResult {
        // TODO: add data
    };

    /// Print detailed information about the result to an output stream.
    std::ostream& operator << (std::ostream& o, AnalysisResult const& analysis_result);

    class StereoVision {
    public:
        struct Settings {
            // TODO: add settings (incl. calibration data!)
        };

        explicit StereoVision(Settings settings);

        /**
         * Analyse an image for the presence of bananas and their properties.
         *
         * @param left_image the left image of the stereo cameras, taken at the same time as `right_image`.
         * @param right_image the right image of the stereo cameras, taken at the same time as `left_image`.
         * @return the result of the analysis and the annotated image, see the description of {@link AnalysisResult} for more details.
         */
        [[nodiscard]]
        auto AnalyzeAndAnnotateImage(cv::Mat const& left_image, cv::Mat const& right_image) const -> std::expected<AnalysisResult, AnalysisError>;

    private:
        /// All externally configurable settings used by the analyzer.
        Settings const settings_;
    };

}

#endif //STEREO_VISION_LIB_HPP
