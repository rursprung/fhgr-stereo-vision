find_package(OpenCV CONFIG REQUIRED)

add_executable(stereo-vision-calibrate main.cpp CalibrationRun.cpp)

target_include_directories(
        stereo-vision-calibrate
        PRIVATE "${PROJECT_SOURCE_DIR}/include"
        PRIVATE "${OpenCV_INCLUDE_DIRS}"
)

target_link_libraries(stereo-vision-calibrate
        PUBLIC ${OpenCV_LIBS}
)
