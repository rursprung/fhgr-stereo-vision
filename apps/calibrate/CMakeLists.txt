find_package(OpenCV CONFIG REQUIRED)

add_executable(stereo-vision-calibrate-static static-image-main.cpp CalibrationRun.cpp)

target_include_directories(
        stereo-vision-calibrate-static
        PRIVATE "${PROJECT_SOURCE_DIR}/include"
        PRIVATE "${OpenCV_INCLUDE_DIRS}"
)

target_link_libraries(stereo-vision-calibrate-static
        PUBLIC ${OpenCV_LIBS}
)

add_executable(stereo-vision-calibrate-livecam livecam-main.cpp CalibrationRun.cpp)

target_include_directories(
        stereo-vision-calibrate-livecam
        PRIVATE "${PROJECT_SOURCE_DIR}/include"
        PRIVATE "${OpenCV_INCLUDE_DIRS}"
)

target_link_libraries(stereo-vision-calibrate-livecam
        PUBLIC ${OpenCV_LIBS}
)
