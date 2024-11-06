find_package(OpenCV CONFIG REQUIRED)

add_library(stereo-vision-lib lib.cpp "${PROJECT_SOURCE_DIR}/include/stereo-vision-lib/lib.hpp")

target_include_directories(
        stereo-vision-lib
        PUBLIC "${PROJECT_SOURCE_DIR}/include"
        PUBLIC "${OpenCV_INCLUDE_DIRS}"
)

target_link_libraries(stereo-vision-lib
        PUBLIC ${OpenCV_LIBS}
)
