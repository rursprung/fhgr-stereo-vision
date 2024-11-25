cmake_minimum_required(VERSION 3.15)
project(FindSaveIMG_FeatureDetect)

#set(CMAKE_CXX_STANDARD 17)

# Include the nlohmann/json header file
include_directories(${PROJECT_SOURCE_DIR}/nlohmann)

add_executable(FindSaveIMG_FeatureDetect SavingIMGs.cpp)

#find_package(OpenCV REQUIRED)
#include_directories(${OpenCV_INCLUDE_DIRS})
#target_link_libraries(FindSaveIMG_FeatureDetect ${OpenCV_LIBS})

target_link_libraries(stereo-vision-static-image
        PRIVATE stereo-vision-lib
)


