### Riaan, 24112024 ; Original CMakeLists commented out below!
cmake_minimum_required(VERSION 3.29)
project(
    stereo_vision
    DESCRIPTION "A demonstration of how basic stereo vision works"
    LANGUAGES CXX
)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)



# Set the toolchain file to use vcpkg for dependency management
set(CMAKE_TOOLCHAIN_FILE /home/riaanlinub/vcpkg/scripts/buildsystems/vcpkg.cmake CACHE STRING "Vcpkg toolchain file")

# Ensure OpenCV is found
#find_package(OpenCV REQUIRED)

# Specify the custom OpenCV installation path
set(OpenCV_DIR "/usr/include/opencv4")
find_package(OpenCV REQUIRED)

# Add OpenCV to your target
#target_include_directories(${PROJECT_NAME} PRIVATE ${OpenCV_INCLUDE_DIRS})
#target_link_libraries(${PROJECT_NAME} PRIVATE ${OpenCV_LIBS})

# Include OpenCV headers
#include_directories(${OpenCV_INCLUDE_DIRS})

# Add the src folder
add_subdirectory(src)

# Add the apps folder (e.g., the calibration program)
add_subdirectory(apps)

# Uncomment this line if you wish to include tests
# add_subdirectory(test)

# Link libraries for the project
# This is optional if linking is done within subdirectories

###


###-------------Original CMakeLists.txt-------------###
#cmake_minimum_required(VERSION 3.29)
#project(
#    stereo_vision
#    DESCRIPTION "A demonstration of how basic stereo vision works"
#    LANGUAGES CXX)



#set(CMAKE_CXX_STANDARD 23)
#set(CMAKE_CXX_STANDARD_REQUIRED ON)
#set(CMAKE_CXX_EXTENSIONS OFF)

##include(CTest) # this automatically enables testing as well

#add_subdirectory(src)
#add_subdirectory(apps)
##add_subdirectory(test)
