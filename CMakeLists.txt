cmake_minimum_required(VERSION 3.29)
project(
    stereo_vision
    DESCRIPTION "A demonstration of how basic stereo vision works"
    LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(CTest) # this automatically enables testing as well

add_subdirectory(src)
add_subdirectory(apps)
add_subdirectory(test)
