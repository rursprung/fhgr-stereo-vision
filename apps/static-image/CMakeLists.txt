add_executable(stereo-vision-static-image main.cpp Viewer.cpp)

target_link_libraries(stereo-vision-static-image
        PRIVATE stereo-vision-lib
)
