add_executable(stereo-vision-static-image static-image-main.cpp Viewer.cpp)
add_executable(stereo-vision-livecam livecam-main.cpp Viewer.cpp)

target_link_libraries(stereo-vision-static-image
        PRIVATE stereo-vision-lib
)

target_link_libraries(stereo-vision-livecam
        PRIVATE stereo-vision-lib
)
