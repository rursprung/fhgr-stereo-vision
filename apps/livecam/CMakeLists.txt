add_executable(stereo-vision-livecam main.cpp)

target_link_libraries(stereo-vision-livecam
        PRIVATE stereo-vision-lib
)
