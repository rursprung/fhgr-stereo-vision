find_package(GTest CONFIG REQUIRED)
include(GoogleTest)

add_executable(stereo-vision-lib-test stereo-vision-lib-test.cpp)
target_link_libraries(stereo-vision-lib-test stereo-vision-lib GTest::gtest_main)
gtest_discover_tests(stereo-vision-lib-test)
