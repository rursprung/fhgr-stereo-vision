find_package(GTest CONFIG REQUIRED)
include(GoogleTest)

add_executable(stereo-vision-lib-test stereo-vision-lib-test.cpp)
target_link_libraries(stereo-vision-lib-test stereo-vision-lib GTest::gtest_main)
gtest_discover_tests(stereo-vision-lib-test)

# the resources are used in the tests, so they need to be present in a folder where the test can access them
# with a known location.
file(COPY ${PROJECT_SOURCE_DIR}/resources DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
