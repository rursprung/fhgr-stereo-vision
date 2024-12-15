# Stereo Vision Project

This is a mini-project for the image processing course of the [FHGR B.Sc. Mobile Robotics](https://fhgr.ch/mr),
implemented by [Dominic Eicher](https://github.com/Nic822), [Riaan Kämpfer](https://github.com/RiaanGitHub)
and [Ralph Ursprung](https://github.com/rursprung).

The goal of this project is to implement the basics of stereo vision:
- calibrate two cameras using [ChArUco boards](https://docs.opencv.org/4.10.0/df/d4a/tutorial_charuco_detection.html)
- rectify the images
- identify key features in both images
- match key features
- calculate depth map
- optional: create 3D point cloud

To show how/that this works we plan to identify the width of an (otherwise isolated) object.

Note that this is purely an educational project. For real-world applications you should really use pre-existing
libraries which implement this with much more accuracy and cover more use-cases!

## Project Structure

The project consists of the following parts:
* [A camera calibration tool](apps/calibrate/README.md) (which is independent of the library)
* The stereo vision implementation:
  * The library (everything in [`src/`](src/)), including test coverage (everything in [`test/`](src/))
  * [An application using the library](apps/stereo-vision/README.md)

## Building

To build this project you will need:

* A modern C++ compiler supporting C++23
* [CMake](https://cmake.org/) incl. CTest - this might well come included with your favourite IDE
* [vcpkg](https://vcpkg.io/)

## License
As this is purely an educational project there's no need for others to include it in their commercial works.
Accordingly, this is licensed under the **GNU General Public License v3.0 or later** (SPDX: `GPL-3.0-or-later`).
See [LICENSE](LICENSE) for the full license text.
