# Stereo Vision: Static Image Reader

This loads existing images and runs them through the stereo vision algorithm. This does not require the physical cameras
to be connected. The calibration data must match the hardware setup used to generate the images.

## Usage
### Single Image Pair
To use the program with a single pair of images you must launch it with the two image paths as well as the calibration file path:
```bash
stereo-vision-static-image left_image_path right_image_path config_file_path
```

### Image Folders
You can also launch it with the path to a folder which contains two sub-folders, `left` and `right` which contain images
in the same order for the left/right image pairs. If you do not specify the optional config file path the config file must
exist in the specified folder and be named `calibration.yml`.
```bash
stereo-vision-static-image image_folder_path [config_file_path]
```
