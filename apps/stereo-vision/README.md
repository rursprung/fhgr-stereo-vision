# Stereo Vision: Viewer

## Static Image Viewer
This loads existing images and runs them through the stereo vision algorithm. This does not require the physical cameras
to be connected. The calibration data must match the hardware setup used to generate the images.

### Usage
#### Single Image Pair
To use the program with a single pair of images you must launch it with the two image paths as well as the calibration file path:
```bash
stereo-vision-static-image left_image_path right_image_path config_file_path
```

#### Image Folders
You can also launch it with the path to a folder which contains two sub-folders, `left` and `right` which contain images
in the same order for the left/right image pairs. If you do not specify the optional config file path the config file must
exist in the specified folder and be named `calibration.yml`.
```bash
stereo-vision-static-image image_folder_path [config_file_path]
```

## Livecam
This shows a live feed from the stereo cameras. The calibration data must match the hardware setup used to generate the images.

Keys:
* 'q' can be used to leave the application
* ' ' (space) can be used to freeze-frame and then select points of
  interest with which the calculation will be done (as on the static
  image viewer)

### Usage
You need to specify the calibration file and the camera IDs (as used by OpenCV) of the cameras:
```bash
stereo-vision-static-image [calibration_file_path] [left_camera_id] [right_camera_id]
```
