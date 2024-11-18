# Stereo Vision: Calibration Tool

This tool uses images of a [ChArUco board](https://docs.opencv.org/4.10.0/df/d4a/tutorial_charuco_detection.html) to generate
the calibration data for the cameras and persist for future use by the actual stereo vision application.

## Usage
You need to launch it with the path to a folder with two sub-folders, `left` and `right` which contain images
in the same order for the left/right image pairs. Additionally, the folder must contain a `config.yml` with the following content (values as examples):
```yaml
board:
  board_size: [ 6, 5 ]
  square_length: 75
  marker_length: 37.5
  aruco_marker_dictionary: 10
  legacy_pattern: false
```

Parameter description:
* `board_size`: the amount of squares on the ChArUco board in x & y dimension
* `square_length`: the length of one side of a square, in mm
* `marker_length`: the length of one side of a marker, in mm
* `aruco_marker_dictionary`: constant ID from [`cv::aruco::PredefinedDictionaryType`](https://docs.opencv.org/4.10.0/de/d67/group__objdetect__aruco.html#ga4e13135a118f497c6172311d601ce00d)
* `legacy_pattern`: set to `true` if the pattern starts with a white box in the upper left corner.

Usage:
```bash
stereo-vision-calibrate image_folder_path
```

Once the calibration is finished the result will be saved in a file named `calibration.yml` in the same folder.
In case a file with the same name already exists you will be prompted whether the file should be overwritten. If not,
the result will *not* be saved!
