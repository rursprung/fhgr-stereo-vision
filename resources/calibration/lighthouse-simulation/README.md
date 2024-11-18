# Simulated Cameras from the [Lighthouse Project](https://gitlab.com/proebrock/lighthouse)

These files have been used to have a "ground truth" and comparison to another implementation (which is also based on OpenCV).

Note that the right camera in this test data does not have the same focal length for x & y while the left does!

To generate the images run [`generate_images.py`](https://gitlab.com/proebrock/lighthouse/-/blob/devel/2d_calibrate_stereo/generate_images.py?ref_type=heads)
from the lighthouse project.

The generated files will have to be renamed. To do this copy the `*.png` files from the output folder of the lighthouse script
to this folder and delete any existing `left` and `right` folders. Then run `rename_images.sh`.
