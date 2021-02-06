# bugcar_image_segmenation

### Warning: this codebase relies heavily on Intel Realsense's functionality, so feel free to tweak the code if you are using a different type of camera.

## Steps for using this repository:

- Run calibration process to store bird eye's view matrix into json file.
  - `python3 calibration.py`
  - Press C to calibrate, and S to save the matrix into a json file.

### FOR TESTING:

- Run `test_straight_line.py` to see for yourself if the bev_matrix is functioning as intended.
- Run `evaluate_model.py` (test set not included) to evaluate the accuracy of this ENET model.

### FOR INFERENCING:

- Fire up `roscore` first.
- Run ` python inference_video.py` for inferencing and publishing occupancy grid.
  (This script does not support tensorflow-CPU)
