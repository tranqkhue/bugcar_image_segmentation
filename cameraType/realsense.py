import cv2
import numpy as np
from cameraType.baseCamera import BaseCamera
import pyrealsense2 as rs

# INPUT_SHAPE = {"30fps": (1920, 1080), "60fps": (960, 540)}
class RealsenseCamera(BaseCamera):
    def __init__(self,serial_no,input_shape,fps):
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(serial_no)
        config.enable_stream(rs.stream.color, input_shape[0], input_shape[1],
                             rs.format.rgb8, fps)
        self.cfg = config
        self.profile = self.cfg.get_stream(rs.stream.color)
        self.cfg = self.pipeline.start(config)

    def get_bgr_frame(self):
        pipeline_frames = self.pipeline.wait_for_frames()
        pipeline_rgb_frame = pipeline_frames.get_color_frame()
        frame = np.asanyarray(pipeline_rgb_frame.get_data())
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        return frame
    def get_intrinsic_matrix(self):
        fx = self.profile.as_video_stream_profile().intrinsics.fx
        fy = self.profile.as_video_stream_profile().intrinsics.fy
        ppx =self.profile.as_video_stream_profile().intrinsics.ppx
        ppy =self.profile.as_video_stream_profile().intrinsics.ppy        
        K = np.array([[fx, 0, ppx], [0, fy, ppy], [0, 0, 1]])
        return K

    def get_distortion_coeff(self):
        distortion_coeffs = np.array(
            self.profile.as_video_stream_profile().intrinsics.coeffs)
        return distortion_coeffs
    def stop(self):
        self.pipeline.stop()