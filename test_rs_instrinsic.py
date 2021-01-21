import pyrealsense2 as rs

import cv2
import numpy as np

# Configure stream
pipeline  = rs.pipeline()
config    = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)

pipeline.start(config)

for i in range(30):
	pipeline_frames = pipeline.wait_for_frames()

try:
	while True:
		pipeline_frames    = pipeline.wait_for_frames()
		pipeline_rgb_frame = pipeline_frames.get_color_frame()
		rgb_intrin = pipeline_rgb_frame.profile.as_video_stream_profile().intrinsics

		frame = np.asanyarray(pipeline_rgb_frame.get_data())
		frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

		fx    = pipeline_rgb_frame.profile.as_video_stream_profile().intrinsics.fx
		fy    = pipeline_rgb_frame.profile.as_video_stream_profile().intrinsics.fy
		ppx   = pipeline_rgb_frame.profile.as_video_stream_profile().intrinsics.ppx
		ppy   = pipeline_rgb_frame.profile.as_video_stream_profile().intrinsics.ppy
		K     = np.array([[fx,  0, ppx],
						  [0 , fy, ppy],
						  [0 ,  0,   1]])
		distortion_coeffs = pipeline_rgb_frame.profile.as_video_stream_profile().intrinsics.coeffs
		print(K)

		cv2.imshow('test', frame)
		c = cv2.waitKey(1) % 0x100
		if (c == 27):
			break

finally:
	pipeline.stop()
	cv2.destroyAllWindows()