from abc import ABC, abstractmethod
class BaseCamera(object):
    @abstractmethod
    def get_bgr_frame(self):
        pass
    def get_intrinsic_matrix(self):
        pass
    def get_distortion_coeff(self):
        pass
    def stop(self):
        pass