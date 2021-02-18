from bev import bev_transform_tools
import numpy as np
transformer = bev_transform_tools.fromJSON('calibration_data.json')
import pytest
import cv2


class TestOccupancy_grid:
    @pytest.fixture
    def seeImgandShutdown(self):
        yield 1
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def test_1(self, seeImgandShutdown):
        segmap = np.ones(shape=(600, 600))
        new = transformer.create_occupancy_grid(
            segmap, transformer._bev_matrix, transformer.width,
            transformer.height, transformer.map_size,
            transformer.map_resolution, transformer.cm_per_px)
        cv2.imshow("new1", new)
        a = seeImgandShutdown

    def test_2(self, seeImgandShutdown):
        segmap_2 = np.zeros(shape=(1200, 1200))
        new = transformer.create_occupancy_grid(
            segmap_2, transformer._bev_matrix, transformer.width,
            transformer.height, transformer.map_size,
            transformer.map_resolution, transformer.cm_per_px)
        cv2.imshow("new2", new)

        a = seeImgandShutdown

    def test_3(self, seeImgandShutdown):
        segmap_3 = np.ones(shape=(400, 400))
        new = transformer.create_occupancy_grid(
            segmap_3, transformer._bev_matrix, transformer.width,
            transformer.height, transformer.map_size,
            transformer.map_resolution, transformer.cm_per_px)
        cv2.imshow("new3", new)

        a = seeImgandShutdown

    def test_4(self, seeImgandShutdown):
        segmap_4 = np.ones(shape=(400, 1200))
        new = transformer.create_occupancy_grid(
            segmap_4_, transformer._bev_matrix, transformer.width,
            transformer.height, transformer.map_size,
            transformer.map_resolution, transformer.cm_per_px)
        cv2.imshow("new4", new)
        a = seeImgandShutdown

    def test_5(self, seeImgandShutdown):
        segmap_5 = np.ones(shape=(512, 1024))
        new = transformer.create_occupancy_grid(
            segmap_5, transformer._bev_matrix, transformer.width,
            transformer.height, transformer.map_size,
            transformer.map_resolution, transformer.cm_per_px)
        cv2.imshow("new5", new)
        a = seeImgandShutdown


# segmap = np.ones(shape=(512, 1024))
# new = transformer.create_occupancy_grid(segmap)
# cv2.imshow("new1", new)
# cv2.waitKey(0)
segmap = np.ones(shape=(512, 1024))
new = transformer.create_occupancy_grid(segmap, transformer._bev_matrix,
                                        transformer.width, transformer.height,
                                        transformer.map_size,
                                        transformer.map_resolution,
                                        transformer.cm_per_px)
cv2.imshow("new1", new)
cv2.waitKey(0)