import cv2
import numpy as np
from numpy import pi, cos, sin
import json
from utils import order_points


class bev_transform_tools:

    # dist2target : distance from camera to the target: denoted (x,y) (cm)
    # x is the horizontal distance
    # y is the vertical distance
    def __init__(self, image_shape, dist2target, tile_length, cm_per_px, yaw):
        self.width = image_shape[0]
        self.height = image_shape[1]
        self.dist2target = dist2target
        self.tile_length = tile_length  # in cm
        self.cm_per_px = cm_per_px
        self.yaw = yaw
        self.skeleton_image = cv2.imread(
            "segmap_boundaries_in_occ_grid.jpg", cv2.IMREAD_GRAYSCALE)

    @classmethod
    def fromJSON(cls, filepath):
        f = open(filepath, mode="r")
        print(f)
        data = json.load(f)

        shape = data["size"]
        bev_matrix = np.reshape(np.array(data['bev matrix']),
                                (3, 3))
        dist2target = data["distance to target"]
        tile_length = data["tile_length"]
        cm_per_px = data['cm_per_px']
        yaw = data['yaw']
        bev = cls(shape, dist2target, tile_length, cm_per_px, yaw)
        bev._bev_matrix = bev_matrix
        return bev

    # --------------------------------------------------------------------------------------
    def save_to_JSON(self, file_path):
        f = open(file_path, mode="w")

        data = {
            "size": (self.width, self.height),
            "bev matrix": self._bev_matrix.tolist(),
            "distance to target": self.dist2target,
            "tile_length": self.tile_length,
            "cm_per_px": self.cm_per_px,
            "yaw": self.yaw
        }
        json.dump(data, f)

    def calculate_transform_matrix(self, tile_coords, dist2target, cm_per_px,
                                   width, height, tile_length, yaw):
        dist2target_px = (dist2target[0] / cm_per_px,
                          dist2target[1] / cm_per_px)
        max_height = tile_length / cm_per_px
        original_pts = np.array([[max_height / 2, max_height / 2],
                                 [max_height / 2, -max_height / 2],
                                 [-max_height / 2, -max_height / 2],
                                 [-max_height / 2, max_height / 2]])

        # Then rotate the point around the origin by *yaw* angle
        yaw_fid2bev_mat = np.array([[np.cos(yaw), -np.sin(yaw)],
                                    [np.sin(yaw), np.cos(yaw)]])
        unit_vec_along_x = np.array([100, 0])
        rotated_unit_vec = np.matmul(yaw_fid2bev_mat, unit_vec_along_x)
        target_in_img = (width / 2 + dist2target_px[0],
                         height - dist2target_px[1])
        rotated_unit_vec += target_in_img
        bev_fiducial_axis = np.stack(
            [np.asarray(target_in_img), rotated_unit_vec], axis=0)

        print("bev_fiducial_axis", bev_fiducial_axis)
        rotated_pts = np.zeros(original_pts.shape)
        for i in range(len(original_pts)):
            rotated_pts[i] = np.matmul(yaw_fid2bev_mat, original_pts[i])
        rotated_pts += target_in_img
        rotated_pts = order_points(rotated_pts, bev_fiducial_axis)
        print("transformed corners", rotated_pts)

        M = cv2.getPerspectiveTransform(tile_coords.astype(np.float32),
                                        rotated_pts.astype(np.float32))

        self._bev_matrix = M
        return M
    # --------------------------------------------------------------------------------------

    # --------------------------------------------------------------------------------------

    def create_occupancy_grid(self, segmap, bev_matrix, width, height,
                              occupancy_grid_width_in_m, occupancy_grid_height_in_m, cell_size_in_m,
                              cm_per_px):
        # segmap must have the same size
        cell_size = (cell_size_in_m * 100 / cm_per_px)
        occ_grid_width = int(occupancy_grid_width_in_m / cell_size_in_m)
        occ_width_pixel = int(occ_grid_width * cell_size)
        occ_grid_height = int(occupancy_grid_height_in_m / cell_size_in_m)
        occ_height_pixel = int(occ_grid_height*cell_size)
        segmap = np.add(segmap, 1)
        warped_img = cv2.warpPerspective(segmap, bev_matrix, (width, height))
        left_x = int((width - occ_width_pixel) / 2)
        top_y = height - occ_height_pixel
        warped_left_x = int(np.clip(left_x, 0, np.inf))
        warped_img = warped_img[int(np.clip(top_y, 0, np.Inf)):height,
                                warped_left_x:warped_left_x + occ_width_pixel]
        occ_grid_left_x = int(np.clip(-left_x, 0, np.inf))
        occ_grid_top_y = int(np.clip(-top_y, 0, np.inf))
        template_occ_grid = np.zeros(shape=(occ_height_pixel, occ_width_pixel))

        template_occ_grid[occ_grid_top_y:occ_height_pixel,
                          occ_grid_left_x:occ_grid_left_x +
                          warped_img.shape[1]] = warped_img
        template_occ_grid = template_occ_grid.astype(np.uint8)
        isOccupiedGrid = (template_occ_grid == 1).astype(np.uint8)
        morphKernel = np.ones((3, 3))
        morphGrid = cv2.morphologyEx(
            isOccupiedGrid, cv2.MORPH_OPEN, kernel=morphKernel)
        mask1 = (morphGrid > 0).astype(np.uint8)
        subtract_mask = cv2.subtract(isOccupiedGrid, mask1)
        template_occ_grid = np.where(subtract_mask > 0, 2, template_occ_grid)
        occupancy_grid = cv2.resize(
            template_occ_grid,
            (occ_grid_width, occ_grid_height), interpolation=cv2.INTER_NEAREST
        ) * 100
        occupancy_grid = np.where(occupancy_grid == 0, -1,
                                  200 - occupancy_grid).astype(np.int8)
        return occupancy_grid
