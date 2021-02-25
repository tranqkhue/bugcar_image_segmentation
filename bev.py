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

    @classmethod
    def fromJSON(cls, filepath):
        f = open(filepath, mode="r")
        print(f)
        data = json.load(f)

        shape = data["size"]
        intrinsic_matrix = np.reshape(np.array(data['intrinsic matrix']),
                                      (3, 3))

        dist2target = data["distance to target"]
        tile_length = data["tile_length"]
        occ_grid_size_in_m = data["occ_grid_size"]
        cell_size_in_m = data['cell_size_in_m']
        cm_per_px = data['cm_per_px']
        yaw = data['yaw']
        bev = cls(shape, dist2target, tile_length, cm_per_px, yaw)
        bev._bev_matrix = intrinsic_matrix
        bev.create_occ_grid_param(occ_grid_size_in_m, cell_size_in_m)
        return bev

    @property
    def map_size(self):
        return self.__cell_size_in_m * self.__occ_grid

    @property
    def map_resolution(self):
        return self.__cell_size_in_m

    # --------------------------------------------------------------------------------------
    def save_to_JSON(self, file_path):
        f = open(file_path, mode="w")

        data = {
            "size": (self.width, self.height),
            "intrinsic matrix": self._bev_matrix.tolist(),
            "distance to target": self.dist2target,
            "tile_length": self.tile_length,
            "occ_grid_size": self.__occ_grid * self.__cell_size_in_m,
            "cell_size_in_m": self.__cell_size_in_m,
            "cm_per_px": self.cm_per_px,
            "yaw": self.yaw
        }
        json.dump(data, f)

    def calculate_transform_matrix(self, tile_coords, dist2target, cm_per_px,
                                   width, height, tile_length, yaw):
        dist2target_px = (dist2target[0] / cm_per_px,
                          dist2target[1] / cm_per_px)
        target_in_img = (width / 2 + dist2target_px[0],
                         height - dist2target_px[1])
        max_height = tile_length / cm_per_px
        original_pts = np.array([[max_height / 2, max_height / 2],
                                 [max_height / 2, -max_height / 2],
                                 [-max_height / 2, -max_height / 2],
                                 [-max_height / 2, max_height / 2]])

        #Then rotate the point around the origin by *yaw* angle
        yaw_fid2cam_mat = np.array([[np.cos(yaw), -np.sin(yaw)],
                                    [np.sin(yaw), np.cos(yaw)]])
        unit_vec_along_x = np.array([100, 0])
        rotated_unit_vec = np.matmul(yaw_fid2cam_mat, unit_vec_along_x)
        rotated_unit_vec += target_in_img
        bev_fiducial_axis = np.stack(
            [np.asarray(target_in_img), rotated_unit_vec], axis=0)
        print("bev_fiducial_axis", bev_fiducial_axis)
        rotated_pts = np.zeros(original_pts.shape)
        for i in range(len(original_pts)):
            rotated_pts[i] = np.matmul(yaw_fid2cam_mat, original_pts[i])
        rotated_pts += target_in_img
        rotated_pts = order_points(rotated_pts, bev_fiducial_axis)
        print("transformed corners", rotated_pts)

        M = cv2.getPerspectiveTransform(tile_coords.astype(np.float32),
                                        rotated_pts.astype(np.float32))

        self._bev_matrix = M
        return M

    # --------------------------------------------------------------------------------------

    # --------------------------------------------------------------------------------------

    def create_occ_grid_param(self, occupancy_grid_size_in_m, cell_size_in_m):
        # Occupancy grid should be square
        # Should show every object in the 5m radius
        self.__cell_size_in_m = cell_size_in_m  # except this
        # all of these size are in pixels unit
        self.__cell_size = (self.__cell_size_in_m * 100 / self.cm_per_px)
        self.__occ_grid = int(occupancy_grid_size_in_m / self.__cell_size_in_m)
        # this describe the length of the occupancy grid's edges in pixels
        self.__occ_edge_pixel = int(self.__occ_grid * self.__cell_size)

    # --------------------------------------------------------------------------------------
    @staticmethod
    def create_occupancy_grid(segmap, bev_matrix, width, height,
                              occupancy_grid_size_in_m, cell_size_in_m,
                              cm_per_px):
        # segmap must have the same size
        seg_width = segmap.shape[1]
        seg_height = segmap.shape[0]
        cell_size = (cell_size_in_m * 100 / cm_per_px)
        occ_grid = int(occupancy_grid_size_in_m / cell_size_in_m)
        occ_edge_pixel = int(occ_grid * cell_size)
        segmap = np.add(segmap, 1)
        warped_img = cv2.warpPerspective(segmap, bev_matrix, (width, height))
        left_x = int((width - occ_edge_pixel) / 2)
        top_y = height - occ_edge_pixel
        warped_left_x = int(np.clip(left_x, 0, np.inf))
        warped_img = warped_img[int(np.clip(top_y, 0, np.Inf)):height,
                                warped_left_x:warped_left_x + occ_edge_pixel]
        occ_grid_left_x = int(np.clip(-left_x, 0, np.inf))
        occ_grid_top_y = int(np.clip(-top_y, 0, np.inf))
        template_occ_grid = np.zeros(shape=(occ_edge_pixel, occ_edge_pixel))

        template_occ_grid[occ_grid_top_y:occ_edge_pixel,
                          occ_grid_left_x:occ_grid_left_x +
                          warped_img.shape[1]] = warped_img
        template_occ_grid = template_occ_grid.astype(np.uint8)
        occ_grid_size = template_occ_grid.shape[0]

        image_bottom_vertices = np.transpose(
            np.array([[seg_width, seg_height, 1], [0, seg_height, 1]]))
        vertices_after_transform = np.matmul(bev_matrix, image_bottom_vertices)

        vertices_after_transform[:, 0] /= vertices_after_transform[2, 0]
        vertices_after_transform[:, 1] /= vertices_after_transform[2, 1]
        vertices_after_transform = vertices_after_transform[0:2, :]
        vertices_after_transform[0] -= left_x
        vertices_after_transform[1] -= top_y
        vertices_after_transform_x_projection = np.copy(
            vertices_after_transform)
        vertices_after_transform_x_projection[1] = occ_grid_size
        '''
        front_value = template_occ_grid[
            int(occ_grid_size - 350 / cm_per_px):int(occ_grid_size -
                                                     300 / cm_per_px),
            int(occ_grid_size / 2 - 100 / cm_per_px):int(occ_grid_size / 2 +
                                                         100 / cm_per_px)]
        front_value = np.mean(front_value)
        front_value = int(np.round(front_value))

        unknown_area_poly = np.append(
            vertices_after_transform,
            np.flip(vertices_after_transform_x_projection, axis=1),
            axis=1)
        unknown_area_poly = np.transpose(unknown_area_poly)
        unknown_area_poly = unknown_area_poly.astype(np.int32)
        cv2.fillConvexPoly(template_occ_grid, unknown_area_poly, front_value)
        '''

        occupancy_grid = cv2.resize(
            template_occ_grid,
            (occ_grid, occ_grid),
        ) * 100
        occupancy_grid = np.where(occupancy_grid == 0, -1,
                                  200 - occupancy_grid)
        occupancy_grid = cv2.flip(occupancy_grid, 0)
        occupancy_grid = cv2.rotate(occupancy_grid,
                                    cv2.ROTATE_90_COUNTERCLOCKWISE)

        occupancy_grid = occupancy_grid.astype('int8')
        return occupancy_grid
