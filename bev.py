import cv2
import numpy as np
from numpy import pi, cos, sin
import json


class bev_transform_tools_legacy:

    # dist2target : distance from camera to the target: denoted (x,y) (cm)
    # x is the horizontal distance
    # y is the vertical distance
    def __init__(self, image_shape, dist2target, tile_length, cm_per_px):
        self.width = image_shape[0]
        self.height = image_shape[1]
        self.dist2target = dist2target
        self.tile_length = tile_length  # in cm
        self.cm_per_px = cm_per_px

    @classmethod
    def fromJSON(cls, filepath):
        f = open(filepath, mode="r")
        print(f)
        data = json.load(f)

        shape = data["size"]
        intrinsic_matrix = np.reshape(np.array(data['intrinsic matrix']),
                                      (3, 3))
        dero_matrix = np.reshape(np.array(data["derotation"]), (3, 3))
        M = np.reshape(np.array(data["M"]), (3, 3))
        detrans = np.reshape(np.array(data["detranslation"]), (3, 3))
        dist2target = data["distance to target"]
        tile_length = data["tile_length"]
        occ_grid_size_in_m = data["occ_grid_size"]
        cell_size_in_m = data['cell_size_in_m']
        cm_per_px = data['cm_per_px']
        bev = cls(shape, dist2target, tile_length, cm_per_px)
        bev._intrinsic_matrix = intrinsic_matrix
        bev.M = M
        bev.dero = dero_matrix
        bev.detran = detrans
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
            "intrinsic matrix": self._intrinsic_matrix.tolist(),
            "distance to target": self.dist2target,
            "tile_length": self.tile_length,
            "occ_grid_size": self.__occ_grid * self.__cell_size_in_m,
            "cell_size_in_m": self.__cell_size_in_m,
            "M": self.M.tolist(),
            "derotation": self.dero.tolist(),
            "detranslation": self.detran.tolist(),
            "cm_per_px": self.cm_per_px
        }
        json.dump(data, f)

    def _calculate_derotation_matrix(self, M):
        rotation = np.array([[np.cos(self.yaw), -np.sin(self.yaw), 0],
                             [np.sin(self.yaw),
                              np.cos(self.yaw), 0], [0, 0, 1]])
        self.dero = rotation
        return rotation

    # --------------------------------------------------------------------------------------

    def _calculate_detranslation_matrix(self, rotation_matrix,
                                        target_in_pixels):
        # There are 4 steps to calculate_detranlation_matrix:
        # Given the target location (in pixels) after transformation. Derotate it with 'rotation_matrix'
        # Calculate distance from target to POV origin ( in pixels).
        # Hence, we derive the new origin from subtracting target location to distance from target to POV origin.
        # The returned detranlation will translate the new origin back to its original location, which is (width/2, height)f

        dist_origin_from_target_in_pixel = np.array([-self.dist2target[0],self.dist2target[1]]) \
            / self.cm_per_px
        print("distance in px is", dist_origin_from_target_in_pixel)
        target_after_derotation = np.dot(rotation_matrix, target_in_pixels)
        print("target", target_after_derotation)
        origin_after_derotation = target_after_derotation[0:2] + \
            dist_origin_from_target_in_pixel
        print(origin_after_derotation)
        trans_x = self.width / 2 - origin_after_derotation[0]
        trans_y = self.height - origin_after_derotation[1]
        print(trans_x, trans_y)
        translation = np.array([[1, 0, trans_x], [0, 1, trans_y], [0, 0, 1]])

        self.detran = translation
        self.origin_in_bev = origin_after_derotation
        return translation

    # --------------------------------------------------------------------------------------

    def calculate_transform_matrix(self, tile_coords, yaw):
        self.yaw = yaw
        # Choose 4 corner of the white tile
        top_left_tile = tile_coords[0]
        bot_left_tile = tile_coords[1]
        top_right_tile = tile_coords[2]
        bot_right_tile = tile_coords[3]

        # Find the longest edge of the tile quadrilateral
        max_width_tile = max(np.linalg.norm(top_left_tile - top_right_tile),
                             np.linalg.norm(bot_left_tile - bot_right_tile))
        max_height_tile = max(np.linalg.norm(top_left_tile - bot_left_tile),
                              np.linalg.norm(bot_right_tile - top_right_tile))

        max_height = self.tile_length / self.cm_per_px

        dest = np.array([[0, 0], [0, max_height - 1], [max_height - 1, 0],
                         [max_height - 1, max_height - 1]],
                        dtype=np.float32)
        top_left_tile_array = np.stack([top_left_tile for i in range(4)])
        dest += top_left_tile_array
        # this is the location of target in pixels
        target_after_transform = (dest[0] + dest[3]) / 2
        target_after_transform = np.array(
            [target_after_transform[0], target_after_transform[1], 1])
        M = cv2.getPerspectiveTransform(tile_coords.astype(np.float32), dest)
        self.M = M
        rotation = self._calculate_derotation_matrix(M)
        translation = self._calculate_detranslation_matrix(
            rotation, target_after_transform)

        # Multiply 3 matrix together in backward order: M rotation translation
        rot_trans = np.matmul(translation, rotation)

        self._intrinsic_matrix = np.matmul(rot_trans, M)
        return self._intrinsic_matrix

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

    def create_occupancy_grid(self, segmap):
        # segmap must have the same size
        segmap = np.add(segmap, 1)
        warped_img = cv2.warpPerspective(segmap, self._intrinsic_matrix,
                                         (self.width, self.height))
        left_x = int((self.width - self.__occ_edge_pixel) / 2)
        top_y = self.height - self.__occ_edge_pixel
        warped_left_x = int(np.clip(left_x, 0, np.inf))
        warped_img = warped_img[int(np.clip(top_y, 0, np.Inf)):self.height,
                                warped_left_x:warped_left_x +
                                self.__occ_edge_pixel]

        occ_grid_left_x = int(np.clip(-left_x, 0, np.inf))
        occ_grid_top_y = int(np.clip(-top_y, 0, np.inf))

        template_occ_grid = np.zeros(shape=(self.__occ_edge_pixel,
                                            self.__occ_edge_pixel))

        template_occ_grid[occ_grid_top_y:self.__occ_edge_pixel,
                          occ_grid_left_x:occ_grid_left_x +
                          self.__occ_edge_pixel] = warped_img
        template_occ_grid = template_occ_grid.astype(np.uint8)

        occ_grid_size = template_occ_grid.shape[0]

        image_bottom_vertices = np.transpose(
            np.array([[self.width, self.height, 1], [0, self.height, 1]]))
        vertices_after_transform = np.matmul(self._intrinsic_matrix,
                                             image_bottom_vertices)

        vertices_after_transform[:, 0] /= vertices_after_transform[2, 0]
        vertices_after_transform[:, 1] /= vertices_after_transform[2, 1]
        vertices_after_transform = vertices_after_transform[0:2, :]
        vertices_after_transform[0] -= left_x
        vertices_after_transform[1] -= top_y
        vertices_after_transform_x_projection = np.copy(
            vertices_after_transform)
        vertices_after_transform_x_projection[1] = occ_grid_size
        front_value = template_occ_grid[
            int(occ_grid_size -
                250 / self.cm_per_px):int(occ_grid_size -
                                          200 / self.cm_per_px),
            int(occ_grid_size / 2 -
                100 / self.cm_per_px):int(occ_grid_size / 2 +
                                          100 / self.cm_per_px)]
        front_value = np.mean(front_value)
        front_value = int(np.round(front_value))

        unknown_area_poly = np.append(
            vertices_after_transform,
            np.flip(vertices_after_transform_x_projection, axis=1),
            axis=1)
        unknown_area_poly = np.transpose(unknown_area_poly)
        unknown_area_poly = unknown_area_poly.astype(np.int32)
        cv2.fillConvexPoly(template_occ_grid, unknown_area_poly, front_value)

        occupancy_grid = cv2.resize(
            template_occ_grid,
            (self.__occ_grid, self.__occ_grid),
        ) * 100
        occupancy_grid = np.where(occupancy_grid == 0, -1,
                                  200 - occupancy_grid)
        occupancy_grid = cv2.flip(occupancy_grid, 0)
        occupancy_grid = cv2.rotate(occupancy_grid,
                                    cv2.ROTATE_90_COUNTERCLOCKWISE)

        occupancy_grid = occupancy_grid.astype('int8')
        return occupancy_grid


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
        bev._intrinsic_matrix = intrinsic_matrix
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
            "intrinsic matrix": self._intrinsic_matrix.tolist(),
            "distance to target": self.dist2target,
            "tile_length": self.tile_length,
            "occ_grid_size": self.__occ_grid * self.__cell_size_in_m,
            "cell_size_in_m": self.__cell_size_in_m,
            "cm_per_px": self.cm_per_px,
            "yaw": self.yaw
        }
        json.dump(data, f)

    def calculate_transform_matrix(self, tile_coords):

        # Find the longest edge of the tile quadrilateral
        dist2target_px = (self.dist2target[0] / self.cm_per_px,
                          self.dist2target[1] / self.cm_per_px)
        target_in_img = (self.width / 2 - dist2target_px[0],
                         self.height - dist2target_px[1])
        max_height = self.tile_length / self.cm_per_px
        small_edge = max_height / np.sqrt(2)
        yaw = -self.yaw + np.pi / 4

        right_angle = np.pi / 2
        top_left = (target_in_img[0] +
                    small_edge * np.cos(yaw - 2 * right_angle),
                    target_in_img[1] +
                    small_edge * np.sin(yaw - 2 * right_angle))
        bot_right = (target_in_img[0] + small_edge * np.cos(yaw),
                     target_in_img[1] + small_edge * np.sin(yaw))
        top_right = (target_in_img[0] + small_edge * np.cos(yaw - right_angle),
                     target_in_img[1] + small_edge * np.sin(yaw - right_angle))
        bot_left = (target_in_img[0] + small_edge * np.cos(yaw + right_angle),
                    target_in_img[1] + small_edge * np.sin(yaw + right_angle))

        dest = np.asarray([top_left, bot_left, top_right,
                           bot_right]).astype(np.float32)
        print(dest)

        M = cv2.getPerspectiveTransform(tile_coords.astype(np.float32), dest)

        self._intrinsic_matrix = M
        return self._intrinsic_matrix

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

    def create_occupancy_grid(self, segmap):
        # segmap must have the same size
        segmap = np.add(segmap, 1)
        warped_img = cv2.warpPerspective(segmap, self._intrinsic_matrix,
                                         (self.width, self.height))
        left_x = int((self.width - self.__occ_edge_pixel) / 2)
        top_y = self.height - self.__occ_edge_pixel
        warped_left_x = int(np.clip(left_x, 0, np.inf))
        warped_img = warped_img[int(np.clip(top_y, 0, np.Inf)):self.height,
                                warped_left_x:warped_left_x +
                                self.__occ_edge_pixel]

        occ_grid_left_x = int(np.clip(-left_x, 0, np.inf))
        occ_grid_top_y = int(np.clip(-top_y, 0, np.inf))

        template_occ_grid = np.zeros(shape=(self.__occ_edge_pixel,
                                            self.__occ_edge_pixel))

        template_occ_grid[occ_grid_top_y:self.__occ_edge_pixel,
                          occ_grid_left_x:occ_grid_left_x +
                          self.__occ_edge_pixel] = warped_img
        template_occ_grid = template_occ_grid.astype(np.uint8)

        occ_grid_size = template_occ_grid.shape[0]

        image_bottom_vertices = np.transpose(
            np.array([[self.width, self.height, 1], [0, self.height, 1]]))
        vertices_after_transform = np.matmul(self._intrinsic_matrix,
                                             image_bottom_vertices)

        vertices_after_transform[:, 0] /= vertices_after_transform[2, 0]
        vertices_after_transform[:, 1] /= vertices_after_transform[2, 1]
        vertices_after_transform = vertices_after_transform[0:2, :]
        vertices_after_transform[0] -= left_x
        vertices_after_transform[1] -= top_y
        vertices_after_transform_x_projection = np.copy(
            vertices_after_transform)
        vertices_after_transform_x_projection[1] = occ_grid_size
        front_value = template_occ_grid[
            int(occ_grid_size -
                250 / self.cm_per_px):int(occ_grid_size -
                                          200 / self.cm_per_px),
            int(occ_grid_size / 2 -
                100 / self.cm_per_px):int(occ_grid_size / 2 +
                                          100 / self.cm_per_px)]
        front_value = np.mean(front_value)
        front_value = int(np.round(front_value))

        unknown_area_poly = np.append(
            vertices_after_transform,
            np.flip(vertices_after_transform_x_projection, axis=1),
            axis=1)
        unknown_area_poly = np.transpose(unknown_area_poly)
        unknown_area_poly = unknown_area_poly.astype(np.int32)
        cv2.fillConvexPoly(template_occ_grid, unknown_area_poly, front_value)

        occupancy_grid = cv2.resize(
            template_occ_grid,
            (self.__occ_grid, self.__occ_grid),
        ) * 100
        occupancy_grid = np.where(occupancy_grid == 0, -1,
                                  200 - occupancy_grid)
        occupancy_grid = cv2.flip(occupancy_grid, 0)
        occupancy_grid = cv2.rotate(occupancy_grid,
                                    cv2.ROTATE_90_COUNTERCLOCKWISE)

        occupancy_grid = occupancy_grid.astype('int8')
        return occupancy_grid


# ==========================================================================================
if __name__ == "__main__":
    img = cv2.resize(cv2.imread('test05.jpg', cv2.IMREAD_GRAYSCALE),
                     (1024, 512))
    cv2.imshow('input', img)

    target = (0, 270)
    tile_length = 60
    top_left_tile = np.array([426, 370])
    bot_left_tile = np.array([406, 409])
    top_right_tile = np.array([574, 371])
    bot_right_tile = np.array([592, 410])
    tile = np.stack(
        (top_left_tile, bot_left_tile, top_right_tile, bot_right_tile), axis=0)

    perspective_transformer = bev_transform_tools_legacy(
        img.shape, target, tile_length)
    matrix = perspective_transformer.calculate_transform_matrix(tile)
    M, dero, detran = (perspective_transformer.M, perspective_transformer.dero,
                       perspective_transformer.detran)
    warped_img = cv2.warpPerspective(img, matrix, (1024, 512))
    cv2.imshow('bev', warped_img)

    perspective_transformer.create_occ_grid_param()
    occ_grid = perspective_transformer.create_occupancy_grid(img).astype(
        'float32')
    resized_occ_grid = cv2.resize(occ_grid, (360, 360))
    cv2.imshow('occupancy_grid', resized_occ_grid)
    cv2.waitKey(0)
