import cv2
import numpy as np
from numpy import pi, cos, sin
import json

# When dealing with OpenCV functions
# Numpy dimension order should be [height,width]
# Not to be confused with [x,y] axis


def setupBEVLongRange():
    img_shape = (512, 1024)
    # first number is x , second number is y
    top_left_tile = np.array([489, 380])
    bot_left_tile = np.array([483, 396])
    top_right_tile = np.array([580, 380])
    bot_right_tile = np.array([578, 396])
    tile = np.stack(
        (top_left_tile, bot_left_tile, top_right_tile, bot_right_tile), axis=0)
    target = (0, 510)
    tile_length = 60
    perspective_transformer = bev_transform_tools(img_shape, target,
                                                  tile_length)
    matrix = perspective_transformer.calculate_transform_matrix(tile)
    perspective_transformer.create_occ_grid_param()
    print("cm per pixel", perspective_transformer.cm_per_px)
    return perspective_transformer, matrix


def setupBEVShortRange():
    # there should be 2 bev parameter for each range, this is for distance from 3m
    img_shape = (512, 1024)
    # first number is x , second number is y
    top_left_tile = np.array([441, 437])
    bot_left_tile = np.array([437, 471])
    top_right_tile = np.array([568, 437])
    bot_right_tile = np.array([588, 471])
    tile = np.stack(
        (top_left_tile, bot_left_tile, top_right_tile, bot_right_tile), axis=0)
    target = (0, 330)
    tile_length = 60
    perspective_transformer = bev_transform_tools(img_shape, target,
                                                  tile_length)
    matrix = perspective_transformer.calculate_transform_matrix(tile)
    perspective_transformer.create_occ_grid_param()
    print("cm per pixel", perspective_transformer.cm_per_px)
    return perspective_transformer, matrix


def setupBEV():
    # bugs: setting long range bev will distort short range object.
    # idea : setting different bev ratio for different distance?
    return setupBEVShortRange()


class bev_transform_tools:

    # dist2target : distance from camera to the target: denoted (x,y) (cm)
    # x is the horizontal distance
    # y is the vertical distance
    def __init__(self, image_shape, dist2target, tile_length):
        self.width = image_shape[1]
        self.height = image_shape[0]
        self.dist2target = dist2target
        self.tile_length = tile_length  # in cm

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

        bev = cls(shape, dist2target, tile_length)
        bev.tile_length_pixel = data["tile_length_pixel"]
        bev.cm_per_px = (bev.tile_length / bev.tile_length_pixel)
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
            "size": (self.height, self.width),
            "intrinsic matrix": self._intrinsic_matrix.tolist(),
            "distance to target": self.dist2target,
            "tile_length": self.tile_length,
            "tile_length_pixel": self.tile_length_pixel,
            "occ_grid_size": self.__occ_grid * self.__cell_size_in_m,
            "cell_size_in_m": self.__cell_size_in_m,
            "M": self.M.tolist(),
            "derotation": self.dero.tolist(),
            "detranslation": self.detran.tolist(),
        }
        json.dump(data, f)

    def _calculate_derotation_matrix(self, M):
        # Find the rotation matrix to make the bottom line straight again
        # Counter-clockwise rotation
        bot_left_corner = np.dot(M, np.array([0, self.height, 1]))
        bot_left_corner = bot_left_corner / bot_left_corner[2]
        bot_right_corner = np.dot(M, np.array([self.width, self.height, 1]))
        bot_right_corner = bot_right_corner / bot_right_corner[2]
        diagonal_line = np.stack((bot_left_corner, bot_right_corner), axis=0)
        angle = np.arctan2((diagonal_line[1][1] - diagonal_line[0][1]),
                           (diagonal_line[1][0] - diagonal_line[0][0]))
        rotation = np.array([[np.cos(angle), -np.sin(angle), 0],
                             [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
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

        target_to_origin_in_pixel = np.asarray((self.dist2target))\
            / self.cm_per_px
        print("distance in px is", target_to_origin_in_pixel)
        target_after_derotation = np.dot(rotation_matrix, target_in_pixels)
        print("target", target_after_derotation)
        origin_after_derotation = target_after_derotation[0:2] + \
            target_to_origin_in_pixel
        print(origin_after_derotation)
        trans_x = self.width / 2 - origin_after_derotation[0]
        trans_y = self.height - origin_after_derotation[1]
        translation = np.array([[1, 0, trans_x], [0, 1, trans_y], [0, 0, 1]])

        self.detran = translation
        self.origin_in_bev = origin_after_derotation
        return translation

    # --------------------------------------------------------------------------------------

    def calculate_transform_matrix(self, tile_coords):
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
        zoom_out_ratio = 3

        # print(max_height_tile)
        # int(max_height_tile/zoom_out_ratio) #max height should be 12
        max_height = 12
        self.tile_length_pixel = max_height
        self.cm_per_px = self.tile_length / self.tile_length_pixel
        # print(self.tile_length)

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
        self.__occ_edge_pixel = int(self.__occ_grid * self.__cell_size)

    # --------------------------------------------------------------------------------------

    def create_occupancy_grid(self, segmap):
        # segmap must have the same size
        segmap = np.add(segmap, 1)
        warped_img = cv2.warpPerspective(segmap, self._intrinsic_matrix,
                                         (self.width, self.height))
        x = int((self.width - self.__occ_edge_pixel) / 2)
        y = self.height
        warped_img = warped_img[y - self.__occ_edge_pixel:y,
                                x:x + self.__occ_edge_pixel]
        warped_width, warped_height = warped_img.shape

        front_value = warped_img[
            int(warped_height -
                175 / self.cm_per_px):int(warped_height -
                                          170 / self.cm_per_px),
            int(warped_width / 2 -
                100 / self.cm_per_px):int(warped_width / 2 +
                                          100 / self.cm_per_px)]
        front_value = np.mean(front_value)
        front_value = int(np.round(front_value))
        warped_img = cv2.rectangle(
            warped_img, (int(warped_width / 2 - 100 / self.cm_per_px),
                         int(warped_height - 170 / self.cm_per_px)),
            (int(warped_width / 2 + 100 / self.cm_per_px),
             int(warped_height - 50 / self.cm_per_px)), front_value, -1)
        #warped_img = cv2.rectangle(warped_img, (0,0),(100,100),1,-1)
        #occupancy_grid = (np.ones((self.__occ_grid,self.__occ_grid))* -1).astype(np.int8)
        #visible_version_grid = np.zeros((self.occ_gridY,self.occ_gridX,3))
        occupancy_grid = cv2.resize(warped_img,
                                    (self.__occ_grid, self.__occ_grid),
                                    interpolation=cv2.INTER_NEAREST) * 100
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

    perspective_transformer = bev_transform_tools(img.shape, target,
                                                  tile_length)
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
