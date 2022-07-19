import numpy as np
import cv2

def contour_noise_removal(segmap):
    # Close small gaps by a kernel with shape proporition to the image size
    h_segmap, w_segmap = segmap.shape
    min_length = min(h_segmap, w_segmap)
    kernel = np.ones((int(min_length / 50), int(min_length / 50)), np.uint8)
    closed = cv2.morphologyEx(segmap, cv2.MORPH_CLOSE, kernel)

    # Find contours of segmap
    cnts, hie = cv2.findContours(closed, 1, 2)
    cnts = list(filter(lambda cnt: cnt.shape[0] > 2, cnts))

    # Create a rectangular mask in lower part of the frame
    # If a contour intersect with this lower part above a threshold
    # then that contour will be kept as a valid one

    LENGTH_RATIO = 0.1
    x_left = 0
    x_right = w_segmap
    y_top = int(h_segmap * (1 - LENGTH_RATIO))
    y_bot = h_segmap
    bottom_rect = np.array([(x_left,  y_top), (x_right, y_top),
                            (x_right, y_bot), (x_left,  y_bot)])
    bottom_mask = np.zeros_like(segmap)
    mask_area = (x_right - x_left) * (y_bot - y_top)
    cv2.fillPoly(bottom_mask, [bottom_rect], 1)

    # Iterate through contour[S]
    MASK_AREA_THRESH = 0.4  # The threshold of intersect over whole mask area
    main_road_cnts = []
    for cnt in cnts:
        cnt_map = np.zeros_like(segmap)
        cv2.fillPoly(cnt_map, [cnt], 1)
        masked = cv2.bitwise_and(cnt_map, bottom_mask).astype(np.uint8)
        intersected_area = np.count_nonzero(masked)
        if (intersected_area > (mask_area * MASK_AREA_THRESH)):
            main_road_cnts.append(cnt)

    contour_noise_removed = np.zeros(segmap.shape).astype(np.uint8)
    cv2.fillPoly(contour_noise_removed, main_road_cnts, 1)

    return contour_noise_removed

def clahe(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    # -----Splitting the LAB image to different channels-------------------------
    l, a, b = cv2.split(lab)

    # -----Applying CLAHE to L-channel-------------------------------------------
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)

    # -----Merge the CLAHE enhanced L-channel with the a and b channel-----------
    limg = cv2.merge((cl, a, b))

    # -----Converting image from LAB Color model to RGB model--------------------
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return final

def find_intersection_line(line1, line2):
    # line: 2x2 array representing 2 point on the line (x1,y1),(x2,y2).
    (x1, y1) = line1[0]
    (x2, y2) = line1[1]
    if x2 - x1 == 0:
        b1 = 0
        a1 = 1
        c1 = x1
    else:
        b1 = -1
        a1 = (y2 - y1) / (x2 - x1)
        c1 = (x1 * y2 - x2 * y1) / (x2 - x1)

    (x3, y3) = line2[0]
    (x4, y4) = line2[1]
    if x4 == x3:
        b2 = 0
        a2 = 1
        c2 = x3
    else:
        b2 = -1
        a2 = (y4 - y3) / (x4 - x3)
        c2 = (x3 * y4 - x4 * y3) / (x4 - x3)
    if a1 == a2:
        return
    coeff = np.array([[a1, b1], [a2, b2]])
    res = np.array([c1, c2])
    intersection = np.linalg.solve(coeff, res)
    return intersection



def create_skeleton(perspective_transformer, input_shape):
    width, height = input_shape
    free_img = np.ones((height, width))
    occ_grid = perspective_transformer.create_occupancy_grid(
        free_img, perspective_transformer._bev_matrix,
        perspective_transformer.width, perspective_transformer.height,
        perspective_transformer.map_size,
        perspective_transformer.map_resolution,
        perspective_transformer.cm_per_px)
    edges = cv2.Canny(occ_grid.astype(np.uint8), 50, 150, apertureSize=3)
    return edges