import numpy as np
import cv2

# for edges
sobel_kernel = 3
abs_thresh = (20, 100)
mag_thresh = (10, 100)
direc_thresh = (0.7, 1.3)

# for colour spaces
gray_thresh = (180, 255)
hue_thresh = (15, 100)
lit_thresh = (220, 255)
sat_thresh = (90, 255)
red_thresh = (200, 255)
b_thresh = (190, 255)


def threshold_mag(sobelx, sobely):
    abs_sobelxy = np.sqrt(np.square(sobelx) + np.square(sobely))

    condition = (abs_sobelxy >= mag_thresh[0]) & (abs_sobelxy <= mag_thresh[1])
    binary = np.zeros_like(abs_sobelxy)
    binary[condition] = 1
    return binary


def threshold_direc(sobelx, sobely):
    direction = np.arctan2(sobely, sobelx)

    condition = (direction >= direc_thresh[0]) & (direction <= direc_thresh[1])
    binary = np.zeros_like(direction)
    binary[condition] = 1
    return binary


def threshold_abs(sobel):
    scaled = np.uint8((255 * sobel)/np.max(sobel))
    condition = (scaled >= abs_thresh[0]) & (scaled <= abs_thresh[1])
    binary = np.zeros_like(scaled)
    binary[condition] = 1
    return binary


def threshold_grad(image, do_mag=False, do_direc=False, do_grad=False):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sobelx = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
    sobely = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel))

    mag_map = np.zeros_like(gray)
    direc_map = mag_map
    abs_x_map = abs_y_map = direc_map

    if(do_mag):
        mag_map = threshold_mag(sobelx, sobely)
    if(do_direc):
        direc_map = threshold_direc(sobelx, sobely)
    if(do_grad):
        abs_x_map, abs_y_map = threshold_abs(sobelx), threshold_abs(sobely)

    gradient_binary = np.zeros_like(mag_map)
    condition = ((mag_map == 1) & (direc_map == 1)) | ((abs_x_map == 1) & (abs_y_map == 1))
    gradient_binary[condition] = 1

    return gradient_binary


def threshold_colour(image, do_gray=False, do_r=False, do_s=False, do_h=False, do_l=False, do_b=False):
    gray = np.zeros_like(image[:, :, 0])  # we only need one channel
    b = red = sat = lit = hue = gray

    if(do_gray):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if(do_r):
        red = image[:, :, 0]
    if(do_h and do_l and do_s):
        hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
        hue = hls[:, :, 0]
        lit = hls[:, :, 1]
        sat = hls[:, :, 2]  # more efficient to index into the array thrice rather than have three function calls per frame!
    elif(do_s):
        sat = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)[:, :, 2]
    elif(do_l):
        lit = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)[:, :, 1]
    elif(do_h):
        hue = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)[:, :, 0]
    if(do_b):
        b = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)[:, :, 2]

    colour_binary = np.zeros_like(gray)
    gray_condition = ((gray >= gray_thresh[0]) & (gray <= gray_thresh[1]))
    red_condition = ((red >= red_thresh[0]) & (red <= red_thresh[1]))
    hue_condition = ((hue >= hue_thresh[0]) & (hue <= hue_thresh[1]))
    lit_condition = ((lit >= lit_thresh[0]) & (lit <= lit_thresh[1]))
    sat_condition = ((sat >= sat_thresh[0]) & (sat <= sat_thresh[1]))
    b_condition = ((b >= b_thresh[0]) & (b <= b_thresh[1]))
    condition = (gray_condition & red_condition) | (hue_condition | lit_condition | sat_condition) | (b_condition)
    colour_binary[condition] = 1

    return colour_binary


# Loads of flags, but keeps things easy to tweak further down.
def threshold(image, do_mag=False, do_direc=False, do_grad=False, do_gray=False, do_r=False, do_s=False, do_h=False, do_l=False, do_b=False):
    gradient_binary = threshold_grad(image, do_direc=do_direc, do_mag=do_mag, do_grad=do_grad)
    colour_binary = threshold_colour(image, do_gray=do_gray, do_r=do_r, do_s=do_s, do_h=do_h, do_l=do_l, do_b=do_b)

    ultimate_threshold_to_end_all_thresholds = np.zeros_like(image[:, :, 0])
    ultimate_condition = (gradient_binary == 1) | (colour_binary == 1)
    ultimate_threshold_to_end_all_thresholds[ultimate_condition] = 1

    return ultimate_threshold_to_end_all_thresholds


def final_threshold(image):
    binary = np.zeros_like(image[:, :, 0])

    b = cv2.cvtColor(image, cv2.COLOR_RGB2Lab)[:, :, 2]
    lit = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)[:, :, 1]
    lit = lit * (255 / np.max(lit))
    if np.max(b) > 175:
        b = b * (255 / np.max(b))

    b_condition = (b > b_thresh[0]) & (b <= b_thresh[1])
    lit_condition = (lit > lit_thresh[0]) & (lit <= lit_thresh[1])

    condition = (b_condition == 1) | (lit_condition == 1)
    binary[condition] = 1

    return binary
