import numpy as np
import cv2
from collections import deque
import matplotlib.pyplot as plt


# Define a class to receive the characteristics of each line detection
class Line():
    def __init__(self):
        # let's set n to be internal to the line class
        self.n = 5
        # was the line detected in the last iteration?
        self.detected = False
        # polynomial coefficients over last n iterations
        self.recent_coefs = deque(maxlen=self.n)
        # polynomial coefficients averaged over the last n iterations
        self.best_coefs = None
        # radii in metres over last n iterations
        self.radii = deque(maxlen=self.n)
        # radius of curvature of the line in metres.
        self.radius_of_curvature = None
        # difference in fit coefficients between last and new fits
        self.diffs = np.array([0, 0, 0], dtype='float')

    def add_coefs(self, coefs):
        if coefs is not None and self.best_coefs is not None:
            self.diffs = abs(coefs - self.best_coefs)
            if (self.diffs[0] > 0.1 or self.diffs[1] > 10.0 or self.diffs[2] > 100.) and len(self.recent_coefs) > 0:
                self.detected = False
                self.recent_coefs.popleft()
                if len(self.recent_coefs) > 0:
                    self.best_coefs = np.average(np.asarray(self.recent_coefs), axis=0)
                else:
                    self.best_coefs = None
            else:
                self.detected = True
                self.recent_coefs.append(coefs)
                self.best_coefs = np.average(np.asarray(self.recent_coefs), axis=0)
        elif coefs is not None:
            self.detected = True
            self.recent_coefs.append(coefs)
            self.best_coefs = np.average(np.asarray(self.recent_coefs), axis=0)
        else:
            self.detected = False
            if len(self.recent_coefs) > 1:
                self.recent_coefs.popleft()
                self.best_coefs = np.average(np.asarray(self.recent_coefs), axis=0)
            elif len(self.recent_coefs) == 1:
                self.recent_coefs.popleft()
                self.best_coefs = None

    def add_curve(self, curve):
        if curve is not None:
            self.radii.append(curve)
            self.radius_of_curvature = np.average(np.asarray(self.radii), axis=0)
        elif len(self.radii) > 1:
            self.radii.popleft()
            self.radius_of_curvature = np.average(np.asarray(self.radii), axis=0)
        elif len(self.radii) == 1:
            self.radii.popleft
            self.radius_of_curvature = None

    def reset(self):
        self.detected = False
        self.recent_coefs = deque(maxlen=self.n)
        self.best_coefs = None
        self.radii = deque(maxlen=self.n)
        self.radius_of_curvature = None
        self.diffs = np.array([0, 0, 0], dtype='float')


# for histograms
num_windows = 9
min_pix = 50
window_margin = 80
search_margin = 100

# for display
font = cv2.FONT_HERSHEY_DUPLEX
ym_per_pix = 30./720  # metres per pixel in y dimension
xm_per_pix = 3.7/600  # metres per pixel in x dimension

LEFT = Line()
RIGHT = Line()


def get_histogram(bird):
    return np.sum(bird[bird.shape[0]//2:, :], axis=0)


def print_histogram(bird):
    histogram = get_histogram(bird)
    plt.plot(histogram)


def blind_search_and_boxes(bird):
    histogram = get_histogram(bird)
    out_img = np.stack((bird, bird, bird), 2)*255

    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    if(leftx_base == 0):
        leftx_base = midpoint // 2
    if(rightx_base == 0):
        rightx_base = (midpoint + histogram.shape[0]) // 2  # if no base found, assume a sensible default and hope we pick something up

    window_height = np.int(bird.shape[0] / num_windows)

    nonzero = bird.nonzero()
    nonzerox = np.array(nonzero[1])
    nonzeroy = np.array(nonzero[0])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    for window in range(num_windows):
        win_y_low, win_y_high = bird.shape[0] - (window+1)*window_height, bird.shape[0] - window*window_height
        win_xleft_low, win_xleft_high = leftx_current - window_margin, leftx_current + window_margin
        win_xright_low, win_xright_high = rightx_current - window_margin, rightx_current + window_margin

        cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 5)
        cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 5)

        good_left = ((nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high) &
                     (nonzeroy >= win_y_low) & (nonzeroy < win_y_high)).nonzero()[0]
        good_right = ((nonzerox >= win_xright_low) & (nonzerox < win_xright_high) &
                     (nonzeroy >= win_y_low) & (nonzeroy < win_y_high)).nonzero()[0]

        left_lane_inds.append(good_left)
        right_lane_inds.append(good_right)

        if len(good_left) > min_pix:
            leftx_current = np.int(np.mean(nonzerox[good_left]))
        if len(good_right) > min_pix:
            rightx_current = np.int(np.mean(nonzerox[good_right]))

    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    leftx, lefty = nonzerox[left_lane_inds], nonzeroy[left_lane_inds]
    rightx, righty = nonzerox[right_lane_inds], nonzeroy[right_lane_inds]

    return out_img, leftx, lefty, rightx, righty


def blind_search(bird):
    histogram = get_histogram(bird)

    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    if(leftx_base == 0):
        leftx_base = midpoint / 2
    if(rightx_base == 0):
        rightx_base = (midpoint + histogram.shape[0]) / 2  # if no base found, assume a sensible default and hope we pick something up

    window_height = np.int(bird.shape[0]/num_windows)

    nonzero = bird.nonzero()
    nonzerox = np.array(nonzero[1])
    nonzeroy = np.array(nonzero[0])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    for window in range(num_windows):
        win_y_low, win_y_high = bird.shape[0] - (window+1)*window_height, bird.shape[0] - window*window_height
        win_xleft_low, win_xleft_high = leftx_current - window_margin, leftx_current + window_margin
        win_xright_low, win_xright_high = rightx_current - window_margin, rightx_current + window_margin

        good_left = ((nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high) &
                     (nonzeroy >= win_y_low) & (nonzeroy < win_y_high)).nonzero()[0]
        good_right = ((nonzerox >= win_xright_low) & (nonzerox < win_xright_high) &
                     (nonzeroy >= win_y_low) & (nonzeroy < win_y_high)).nonzero()[0]

        left_lane_inds.append(good_left)
        right_lane_inds.append(good_right)

        if len(good_left) > min_pix:
            leftx_current = np.int(np.mean(nonzerox[good_left]))
        if len(good_right) > min_pix:
            rightx_current = np.int(np.mean(nonzerox[good_right]))

    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    leftx, lefty = nonzerox[left_lane_inds], nonzeroy[left_lane_inds]
    rightx, righty = nonzerox[right_lane_inds], nonzeroy[right_lane_inds]

    return leftx, lefty, rightx, righty


def targeted_search(bird, left_coef, right_coef):
    nonzero = bird.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = search_margin

    left_lane_inds = ((nonzerox > (left_coef[0]*(nonzeroy**2) + left_coef[1]*nonzeroy + left_coef[2] - margin))
                      & (nonzerox < (left_coef[0]*(nonzeroy**2) + left_coef[1]*nonzeroy + left_coef[2] + margin)))

    right_lane_inds = ((nonzerox > (right_coef[0]*(nonzeroy**2) + right_coef[1]*nonzeroy + right_coef[2] - margin))
                       & (nonzerox < (right_coef[0]*(nonzeroy**2) + right_coef[1]*nonzeroy + right_coef[2] + margin)))

    # extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    return leftx, lefty, rightx, righty


def get_targeted_coloured_points(bird, left_coef, right_coef):
    out_img = np.stack((bird, bird, bird), 2)*255
    nonzero = bird.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = 100
    left_lane_inds = ((nonzerox > (left_coef[0]*(nonzeroy**2) + left_coef[1]*nonzeroy + left_coef[2] - margin))
                      & (nonzerox < (left_coef[0]*(nonzeroy**2) + left_coef[1]*nonzeroy + left_coef[2] + margin)))

    right_lane_inds = ((nonzerox > (right_coef[0]*(nonzeroy**2) + right_coef[1]*nonzeroy + right_coef[2] - margin))
                       & (nonzerox < (right_coef[0]*(nonzeroy**2) + right_coef[1]*nonzeroy + right_coef[2] + margin)))

    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    return out_img, leftx, lefty, rightx, righty


def not_zero(size1, size2):
    return size1 > 0 and size2 > 0


def display_lines(bird):
    out_img, leftx, lefty, rightx, righty = blind_search_and_boxes(bird)
    if not_zero(leftx.size, rightx.size):
        height = bird.shape[0]
        left_coef = np.polyfit(lefty, leftx, 2)
        right_coef = np.polyfit(righty, rightx, 2)

        ploty = np.linspace(0, height-1, height)
        left_fitted_line = left_coef[0]*ploty**2 + left_coef[1]*ploty + left_coef[2]
        right_fitted_line = right_coef[0]*ploty**2 + right_coef[1]*ploty + right_coef[2]

        LEFT.a, LEFT.b, LEFT.c = left_coef[0], left_coef[1], left_coef[2]
        RIGHT.a, RIGHT.b, RIGHT.c = right_coef[0], right_coef[1], right_coef[2]

        out_img[lefty, leftx] = [255, 0, 0]
        out_img[righty, rightx] = [0, 0, 255]

        plt.imshow(out_img)
        plt.plot(left_fitted_line, ploty, color='yellow')
        plt.plot(right_fitted_line, ploty, color='yellow')
        plt.xlim(0, 1280)
        plt.ylim(720, 0)
        plt.show()
        return out_img
    else:
        return bird


def get_lines(bird):
    out_img, leftx, lefty, rightx, righty = blind_search_and_boxes(bird)
    if not_zero(leftx.size, rightx.size):
        height = bird.shape[0]
        left_coef = np.polyfit(lefty, leftx, 2)
        right_coef = np.polyfit(righty, rightx, 2)

        ploty = np.linspace(0, height-1, height)
        left_fitted_line = left_coef[0]*ploty**2 + left_coef[1]*ploty + left_coef[2]
        right_fitted_line = right_coef[0]*ploty**2 + right_coef[1]*ploty + right_coef[2]

        LEFT.a, LEFT.b, LEFT.c = left_coef[0], left_coef[1], left_coef[2]
        RIGHT.a, RIGHT.b, RIGHT.c = right_coef[0], right_coef[1], right_coef[2]

        out_img[lefty, leftx] = [255, 0, 0]
        out_img[righty, rightx] = [0, 0, 255]

        plt.plot(left_fitted_line, ploty, color='yellow')
        plt.plot(right_fitted_line, ploty, color='yellow')
        return out_img
    else:
        return bird


def display_lines_targeted(bird):
    left_coef = [LEFT.a, LEFT.b, LEFT.c]
    right_coef = [RIGHT.a, RIGHT.b, RIGHT.c]
    out_img, leftx, lefty, rightx, righty = get_targeted_coloured_points(bird, left_coef, right_coef)
    if not_zero(leftx.size, rightx.size):
        window_img = np.zeros_like(out_img)

        height = bird.shape[0]

        ploty = np.linspace(0, height - 1, height)

        left_coef = np.polyfit(lefty, leftx, 2)
        right_coef = np.polyfit(righty, rightx, 2)
        left_fitted_line = left_coef[0]*ploty**2 + left_coef[1]*ploty + left_coef[2]
        right_fitted_line = right_coef[0]*ploty**2 + right_coef[1]*ploty + right_coef[2]

        left_line_window1 = np.array([np.transpose(np.stack([left_fitted_line - search_margin, ploty], axis=0))])
        left_line_window2 = np.array([np.flipud(np.transpose(np.stack([left_fitted_line + search_margin, ploty], axis=0)))])
        left_line_pts = np.hstack((left_line_window1, left_line_window2))

        right_line_window1 = np.array([np.transpose(np.stack([right_fitted_line - search_margin, ploty], axis=0))])
        right_line_window2 = np.array([np.flipud(np.transpose(np.stack([right_fitted_line + search_margin, ploty], axis=0)))])
        right_line_pts = np.hstack((right_line_window1, right_line_window2))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(window_img, np.int_([left_line_pts]), (0, 255, 0))
        cv2.fillPoly(window_img, np.int_([right_line_pts]), (0, 255, 0))
        result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
        # plt.imshow(result)
        plt.plot(left_fitted_line, ploty, color='yellow')
        plt.plot(right_fitted_line, ploty, color='yellow')
        plt.xlim(0, 1280)
        plt.ylim(720, 0)
        return result
    else:
        return bird


def get_radius(undist, bird, Minv):
    leftx, lefty, rightx, righty = blind_search(bird)
    width, height = bird.shape[1], bird.shape[0]
    mid = width / 2
    ploty = np.linspace(0, height - 1, height)

    left_coef, right_coef = np.polyfit(lefty, leftx, 2), np.polyfit(righty, rightx, 2)

    left_fitted_line = left_coef[0]*ploty**2 + left_coef[1]*ploty + left_coef[2]
    right_fitted_line = right_coef[0]*ploty**2 + right_coef[1]*ploty + right_coef[2]

    y_eval = height
    left_fitted_point = left_coef[0]*height**2 + left_coef[1]*height + left_coef[2]
    right_fitted_point = right_coef[0]*height**2 + right_coef[1]*height + right_coef[2]

    left_fit_cr = np.polyfit(lefty * ym_per_pix, leftx * xm_per_pix, 2)
    right_fit_cr = np.polyfit(righty * ym_per_pix, rightx * xm_per_pix, 2)
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    curverad = (left_curverad + right_curverad) / 2

    centre_dist = (abs(mid - ((right_fitted_point + left_fitted_point) / 2))) * xm_per_pix

    # Create an image to draw the lines on
    bird_zero = np.zeros_like(bird).astype(np.uint8)
    color_warp = np.stack((bird_zero, bird_zero, bird_zero), 2)

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitted_line, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitted_line, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane area and lines onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))
    cv2.polylines(color_warp, np.int_([pts_left]), 0, (255, 0, 0), thickness=30)
    cv2.polylines(color_warp, np.int_([pts_right]), 0, (255, 0, 0), thickness=30)

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (width, height))
    # Combine the result with the original image and add text
    result = cv2.addWeighted(undist, 1, newwarp, 0.5, 0)

    # Create and append text regarding distance from centre and radii
    dist_string = 'Centre offset: ' + str(round(centre_dist, 3)) + 'm'
    curve_string = 'Radius: ' + str(round(curverad, 3)) + 'm'

    cv2.putText(result, dist_string, (670, 70), font, 1.5, (255, 255, 255), 10)
    cv2.putText(result, curve_string, (670, 140), font, 1.5, (255, 255, 255), 10)

    return result


def video_pipeline(undist, bird, Minv):
    if not LEFT.detected and not RIGHT.detected:
        leftx, lefty, rightx, righty = blind_search(bird)
    else:
        leftx, lefty, rightx, righty = targeted_search(bird, LEFT.best_coefs, RIGHT.best_coefs)

    left_coef, right_coef = None, None
    if not_zero(leftx.size, rightx.size):
        left_coef, right_coef = np.polyfit(lefty, leftx, 2), np.polyfit(righty, rightx, 2)
    else:
        return undist

    LEFT.add_coefs(left_coef)
    RIGHT.add_coefs(right_coef)

    if LEFT.best_coefs is not None and RIGHT.best_coefs is not None:
        width, height = bird.shape[1], bird.shape[0]
        mid = width / 2

        ploty = np.linspace(0, height - 1, height)
        left_fitted_line = LEFT.best_coefs[0]*ploty**2 + LEFT.best_coefs[1]*ploty + LEFT.best_coefs[2]
        right_fitted_line = RIGHT.best_coefs[0]*ploty**2 + RIGHT.best_coefs[1]*ploty + RIGHT.best_coefs[2]

        left_fitted_point = LEFT.best_coefs[0]*height**2 + LEFT.best_coefs[1]*height + LEFT.best_coefs[2]
        right_fitted_point = RIGHT.best_coefs[0]*height**2 + RIGHT.best_coefs[1]*height + RIGHT.best_coefs[2]

        left_fit_cr = np.polyfit(lefty * ym_per_pix, leftx * xm_per_pix, 2)
        right_fit_cr = np.polyfit(righty * ym_per_pix, rightx * xm_per_pix, 2)
        left_curverad = ((1 + (2*left_fit_cr[0]*height*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
        LEFT.add_curve(left_curverad)
        right_curverad = ((1 + (2*right_fit_cr[0]*height*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
        RIGHT.add_curve(right_curverad)
        curverad = (LEFT.radius_of_curvature + RIGHT.radius_of_curvature) / 2

        centre_dist = (abs(mid - ((right_fitted_point + left_fitted_point) / 2))) * xm_per_pix

        # Create an image to draw the lines on
        bird_zero = np.zeros_like(bird).astype(np.uint8)
        color_warp = np.stack((bird_zero, bird_zero, bird_zero), 2)

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left_fitted_line, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitted_line, ploty])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane area and lines onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))
        cv2.polylines(color_warp, np.int_([pts_left]), 0, (255, 0, 0), thickness=30)
        cv2.polylines(color_warp, np.int_([pts_right]), 0, (255, 0, 0), thickness=30)

        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        newwarp = cv2.warpPerspective(color_warp, Minv, (width, height))
        # Combine the result with the original image and add text
        result = cv2.addWeighted(undist, 1, newwarp, 0.5, 0)

        # Create and append text regarding distance from centre and radii
        dist_string = 'Centre offset: ' + str(round(centre_dist, 3)) + 'm'
        curve_string = 'Radius: ' + str(round(curverad, 3)) + 'm'

        cv2.putText(result, dist_string, (670, 70), font, 1.5, (255, 255, 255), 10)
        cv2.putText(result, curve_string, (670, 140), font, 1.5, (255, 255, 255), 10)

        return result
    else:
        return undist
