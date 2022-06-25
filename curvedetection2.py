import cv2
import edge_detection as edge
import os
import laneDetection as ld
import numpy as np
import matplotlib.pyplot as plt

def finding_roi(img):
    # finding the region of intrest so at to focus only on the required zone:-
    # I initally tried hardcoding the region, but I found a dynamic approch to overcome the change in input image size.
    # this approch crops the region according to the percentage.
    mask = np.zeros_like(img)
    r, c = img.shape[:2]
    bl = [c * 0.1, r * 0.90]
    tl = [c * 0.5, r * 0.6]
    br = [c * 0.9, r * 0.90]
    tr = [c * 0.55, r * 0.6]
    cv2.fillPoly(mask, np.array([[bl, tl, tr, br]], dtype=np.int32), color=(255, 255, 255))
    roi = cv2.bitwise_and(img, mask)
    return roi

def perspective_transform(img):
    r, c = img.shape[:2]
    bl = [c * 0.1, r * 0.90]
    tl = [c * 0.5, r * 0.6]
    br = [c * 0.9, r * 0.90]
    tr = [c * 0.55, r * 0.6]
    roi_points = np.float32([tl,bl,br,tr])
    padding = int(0.2 * r)
    desired_roi_points = np.float32([
        [padding, 0],  # Top-left corner
        [padding, c],  # Bottom-left corner
        [r - padding, c],  # Bottom-right corner
        [r - padding, 0]  # Top-right corner
    ])

    transformation_matrix = cv2.getPerspectiveTransform(roi_points, desired_roi_points)
    inv_transformation_matrix = cv2.getPerspectiveTransform(desired_roi_points, roi_points)

    warped_frame = cv2.warpPerspective(img, transformation_matrix, (img.shape[0],img.shape[1]), flags=(cv2.INTER_LINEAR))
    grey_roi_transformed = cv2.cvtColor(warped_frame,cv2.COLOR_BGR2GRAY)

    # Convert image to binary
    (thresh, binary_warped) = cv2.threshold(grey_roi_transformed, 127, 255, cv2.THRESH_BINARY)
    # cv2.imshow('img',warped_frame)
    return binary_warped,inv_transformation_matrix

def get_lane_indices(roi_transformed,img,inv_transformation_matrix):
    no_of_windows = 20
    margin = int((1 / 12) * roi_transformed.shape[1])  # Window width is +/- margin
    minpix = int((1 / 34) * roi_transformed.shape[1])  # Min no. of pixels to recenter window

    frame_sliding_window = roi_transformed.copy()
    # cv2.imshow("imag", frame_sliding_window)
    # cv2.waitKey(1000)

    # Set the height of the sliding windows
    window_height = int(roi_transformed.shape[0] / no_of_windows)

    nonzero = roi_transformed.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    # Store the pixel indices for the left and right lane lines
    left_lane_inds = []
    right_lane_inds = []

    # Generate the histogram
    histogram = np.sum(roi_transformed[int(roi_transformed.shape[0] / 2):, :], axis=0)

    # Current positions for pixel indices for each window, which we will continue to update

    leftx_base, rightx_base = histogram_peak(histogram)
    leftx_current = leftx_base
    rightx_current = rightx_base

    for window in range(no_of_windows):

        # Identify window boundaries in x and y (and right and left)
        win_y_low = roi_transformed.shape[0] - (window + 1) * window_height
        win_y_high = roi_transformed.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        cv2.rectangle(frame_sliding_window, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (255, 255, 255), 10)
        cv2.rectangle(frame_sliding_window, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (255, 255, 255), 10)

        # cv2.imshow("imag", frame_sliding_window)
        # cv2.waitKey(1000)

        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= win_xleft_low) & (
                                  nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                           (nonzerox >= win_xright_low) & (
                                   nonzerox < win_xright_high)).nonzero()[0]

        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        # If you found > minpix pixels, recenter next window on mean position
        if len(good_left_inds) > minpix:
            leftx_current = int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = int(np.mean(nonzerox[good_right_inds]))
            # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract the pixel coordinates for the left and right lane lines
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit a second order polynomial curve to the pixel coordinates for
    # the left and right lane lines
    if (len(lefty)> 0 and len(leftx) > 0 and len(righty) > 0 and len(rightx) > 0):
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

        out_img = np.dstack((frame_sliding_window, frame_sliding_window, (frame_sliding_window))) * 255
        window_img = np.zeros_like(out_img)
        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

        # Create the x and y values to plot on the image
        ploty = np.linspace(0, roi_transformed.shape[0] - 1, roi_transformed.shape[0])
        left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

        # Generate an image to draw the lane lines on
        warp_zero = np.zeros_like(roi_transformed).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw lane on the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

        # Warp the blank back to original image space using inverse perspective
        # matrix (Minv)
        newwarp = cv2.warpPerspective(color_warp, inv_transformation_matrix, (img.shape[1], img.shape[0]))

        # Combine the result with the original image
        result = cv2.addWeighted(img, 1, newwarp, 0.3, 0)

    else:
        result = img
    return result


def histogram_peak(histogram):
    midpoint = int(histogram.shape[0] / 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # (x coordinate of left peak, x coordinate of right peak)
    return leftx_base, rightx_base

if __name__ =="__main__":
    # img = cv2.imread('/Users/saraf/PycharmProjects/laneDetection/venv/images/01. solidWhiteCurve.jpeg')
    #
    # hls_img = cv2.cvtColor(img,cv2.COLOR_BGR2HLS)
    #
    # # thresholding the ligtness channel to create a binary mask that has great contrast. This would be helpful for edge
    # # generation, this helps overcome shadows.
    # _, lthresh = cv2.threshold(hls_img[:,:,1], 120, 255, cv2.THRESH_BINARY)
    # sxbinary = edge.blur_gaussian(lthresh, ksize=3)  # Reduce noise
    #
    # # sxbinary = edge.mag_thresh(sxbinary, sobel_kernel=3, thresh=(110, 255))
    # sxbinary = cv2.Canny(sxbinary,110,255)
    # # sxbinary = cv2.bitwise_not(sxbinary)
    #
    # # Perform binary thresholding on the S (saturation) channel
    # # of the video frame. A high saturation value means the hue color is pure.
    # # We expect lane lines to be nice, pure colors (i.e. solid white, yellow)
    # # and have high saturation channel values.
    # # s_binary is matrix full of 0s (black) and 255 (white) intensity values
    # # White in the regions with the purest hue colors (e.g. >80...play with
    # # this value for best results).
    # s_channel = hls_img[:, :, 2]  # use only the saturation channel data
    # _, s_binary = cv2.threshold(s_channel, 80, 255, cv2.THRESH_BINARY)
    #
    # # Perform binary thresholding on the R (red) channel of the
    # # original BGR video frame.
    # # r_thresh is a matrix full of 0s (black) and 255 (white) intensity values
    # # White in the regions with the richest red channel values (e.g. >120).
    # # Remember, pure white is bgr(255, 255, 255).
    # # Pure yellow is bgr(0, 255, 255). Both have high red channel values.
    # _, r_binary = cv2.threshold(img[:,:,2], 200, 255, cv2.THRESH_BINARY)
    #
    # # Lane lines should be pure in color and have high red channel values
    # # Bitwise AND operation to reduce noise and black-out any pixels that
    # # don't appear to be nice, pure, solid colors (like white or yellow lane
    # # lines.)
    # rs_binary = cv2.bitwise_and(s_binary, r_binary)
    #
    # lanes = cv2.bitwise_or(sxbinary,r_binary)
    #
    # roi_transformed, inv_transform = perspective_transform(img)
    # get_lane_indices(roi_transformed,img,inv_transform)
    #
    # cv2.imshow('img', roi_transformed)
    # cv2.waitKey(0)

    cap = cv2.VideoCapture(os.getcwd() +'/Videos/challenge.mp4')
    while (cap.isOpened()):
        # Capture frame-by-frame
        ret, img = cap.read()
        if ret == True:
            hls_img = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
            hls_img = finding_roi(hls_img)
            # cv2.imshow('roi',hls_img )
            # thresholding the ligtness channel to create a binary mask that has great contrast. This would be helpful for edge
            # generation, this helps overcome shadows.
            _, lthresh = cv2.threshold(hls_img[:,:,1], 120, 255, cv2.THRESH_BINARY)
            sxbinary = edge.blur_gaussian(lthresh, ksize=3)  # Reduce noise
            # cv2.imshow('lightning threshold',lthresh )

            # sxbinary = edge.mag_thresh(sxbinary, sobel_kernel=3, thresh=(110, 255))
            sxbinary = cv2.Canny(sxbinary,110,255)
            # sxbinary = cv2.bitwise_not(sxbinary)

            # Perform binary thresholding on the S (saturation) channel
            # of the video frame. A high saturation value means the hue color is pure.
            # We expect lane lines to be nice, pure colors (i.e. solid white, yellow)
            # and have high saturation channel values.
            # s_binary is matrix full of 0s (black) and 255 (white) intensity values
            # White in the regions with the purest hue colors (e.g. >80...play with
            # this value for best results).
            s_channel = hls_img[:, :, 2]  # use only the saturation channel data
            _, s_binary = cv2.threshold(s_channel, 80, 255, cv2.THRESH_BINARY)
            # cv2.imshow('lightning threshold', s_binary)

            # Perform binary thresholding on the R (red) channel of the
            # original BGR video frame.
            # r_thresh is a matrix full of 0s (black) and 255 (white) intensity values
            # White in the regions with the richest red channel values (e.g. >120).
            # Remember, pure white is bgr(255, 255, 255).
            # Pure yellow is bgr(0, 255, 255). Both have high red channel values.
            rgb_roi = finding_roi(img)
            _, r_binary = cv2.threshold(rgb_roi[:,:,2], 200, 255, cv2.THRESH_BINARY)
            # cv2.imshow('red threshold', r_binary)

            # Lane lines should be pure in color and have high red channel values
            # Bitwise AND operation to reduce noise and black-out any pixels that
            # don't appear to be nice, pure, solid colors (like white or yellow lane
            # lines.)
            rs_binary = cv2.bitwise_and(s_binary, r_binary)
            # cv2.imshow('and of threshold', rs_binary)

            lanes = cv2.bitwise_or(sxbinary,r_binary)
            # cv2.imshow('lanes', lanes)

            roi_transformed, inv_transform = perspective_transform(img)
            output = get_lane_indices(roi_transformed,img,inv_transform)
            cv2.imshow('Frame', output)

            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        else:
            break




