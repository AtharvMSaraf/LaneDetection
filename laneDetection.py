import cv2
import numpy as np
import os

def finding_roi(img):
    # finding the region of intrest so at to focus only on the required zone:-
    # I initally tried hardcoding the region, but I found a dynamic approch to overcome the change in input image size.
    # this approch crops the region according to the percentage.
    mask = np.zeros_like(img)
    r, c = img.shape[:2]
    bl = [c * 0.1, r * 0.90]
    tp = [c * 0.4, r * 0.6]
    br = [c * 0.9, r * 0.90]
    tr = [c * 0.6, r * 0.6]
    cv2.fillPoly(mask, np.array([[bl, tp, tr, br]], dtype=np.int32), color=(255, 255, 255))
    roi = cv2.bitwise_and(img, mask)
    return roi

def create_colourmask(hls_img):
    # white colour mask
    lower_threshold = np.uint8([0, 200, 0])
    upper_threshold = np.uint8([255, 255, 255])
    white_mask = cv2.inRange(hls_img, lower_threshold, upper_threshold)

    # Yellow color mask
    lower_threshold = np.uint8([50, 0, 50])
    upper_threshold = np.uint8([100, 255, 255])
    yellow_mask = cv2.inRange(hls_img, lower_threshold, upper_threshold)

    # Combine white and yellow masks
    mask = cv2.bitwise_or(white_mask, yellow_mask)
    masked_image = cv2.bitwise_and(roi, roi, mask=mask)
    return  masked_image


def convert_to_pixel(y1, y2, line):
    if line is not None:
        slope, intercept = line
        if slope != float("-inf"):
            # print(slope)
            x1 = int((y1 - intercept) / slope)
            x2 = int((y2 - intercept) / slope)
            y1 = int(y1)
            y2 = int(y2)
            return ((x1, y1), (x2, y2))

def get_lines(edge_image):
    lines = cv2.HoughLinesP(edge_image, rho=2, theta=np.pi / 180, threshold=20, minLineLength=20, maxLineGap=200)

    left_marking = []
    left_length = []
    right_marking = []
    right_length = []

    if lines is not None:

        for line in lines:
            for x1, y1, x2, y2 in line:
                if (x1 != x2):
                    slope = (y2 - y1) / (x2 - x1)
                    intercept = y1 - (slope * x1)
                    length = np.sqrt(((y2 - y1) ** 2) + ((x2 - x1) ** 2))
                    if slope < 0:
                        left_marking.append((slope, intercept))
                        left_length.append((length))
                    else:
                        right_marking.append((slope, intercept))
                        right_length.append((length))
    left = np.dot(left_length, left_marking) / np.sum(left_length) if len(left_length) > 0 else None
    right = np.dot(right_length, right_marking) / np.sum(right_length) if len(right_length) > 0 else None

    final_marking = []
    y1 = img.shape[0]
    y2 = y1 * 0.65
    left_line = convert_to_pixel(y1, y2, left)
    right_line = convert_to_pixel(y1, y2, right)
    final_marking.append(left_line)
    final_marking.append(right_line)

    for line in final_marking:
        if line is not None:
            cv2.line(img, *line, (255, 0, 0), 5)
    return img



if __name__=="__main__":
#------------------------Images----------------------------------
    # arr = os.listdir(os.getcwd() + "/images")
    # for filename in arr:
    #     img = cv2.imread(os.getcwd() + "/images" + "/" + filename)
    #     hls_img = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    #
    #     roi = finding_roi(img)
    #
    #     masked_image = create_colourmask(hls_img)
    #
    #     # for edge detection convert the image to gray scale:
    #     grey_img = cv2.cvtColor(masked_image,cv2.COLOR_RGB2GRAY)
    #     blur_grey_img = cv2.GaussianBlur(grey_img, (11, 11), 0)
    #     edge_image = cv2.Canny(blur_grey_img, 50, 150)
    #
    #     final_image = get_lines(edge_image)
    #     cv2.imshow('img',final_image)
    #     cv2.waitKey(0)

#------------------------ Video ---------------------------------

    cap = cv2.VideoCapture(os.getcwd() +'/solidWhiteRight.mp4')
    while (cap.isOpened()):
        # Capture frame-by-frame
        ret, img = cap.read()
        if ret == True:
            hls_img = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)

            roi = finding_roi(img)

            masked_image = create_colourmask(hls_img)

            # for edge detection convert the image to gray scale:
            grey_img = cv2.cvtColor(masked_image, cv2.COLOR_RGB2GRAY)
            blur_grey_img = cv2.GaussianBlur(grey_img, (11, 11), 0)
            edge_image = cv2.Canny(blur_grey_img, 50, 150)

            final_image = get_lines(edge_image)
            cv2.imshow('Frame', final_image)
            
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        else:
            break



