Lane detection using opencv

Pipeline architecture:

    1] Load video.

    2] Extract the region of interest.

    3] Color masking. 

    4] Apply Canny edge detection.
        -Convert the image to grey scale.
        -Apply Gaussian smoothing to remove any noise.
        -Perform Canny edge detection.

    5] Apply Hough transform.

    6] Average and extrapolating the lane lines.


Extracting the region of interest:- 
  Since out goal is so specific as to detect the lanes in the given camera feed, the region on which we do the further processing should also be
  confined. Camera projects a 3D world to a 2D plane, while doing so the parallel lines converge at a vanishing point on the 2D plane. Similarly 
  the the lanes being parallel get narrow as we move up the the image. They usually lie in a trapezoidal zone with side closer to car being broad
  and the further away being narrow. To crop the image feed only to this limited trapezoidal zone i made a mask of similar size as of the image. 
  Initially i hardcoded the corner points of the trapezoid but eventually figured out I can make it dynamic by constraint the size by certain 
  percentage of it. This help overcome the challenge of dimension change with different image feed. I then used the cv2's fillPoly function to 
  overlay roi on to the mask. The function takes in the mask on which we want create the polynomial, the corners of the polynomial and the color 
  of it. We then simply bitwise-and with the raw image to extract the pixel that are only in our trapezoid.
  
  
Color Masking:-
  Once we have got our region of interest we the move to extracting just the lanes out of them. The lanes are usually of two colors yellow and 
  white. There are different color formats used to represent an image, each having its own characteristic. I played around with 3 of them to see 
  which one of them helps me segregate lanes out of image the most. Color schemes that I tried were RGB, HSV, HLS and I used the HLS color space,
  which divides all colors into hue, saturation, and lightness values. It helps eleminate detection problems due to lighting, such as shadows, 
  glare from the sun, headlights, etc. By trial and error I set the higher and lower threshold for each color separately and create a mask using 
  them. To create mask I have used CV2's inRange function which takes values only in the given range. Later i bitwise or the masks to create a 
  final mask which thresholds for both white and yellow. This final mask is bitwise-and with the roi to extract just the lanes out of.
  
  
Canny Edge Detection:- 
The CannyEdge detection function of opencv requires a grey format image input. Thus I converted the masked image to 
greyscale and to remove any noise present applied Gaussian blur. In background the canny edge detection uses 2 techniques Non-maximum Suppression
and Hysteresis thresholding, the non-maximum suppression is responsible for making the edges thin whereas hysteresis decided the continuity of the edges. Hysteresis takes in 2 threshold values, maxthresold and min threshold.Any edges with intensity gradient more than maxVal are sure to be edges and those below minVal are sure to be non-edges, so discarded. Those who lie between these two thresholds are classified edges or non-edges based on their connectivity. If they are connected to "sure-edge" pixels, they are considered to be part of edges. Otherwise, they are also 
discarded. This threshold values where decided by experimentation.


Hough Transform:- 
  Hough transform, transforms the coordinate space into parametric space. In case of line it uses the normal line representation for mapping it 
  into the parametric space using the equation ρ = x cos(θ) + y sin(θ) where ρ is the length of the normal line and θ is the angle between the 
  normal line and the x axis. Cv2's HoughLinesP makes it easy to to extract lines from the image, but these are line formed using only the edges,
  therefore they are no continuous. To extract 2 continues lines indicating lanes on each side, we need to average and extrapolate lines from all
  the detected lines
  
  
Average and extrapolating the lane lines:- 
  Given all the lines detected by the hough transform how do we segregate the left lane and right lane lines? The answer lines in the slope, 
  since the lane seems to be converging to a point, the line representing the left lane would have a negative slope whereas the right lane line 
  would have a positive slope. We thus find slope and length for each detected line ( hough return the end coordinates). We save this sloped and 
  intercept. Since slope and intercept cannot be used to draw a line using the cv2.line function we convert the slope-intercept format to points.
  The y1 and the y2 points are easy to extract as one is the last row of the image where as the other would be some percent of the y1. The 
  challenge was extracting the x1,x2 coordinates. This could be done using the equation of line. We the finally plot the lines indicating the 
  lanes.

This is a trivial scenario taken into consideration, it cannot handle the curvature of the road which would be the challenge for next code.

The pipeline for which would be:-

Detection of lane pixel invariant to external factors:-
    
