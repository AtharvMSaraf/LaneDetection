# LandDetection
Lane detection using opencv

Pipeline architecture:

  1] Load video.
  2] Extract the region of interest.
  3] Color masking
  3] Apply Canny edge detection.
      -Convert the image to grey scale.
      -Apply Gaussian smoothing to remove any noise.
      -Perform Canny edge detection.
  4] Apply Hough transform.
  5] Average and extrapolating the lane lines.

Extracting the region of intrest:-
  Since out goal is so specific as to detect the lanes in the given camera feed, the region on which we do the further processing should aslo be
  confined. Camera projects a 3D world to a 2D plane, while doing so the parrelel lines converge at a vanishing point on the 2D plane. Similarly
  the the lanes being parrelel get narrow as we move up the the image. They usually lie in a tropozoidal zone with side closer to car being broad
  and the further away being narrow. 
    To crop the image feed only to this limited trapozoidal zone i made a mask of similar size as of the image. Initally i hardcoded the corner
  points of the trapozoid but eventually figured out I can make it dynamic by constraing the size by certain percentage of it. This help overcome
  the challenge of dimmension change with different image feed. I then used the cv2's fillPoly fuction to overlay roi on to the mask. The function
  takes in the mask on which we want create the polynomial, the corners of the polynomial and the colour of it. We then simply bitwise-and with
  the raw image to extract the pixel that are only in our trapezoid. 

Color Masking:-
  Once we have got our region of intrest we the move to extracting just the lanes out of them. The lanes are usually of two colors yellow and 
  white. There are different color formats used to represent an image, each having its own charaterstic. I played around with 3 of them to see
  which one of them helps me segrigate lanes out of image the most. Colour sceams that I tried were RGB, HSV, HSL and finalised the HLS (hue,
  saturation,lightess) format. By trial and error I set the higher and lower threshold for each color seperately and create a mask using them. 
  To create mask I have used CV2's inRange function which takes values only in the given range. Later i bitwise or the masks to creat a final 
  mask which thresholds for both white and yellow. This final mask is bitwise-and with the roi to extract just the lanes out of.
  
Canny Edge Detection:-
  The CannyEdge detection function of opencv requires a grey format image input. Thus I converted the masked image to greyscale and to remove any
  noise present applied guassian blur. In background the canny edge detection uses 2 techniques Non-maximum Supression and Hysteresis thresholding,
  the non-maximum supression is resposnisble for making the edges thin whereas hysteresis decied the continuity of the edges. Hysteresis takes in
  2 thresold values, maxthresold and min threshold.Any edges with intensity gradient more than maxVal are sure to be edges and those below minVal 
  are sure to be non-edges, so discarded. Those who lie between these two thresholds are classified edges or non-edges based on their connectivity. 
  If they are connected to "sure-edge" pixels, they are considered to be part of edges. Otherwise, they are also discarded. This thresold values
  where decided by expreimentation.
  
Hough Transform:- 
  hough transform, transforms the coordinate space into parametric space. In case of line it uses the normal line represntation for mapping it 
  into the parametric space using the equation ρ = x cos(θ) + y sin(θ) where ρ is the length of the normal line and θ is the angle between the 
  normal line and the x axis. Cv2's HoughLinesP makes it easy to to extract lines from the image, but these are line formed using oinly the edges
  , therefore they are no continuous. To extract 2 continuies lines indicating lanes on each side, we need to average and extrapolate lines
  from all the detected lines

Average and extrapolating the lane lines:-
  Given all the lines detected by the hough transform how do we segregate the left lane and right lane lines? The answer lines in the slope, since
  the lane seeems to be converging to a point, the line represnting the left lane would have a negative slope whereas the right lane line would 
  have a positive slope. We thus find slope and length for each detected line ( hough return the end cooridates). We save this sloped and 
  intercept. Since slope and intercept cannot be used to draw a line using the cv2.line functio we convert the slope-intercept format to points.
  The y1 and the y2 points are easy to extract as one is the last row of the image where as the other would be some percent of the y1. The 
  challenge was extracting the x1,x2 coordinates. This could be done using the equation of line. We the finally plot the lines indicating the 
  lanes.
  
  
  This is a trivial scenario taken into consideration, it cannot handle the curvature of the road which would be the challenge for next code. 
 
  
  
