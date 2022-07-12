import cv2
import imutils


def ObjectTrackHSV():

    hsv_lower, hsv_upper = (150, 69, 127), (179, 255, 255)  # HSV color space bound for RED COLOR..hmm my perfume bottle cap
    video = cv2.VideoCapture(0)  # using system primary camera

    while True:

        flag, live = video.read()

        if flag:

            live = imutils.resize(live, width=600)
            blur_image = cv2.GaussianBlur(live, (11, 11), 0)  # help to smoothen the image
            hsv_image = cv2.cvtColor(blur_image, 40)  # cv.COLOR_BGR2HSV = 40, hsv color space

            hsv_extract_img = cv2.inRange(hsv_image, hsv_lower, hsv_upper)  # extract threshold image
            hsv_extract_img1 = cv2.erode(hsv_extract_img, None, iterations=3)  # erode and dilate helps to get smooth shape
            hsv_extract_img2 = cv2.dilate(hsv_extract_img1, None, iterations=3)

            contours = cv2.findContours(hsv_extract_img2.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]

            centre = None

            if len(contours) > 0:

                maximum_contour = max(contours, key=cv2.contourArea)

                ((x, y), radius) = cv2.minEnclosingCircle(maximum_contour)

                moments = cv2.moments(maximum_contour)
                centre = (int(moments["m10"]/moments["m00"]), int(moments["m01"]/moments["m00"]))

                if radius > 10:
                    cv2.circle(live, (int(x), int(y)), int(radius), (0, 255, 0), 1)
                    cv2.circle(live, centre, 5, (0, 0, 255), -2)

            cv2.imshow("Object Tracking", live)
            if cv2.waitKey(1) == 32:
                break

    video.release()
    cv2.destroyAllWindows()

