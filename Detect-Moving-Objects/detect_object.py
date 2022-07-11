import cv2
import imutils


def detect_moving_object():
    video = cv2.VideoCapture(0)  # using system primary camera

    last_frame = None
    area = 5000

    while True:
        flag, live = video.read()

        if flag:
            gray_image = cv2.cvtColor(live, 6)  # module attribute cv2.COLOR_BGR2GRAY = 6

            if last_frame is None:
                last_frame = gray_image
                continue

            subtracted_image = cv2.absdiff(last_frame, gray_image)

            last_frame = gray_image

            thresh_hold_image = cv2.threshold(subtracted_image, 25, 255, 0)[1]

            dilated_image = cv2.dilate(thresh_hold_image, None, iterations=2)

            contours = cv2.findContours(dilated_image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = imutils.grab_contours(contours)

            count = 0
            for i in contours:

                if cv2.contourArea(i) < area:
                    continue

                count += 1

                x, y, w, h = cv2.boundingRect(i)
                cv2.rectangle(live, (x, y), (x + w, y + h), (0, 255, 0), 2)

            cv2.putText(live, f"{count} OBJECT MOVING...", (10, 20), 0, 0.5, (0, 255, 0), 1)

            cv2.imshow("detect moving", live)

            if cv2.waitKey(1) == 32:  # ascii value of space bar is 32, press to exit
                break

    video.release()
    cv2.destroyAllWindows()
