import cv2
import imutils
from imutils.object_detection import non_max_suppression
import numpy as np

#cam = cv2.VideoCapture('BodyMechanics.mp4')
cam = cv2.VideoCapture(1)
#faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#tracker = cv2.TrackerMedianFlow_create()
onTracking = False
# initialize the HOG descriptor/person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

def find_background_image(frame_RGB):
    """Given a directory and a class name, search for a viable background image"""
    # This method based on: http://www.pyimagesearch.com/2015/11/09/pedestrian-detection-opencv/

    # loop over the image paths for each class
    image = np.copy(frame_RGB)
    #image = imutils.resize(frame_RGB, width=min(400, frame_RGB.shape[1]))
    orig = frame_RGB.copy()
    # detect people in the image
    (rects, weights) = hog.detectMultiScale(frame_RGB, winStride=(4, 4), padding=(8, 8), scale=1.05)

    # draw bounding boxes
    for (x, y, w, h) in rects:
        cv2.rectangle(orig, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # apply non-maxima suppression to the bounding boxes using a
    # fairly large overlap threshold to try to maintain overlapping
    # boxes that are still people
    rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
    pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)
    # draw the final bounding boxes
    for (xA, yA, xB, yB) in pick:
        cv2.rectangle(image, (xA, yA), (xB, yB), (0, 255, 0), 2)

    # show the output images
    cv2.imshow("Before NMS", orig)
    cv2.imshow("After NMS", image)

while True:
    ret, frame = cam.read()
    if not onTracking:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        find_background_image(frame)
        '''
        #faces = faceDetect.detectMultiScale(gray, scaleFactor=1.4, minNeighbors=1, minSize=(100, 100))
        faces = faceDetect.detectMultiScale(gray,1.4,5)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            if tracker.init(frame, (x, y, w, h)):
                onTracking = True

    else:
        ok, bbox = tracker.update(frame)
        if ok:
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(frame, p1, p2, (255, 0, 0), 2)
        else:
            onTracking = False
            tracker = cv2.TrackerMedianFlow_create()'''

    #cv2.imshow("Faces", frame)
    if cv2.waitKey(1) == ord('q'):
        break
        cam.release()
        cv2.destroyAllWindows()
