import cv2
import numpy as np
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt

class YOLO_Human:
    def __init__(self):
        # ------------------------------------------------#
        #self.cap = cv2.VideoCapture(1)
        self.cap = cv2.VideoCapture('vtest.avi')
        #self.cap = cv2.VideoCapture('BodyMechanics.mp4')
        self.out_frame = None

        # ------------------------------------------------#

        self.tracker = cv2.TrackerMedianFlow_create()
        self.onTracking = False

        # Write down conf, nms thresholds,inp width/height
        self.confThreshold = 0.25
        self.nmsThreshold = 0.40
        # Load names of classes and turn that into a list
        classesFile = "./yolo_model/coco.names"
        modelConf = './yolo_model/yolov3-tiny.cfg'
        modelWeights = './yolo_model/yolov3-tiny.weights'
        # Set up the net
        self.net = cv2.dnn.readNetFromDarknet(modelConf, modelWeights)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        classes = None
        with open(classesFile, 'rt') as f:
            self.classes = f.read().rstrip('\n').split('\n')

        # =============== Variable Mouse ==================#
        self.drawing = False
        self.DownPoint1 = ()
        self.DownPoint2 = ()
        self.DownPoint3 = ()
        self.DownPoint4 = ()
        self.DownPoint = []
        self.Click = 0
        self.Mouse_count = False
        # =============== Variable Mouse ==================#

    def postprocess(self, frame, outs):
        frameHeight = frame.shape[0]
        frameWidth = frame.shape[1]

        classIDs = []
        confidences = []
        boxes = []

        for out in outs:
            for detection in out:
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]
                if confidence > self.confThreshold:
                    centerX = int(detection[0] * frameWidth)
                    centerY = int(detection[1] * frameHeight)

                    width = int(detection[2] * frameWidth)
                    height = int(detection[3] * frameHeight)

                    left = int(centerX - width / 2)
                    top = int(centerY - height / 2)

                    classIDs.append(classID)
                    confidences.append(float(confidence))
                    boxes.append([left, top, width, height])

        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.confThreshold, self.nmsThreshold)
        # print('ClassIDs' +str(classes)) # 2 = car, 7= 'truck'
        # ------------ Count --------------#
        boxes = np.array(boxes)
        classIDs = np.array(classIDs)
        confidences = np.array(confidences) # %
        selected_boxes = boxes[indices]

        # --- person --#
        person_deteced_indices = np.where(classIDs[indices] == 0)
        #person_ALL = len(person_deteced_indices)
        #print(person_ALL)
        selected_boxes = selected_boxes[person_deteced_indices]
        # confidences = confidences[car_deteced_indices[0]]
        # classIDs = classIDs[car_deteced_indices[0]]
        # ------------ Count --------------#

        # for i in indices:
        centerCount_list = []
        subtract = False
        for i, box in enumerate(selected_boxes):
            # i = i[0]
            # box = boxes[i]
            left = box[0]
            top = box[1]
            width = box[2]
            height = box[3]
            '''print('left: ' + str(left))
            print('top: ' + str(top))
            print('width: ' + str(width))
            print('height: ' + str(height))'''

            # ----------------- Center ---------------- #
            right = left + width
            bottom = top + height
            #center = ((left + right) / 2, (top + bottom) / 2)
            center = (int((left + right) / 2), int((top + bottom) / 2))
            centerCount_list.append(center)

            # ============ distances sklearn ============#
            ''''
            if len(centerCount_list) > 1:
                #print('centerCount_list: ' + str(centerCount_list))
                distance_sklearn = pairwise_distances(centerCount_list)
                print(distance_sklearn)
                #plt.plot(distance_sklearn, 'ro')
                #plt.show()
                for i, output_i in enumerate(distance_sklearn):
                    for ii, output_ii in enumerate(output_i):
                        if 0 < output_ii < 160:
                            #print(i)
                            #print(ii)
                            #print(output_ii)
                            cv2.line(self.out_frame, centerCount_list[i], centerCount_list[ii], (0, 0, 255), 2)
            '''
            # ============ distances sklearn ============#


            if len(centerCount_list) > 1:
                for C, center_First in enumerate(centerCount_list):
                    print(center_First)
                    #print(center_First[0])
                    print("...........")
                    for CC, center_Second in enumerate(centerCount_list):
                        if C != CC:
                            print(center_Second)
                            # P = (x,y)
                            P1 = (int(center_First[0]), int(center_First[1]))  # 0
                            P2 = (int(center_Second[0]), int(center_Second[1]))  # 1
                            cv2.line(self.out_frame, P1, P2, (255, 0, 0), 1)
                            distance_line = np.sqrt((P2[0] - P1[0]) ** 2 + (P2[0] - P1[1]) ** 2)  # (x2-x1)**2 + (y2-y1)**2
                            print('distance_line = {}'.format(str(distance_line)))
                            if distance_line < 160:
                                cv2.circle(self.out_frame, center, 3, (0, 2, 255), -1)
                                cv2.line(self.out_frame, P1, P2, (0, 0, 255), 2)
                                #cv2.rectangle(self.out_frame, (left, top), (right, bottom), (0, 0, 255), 3)

                            else:
                                #cv2.line(self.out_frame, P1, P2, (0, 255, 255), 1)
                                cv2.circle(self.out_frame, center, 3, (0, 255, 255), -1)
                    print('_____________________________________________')


            # ----------------- Center ---------------- #

            #self.drawPred(classIDs[i], confidences[i], left, top, left + width, top + height, width, height, len(selected_boxes))
            self.drawPred(classIDs[i], confidences[i], left, top, left + width, top + height, subtract)

        return len(selected_boxes), left, top, width, height  # --count--#

    def drawPred(self, classId, conf, left, top, right, bottom, subtract):

        if self.classes[classId] == 'person':
            Percent = '%.2f' % conf
            assert (classId < len(self.classes))
            label = '%s:%s' % (self.classes[classId], Percent)
            cv2.putText(self.out_frame, str(label), (left, top), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)

    def getOutputsNames(self):
        # Get the names of all the layers in the network
        layersNames = self.net.getLayerNames()
        # Get the names of the output layers, i.e. the layers with unconnected outputs
        return [layersNames[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]

    # ================================================#
    def mouse_drawing(self, event, x, y, flags, params):
        # ----------Mouse 1------- #
        if not self.Mouse_count:
            if event == cv2.EVENT_LBUTTONDOWN:
                if self.drawing is False:
                    if self.Click == 0:
                        self.DownPoint1 = (x, y)
                        print("P1:", self.DownPoint1)
                        cv2.circle(self.frame_First, (x, y), 5, (25, 255, 255), -1)
                        cv2.imshow("Detecion ROI", self.frame_First)
                        self.Click = 1

                    elif self.Click == 1:
                        self.DownPoint2 = (x, y)
                        print("P2:", self.DownPoint2)
                        cv2.circle(self.frame_First, (x, y), 5, (25, 255, 255), -1)
                        cv2.imshow("Detecion ROI", self.frame_First)
                        self.Click = 2

                    elif self.Click == 2:
                        self.DownPoint3 = (x, y)
                        print("P3:", self.DownPoint3)
                        cv2.circle(self.frame_First, (x, y), 5, (25, 255, 255), -1)
                        cv2.imshow("Detecion ROI", self.frame_First)
                        self.Click = 3

                    elif self.Click == 3:
                        self.DownPoint4 = (x, y)
                        print("P4:", self.DownPoint4)
                        cv2.circle(self.frame_First, (x, y), 5, (25, 255, 255), -1)
                        cv2.imshow("Detecion ROI", self.frame_First)
                        self.Click = 0
                        self.Mouse_count = True

    def LoopRunCam(self):
        # ================================================#
        # -----Image----#
        _, self.frame_First = self.cap.read()
        cv2.namedWindow("Detecion ROI")
        cv2.setMouseCallback("Detecion ROI", self.mouse_drawing)
        cv2.imshow("Detecion ROI", self.frame_First)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        inpWidth = 416
        inpHeight = 416

        while cv2.waitKey(1) < 0:
            # get frame from video
            hasFrame, frame = self.cap.read()

            if self.Mouse_count:  # True
                # --------------------Roi Mouse--------------------#
                pts = np.array([
                    [self.DownPoint1[0], self.DownPoint1[1]],
                    [self.DownPoint2[0], self.DownPoint2[1]],
                    [self.DownPoint3[0], self.DownPoint3[1]],
                    [self.DownPoint4[0], self.DownPoint4[1]]])

                canvas = np.zeros_like(frame)  # ---black in RGB---#
                frame_crop = cv2.drawContours(canvas, [pts], -1, (255, 255, 255), -1, cv2.LINE_AA)

                self.out_frame = np.zeros_like(frame)  # Extract out the object and place into output image
                self.out_frame[canvas == 255] = frame[canvas == 255]

                if hasFrame:
                    # Create a 4D blob from a frame
                    blob = cv2.dnn.blobFromImage(self.out_frame, 1 / 255, (inpWidth, inpHeight), [0, 0, 0], 1,
                                                 crop=False)

                    # Set the input the the net
                    self.net.setInput(blob)
                    outs = self.net.forward(self.getOutputsNames())

                    # postprocess (out_frame, outs)
                    selected_boxes, left, top, width, height = self.postprocess(frame, outs)
                    # cv.putText(out_frame, str(selected_boxes_car), (10,100), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(self.out_frame, str(selected_boxes), (10, 100), font, 1, (255, 200, 55), 2,
                                cv2.LINE_AA)
                    print("Number of classes: ", selected_boxes)

                    # show the image
                    cv2.imshow('DL OD with OpenCV', self.out_frame)


if __name__ == "__main__":
    RunClasYOLO_Humans = YOLO_Human()
    RunClasYOLO_Humans.LoopRunCam()
