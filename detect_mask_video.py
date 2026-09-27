# USAGE
# python detect_mask_video.py

# import the necessary packages
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import argparse
import imutils
import time
import cv2
import os


def detect_and_predict_mask(frame, faceCascade, maskNet):
    # convert the frame to grayscale for Haar cascade detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    # initialize our list of faces, their corresponding locations,
    # and the list of predictions from our face mask network
    faceROIs = []
    locs = []
    preds = []

    for (x, y, w, h) in faces:
        # compute the (x, y)-coordinates of the bounding box for the face
        startX, startY = max(0, x), max(0, y)
        endX, endY = min(frame.shape[1] - 1, x +
                         w), min(frame.shape[0] - 1, y + h)

        # extract the face ROI, convert it from BGR to RGB channel ordering,
        # resize it to 224x224, and preprocess it
        face = frame[startY:endY, startX:endX]
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = cv2.resize(face, (224, 224))
        face = img_to_array(face)
        face = preprocess_input(face)

        # add the face and bounding boxes to their respective lists
        faceROIs.append(face)
        locs.append((startX, startY, endX, endY))

    # only make predictions if at least one face was detected
    if len(faceROIs) > 0:
        faceROIs = np.array(faceROIs, dtype="float32")
        preds = maskNet.predict(faceROIs, batch_size=32)

    # return a 2-tuple of the face locations and their corresponding
    # locations
    return (locs, preds)


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--face", type=str,
                default="",
                help="unused in the current OpenCV build; kept for compatibility")
ap.add_argument("-m", "--model", type=str,
                default="facemask_detector.keras",
                help="path to trained face mask detector model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
                help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# load the Haar cascade face detector shipped with OpenCV
print("[INFO] loading face detector model...")
faceCascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
if faceCascade.empty():
    raise FileNotFoundError(
        "Unable to load Haar cascade face detector. Check your OpenCV installation.")

# load the face mask detector model from disk
print("[INFO] loading face mask detector model...")
maskNet = load_model(args["model"])

# initialize the video stream and allow the camera sensor to warm up
print("[INFO] starting video stream...")
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError(
        "Could not open webcam. Check camera permissions or your camera device.")
time.sleep(2.0)

# loop over the frames from the video stream
while True:
    # grab the frame from the webcam and resize it
    # to have a maximum width of 400 pixels
    ret, frame = cap.read()
    if not ret or frame is None:
        print("[ERROR] Unable to read frame from webcam. Exiting...")
        break
    frame = imutils.resize(frame, width=400)

    # detect faces in the frame and determine if they are wearing a
    # face mask or not
    (locs, preds) = detect_and_predict_mask(frame, faceCascade, maskNet)

    # loop over the detected face locations and their corresponding
    # locations
    for (box, pred) in zip(locs, preds):
        # unpack the bounding box and predictions
        (startX, startY, endX, endY) = box
        (mask, withoutMask) = pred

        # determine the class label and color we'll use to draw
        # the bounding box and text
        label = "Mask" if mask > withoutMask else "No Mask"
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

        # include the probability in the label
        # label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

        # display the label and bounding box rectangle on the output
        # frame
        if (label == "Mask"):

            cv2.putText(frame, "Mask", (startX, startY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
        elif (label == "No Mask"):
            lab = "No Mask"
            cv2.putText(frame, lab, (startX, startY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

    # show the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

# do a bit of cleanup
cv2.destroyAllWindows()
cap.release()
