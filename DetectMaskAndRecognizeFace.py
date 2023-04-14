# import the necessary packages
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
from cv2 import cv2
import numpy as np
import face_recognition
import imutils
import argparse
import pickle
import time
import os
from openpyxl import Workbook
from datetime import date
from datetime import datetime

# function to select largest face from the frame
def select_largest_face(faces):
    height = []
    for face in faces:
        (top, right, bottom, left) = face
        h = bottom-top
        height.append(h)
    # (i, h) = max(enumerate(height))
    i = height.index(max(height))
    return(faces[i:i+1])

# define function to detect face and predict mask
def predict_mask(frame, box, model):
    # grab the dimension of the frame and construct a blob
    (h, w) = frame.shape[:2]

    # compute (x,y) coordinates of bounding box
    (startY, endX, endY, startX) = box[0]
        
    # ensure that bounding box fall within dimension of the frame
    (startX, startY) = (max(0, startX), max(0, startY))
    (endX, endY) = (min((w-1), endX), min((h-1), endY))
        
    # extract the face ROI, convert from BGR to RGB
    face = frame[startY:endY, startX:endX]
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    face = cv2.resize(face, (224, 224))
    face = img_to_array(face)
    face = preprocess_input(face)
    face = np.expand_dims(face, axis=0)

    # set the bounding box
    location = (startX, startY, endX, endY)

    # predict mask
    (mask, withoutMask) = model.predict(face)[0]

    # determine the class label and color we will use to draw the bounding box
    if mask>withoutMask:
        label = "Mask"
        color = (0, 225, 0)
    else:
        label = "No Mask"
        color = (0, 0, 225)

    # return te face location and the corresponding prediction
    return(location, label, color)

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-e", "--encodings", required=True, help="path to serialized db of facial encodings")
ap.add_argument("-m", "--model", type=str, default="mask_detector.model", help="path to face mask detector model")
ap.add_argument("-d", "--detection-method", type=str, default="dnn", help="face detection model to use: either `hog` or `cnn`")
args = vars(ap.parse_args())

# create workbook
mywb = Workbook()
today = date.today()
today = today.strftime("%d-%m-%Y")

# create a sheet with today's date
mywb.create_sheet(index=0, title=today)

# grab the sheet created
mysheet = mywb[today]

# insert the headings
mysheet['A1'] = 'Name'
mysheet['C1'] = 'Mask'
mysheet['E1'] = 'Time of entry'
mysheet.append([])

# load our mask detector model
print("[INFO] loading mask detector model...")
model = load_model(args["model"])

# load the known faces and embeddings
print("[INFO] loading encodings...")
data = pickle.loads(open(args["encodings"], "rb").read())

# initialize frame counter
frame_count = 0
frame_thresh = 20

# initialize names list
names = []

# initialize video stream and allow camera sensor to warm up
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

# outer loop to run the entire process repeatedly
while True:
    # reset mask value
    mask_value = "No"

    # loop over the frames in video stream
    while True:
        # grab the frame, change it from BGR to RGB and resize it
        frame = vs.read()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # detect the (x, y)-coordinates of the bounding boxes corresponding to each face in the input image
        faces = face_recognition.face_locations(rgb, model=args["detection_method"])


        # if one face is detected
        if len(faces) <= 1:
            face = faces

        # if more than one face is detected
        else:
            text = "Please be only one person in the frame"
            cv2.putText(frame, text, (15,15), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 140, 0), 2)
            # call function to select largest face
            face = select_largest_face(faces)

        if len(face)>0:
        # while True:
            # call the function to predict mask
            (box, label, color) = predict_mask(frame, face, model)

            # unpack the bounding box
            (startX, startY, endX, endY) = box

            # drawing rectangle around faces
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

            # display labels with probability
            cv2.putText(frame, label, (startX, startY-10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 140, 0), 2)

            # checking frame count for continuous presence of mask
            if label == "Mask":
                # increment frame count
                frame_count += 1

                # if frame count is greater than threshold then display the message
                if frame_count >= frame_thresh:
                    cv2.putText(frame, "Proceed for face recognition", (15, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (140, 220, 0), 2)
                    mask_value = "Yes"

            else:
                frame_count = 0

        # show the output frame
        cv2.imshow("Frame",frame)

        # exit condition
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    # loop over the frames to perform face recognition
    while True:
        # grab the frame, change it from BGR to RGB and resize it
        frame = vs.read()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # detect the (x, y)-coordinates of the bounding boxes corresponding to each face in the input image
        faces = face_recognition.face_locations(rgb, model=args["detection_method"])


        # if one face is detected
        if len(faces) <= 1:
            face = faces

        # if more than one face is detected
        else:
            text = "Please be only one person in the frame"
            cv2.putText(frame, text, (15,15), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 140, 0), 2)
            
            # call function to select largest face
            face = select_largest_face(faces)
    
        # find the encodings for the face
        encoding = face_recognition.face_encodings(rgb, face)

        # if a face is there in the frame
        if len(encoding)>0:
            # attempt to match each face in the input image to our known encodings
            matches = face_recognition.compare_faces(data["encodings"], encoding[0])

            # initialize the name as unknown
            name = "Unknown"

            # check to see if we have found a match
            if True in matches:
                # find the indexes of all matched faces then initialize a dictionary to count the total number of times each face was matched
                matchedIdxs = [i for (i, b) in enumerate(matches) if b]
                counts = {}

                # loop over the matched indexes and maintain a count for each recognized face
                for i in matchedIdxs:
                    name = data["names"][i]
                    counts[name] = counts.get(name, 0) + 1

                # determine the recognized face with the largest number of votes (note: in the event of an unlikely tie Python will select first entry in the dictionary)
                name = max(counts, key=counts.get)

            # rescale the face co-ordinates
            (top, right , bottom, left) = face[0]

            # draw the predicted face name on the image
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            y = top - 15 if top - 15 > 15 else top + 15
            cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (220, 140, 0), 2)

        # show the output frame
        cv2.imshow("Frame", frame)

        # exit condition
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == ord("z"):
            break
    
    # check whether there is any known face in the frame
    if name != 'Unknown':
        # get the current time
        now = datetime.now()
        time = now.strftime("%I:%M:%S %p")

        # if new face appears
        if name not in names:
            # update the name list
            names.append(name)

            # update the excel sheet with name, mask_value, time of entry
            mysheet.append([name, '', mask_value, '', time])
        
        # if the name already exists in the list
        else:
            index = names.index(name) + 3
            mysheet['C{}'.format(index)] = mask_value
            mysheet['E{}'.format(index)] = time

    # condition to close the program
    if key == ord("z"):
        break

# clean up
cv2.destroyAllWindows()
vs.stop()

# save the workboox(excel file) into Attendence/Month_name/Date.xlsx
today = date.today()
month = today.strftime("%B")
today = today.strftime("%d-%b-%Y")
mywb.save("Attendence/{}/{}.xlsx".format(month,today))
