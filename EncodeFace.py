# import necessary packages
from imutils import paths
from cv2 import cv2
import face_recognition
import pickle
import argparse
import os

# parse arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--dataset", required=True, help= "path to the directory of images")
ap.add_argument("-e", "--encoding", required=True, help= "path to the  facial encodings of images")
ap.add_argument("-d", "--detection-method", type=str, default="cnn", help= "detection method")
args = vars(ap.parse_args())

# grab the path of input image inour dataset
imagePaths = list(paths.list_images(args["dataset"]))

# initialize the list of encodings and faces
knownEncodings = []
knownNames = []

# loop over the images
for (i, imagePath) in enumerate(imagePaths):
    print("[INFO] processing image :{}/{}".format(i+1, len(imagePaths)))
    # take out the names
    name = imagePath.split(os.path.sep)[-2]

    # read the image and convert BGR to RGB
    image = cv2.imread(imagePath)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # detect the bounding box for detected faces
    box = face_recognition.face_locations(rgb, model=args["detection_method"])

    # compute the facial encodings
    encodings = face_recognition.face_encodings(rgb, box)

    # loop over the encodings
    for encoding in encodings:
        # append encoding and face name for each detected face
        knownEncodings.append(encoding)
        knownNames.append(name)

# dump the facial encodings and names to the disk
print("[INFO] dumping encodings to disk...")
data = {"encodings": knownEncodings, "names": knownNames}
f = open(args["encoding"], "wb")
f.write(pickle.dumps(data))
f.close()