# import necessary packages
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import numpy as np
import argparse
import os

# argument parsing
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="path to input dataset")
ap.add_argument("-m", "--model", required=True, help="path to output model")
args = vars(ap.parse_args())

# initialize learning rate, epochs to train for and batch size
LR = 1e-4
EPOCHS = 20
BS = 32

# grab the list of images in our dataset and then initialize the list of data and class images
print("[INFO] loading images...")
imagePaths = list(paths.list_images(args["dataset"]))
data = []
labels = []

# loop over the imag paths
for imagePath in imagePaths:
    # extract the class label
    label = imagePath.split(os.path.sep)[-2]

    # load the input image and preprocess it
    image = load_img(imagePath, target_size=(224, 224))
    image = img_to_array(image)
    image = preprocess_input(image)

    # update the data and label list
    data.append(image)
    labels.append(label)

# convert data and labels to NumPy array
data = np.array(data, dtype="float32" )
labels = np.array(labels)

# perform one-hot encoding of the labels
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)

# partitioning the dataset into train data(80%) and test data(20%)
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.20, stratify=labels, random_state=42)

# construct the training image generator for data augmentation
aug = ImageDataGenerator(rotation_range=20, zoom_range=0.15, width_shift_range=0.2,  height_shift_range=0.2, shear_range=0.15, horizontal_flip=True, fill_mode="nearest")

# load the MobileNetV2 network, ensuring the head FC layer sets are left off
baseModel = MobileNetV2(weights="imagenet", include_top=False, input_tensor=Input(shape=(224, 224, 3)))

# construct head of the model that will be placed on top of base model
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7,7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(128, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2, activation="softmax")(headModel)

# place the head FC model on top of the base model (this will be the actual model we will train)
model = Model(inputs=baseModel.input, outputs=headModel)

# loop over all the layers in the base model and freeze them so that they will not be updated during first training process
for layer in baseModel.layers:
    layer.trainable = False

# compile the model
print("[INFO] compiling model...")
opt = Adam(lr=LR, decay=LR/EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

# train the head of the network
print("[INFO] training head...")

H = model.fit(aug.flow(trainX,trainY, batch_size=BS), steps_per_epoch=len(trainX)//BS, validation_data=(testX, testY), validation_steps=len(testX)//BS, epochs=EPOCHS)

# make prediction on the test set
print("[INFO] evaluating network...")
predIdxs = model.predict(testX, batch_size=BS)

# for each image in test set we need to find the index of label with maximum probability
predIdxs = np.argmax(predIdxs, axis=1)

# print a formatted classification report
print(classification_report(testY.argmax(axis=1), predIdxs, target_names=lb.classes_))

# serialize the model to disk
print("[INFO] saving mask detector model...")
model.save(args["model"], save_format="h5")