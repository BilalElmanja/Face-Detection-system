# In that project we will be using OpenCv and keras to build a Face Detection program
# importing the libraries
import os
import cv2 as cv
import numpy as np
import tensorflow.keras.models as models
import tensorflow.keras.layers as layers
from tensorflow.keras.models import load_model


# function to train the model
def build_model(input_image):

    
    # we will use convolutional neural networks in that model so we can end up with better results
    model = models.Sequential()

    # building the model structure as a simple format of the AlexNet classifier
    # first layer is ConvLayer with 96 11*11 filters at stride 4 and pad 0 with relu activation function
    model.add(layers.Conv2D(32 , (6 , 6) , input_shape=(input_image.shape[1] , input_image.shape[2] , 1) , activation='relu' , strides=(2 , 2)))

    # second layers is a pooling one with 3*3 filter at stride 2
    model.add(layers.MaxPooling2D((3 , 3) , strides=(2 , 2)))

    #normalization layer
    model.add(layers.BatchNormalization())

    # second ConvLayer with 256 5*5 filters at stride 1 with pad 2
    model.add(layers.Conv2D(128 , (5 , 5) , activation='relu' , padding='same'))

    # pooling layer with 3*3 filter at stride 2
    model.add(layers.MaxPooling2D((3 , 3) , strides=(2 , 2)))

    #normalization layer
    model.add(layers.BatchNormalization())

    #ConvLayers with 256 3*3 filters at stride 1 and pad 1
    model.add(layers.Conv2D(256 , (3 , 3) , activation='relu', strides=(1 , 1) , padding='same'))
    model.add(layers.Conv2D(256 , (3 , 3) , activation='relu', strides=(1 , 1) , padding='same'))
    model.add(layers.Conv2D(256 , (3 , 3) , activation='relu', strides=(1 , 1) , padding='same'))

    # pooling layer with 3*3 filter at stride 2
    model.add(layers.MaxPooling2D((3 , 3) , strides=(2 , 2)))
    model.add(layers.Flatten())

    # fully connected layers
    model.add(layers.Dense(128 , activation='relu'))
    model.add(layers.Dense(128 , activation='relu'))

    #output layer
    model.add(layers.Dense(1 , activation='sigmoid'))

    #compiling the model and choosing the optimization
    model.compile(optimizer='adam', loss='binary_crossentropy' , metrics=['accuracy'])

    print(model.summary())

    return model

# training the model function
def train_model(X_train , y_train , n_samples , face_number):

    # getting the training data ready
    # first we take the training shape and adjust the training data so we end up
    # with training data with 4 dimension (n_samples , width_len , height_len , 1)
    # the 4th dimension is 1 because we train the model on gray_scale images
    # just to simplify and speed up the process of training 
    training_shape = X_train.shape

    # reshaping the input
    X_input = X_train.reshape(n_samples , training_shape[1] , training_shape[2] , 1)
    output = y_train.reshape(n_samples , 1)

    # building the model
    input_example = X_input[0].reshape(1 , training_shape[1] , training_shape[2] , 1 )
    model = build_model(input_example)

    #training the model
    iters = 1
    model.fit(X_input , output , epochs=iters)

    #saving the model 
    model.save('faces_models/face{}_model/model.h5'.format(face_number))
    #delete the model to save memory
    del model


# constructing the training unput function given the face images directory and outputting the X_train and y_train
def construct_training_data(face_number):
    # image_list
    image_list = []

    # loading the images in a list
    for img_dir in os.listdir('images/face{}'.format(face_number)):
        print(img_dir)
        
        # get pixels version of image
        image = cv.imread('images/face{}'.format(face_number)+'/{}'.format(img_dir) , 0)
        # save it to the list
        image_list.append(image.reshape(1 , 52900))
    print(image_list)
    X_train = np.concatenate(image_list).reshape(200 , 230 , 230)
    if face_number == 0:
        y_train = np.ones(len(X_train)).reshape(len(X_train) , 1)
    else:
        y_train = np.zeros(len(X_train)).reshape(len(X_train) , 1)


    return X_train , y_train

# loading model function
def load_model_function(model_directory):
    model = load_model(model_directory)
    return model
    


# load the frontal face detector
face_cascade = cv.CascadeClassifier('haarcascade_frontalface_alt2.xml')
# turn on camera
cap = cv.VideoCapture(0)

while True:

    ret , frame = cap.read()
    # getting the gray_scale version of the frame
    gray_frame = cv.cvtColor(frame , cv.COLOR_BGR2GRAY)

    #detect faces
    faces = face_cascade.detectMultiScale(gray_frame , 1.3 , 5)

    for i in range(len(faces)) :
        (x , y , w , h) = faces[i]
        # displaying a blue rectangle on the camera (BGR version)
        cv.rectangle(frame , (x , y) , (x+230 , y+230) , (255 , 0 , 0) , 2)
        # isolating and saving the face_image to either recognize or train a new model
        face_image = gray_frame[y:y+230 , x:x+230]

        test_image = np.array(face_image).reshape(1 , 230 , 230 , 1)
        cv.imwrite('test_.png' , face_image)

        model_directory = 0
        print('hhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhh')
        while len(os.listdir('./faces_models')) < len(faces):
            t = i
            while ('face{}_model'.format(i)) in os.listdir('faces_models'):
                t += 1
            os.mkdir('./faces_models/face{}_model'.format(t))

        while len(os.listdir('./images')) < len(faces):
            t = i
            while ('face{}'.format(i)) in os.listdir('images'):
                t += 1
            os.mkdir('./images/face{}'.format(t))

        for root , dirname , filenames in os.walk('faces_models/face{}_model'.format(i)):
            
            if len(filenames) < 1:
                
                # saving some training data
                for j in range(1 , 201):
                    cv.imwrite(r'images\face{}\img_{}.png'.format(i , j) , face_image)
                
                # constructing the data
                X_train , y_train = construct_training_data(i)
                print('training shape is: {}'.format(X_train.shape))
                # building a model
                train_model(X_train , y_train , X_train.shape[0] , i)
                    
            else:
                
                model_directory = 'faces_models/face{}_model/model.h5'.format(i)

            # loading the model
            model = load_model_function(model_directory)

            # testing the model
            prediction = model.predict(test_image)
            print(prediction)
        print('gggggggggggggggggggggggggggggggggggggggggg')
    cv.imshow('frame' , frame)

    k = cv.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv.destroyAllWindows()

        










