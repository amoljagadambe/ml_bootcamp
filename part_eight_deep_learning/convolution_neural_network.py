# Convolution neural network

# Importing Libraries
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPool2D, Flatten, Dense

# Initialize the CNN
classifier = Sequential()

# First Layer: Convolution
classifier.add(Convolution2D(filters=32, kernel_size=(3, 3), input_shape=(64, 64, 3), activation='relu'))

# Second Layer: Pooling
classifier.add(MaxPool2D(pool_size=(2, 2), strides=2))

# Adding a second convolutional layer
classifier.add(Convolution2D(filters=32, kernel_size=3, activation='relu'))
classifier.add(MaxPool2D(pool_size=2, strides=2))

# Third Layer: Flattening
classifier.add(Flatten())

# Fourth Layer: Dense (i.e. Full connection)
classifier.add(Dense(units=128, activation='relu'))

# Final Layer
classifier.add(Dense(units=1, activation='sigmoid'))

# Compile the CNN
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Fitting CNN to images
from keras.preprocessing.image import ImageDataGenerator

# Preprocessing the Training set
train_datagen = ImageDataGenerator(rescale=1. / 255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)
training_set = train_datagen.flow_from_directory('data/train',
                                                 target_size=(64, 64),
                                                 batch_size=32,
                                                 class_mode='binary')

# Preprocessing the Test set
test_datagen = ImageDataGenerator(rescale=1. / 255)
test_set = test_datagen.flow_from_directory('data/test',
                                            target_size=(64, 64),
                                            batch_size=32,
                                            class_mode='binary')

# Training the CNN on the Training set and evaluating it on the Test set
classifier.fit(x=training_set, validation_data=test_set, epochs=25)

# Part 4 - Making a single prediction

import numpy as np
import tensorflow as tf

image = tf.keras.utils.load_img('dataset/single_prediction/cat_or_dog_1.jpg', target_size=(64, 64))
test_image = tf.keras.utils.img_to_array(image)
test_image = np.expand_dims(test_image, axis=0)
result = classifier.predict(test_image)

print(training_set.class_indices)

if result[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'
print(prediction)
