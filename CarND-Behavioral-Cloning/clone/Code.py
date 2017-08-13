import csv
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Flatten, Lambda, Activation, Cropping2D
from keras.layers.convolutional import Convolution2D
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

samples = []
with open('/home/gokul/Cloning/data/data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

train_samples, validation_samples = train_test_split(samples,test_size=0.2)

def generator(samples,batch_size=32):
    num_samples = len(samples)
    while 1:
        shuffle(samples)
        for offset in range(0,num_samples,batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            measurements = []
            for line in batch_samples:
                for i in range(3):
                    source_path = line[i]
                    filename = source_path.split('/')[-1]
                    current_path = '/home/gokul/Cloning/data/data/IMG/' + filename
                    image = cv2.imread(current_path)
                    images.append(image)
                    if i == 0:
                        measurement = float(line[3])
                        measurements.append(measurement)
                    if i == 1:
                        measurement = float(line[3]) + 0.2
                        measurements.append(measurement)
                    if i == 2:
                        measurement = float(line[3]) - 0.2
                        measurements.append(measurement)
            augmented_measurements = []
            augmented_images = []
            for image, measurement in zip(images, measurements):
                augmented_measurements.append(measurement)
                augmented_images.append(image)
                augmented_measurements.append(measurement * -1.0)
                image_flipped = np.fliplr(image)
                augmented_images.append(image_flipped)
            X_train = np.array(augmented_images)
            y_train = np.array(augmented_measurements)
            yield shuffle(X_train,y_train)

train_generator = generator(train_samples,batch_size=32)
validation_generator = generator(validation_samples,batch_size=32)

model = Sequential()
model.add(Lambda(lambda x: x/127.5 - 1., input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25),(0,0))))
model.add(Convolution2D(filters=24,kernel_size=(5,5),strides=(2,2),activation="relu"))
model.add(Convolution2D(filters=36,kernel_size=(5,5),strides=(2,2),activation="relu"))
model.add(Convolution2D(filters=48,kernel_size=(5,5),strides=(2,2),activation="relu"))
model.add(Convolution2D(filters=64,kernel_size=(3,3),activation="relu"))
model.add(Convolution2D(filters=64,kernel_size=(3,3),activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))
model.add(Activation('softmax'))
model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator,steps_per_epoch=len(train_samples),validation_data=validation_generator, validation_steps=len(validation_samples), nb_epoch=3)
model.save('model.h5')


'''
for line in samples:
    for i in range(3):
        source_path = line[i]
        filename = source_path.split('/')[-1]
        current_path = '/home/gokul/Cloning/data/data/IMG/' + filename
        image = cv2.imread(current_path)
        images.append(image)
        if i==0:
            measurement = float(line[3])
            measurements.append(measurement)
        if i==1:
            measurement = float(line[3])+0.2
            measurements.append(measurement)
        if i==2:
            measurement = float(line[3])-0.2
            measurements.append(measurement)

for image, measurement in zip(images,measurements):
    augmented_measurements.append(measurement)
    augmented_images.append(image)
    augmented_measurements.append(measurement*-1.0)
    image_flipped = np.fliplr(image)
    augmented_images.append(image_flipped)

X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)

model = Sequential()
model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25),(0,0))))
model.add(Convolution2D(filters=24,kernel_size=(5,5),strides=(2,2),activation="relu"))
model.add(Convolution2D(filters=36,kernel_size=(5,5),strides=(2,2),activation="relu"))
model.add(Convolution2D(filters=48,kernel_size=(5,5),strides=(2,2),activation="relu"))
model.add(Convolution2D(filters=64,kernel_size=(3,3),activation="relu"))
model.add(Convolution2D(filters=64,kernel_size=(3,3),activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))
model.add(Activation('softmax'))
model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=2)
model.save('model.h5')
'''


