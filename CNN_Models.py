import keras.metrics
import tensorflow as tf
from keras.layers import Dropout
#from sklearn import metrics
import cv2
import os
import numpy as np
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt



def Load_Images(Folder_Path, img_name):
    img = cv2.imread(os.path.join(Folder_Path, img_name))
    img2 = cv2.resize(img, (64, 64))
    imgCol = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    return imgCol


def Load_TrainingData(alphbet, Folder_Path):
    training_data = []
    labels = []
    for i in alphbet:
        Class_Number = alphbet.index(i)
        Data_Folder = os.path.join(Folder_Path, i)
        print(Data_Folder)
        for img_name in os.listdir(Data_Folder):
            try:
                img = Load_Images(Data_Folder, img_name)
                training_data.append(img)
                labels.append(Class_Number)
            except Exception as e:
                pass
    return training_data, labels


def Load_testData(alphbet, Folder_Path):
    test_data = []
    labeltest = []
    # images=Load_Images(Folder_Path)
    # Class_Number=alphbet.index(i)
    i = 0
    for img_name in os.listdir(Folder_Path):
        # if alphbet[i]=='del':
        #     test_data.append()
        #     labeltest.append(i)
        #     i+=1
        #     continue
        try:
            img = Load_Images(Folder_Path, img_name)
            test_data.append(img)
            labeltest.append(i)
        except Exception as e:
            pass
        i += 1
    return test_data, labeltest


def vectorization(Data):
    return np.array(Data)


def Getxy_Train(training_data, labels):
    X_Train = []
    Y_Train = []
    X_Train = training_data
    Y_Train = labels
    X_Train = vectorization(X_Train)
    X_Train = X_Train.astype('float32') / 255.0  # to normalize images
    Y_Train = np.array(Y_Train)
    return X_Train, Y_Train


def Getxy_Test(testdata, labeltest):
    X_Test = []
    Y_Test = []
    X_Test = testdata
    Y_Test = labeltest
    X_Test = vectorization(X_Test)
    X_Test = X_Test.astype('float32') / 255.0  # to normalize images
    Y_Test = np.array(Y_Test)
    return X_Test, Y_Test


alphbet = ['A', 'B', 'C', 'D', 'del', 'E', 'F', 'G', 'H', 'I', 'J', 'K',
           'L', 'M', 'N', 'nothing', 'O', 'P', 'Q', 'R', 'S',
           'space', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
Folder_Path = "D:/Iseul/Education/College/(4)_1st_Semester/Machine and Bioinformatics/Assignment_3/asl_alphabet_train"
training_data, labels = Load_TrainingData(alphbet, Folder_Path)
train_images, train_labels = Getxy_Train(training_data, labels)

Folder_Path2 = "D:/Iseul/Education/College/(4)_1st_Semester/Machine and Bioinformatics/Assignment_3/asl_alphabet_test"
testdata, labeltest = Load_testData(alphbet, Folder_Path2)
test_images, test_labels = Getxy_Test(testdata, labeltest)

train_images = np.array(train_images)
test_images = np.array(test_images)

plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i])


# plt.show()

def first_model():
    model = models.Sequential()
    # data_dim=3600384
    # model.add(LSTM(units=50, input_shape=(1, data_dim), return_sequences=True))
    model.add(layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(64, 64, 3), strides=(1, 1),
                            padding="same"))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', input_shape=(64, 64, 3), strides=(1, 1),
                            padding="same"))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu', input_shape=(64, 64, 3), strides=(1, 1),
                            padding="same"))
    model.add(layers.MaxPooling2D((2, 2)))

    model.summary()

    model.add(layers.Flatten())
    model.add(layers.Dense(200, activation='relu'))  # to add hidden layer
    model.add(Dropout(0.2))  # drop out some hidden layer to prevent overfitting
    model.add(layers.Dense(300, activation='relu'))
    model.add(Dropout(0.3))
    model.add(layers.Dense(400, activation='relu'))
    model.add(Dropout(0.4))
    model.add(layers.Dense(29, activation='softmax'))

    model.summary()

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    history = model.fit(train_images, train_labels, epochs=10,
                        validation_data=(test_images, test_labels))

    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.5, 1])
    plt.legend(loc='lower right')
    test_loss_one, test_acc_one = model.evaluate(test_images, test_labels, verbose=2)
    return test_loss_one, test_acc_one


def second_model():
    model2 = models.Sequential()

    model2.add(layers.Conv2D(filters=32, kernel_size=(4, 4), activation='relu', input_shape=(64, 64, 3), strides=(1, 1),
                             padding="same"))
    model2.add(layers.MaxPooling2D((2, 2)))
    model2.add(layers.Conv2D(filters=64, kernel_size=(4, 4), activation='relu', input_shape=(64, 64, 3), strides=(1, 1),
                             padding="same"))
    model2.add(layers.MaxPooling2D((2, 2)))
    model2.add(
        layers.Conv2D(filters=128, kernel_size=(4, 4), activation='relu', input_shape=(64, 64, 3), strides=(1, 1),
                      padding="same"))
    model2.add(layers.MaxPooling2D((2, 2)))
    model2.add(
        layers.Conv2D(filters=265, kernel_size=(3, 3), activation='relu', input_shape=(64, 64, 3), strides=(1, 1),
                      padding="same"))
    model2.add(layers.MaxPooling2D((2, 2)))

    model2.summary()

    model2.add(layers.Flatten())
    model2.add(layers.Dense(500, activation='relu'))  # to add hidden layer
    model2.add(Dropout(0.2))  # drop out some hidden layer to prevent overfitting
    model2.add(layers.Dense(700, activation='relu'))
    model2.add(Dropout(0.2))
    model2.add(layers.Dense(29, activation='softmax'))

    model2.summary()
    metricss = [keras.metrics.BinaryAccuracy(name='accuracy'), keras.metrics.Precision(name='precision'),
                keras.metrics.Recall(name='recall')]
    model2.compile(optimizer='adam',
                   loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                   metrics=metricss)

    history = model2.fit(train_images, train_labels, epochs=10,
                         validation_data=(test_images, test_labels))

    plt.plot(history.history['accuracy'], label='accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.5, 1])
    plt.legend(loc='lower right')
    test_loss_two, test_acc_two = model2.evaluate(test_images, test_labels, verbose=2)
    return test_loss_two, test_acc_two



test_loss_two, test_acc_two = first_model()
print(test_acc_two)
# test_loss_one, test_acc_one = first_model()
# print(test_acc_one)
#with tf.device('/GPU:0'):

