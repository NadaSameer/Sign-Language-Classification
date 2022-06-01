import os
import cv2
import keras
import random
from cv2 import *
import numpy as np
from matplotlib import *
import tensorflow as tf
from sklearn import metrics
from keras import backend as K
from keras import regularizers
import matplotlib.pyplot as plt
from keras.layers import Dropout
from keras.callbacks import TensorBoard
from tensorflow.keras.optimizers import Adam
from keras.initializers import glorot_uniform
from keras.models import Sequential, load_model
from keras.losses import categorical_crossentropy
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report
from keras.models import Sequential, Model,load_model
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras import datasets, layers, models
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Conv2D, Dense, Flatten, MaxPool2D
from tensorflow.keras.applications.resnet50 import ResNet50
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D,MaxPool2D

def Load_Images(Folder_Path, img_name):
    img = cv2.imread(os.path.join(Folder_Path, img_name))
    img2 = cv2.resize(img, (64, 64))
    imgCol = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    return imgCol

def load_training_data(Alphbet,DATAPath):
    training_data = []
    for a in Alphbet:
        path = os.path.join(DATAPath, a) #path to alphabets
        label = Alphbet.index(a)
        print(path)
        for img in os.listdir(path):
              try:
                  img1= cv2.imread(os.path.join(path,img), cv2.IMREAD_COLOR)
                  newimg = cv2.resize(img1, (64, 64))
                  img = cv2.cvtColor(newimg, cv2.COLOR_BGR2GRAY)
                  training_data.append([img, label])
              except Exception as e:
                  pass
    return training_data

def Load_TrainingData(alphbet, Folder_Path):
    training_data2 = []
    labels2 = []
    k = 0
    for i in alphbet:
        Class_Number = alphbet.index(i)
        Data_Folder = os.path.join(Folder_Path, i)
        training_data = []
        labels = []
        print(Data_Folder)
        for img_name in os.listdir(Data_Folder):
            try:
                img = Load_Images(Data_Folder, img_name)
                img = np.array(img)
                training_data.append(img)
                labels.append(Class_Number)
            except Exception as e:
                pass
    return training_data, labels

dirc= "D:/Iseul/Education/College/(4)_1st_Semester/Machine and Bioinformatics/Assignment_3/asl_alphabet_train"
alphbet = ['A', 'B', 'C', 'D', 'del', 'E', 'F', 'G', 'H', 'I', 'J', 'K',
           'L', 'M', 'N', 'nothing', 'O', 'P', 'Q', 'R', 'S',
           'space', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

def loading(model_name):
    if model_name == 'vgg16':
        training_data2= load_training_data(alphbet, dirc)
        return training_data2
    if model_name == 'Bvgg16':
        Train_DATA, LabelTrain = Load_TrainingData(alphbet, dirc) #for the built in vgg16
        return Train_DATA, LabelTrain

def Load_testData(alphbet, Folder_Path):
    test_data = []
    i = 0
    labeltest = []
    for img_name in os.listdir(Folder_Path):
        try:
            img = Load_Images(Folder_Path, img_name)
            img = np.array(img)
            test_data.append(img)
            labeltest.append(i)
        except Exception as e:
            pass
        i += 1
    return test_data, labeltest

Folder_Path = "D:/Iseul/Education/College/(4)_1st_Semester/Machine and Bioinformatics/Assignment_3/asl_alphabet_test"
Test_DATA, LabelTest = Load_testData(alphbet, Folder_Path)

def AugmentedData(training_data):
    X=[]
    y=[]
    for features,label in training_data:
        X.append(features)
        y.append(label)

    X = np.array(X)
    X=X.reshape(-1,64, 64, 1)
    X = X.astype('float32')/255.0 #to normalize data
    y = tf.keras.utils.to_categorical(y) #one-hot encoding
    y = np.array(y)
    datagen = ImageDataGenerator(validation_split = 0.1, 
                                     rotation_range=20,
                                     width_shift_range=0.2,
                                     height_shift_range=0.2,
                                     horizontal_flip=True)
    train_data = datagen.flow(X, y, batch_size = 64, shuffle=True, subset='training')
    val_data = datagen.flow(X, y, batch_size = 64, shuffle=True, subset='validation')
    return (train_data, val_data, X, y)



def Getxy_Train(training_data, labels):
    X_Train = training_data
    Y_Train = labels
    X_Train = np.array(X_Train)
    X_Train = X_Train.astype('float32') / 255.0  # to normalize images
    Y_Train = np.array(Y_Train)
    return X_Train, Y_Train

def Getxy_Test(testdata, labeltest):
    X_Test = testdata
    Y_Test = labeltest
    X_Test = np.array(X_Test)
    X_Test = X_Test.astype('float32') / 255.0  # to normalize images
    Y_Test = np.array(Y_Test)
    return X_Test, Y_Test




def fit_model(train_data, val_data, model):
    filepath = "weights.best.h5"
    checkpoint = keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', 
                                                 verbose=1, save_best_only=True,
                                                 save_weights_only=False,
                                                 mode='auto', period=1)
    tensorboard_callback = keras.callbacks.TensorBoard("logs")

    model.fit_generator(train_data,epochs=10,
                        steps_per_epoch = 500, validation_data = val_data,
                        validation_steps= len(val_data),
                        callbacks = [checkpoint, tensorboard_callback])


def Vgg16_model():
    model = Sequential()
    model.add(Conv2D(input_shape=(64, 64, 1), filters=64, kernel_size=(3, 3),
                     padding='same', activation='relu'))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Flatten())
    model.add(Dense(units=4096, activation='relu'))
    # model.add(Dropout(0.2))
    model.add(Dense(units=4096, activation='relu'))
    # model.add(Dropout(0.3))
    model.add(Dense(units=29, activation='softmax'))

    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss=keras.losses.categorical_crossentropy,
                  metrics=['accuracy'])

    filepath = "weights.best.h5"
    checkpoint = keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss',
                                                 verbose=1, save_best_only=True,
                                                 save_weights_only=False,
                                                 mode='auto', period=1)
    tensorboard_callback = keras.callbacks.TensorBoard("logs")

    model.fit_generator(train_data, epochs=10,
                        steps_per_epoch=500, validation_data=val_data,
                        validation_steps=len(val_data),
                        callbacks=[checkpoint, tensorboard_callback])
    input_shape2 = (64, 64, 1)
    validation_labels2, validation_preds2 = show_classification_report(X2, y2, input_shape2, model)
    return (validation_labels2, validation_preds2)

def show_classification_report(X, y, input_shape, model):
    validation = [X[i] for i in range(int(0.1 * len(X)))]
    validation_labels = [ np.argmax(y[i]) for i in range(int(0.1 * len(y)))]
    validation_preds = []
    labels = [i for i in range(29)]
    for img in validation:
        img = img.reshape((1,) + input_shape)
        pred = model.predict_classes(img)
        validation_preds.append(pred[0])
    print(classification_report(validation_labels, validation_preds,labels, target_names=alphbet))
    return (validation_labels, validation_preds)


# validation_labels2, validation_preds2 = show_classification_report(X2, y2, input_shape2, model)
# print(validation_preds2,validation_labels2)

def Vgg16_builtin():
    model = Sequential([VGG16(weights='imagenet', include_top=True, input_shape=(64, 64, 3), pooling='max'),
                        Flatten(),
                        Dense(600, activation='relu'),
                        Dropout(0.2),
                        Dense(29, activation='softmax')])

    model.summary()

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'], run_eagerly=True)

    model.fit(train_images, train_lables, epochs=10,
                        validation_data=(test_images, test_lables))
    test_loss, test_acc = model.evaluate(test_images, test_lables, verbose=2)
    return test_acc

# ----------------------------------------------------------------------------------------------------------------------
def identity_block(X, f, filters, stage, block):

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    F1, F2, F3 = filters

    X_shortcut = X

    X = Conv2D(filters=F1, kernel_size=(1, 1), strides=(1, 1), padding='valid',
               name=conv_name_base + '2a', kernel_initializer=glorot_uniform())(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
    X = Activation('relu')(X)

    X = Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding='same',
               name=conv_name_base + '2b', kernel_initializer=glorot_uniform())(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    X = Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid',
               name=conv_name_base + '2c', kernel_initializer=glorot_uniform())(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2c')(X)

    X = Add()([X, X_shortcut])# SKIP Connection
    X = Activation('relu')(X)

    return X

def convolutional_block(X, f, filters, stage, block, s=2):

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    F1, F2, F3 = filters

    X_shortcut = X

    X = Conv2D(filters=F1, kernel_size=(1, 1), strides=(s, s), padding='valid',
               name=conv_name_base + '2a', kernel_initializer=glorot_uniform())(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
    X = Activation('relu')(X)

    X = Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding='same',
               name=conv_name_base + '2b', kernel_initializer=glorot_uniform())(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    X = Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid',
               name=conv_name_base + '2c', kernel_initializer=glorot_uniform())(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2c')(X)

    X_shortcut = Conv2D(filters=F3, kernel_size=(1, 1), strides=(s, s), padding='valid',
                        name=conv_name_base + '1', kernel_initializer=glorot_uniform())(X_shortcut)
    X_shortcut = BatchNormalization(axis=3, name=bn_name_base + '1')(X_shortcut)

    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)

    return X

def ResNet50(input_shape=(64, 64, 1)):

    X_input = Input(input_shape)

    X = ZeroPadding2D((3, 3))(X_input)

    X = Conv2D(64, (7, 7), strides=(2, 2), name='conv1', kernel_initializer=glorot_uniform())(X)
    X = BatchNormalization(axis=3, name='bn_conv1')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((3, 3), strides=(2, 2))(X)

    X = convolutional_block(X, f=3, filters=[64, 64, 256], stage=2, block='a', s=1)
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='b')
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='c')


    X = convolutional_block(X, f=3, filters=[128, 128, 512], stage=3, block='a', s=2)
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='b')
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='c')
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='d')

    X = convolutional_block(X, f=3, filters=[256, 256, 1024], stage=4, block='a', s=2)
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='b')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='c')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='d')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='e')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='f')

    X = X = convolutional_block(X, f=3, filters=[512, 512, 2048], stage=5, block='a', s=2)
    X = identity_block(X, 3, [512, 512, 2048], stage=5, block='b')
    X = identity_block(X, 3, [512, 512, 2048], stage=5, block='c')

    X = AveragePooling2D(pool_size=(2, 2), padding='same')(X)

    model = Model(inputs=X_input, outputs=X, name='ResNet50')


    return model

# ----------------------------------------------------------------------------------------------------------------------

model_name = 'Bvgg16'

if(model_name == 'vgg16'):
    training_data2 = loading(model_name)
    random.shuffle(training_data2)
    train_data, val_data, X2, y2 = AugmentedData(training_data2)
    Vgg16_model()
elif(model_name == 'Bvgg16'):
    Train_DATA, LabelTrain = loading(model_name)
    X_Train, Y_Train = Getxy_Train(Train_DATA, LabelTrain)
    X_Test, Y_Test = Getxy_Test(Test_DATA, LabelTest)

    train_images = np.array(X_Train)
    test_images = np.array(X_Test)
    train_lables = np.array(Y_Train)
    test_lables = np.array(Y_Test)
    test_acc = Vgg16_builtin()
    print(test_acc)
elif(model_name == 'resnet50'):
    training_data2 = loading(model_name)
    random.shuffle(training_data2)
    train_data, val_data, X2, y2 = AugmentedData(training_data2)

    model2 = ResNet50()

    headModel = model2.output
    headModel = Flatten()(headModel)
    headModel=Dense(256, activation='relu', name='fc1',kernel_initializer=glorot_uniform())(headModel)
    headModel=Dense(128, activation='relu', name='fc2',kernel_initializer=glorot_uniform())(headModel)
    headModel = Dense( 29,activation='sigmoid', name='fc3',kernel_initializer=glorot_uniform())(headModel)

    modelresnet = Model(inputs=model2.input, outputs=headModel)

    modelresnet.summary()

    dir = "D:/Iseul/Education/College/(4)_1st_Semester/Machine and Bioinformatics/Assignment_3/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5"
    model2.load_weights(dir)

    for layer in model2.layers:
       layer.trainable = False

    for layer in modelresnet.layers:
        print(layer, layer.trainable)

    es=EarlyStopping(monitor='val_accuracy', mode='max', verbose=1, patience=20)

    mc = ModelCheckpoint('/content/best_model.h5', monitor='val_accuracy', mode='')

    H = modelresnet.fit_generator(train_data,validation_data=val_data,epochs=100,verbose=1,callbacks=[mc,es])

