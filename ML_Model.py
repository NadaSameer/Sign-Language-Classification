from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn import svm
import cv2
from cv2 import *
import os
import numpy as np


# os.chdir("E://Level Four//First Term//Machine 2//Assignment2//ASL_Alphabet_Dataset//asl_alphabet_train\A")
# print("Current Working Directory " , os.getcwd())


def Canny_EdgeDetection(img):
    img_blur = cv2.GaussianBlur(img, (3, 3), 0)
    edges = cv2.Canny(image=img_blur, threshold1=4, threshold2=100)
    return edges


def Load_Images(Folder_Path, mode):
    images = []
    for filename in os.listdir(Folder_Path):
        img = cv2.imread(os.path.join(Folder_Path, filename))
        img2 = cv2.resize(img, (64, 64))
        # imgCol = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        if mode == 'RGB':
            imgCol = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        elif mode == 'Gray':
            imgCol = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        elif mode == 'Binary':
            r, imgCol = cv2.threshold(img2, 149, 255, cv2.THRESH_BINARY_INV)
        
        img_edges = Canny_EdgeDetection(imgCol)
        #norm = cv2.normalize(img_edges, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        if img is not None:
            images.append(img_edges)

    return images


def get_num_pixels(images):
    width, height = np.shape(images[0])
    return width * height


def Image_Vectorization(ImagesList, pixels):
    ArrayOfImages = []
    for x in ImagesList:
        v = x.reshape(1, pixels)
        ArrayOfImages.append(v)
    return ArrayOfImages


def Load_TrainingData(alphbet, Folder_Path, mode):
    training_data = []
    for i in alphbet:
        path = Folder_Path + "//" + i
        Class_num = alphbet.index(i)
        # print(path)
        images = Load_Images(path, mode)
        # rint(np.shape(images))
        pixels = get_num_pixels(images)
        images2 = Image_Vectorization(images, pixels)
        # print(np.shape(images2))
        for j in images2:
            training_data.append((j, Class_num))

    # print(training_data[0:11])
    # print(np.shape(training_data[0]))
    return training_data, pixels


def Load_testData(Head, Folder_Path, mode):
    test_data = []
    images = Load_Images(Folder_Path, mode)
    pixels = get_num_pixels(images)
    images2 = Image_Vectorization(images, pixels)
    i = 0
    for j in images2:
        test_data.append((j, i))
        i += 1

    return test_data


alphbet = ['A', 'B', 'C', 'D', 'del', 'E', 'F', 'G', 'H', 'I', 'J', 'K',
           'L', 'M', 'N', 'nothing', 'O', 'P', 'Q', 'R', 'S',
           'space', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

Head = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K',
        'L', 'M', 'N', 'nothing', 'O', 'P', 'Q', 'R', 'S',
        'space', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

def GetXY_Train_Test(Train_Data, Test_Data):
    X_Train = []
    Y_Train = []
    X_Test = []
    Y_Test = []

    for features, label in Train_Data:
        X_Train.append(np.array(features))
        Y_Train.append(label)

    Y_Train = np.array(Y_Train)
    for features, label in Test_Data:
        X_Test.append(features)
        Y_Test.append(label)

    Y_Test = np.array(Y_Test)
    return X_Train, Y_Train, X_Test, Y_Test


def SVM_Classifier(X_train, y_train, X_test):
    # Create a svm Classifier
    clf = svm.SVC(kernel='linear')
    # Train the model using the training sets
    clf.fit(X_train, y_train)
    # Predict the response for test dataset
    y_pred = clf.predict(X_test)

    return y_pred


def Logistic_Regression(X_train, y_train, X_test):
    logreg = LogisticRegression()
    logreg.fit(X_train, y_train)
    y_pred = logreg.predict(X_test)
    return y_pred


def Decision_Tree_Model(X_train, y_train, X_test):
    DTC = DecisionTreeClassifier(criterion="entropy")
    DTC = DTC.fit(X_train, y_train)
    y_pred = DTC.predict(X_test)
    return y_pred


def Calc_Accuracy(Y_pred, Y_test):
    return metrics.accuracy_score(Y_test, Y_pred)


def PrecisionANDRecall(Y_pred, Y_test):
    Pre = precision_score(Y_test, Y_pred, average='micro')
    Rec = recall_score(Y_test, Y_pred, average='micro')
    return Pre, Rec

mode = "RGB"
Folder_Path = "D:/Iseul/Education/College/(4)_1st_Semester/Machine and Bioinformatics/Assignment_3/asl_alphabet_train"
TrainData, pixels = Load_TrainingData(alphbet, Folder_Path, mode)

Folder_Path2 = "D:/Iseul/Education/College/(4)_1st_Semester/Machine and Bioinformatics/Assignment_3/asl_alphabet_test"
TestData = Load_testData(Head, Folder_Path2, mode)
X_Train, Y_Train, X_Test, Y_Test = GetXY_Train_Test(TrainData, TestData)

X_Train = np.array(X_Train)
X_Test = np.array(X_Test)

nsamples, nx, ny = X_Train.shape
train_dataset = X_Train.reshape((nsamples, nx * ny))

nsamples2, nx2, ny2 = X_Test.shape
test_dataset = X_Test.reshape((nsamples2, nx2 * ny2))
print("\n\n\n RGB")

# Y_pred2 = Logistic_Regression(train_dataset, Y_Train.T, test_dataset)
#
# print("Logistic_Regression")
# print(Calc_Accuracy(Y_pred2, Y_Test))
# print(PrecisionANDRecall(Y_pred2, Y_Test))
#
# print("SVM_Classifier")
# Y_pred = SVM_Classifier(train_dataset, Y_Train, test_dataset)
# print(Calc_Accuracy(Y_pred, Y_Test))
# print(PrecisionANDRecall(Y_pred, Y_Test))
#
# print("Decision_Tree")
# pred = Decision_Tree_Model(train_dataset, Y_Train, test_dataset)
# print(Calc_Accuracy(pred, Y_Test))
# print(PrecisionANDRecall(pred, Y_Test))

mode = "Gray"
Folder_Path = "D:/Iseul/Education/College/(4)_1st_Semester/Machine and Bioinformatics/Assignment_3/asl_alphabet_train"
TrainData, pixels = Load_TrainingData(alphbet, Folder_Path, mode)

Folder_Path2 = "D:/Iseul/Education/College/(4)_1st_Semester/Machine and Bioinformatics/Assignment_2/asl_alphabet_test"
TestData = Load_testData(Head, Folder_Path2, mode)
X_Train, Y_Train, X_Test, Y_Test = GetXY_Train_Test(TrainData, TestData)

X_Train = np.array(X_Train)
X_Test = np.array(X_Test)

nsamples, nx, ny = X_Train.shape
train_dataset = X_Train.reshape((nsamples, nx * ny))

nsamples2, nx2, ny2 = X_Test.shape
test_dataset = X_Test.reshape((nsamples2, nx2 * ny2))
print("\n\n\n Gray")

Y_pred2 = Logistic_Regression(train_dataset, Y_Train.T, test_dataset)
print("Logistic_Regression")
print(Calc_Accuracy(Y_pred2, Y_Test))
print(PrecisionANDRecall(Y_pred2, Y_Test))

print("SVM_Classifier")
Y_pred = SVM_Classifier(train_dataset, Y_Train, test_dataset)
print(Calc_Accuracy(Y_pred, Y_Test))
print(PrecisionANDRecall(Y_pred, Y_Test))

print("Decision_Tree")
pred = Decision_Tree_Model(train_dataset, Y_Train, test_dataset)
print(Calc_Accuracy(pred, Y_Test))
print(PrecisionANDRecall(pred, Y_Test))


mode = "Binary"
Folder_Path = "D:/Iseul/Education/College/(4)_1st_Semester/Machine and Bioinformatics/Assignment_3/asl_alphabet_train"
TrainData, pixels = Load_TrainingData(alphbet, Folder_Path, mode)

Folder_Path2 = "D:/Iseul/Education/College/(4)_1st_Semester/Machine and Bioinformatics/Assignment_2/asl_alphabet_test"
TestData = Load_testData(Head, Folder_Path2, mode)
X_Train, Y_Train, X_Test, Y_Test = GetXY_Train_Test(TrainData, TestData)

X_Train = np.array(X_Train)
X_Test = np.array(X_Test)

nsamples, nx, ny = X_Train.shape
train_dataset = X_Train.reshape((nsamples, nx * ny))

nsamples2, nx2, ny2 = X_Test.shape
test_dataset = X_Test.reshape((nsamples2, nx2 * ny2))
print("\n\n\n Binary")

Y_pred2 = Logistic_Regression(train_dataset, Y_Train.T, test_dataset)
print("Logistic_Regression")
print(Calc_Accuracy(Y_pred2, Y_Test))
print(PrecisionANDRecall(Y_pred2, Y_Test))

print("SVM_Classifier")
Y_pred = SVM_Classifier(train_dataset, Y_Train, test_dataset)
print(Calc_Accuracy(Y_pred, Y_Test))
print(PrecisionANDRecall(Y_pred, Y_Test))

print("Decision_Tree")
pred = Decision_Tree_Model(train_dataset, Y_Train, test_dataset)
print(Calc_Accuracy(pred, Y_Test))
print(PrecisionANDRecall(pred, Y_Test))

cv2.waitKey(0)
cv2.destroyAllWindows()