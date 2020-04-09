import os
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread
from sklearn.model_selection import train_test_split



def visualisation():
    img_number = 0
    path = "C:\\Users\\micha\\OneDrive\\Documents\\Machine learning\\COVID19_prediction\\COVID-19_Radiography_Database"
    covid_location = path + '\\COVID-19\\' + os.listdir(path + "\\COVID-19")[img_number]
    normal_location = path +'\\NORMAL\\' + os.listdir(path + "\\NORMAL")[img_number]
    vir_pneumonia_location = path + '\\Viral Pneumonia\\' + os.listdir(path + '\\Viral Pneumonia')[img_number]
    plt.imshow(imread(covid_location))
    image_shape = imread(covid_location).shape
    print(image_shape)
    plt.show()


def data_preprocessing():
    path = "C:\\Users\\micha\\OneDrive\\Documents\\Machine learning\\COVID19_prediction\\COVID-19_Radiography_Database"
    how_many = 300
    size = 512
    categories = ['COVID-19', 'NORMAL', 'Viral Pneumonia']
    training_data = []
    X = []
    y = []

    datadir = "C:\\Users\\micha\\OneDrive\\Documents\\Machine learning\\COVID19_prediction\\COVID-19_Radiography_Database"
    for category in categories:
        path = os.path.join(datadir, category)
        class_number = categories.index(category)
        for i in range(0, how_many):
            if category == 'NORMAL':
                try:
                    rd = random.choice(os.listdir(path))
                    while rd in training_data:
                        rd = random.choice(os.listdir(path))
                    img_array = cv2.imread(os.path.join(path, rd), cv2.IMREAD_GRAYSCALE)
                    sc_array = cv2.resize(img_array, (size, size))
                    training_data.append([sc_array, class_number])
                except Exception as e:
                    pass
            if category == 'Viral Pneumonia':
                try:
                    rd = random.choice(os.listdir(path))
                    while rd in training_data:
                        rd = random.choice(os.listdir(path))
                    img_array = cv2.imread(os.path.join(path, rd), cv2.IMREAD_GRAYSCALE)
                    sc_array = cv2.resize(img_array, (size, size))
                    training_data.append([sc_array, class_number])
                except Exception as e:
                    pass
        if category == 'COVID-19':
            path_2 = os.path.join(datadir, 'COVID-19')
            for image in os.listdir(path_2):
                try:
                    img_array = cv2.imread(os.path.join(path_2, image), cv2.IMREAD_GRAYSCALE)
                    sc_array = cv2.resize(img_array, (size, size))
                    training_data.append([sc_array, class_number])
                except Exception as e:
                    pass
    random.shuffle(training_data)
    for features, label in training_data:
        X.append(features)
        y.append(label)
    X = np.array(X).reshape(-1, size, size, 1)
    return {'X': X, 'y': y}


if __name__ == "__main__":
    X = data_preprocessing()['X']
    y = data_preprocessing()['y']
    X_train, X_val, y_train, y_val = train_test_split(X, y, random_state=333, test_size=0.2)
    X_train = X_train/255
    X_val = X_val/255
