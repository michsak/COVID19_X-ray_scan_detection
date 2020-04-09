import os
import random
import matplotlib.pyplot as plt
from matplotlib.image import imread
import numpy as np


def visualisation():
    img_number = 0
    path = "C:\\Users\\micha\\OneDrive\\Documents\\Machine learning\\COVID19_prediction\\COVID-19_Radiography_Database"
    covid_location = path + '\\COVID-19\\' + os.listdir(path + "\\COVID-19")[img_number]
    normal_location = path +'\\NORMAL\\' + os.listdir(path + "\\NORMAL")[img_number]
    vir_pneumonia_location = path + '\\Viral Pneumonia\\' + os.listdir(path + '\\Viral Pneumonia')[img_number]
    plt.imshow(imread(covid_location))
    image_shape = imread(covid_location).shape
    #print(image_shape)
    #plt.show()


def data_preprocessing():
    path = "C:\\Users\\micha\\OneDrive\\Documents\\Machine learning\\COVID19_prediction\\COVID-19_Radiography_Database"
    how_many = 300
    z = []
    k = []
    for i in range(0, how_many):
        rd = random.choice(os.listdir(path + '\\NORMAL'))
        while rd in z:
            rd = random.choice(os.listdir(path + '\\NORMAL'))
        z.append(imread(path + '\\NORMAL\\' + rd))
    for j in range(0, how_many):
        rd = random.choice(os.listdir(path + '\\Viral Pneumonia'))
        while rd in k:
            rd = random.choice(os.listdir(path + '\\Viral Pneumonia'))
        k.append(rd)
    rd = random.choice(os.listdir(path + '\\NORMAL'))
    print(z)

def neural_network_org():
    pass


if __name__ == "__main__":
    data_preprocessing()
    visualisation()