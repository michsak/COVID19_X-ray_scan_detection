import os
import random
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.image import imread
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv2D, Flatten, MaxPooling2D, Dense, BatchNormalization, Activation
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from sklearn.metrics import classification_report
from datetime import datetime


def visualisation():
    img_number = 0
    path = "C:\\Users\\micha\\OneDrive\\Documents\\Machine learning\\COVID19_prediction\\COVID-19_Radiography_Database"
    covid_location = path + '\\COVID-19\\' + os.listdir(path + "\\COVID-19")[img_number]
    normal_location = path + '\\NORMAL\\' + os.listdir(path + "\\NORMAL")[img_number]
    vir_pneumonia_location = path + '\\Viral Pneumonia\\' + os.listdir(path + '\\Viral Pneumonia')[img_number]
    plt.imshow(imread(covid_location))
    plt.show()


def data_preprocessing():
    how_many = 700
    size = 512
    categories = ['COVID-19', 'NORMAL']
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
        if category == 'COVID-19':
            path_2 = os.path.join(datadir, 'COVID-19')
            for image in os.listdir(path_2):
                try:
                    img_array = cv2.imread(os.path.join(path_2, image), cv2.IMREAD_GRAYSCALE)
                    sc_array = cv2.resize(img_array, (size, size))
                    training_data.append([sc_array, class_number])
                except Exception as e:
                    pass
        else:
            pass
    random.shuffle(training_data)
    for features, label in training_data:
        X.append(features)
        y.append(label)
    X = np.array(X).reshape(-1, size, size, 1)
    return {'X': X, 'y': y}


def network():
    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=(4, 4), input_shape=(512, 512, 1), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(filters=32, kernel_size=(4, 4), input_shape=(512, 512, 1), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(filters=64, kernel_size=(4, 4), input_shape=(512, 512, 1), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(filters=64, kernel_size=(4, 4), input_shape=(512, 512, 1), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', metrics=['accuracy'], loss='binary_crossentropy')
    model.save('best_model')
    model.summary()
    return model


def training(training_model, X_train, X_val, y_train, y_val):
    es = EarlyStopping(patience=4, monitor='val_loss')
    filepath = "weights.best.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
    timestamp = datetime.now().strftime('%Y-%m-%d--%H%M')
    log_directory = 'logs\\fit' + '\\' + timestamp
    board = TensorBoard(log_dir=log_directory, histogram_freq=1, write_graph=True, update_freq='epoch', profile_batch=2, embeddings_freq=1)

    training_model.fit(X_train, y_train, epochs=10, batch_size=8, validation_data=(X_val, y_val), callbacks=[es, board, checkpoint])
    metrics = pd.DataFrame(training_model.history.history)
    metrics[['accuracy', 'val_accuracy']].plot()
    plt.show()
    plt.clf()
    metrics[['loss', 'val_loss']].plot()
    plt.show()
    predictions = training_model.predict_classes(X_val)
    clas_report = classification_report(y_val, predictions)
    print(clas_report)


if __name__ == "__main__":
    X = data_preprocessing()['X']
    y = data_preprocessing()['y']
    X_train, X_val, y_train, y_val = train_test_split(X, y, random_state=333, test_size=0.2)
    X_train = X_train/255
    X_val = X_val/255
    training(network(), X_train, X_val, y_train, y_val)
