# COVID19_X-ray_scan_detection
Convolutional neural network created for the purpose of detecting coronavirus disease.

## Table of contents
* [General info](#general-info)
* [Technologies](#technologies)
* [Sources](*sources)

## General info
Project was made during the COVID-19 pandemic.
It is attempt to divide people into two categories - healthy people and those infected by SARS-CoV-2 virus.
It was accomplished by contructing convolutional neural network with Conv and Dense layers (also Maxpooling, Batachormalization, etc.)
Unfortunately due to insufficient number of X-rays made on infected people accuracy looks like below.


Second attempt was made on the same network, but additionaly I used ImageDataGenerator to incerase number of images (especially COVID-19 X-rays). Homever even this actions improved hardly anything.


As it is shown above the best way to improve accurcy is to get a lot more data, network doesn't work well on less than 300 basic photos on each category.
(Project probably to be continued when more data is available).


## Technologies
Python 3.7, libraries:
* pandas
* numpy
* matplotlib
* keras
* sklearn
* cv2
* os
* random
* datetime


## Sources
Dataset taken from https://www.kaggle.com/tawsifurrahman/covid19-radiography-database
