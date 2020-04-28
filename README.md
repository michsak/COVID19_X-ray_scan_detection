# COVID19_X-ray_scan_detection
Convolutional neural network created for the purpose of detecting coronavirus disease.

## Table of contents
* [General info](#general-info)
* [Technologies](#technologies)
* [Sources](#sources)

## General info
Project was made during the COVID-19 pandemic.</br>
The first part is devoted to show how many people this virus infects and how does it look worldwide.
![](https://raw.githubusercontent.com/michsak/COVID19_X-ray_scan_detection/master/results/cases_merged.png?token=AN7LSJ7Z4TJZT3TREWQXLY26URQAM)
![](https://raw.githubusercontent.com/michsak/COVID19_X-ray_scan_detection/master/results/world_death_map.JPG?token=AN7LSJZOQHPLXCMASBVYEPS6UROUS)

Second part is attempt to divide people into two categories - healthy people(1) and those infected by SARS-CoV-2 virus(0).
It was accomplished by contructing convolutional neural network with Conv and Dense layers (also Maxpooling, Batchnormalization, etc.).</br> 
Unfortunately due to insufficient number of X-rays made on infected people accuracy (and loss) looks like below. Actually it's not that bad, but many uninfected people were classified wrong.
![](https://raw.githubusercontent.com/michsak/COVID19_X-ray_scan_detection/master/results/merged.png?token=AN7LSJYOFHNIYHOHPS35IJ26URP6S)

Second attempt was made on the same network, but additionaly I used ImageDataGenerator to incerase number of images (especially COVID-19 X-rays). However even this actions improved hardly anything (actually only repeatability of final results).
![](https://raw.githubusercontent.com/michsak/COVID19_X-ray_scan_detection/master/results/gen_merged.png?token=AN7LSJ3BYL6OGW3OOFSK6SS6UTPDY)

As it is shown above the best way to improve accuracy is to get a lot more data. Network doesn't work well on imbalanced data and less than 300 basic photos of infected category. </br>
(Project probably to be continued when more data is available).


## Technologies
Python 3.7; libraries:
* pandas
* numpy
* matplotlib
* keras
* sklearn
* h5py
* cv2
* os
* random
* datetime


## Sources
Dataset taken from: </br>https://www.kaggle.com/tawsifurrahman/covid19-radiography-database </br> https://github.com/agchung/Figure1-COVID-chestxray-dataset/tree/master/images
