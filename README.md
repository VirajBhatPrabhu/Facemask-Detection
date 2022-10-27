# Facemask-Detection

### Aim of The Project 
The rapid spreading of Coronavirus disease is a major health risk that the whole world is facing for the last two years. One of the main causes of the fast spreading of this virus is the direct contact of people with each other

There are many precautionary measures to reduce the spread of this virus; however, the major one is wearing face masks in public places. Detection of face masks in public places is a real challenge that needs to be addressed to reduce the risk of spreading the virus


### Goal 
To address the challenge of detection of facemask in public, we will build an automated system for face mask detection using deep learning

### Project Explanation

* The dataset consists of 1,376 images belonging to two classes:with_mask and without_mask
* We used tensorflow & keras for Data augmentation, Loading the VGG19 classifier, Pre-processing, Load image data and Train our model
* We used scikit-learn for binarizing class labels & segmenting our dataset
* We fine-tuned the VGG 19 architecture
* We used approach of Tansfer Learning for this project
* The other module we used at this project was OpenCV 
* We also used Haarcascade Classifier to detect the face
