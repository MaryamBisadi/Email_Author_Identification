# Email_Author_Identification
Objective of this code is to classify a set of emails based only on the text of the email. Half of the emails were written by one person and the other half by another person at the same company. The classification has been done by Navie Bayes, SVM and Decision Tree techniques and accuracy and training time of each one has been caculated.  

The data used for this project has been downloaded from https://github.com/udacity/ud120-projects/tree/master/tools

## Naive Bayes

Each unique word is considered as a feature. The relative simplicity of the Naive Baye and the independent features assumption  make it a strong performer for classifying texts.

training time: 1.26 s

accuracy: 0.9732

## SVM
Accuracy and training time of classification of emails depend on parameters that have been used for SVM. To have better understanding of parametes affects on accuracy and training time the following plots have been genertated.

The first plot shows the accuracy and the second one training time based on feature selection percentile.

![svm-accuracy-selectpercentile](https://user-images.githubusercontent.com/39537957/40526997-d0686a56-5f9e-11e8-82fa-53437dbe0dc8.png)

![svm-trainintime-selectpercentile](https://user-images.githubusercontent.com/39537957/40530047-28dbb2ee-5fac-11e8-9c6d-7c41c641c07b.png)

## Decision Tree
The following accuracy and traing time of are generated with default min_sample_split which is 2 and 10% SelectPercentile
training time: 38.27 s

accuracy: 0.9926

### Comparison

|                     | __Navie Bayes__ |   __SVM__  |__Decision Tree__ |
|---------------------|-----------------|------------|------------------|
|__Accuracy (%)__     |      97.32      |    99.08   |       99.26      |
|__Training Time (s)__|      1.26       |   123.52   |       38.27      |
|__Testing Time  (s)__|      0.12       |   11.92    |        0.05      |
