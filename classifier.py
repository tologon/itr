# Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org>
# License: BSD 3 clause

# ------------------------------------------------------------------------------
# Author:   Tologon Eshimkanov (https://github.com/tologon)
# Course:   COMP 3770-01 - Introduction to Artificial Intelligence
# School:   Wentworth Institute of Technology
# Project:  Simplified Digit Recognition
# ------------------------------------------------------------------------------

# Standard scientific Python imports
import matplotlib.pyplot as plt

# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, cross_validation

# The digits dataset
digits = datasets.load_digits()

# The data that we are interested in is made of 8x8 images of digits, let's
# have a look at the first 3 images, stored in the `images` attribute of the
# dataset.  If we were working from image files, we could load them using
# pylab.imread.  Note that each image must have the same size. For these
# images, we know which digit they represent: it is given in the 'target' of
# the dataset.
images_and_labels = list(zip(digits.images, digits.target))
for index, (image, label) in enumerate(images_and_labels[:4]):
    plt.subplot(2, 4, index + 1)
    plt.axis('off')
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title('Training: %i' % label)

# To apply a classifier on this data, we need to flatten the image, to
# turn the data in a (samples, feature) matrix:
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))

# Create a classifier: a support vector classifier
classifier = svm.LinearSVC()

# Cross validate the data
k_fold = 10
scores = cross_validation.cross_val_score(
    classifier, data, digits.target, cv=k_fold
)

print("SVM classifier accuracy (on 10-fold cross-validation): \
      %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

# We learn the digits on the first half of the digits
classifier.fit(data, digits.target)
