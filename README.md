# pulsar-or-not
* Pulsars are neutron stars with extremely high rotational speed. They are magnetic dipoles creating a very powerful magnetic field and emittng electromagnetic radiation in a wide frequency range. They are used as points of
reference for the research of nearby stellar objects. The disruptions in their periodic behaviour could indicate that an event has taken place in the universe. They can also be used for the research of matter in states of 
high density as they are one of the densest objects in the observable universe.
* The purpose of this project is to make several models using different classification algorithms to solve this binary classification problem (is a star pulsar or not?). I have tested LDA, QDA, Logistic Regression, Desicion
Tree and SVM algorithms. I have used grid searchs for hyperparameter optimazation. The problem is unbalanced since for the positive class (1, is pulsar) we have obtained almost 10 times less samples. We can solve this problem
by adding weights to the samples (if the algorithm has a weight option we set to balanced) or we perform oversampling useing SMOTE, but does it improve the accuracy of the model? Another part of the project is feature selection. Using select K best we remove one by one the less important features ranked by select K best until we are left with one. If we observe a large drop of the accuracy it is indicated that that feature was important and cannot be ommited.

The dataset contains NaNs and the last word of the filename of Pulsar_Classification files indicates the method that these values have been filled out. For the feature selection the dataset that the NaN values have been removed is used.
