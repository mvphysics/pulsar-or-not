# pulsar-or-not
* Pulsars are neutron stars with extremely high rotational speed. They are magnetic dipoles that create a very powerful magnetic field and emit electromagnetic radiation in a wide frequency range. They are used as points of
reference for the research of nearby stellar objects. The disruptions in their periodic behaviour could indicate that an event has taken place in the universe. Pulsars can also be used for the research on matter in a state of 
high density as they are one of the densest objects in the observable universe.
* The purpose of this project is to make several models using different classification algorithms to solve this binary classification problem (is a star pulsar or not?). I have tested LDA, QDA, Logistic Regression, Desicion
Tree and SVM algorithms. I have used grid searchs for hyperparameter optimazation. The problem is unbalanced since for the positive class (1, is pulsar) we have obtained almost 10 times fewer samples than those of the negative class (0, is not pulsar). We can solve this problem by adding weights to the samples (if the algorithm has a weight option we set to balanced) or we perform oversampling using SMOTE, but does it improve the accuracy of the model? Another part of the project is feature selection. Using select K best we remove, one by one, the less important features ranked by select K best until we are left with one. If we observe a large drop in accuracy it is indicated that that feature was important and cannot be omitted.

The dataset contains NaNs and the last word of the filename of Pulsar_Classification files indicates the method in which these values have been filled out. For the feature selection, the dataset that the NaN values have been removed is used.
