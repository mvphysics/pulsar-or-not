# Importing data
import pandas as pd
data = pd.read_csv(r'pulsar_data_train.csv')

# Importing useful libraries
import numpy as np
from sklearn import preprocessing
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from statistics import mean
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_validate
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE
from sklearn.utils import shuffle

# Creating array for storing the results
results=np.array(['method','best parameters','validation accuracy','validation f1 score','test accuracy','test f1 score',])

# Scaling data and creating train and test set
data=data.to_numpy()
min_max_scaler = preprocessing.MinMaxScaler()
x_data=min_max_scaler.fit_transform(data[:,:8])

imp_median = SimpleImputer(missing_values=np.nan, strategy='median')
imp_median.fit(x_data)
x_data=imp_median.transform(x_data)

y_data=data[:,8]
x_train,x_test,y_train,y_test=train_test_split(x_data,y_data,test_size=0.25,shuffle=True,random_state=0,stratify=y_data)
x_res,y_res=SMOTE(random_state=0).fit_resample(x_train,y_train)
x_res,y_res=shuffle(x_res,y_res,random_state=0)



# lda models no feature selection
model=LinearDiscriminantAnalysis(solver='svd').fit(x_train,y_train)
scores = cross_validate(model,x_train,y_train,scoring=('accuracy','f1') ,cv=5)
results=np.vstack((results,['lda','-',format(mean(scores['test_accuracy']),'.4f'),format(mean(scores['test_f1']),'.4f'),format(accuracy_score(y_test, model.predict(x_test)),'.4f'),format(f1_score(y_test, model.predict(x_test)),'.4f')]))

model=LinearDiscriminantAnalysis(solver='svd').fit(x_res,y_res)
scores = cross_validate(model,x_res,y_res,scoring=('accuracy','f1') ,cv=5)
results=np.vstack((results,['lda (SMOTE)','-',format(mean(scores['test_accuracy']),'.4f'),format(mean(scores['test_f1']),'.4f'),format(accuracy_score(y_test, model.predict(x_test)),'.4f'),format(f1_score(y_test, model.predict(x_test)),'.4f')]))



# qda models no feature selection
model=QuadraticDiscriminantAnalysis().fit(x_train,y_train)
scores = cross_validate(model,x_train,y_train,scoring=('accuracy','f1') ,cv=5)
results=np.vstack((results,['qda','-',format(mean(scores['test_accuracy']),'.4f'),format(mean(scores['test_f1']),'.4f'),format(accuracy_score(y_test, model.predict(x_test)),'.4f'),format(f1_score(y_test, model.predict(x_test)),'.4f')]))

model=QuadraticDiscriminantAnalysis().fit(x_res,y_res)
scores = cross_validate(model,x_res,y_res,scoring=('accuracy','f1') ,cv=5)
results=np.vstack((results,['qda (SMOTE)','-',format(mean(scores['test_accuracy']),'.4f'),format(mean(scores['test_f1']),'.4f'),format(accuracy_score(y_test, model.predict(x_test)),'.4f'),format(f1_score(y_test, model.predict(x_test)),'.4f')]))



#Logistic regression no feature selection
model=LogisticRegression()
parameters={'penalty':['l2',None],'solver':['lbfgs'],'C':range(1,11),'random_state':[0]}
gs=GridSearchCV(model,parameters,scoring=('accuracy','f1'),refit='f1',cv=5).fit(x_train,y_train)
model=gs.best_estimator_
results=np.vstack((results,['Logistic Regression',gs.best_params_,format(gs.cv_results_['mean_test_accuracy'][gs.best_index_],'.4f'),format(gs.cv_results_['mean_test_f1'][gs.best_index_],'.4f'),format(accuracy_score(y_test, model.predict(x_test)),'.4f'),format(f1_score(y_test, model.predict(x_test)),'.4f')]))

model=LogisticRegression()
parameters={'penalty':['l2',None],'solver':['lbfgs'],'C':range(1,11),'class_weight':['balanced'],'random_state':[0]}
gs=GridSearchCV(model,parameters,scoring=('accuracy','f1'),refit='f1',cv=5).fit(x_train,y_train)
model=gs.best_estimator_
results=np.vstack((results,['Logistic Regression (balanced)',gs.best_params_,format(gs.cv_results_['mean_test_accuracy'][gs.best_index_],'.4f'),format(gs.cv_results_['mean_test_f1'][gs.best_index_],'.4f'),format(accuracy_score(y_test, model.predict(x_test)),'.4f'),format(f1_score(y_test, model.predict(x_test)),'.4f')]))

model=LogisticRegression()
parameters={'penalty':['l2',None],'solver':['lbfgs'],'C':range(1,11),'random_state':[0]}
gs=GridSearchCV(model,parameters,scoring=('accuracy','f1'),refit='f1',cv=5).fit(x_res,y_res)
model=gs.best_estimator_
results=np.vstack((results,['Logistic Regression (SMOTE)',gs.best_params_,format(gs.cv_results_['mean_test_accuracy'][gs.best_index_],'.4f'),format(gs.cv_results_['mean_test_f1'][gs.best_index_],'.4f'),format(accuracy_score(y_test, model.predict(x_test)),'.4f'),format(f1_score(y_test, model.predict(x_test)),'.4f')]))


# Decision Tree no feature selection
model=DecisionTreeClassifier(random_state=0)
ccpp=model.cost_complexity_pruning_path(x_train,y_train)
parameters={'criterion':('gini','entropy','log_loss'),'ccp_alpha':ccpp.ccp_alphas,'random_state':[0]}
gs=GridSearchCV(model,parameters,scoring=('accuracy','f1'),refit='f1',cv=5).fit(x_train,y_train)
model=gs.best_estimator_
results=np.vstack((results,['Decision Tree',gs.best_params_,format(gs.cv_results_['mean_test_accuracy'][gs.best_index_],'.4f'),format(gs.cv_results_['mean_test_f1'][gs.best_index_],'.4f'),format(accuracy_score(y_test, model.predict(x_test)),'.4f'),format(f1_score(y_test, model.predict(x_test)),'.4f')]))

model=DecisionTreeClassifier(random_state=0,class_weight='balanced')
ccpp=model.cost_complexity_pruning_path(x_train,y_train)
parameters={'criterion':('gini','entropy','log_loss'),'ccp_alpha':ccpp.ccp_alphas,'random_state':[0],'class_weight':['balanced']}
gs=GridSearchCV(model,parameters,scoring=('accuracy','f1'),refit='f1',cv=5).fit(x_train,y_train)
model=gs.best_estimator_
results=np.vstack((results,['Decision Tree (balanced)',gs.best_params_,format(gs.cv_results_['mean_test_accuracy'][gs.best_index_],'.4f'),format(gs.cv_results_['mean_test_f1'][gs.best_index_],'.4f'),format(accuracy_score(y_test, model.predict(x_test)),'.4f'),format(f1_score(y_test, model.predict(x_test)),'.4f')]))

model=DecisionTreeClassifier(random_state=0)
ccpp=model.cost_complexity_pruning_path(x_res,y_res)
parameters={'criterion':('gini','entropy','log_loss'),'ccp_alpha':ccpp.ccp_alphas,'random_state':[0]}
gs=GridSearchCV(model,parameters,scoring=('accuracy','f1'),refit='f1',cv=5).fit(x_res,y_res)
model=gs.best_estimator_
results=np.vstack((results,['Decision Tree (SMOTE)',gs.best_params_,format(gs.cv_results_['mean_test_accuracy'][gs.best_index_],'.4f'),format(gs.cv_results_['mean_test_f1'][gs.best_index_],'.4f'),format(accuracy_score(y_test, model.predict(x_test)),'.4f'),format(f1_score(y_test, model.predict(x_test)),'.4f')]))



# svm no feature selection
parameters = {'kernel':['poly'],'C':range(1,11),'degree':range(2,6),'random_state':[0]}
model=SVC()
poly_gs = GridSearchCV(model, parameters,scoring=('accuracy','f1'),refit='f1',cv=5).fit(x_train,y_train)
parameters = {'kernel':('linear','sigmoid','rbf'),'C':range(1,11),'random_state':[0]}
gs=GridSearchCV(model, parameters,scoring=('accuracy','f1'),refit='f1',cv=5).fit(x_train,y_train)
if poly_gs.cv_results_['mean_test_f1'][poly_gs.best_index_]>gs.cv_results_['mean_test_f1'][gs.best_index_]:
    gs=poly_gs
model=gs.best_estimator_
results=np.vstack((results,['svm',gs.best_params_,format(gs.cv_results_['mean_test_accuracy'][gs.best_index_],'.4f'),format(gs.cv_results_['mean_test_f1'][gs.best_index_],'.4f'),format(accuracy_score(y_test, model.predict(x_test)),'.4f'),format(f1_score(y_test, model.predict(x_test)),'.4f')]))
# degree is ignored by all the other kernels except poly

parameters = {'kernel':['poly'],'C':range(1,11),'degree':range(2,6),'random_state':[0],'class_weight':['balanced']}
model=SVC()
poly_gs = GridSearchCV(model, parameters,scoring=('accuracy','f1'),refit='f1',cv=5).fit(x_train,y_train)
parameters = {'kernel':('linear','sigmoid','rbf'),'C':range(1,11),'random_state':[0],'class_weight':['balanced']}
gs=GridSearchCV(model, parameters,scoring=('accuracy','f1'),refit='f1',cv=5).fit(x_train,y_train)
if poly_gs.cv_results_['mean_test_f1'][poly_gs.best_index_]>gs.cv_results_['mean_test_f1'][gs.best_index_]:
    gs=poly_gs
model=gs.best_estimator_
results=np.vstack((results,['svm (balanced)',gs.best_params_,format(gs.cv_results_['mean_test_accuracy'][gs.best_index_],'.4f'),format(gs.cv_results_['mean_test_f1'][gs.best_index_],'.4f'),format(accuracy_score(y_test, model.predict(x_test)),'.4f'),format(f1_score(y_test, model.predict(x_test)),'.4f')]))

parameters = {'kernel':['poly'],'C':range(1,11),'degree':range(2,6),'random_state':[0]}
model=SVC()
poly_gs = GridSearchCV(model, parameters,scoring=('accuracy','f1'),refit='f1',cv=5).fit(x_res,y_res)
gs.cv_results_['mean_test_f1'][gs.best_index_]
parameters = {'kernel':('linear','sigmoid','rbf'),'C':range(1,11),'random_state':[0]}
gs = GridSearchCV(model, parameters,scoring=('accuracy','f1'),refit='f1',cv=5).fit(x_res,y_res)
if poly_gs.cv_results_['mean_test_f1'][poly_gs.best_index_]>gs.cv_results_['mean_test_f1'][gs.best_index_]:
    gs=poly_gs
model=gs.best_estimator_
results=np.vstack((results,['svm (SMOTE)',gs.best_params_,format(gs.cv_results_['mean_test_accuracy'][gs.best_index_],'.4f'),format(gs.cv_results_['mean_test_f1'][gs.best_index_],'.4f'),format(accuracy_score(y_test, model.predict(x_test)),'.4f'),format(f1_score(y_test, model.predict(x_test)),'.4f')]))


