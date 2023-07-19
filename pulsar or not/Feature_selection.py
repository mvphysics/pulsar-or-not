# Importing data
import pandas as pd
data = pd.read_csv(r'pulsar_data_train.csv')
data=data.dropna() #12528 entries reduced to 9273

# Importing useful libraries
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE
from sklearn.utils import shuffle
from sklearn.feature_selection import SelectKBest
import matplotlib.pyplot as plt

# Creating an array for the results
results=np.array(['method','best parameters','num features','features','validation accuracy','validation f1 score','test accuracy','test f1 score',])

# Scaling Data and creating train and test set
min_max_scaler = preprocessing.MinMaxScaler()
x_data=min_max_scaler.fit_transform(data.iloc[:,:8])
x_data=pd.DataFrame(x_data,columns=data.columns[0:8])
y_data=data.iloc[:,8]

x_traini,x_testi,y_train,y_test=train_test_split(x_data,y_data,test_size=0.25,shuffle=True,random_state=0,stratify=y_data)
x_res,y_res=SMOTE(random_state=0).fit_resample(x_traini,y_train)
x_res,y_res=shuffle(x_res,y_res,random_state=0)

# for SMOTE
x_traini=x_res
y_train=y_res

# Feature selection loop until only 1 variable is available
for k in range(8,0,-1):
    x_train=x_traini
    f_selection=SelectKBest(k=k).fit(x_train,y_train)
    x_train=x_traini[x_traini.columns[f_selection.get_support()]]
    x_test=x_testi[x_traini.columns[f_selection.get_support()]]
    features=x_traini.columns[f_selection.get_support()]
    features=str(features)[7:-22]
    
    model=LogisticRegression()
    parameters={'penalty':['l2',None],'solver':['lbfgs'],'C':range(1,11),'random_state':[0]}
    gs=GridSearchCV(model,parameters,scoring=('accuracy','f1'),refit='f1',cv=5,verbose=2).fit(x_train,y_train)
    model=gs.best_estimator_
    results=np.vstack((results,['Logistic Regression',gs.best_params_,k,features,float(format(gs.cv_results_['mean_test_accuracy'][gs.best_index_],'.4f')),float(format(gs.cv_results_['mean_test_f1'][gs.best_index_],'.4f')),float(format(accuracy_score(y_test, model.predict(x_test)),'.4f')),float(format(f1_score(y_test, model.predict(x_test)),'.4f'))]))
    
    model=DecisionTreeClassifier(random_state=0)
    ccpp=model.cost_complexity_pruning_path(x_train,y_train)
    parameters={'criterion':('gini','entropy','log_loss'),'ccp_alpha':ccpp.ccp_alphas,'random_state':[0]}
    gs=GridSearchCV(model,parameters,scoring=('accuracy','f1'),refit='f1',cv=5,verbose=2).fit(x_train,y_train)
    model=gs.best_estimator_
    results=np.vstack((results,['Decision Tree',gs.best_params_,k,features,float(format(gs.cv_results_['mean_test_accuracy'][gs.best_index_],'.4f')),float(format(gs.cv_results_['mean_test_f1'][gs.best_index_],'.4f')),float(format(accuracy_score(y_test, model.predict(x_test)),'.4f')),float(format(f1_score(y_test, model.predict(x_test)),'.4f'))]))
    
    parameters = {'kernel':['poly'],'C':range(1,11),'degree':range(2,6),'random_state':[0],'max_iter':[10000]}
    model=SVC()
    poly_gs = GridSearchCV(model, parameters,scoring=('accuracy','f1'),refit='f1',cv=5,verbose=2).fit(x_train,y_train)
    parameters = {'kernel':('linear','sigmoid','rbf'),'C':range(1,11),'random_state':[0]}
    gs=GridSearchCV(model, parameters,scoring=('accuracy','f1'),refit='f1',cv=5,verbose=2).fit(x_train,y_train)
    if poly_gs.cv_results_['mean_test_f1'][poly_gs.best_index_]>gs.cv_results_['mean_test_f1'][gs.best_index_]:
        gs=poly_gs
    model=gs.best_estimator_
    results=np.vstack((results,['svm',gs.best_params_,k,features,float(format(gs.cv_results_['mean_test_accuracy'][gs.best_index_],'.4f')),float(format(gs.cv_results_['mean_test_f1'][gs.best_index_],'.4f')),float(format(accuracy_score(y_test, model.predict(x_test)),'.4f')),float(format(f1_score(y_test, model.predict(x_test)),'.4f'))]))


logr_=results[np.where(results[:,0]=='Logistic Regression')]
dt_=results[np.where(results[:,0]=='Decision Tree')]
svm_=results[np.where(results[:,0]=='svm')]

# Plots

plt.plot(logr_[:,2],logr_[:,4],linestyle='dashed',color='red',label='Logistic Regression-Validation')
plt.plot(logr_[:,2],logr_[:,6],color='red',label='Logistic Regression-Test')
plt.plot(dt_[:,2],dt_[:,4],linestyle='dashed',color='green',label='Decision Tree-Validation')
plt.plot(dt_[:,2],dt_[:,6],color='green',label='Decision Tree-Test')
plt.plot(svm_[:,2],svm_[:,4],linestyle='dashed',color='blue',label='SVM-Validation')
plt.plot(svm_[:,2],svm_[:,6],color='blue',label='SVM-Test')
plt.legend(prop={'size': 7})
plt.xlabel('Number of features')
plt.ylabel('Accuracy')
plt.xlim([1,8])
plt.ylim([0.85,1])
plt.title('Accuracy after feature selection')
plt.show()

plt.plot(logr_[:,2],logr_[:,5],linestyle='dashed',color='red',label='Logistic Regression-Validation')
plt.plot(logr_[:,2],logr_[:,7],color='red',label='Logistic Regression-Test')
plt.plot(dt_[:,2],dt_[:,5],linestyle='dashed',color='green',label='Decision Tree-Validation')
plt.plot(dt_[:,2],dt_[:,7],color='green',label='Decision Tree-Test')
plt.plot(svm_[:,2],svm_[:,5],linestyle='dashed',color='blue',label='SVM-Validation')
plt.plot(svm_[:,2],svm_[:,7],color='blue',label='SVM-Test')
plt.legend(prop={'size': 7})
plt.xlabel('Number of features')
plt.ylabel('F1-score')
plt.xlim([1,8])
plt.ylim([0.55,1])
plt.title('F1-score after feature selection')
plt.show()
