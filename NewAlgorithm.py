import numpy as np
import csv
import os
from sklearn import decomposition
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pickle as pk
from pandas import DataFrame
from random import randrange, uniform


def loaddata(filename,instancecol):
    file_reader = csv.reader(open(filename,'r'),delimiter=',')
    x = []
    y = []
    for row in file_reader:
        x.append(row[0:instancecol])
        y.append(row[-1])
    return np.array(x[1:]).astype(np.float32), np.array(y[1:]).astype(np.int)


def rf_model(modelname, x_train, y_train, x_test, y_test):
    estim = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='entropy',
                                   max_depth=2, max_features=0.9970139582088121,
                                   max_leaf_nodes=None, min_impurity_decrease=0.0,
                                   min_impurity_split=None, min_samples_leaf=1,
                                   min_samples_split=2, min_weight_fraction_leaf=0.0,
                                   n_estimators=61, n_jobs=1, oob_score=False, random_state=2,
                                   verbose=False, warm_start=False)
    pca = decomposition.PCA()
    pip = Pipeline(steps=[('pca', pca), ('RF', estim)])
    pip.fit(x_train, y_train)
    with open(modelname, 'wb') as f:
        pk.dump(pip, f)
    return f1_score(estim.predict(x_test), y_test, average='micro'), \
           accuracy_score(estim.predict(x_test), y_test), \
           1-accuracy_score(estim.predict(x_test), y_test)


def dt_model(modelname, x_train, y_train, x_test, y_test):
    estim = DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,
                                    max_features='sqrt', max_leaf_nodes=None,
                                    min_impurity_decrease=0.0, min_impurity_split=None,
                                    min_samples_leaf=0.2, min_samples_split=0.5,
                                    min_weight_fraction_leaf=0.0, presort=False, random_state=0,
                                    splitter='random')
    pca = decomposition.PCA()
    pip = Pipeline(steps=[('pca', pca), ('DT', estim)])
    pip.fit(x_train, y_train)
    with open(modelname, 'wb') as f:
        pk.dump(pip, f)
    return f1_score(estim.predict(x_test), y_test, average='micro'), \
           accuracy_score(estim.predict(x_test), y_test), \
           1-accuracy_score(estim.predict(x_test), y_test)


def mlp_model(modelname, x_train, y_train, x_test, y_test):
    estim = MLPClassifier(hidden_layer_sizes=(50,), max_iter=10, alpha=1e-4,
                            solver='sgd', verbose=False, tol=1e-4, random_state=1,
                            learning_rate_init=.1)
    pca = decomposition.PCA()
    pip = Pipeline(steps=[('pca', pca), ('DT', estim)])
    pip.fit(x_train, y_train)
    with open(modelname, 'wb') as f:
        pk.dump(pip, f)
    return f1_score(estim.predict(x_test), y_test, average='micro'), \
           accuracy_score(estim.predict(x_test), y_test), \
           1-accuracy_score(estim.predict(x_test), y_test)


if __name__ == '__main__':
    print("Begin Classificaton....")
    feature_csv = 'D:\\My Source Codes\\Projects-Python' \
                  '\\TextBaseEmotionDetectionWithEnsembleMethod\\Dataset\\features6cl.csv'
    RFmodel_save_csv = 'D:\\My Source Codes\\Projects-Python\\TextBaseEmotionDetectionWithEnsembleMethod\\Models\\RF\\'
    DTmodel_save_csv = 'D:\\My Source Codes\\Projects-Python\\TextBaseEmotionDetectionWithEnsembleMethod\\Models\\DT\\'
    MLPmodel_save_csv = 'D:\\My Source Codes\\Projects-Python\\TextBaseEmotionDetectionWithEnsembleMethod\\Models\\MLP\\'

    pd = DataFrame(columns=('ModelType', 'ModelName', 'Score', 'F1-Score', 'ErrorRate', 'Feature-Count', 'Train-Size'))
    x, y = loaddata(feature_csv, 100)
    np.random.seed(13)
    indices = np.random.permutation(1000)
    test_size = int(0.1 * len(indices))
    X_train = x[indices[:-test_size]]
    Y_train = y[indices[:-test_size]]
    X_test = x[indices[-test_size:]]
    Y_test = y[indices[-test_size:]]
    for i in range(0, 500):
        ModelName = "Model_" + str(i) + ".pkl"
        F1_Score, Score, ErrorRate = rf_model(RFmodel_save_csv + ModelName, X_train.astype('int'), Y_train.astype('int'), X_test.astype('int'), Y_test.astype('int'))
        pd.loc[len(pd)] = ["Random Forest", ModelName , Score, F1_Score, ErrorRate, 0, 0]
        print(ModelName + ", Model Type=Random Forest , With Score Result " + str(Score) + " and Feature Count="
              + str(0))

        F1_Score, Score, ErrorRate = dt_model(DTmodel_save_csv + ModelName, X_train, Y_train, X_test, Y_test)
        pd.loc[len(pd)] = ["Decision Tree", ModelName, Score, F1_Score, ErrorRate, 0, 0]
        print(ModelName + ", Model Type=Decision Tree , With Score Result " + str(Score) + " and Feature Count="
              + str(0))

        F1_Score, Score, ErrorRate = mlp_model(MLPmodel_save_csv + ModelName, X_train, Y_train, X_test, Y_test)
        pd.loc[len(pd)] = ["MLP Neural Network", ModelName, Score, F1_Score, ErrorRate, 0, 0]
        print(ModelName + ", Model Type=Neural Netork , With Score Result " + str(Score) + " and Feature Count="
              + str(0))

    pd.to_csv("D:\\My Source Codes\\Projects-Python\\TextBaseEmotionDetectionWithEnsembleMethod\\Models\dataset.csv",
              mode='a', header=True, index=False)
    print("End Classification...")