import numpy as np
import pandas as pd
import csv
import os.path
import warnings
from sklearn.preprocessing import LabelEncoder
from nltk.corpus import stopwords
from nltk.stem.porter import *
from nltk.tokenize import RegexpTokenizer
from collections import namedtuple
from hpsklearn import HyperoptEstimator, svc, knn, random_forest, decision_tree, gaussian_nb, ada_boost, pca
from sklearn import svm, linear_model
from hyperopt import tpe
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.ensemble import VotingClassifier
from dbn.tensorflow import SupervisedDBNClassification
from sklearn.neural_network import MLPClassifier


warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
from gensim.models.doc2vec import Doc2Vec


def readdata(train_set_path):
    x = []
    y = []
    stop_words = set(stopwords.words('english'))
    with open(train_set_path, encoding="utf8") as infile:
        for line in infile:
            data = []
            data = line.split(",")
            stemmer = PorterStemmer()
            if data[1] != "tweet_id":
                content = re.sub(r"(?:\@|https?\://)\S+", "", data[3].lower())
                toker = RegexpTokenizer(r'((?<=[^\w\s])\w(?=[^\w\s])|(\W))+', gaps=True)
                word_tokens = toker.tokenize(content)
                filtered_sentence = [stemmer.stem(w) for w in word_tokens if not w in stop_words and w.isalpha()]
                x.append(' '.join(filtered_sentence))
                y.append(data[1])

    x, y = np.array(x), np.array(y)
    return x, y


def encode_label(label):
    le = LabelEncoder()
    label_encoded = le.fit(label).transform(label)
    print(le.classes_)
    return label_encoded


def create_model(x, y):
    docs = []
    dfs = []
    features_vectors = pd.DataFrame()
    analyzedDocument = namedtuple('AnalyzedDocument', 'words tags')
    for i, text in enumerate(x):
        words = text.lower().split()
        tags = [i]
        docs.append(analyzedDocument(words, tags))
    model = Doc2Vec(docs, size=100, window=300, min_count=1, workers=4)
    for i in range(model.docvecs.__len__()):
        dfs.append(model.docvecs[i].transpose())

    features_vectors = pd.DataFrame(dfs)
    features_vectors['label'] = y
    return features_vectors, model


def loaddata(filename,instancecol):
    file_reader = csv.reader(open(filename,'r'),delimiter=',')
    x = []
    y = []
    for row in file_reader:
        x.append(row[0:instancecol])
        y.append(row[-1])
    return np.array(x[1:]).astype(np.float32), np.array(y[1:]).astype(np.int)


def extract_features(dataset_csv, feature_csv):
    if not os.path.exists(feature_csv):
        print('Beginning Extract Features.......')
        x, y = readdata(dataset_csv)
        y = encode_label(y)
        features_vactors, model = create_model(x, y)
        features_vactors.to_csv(feature_csv, mode='a', header=False, index=False)
        print('Ending Extract Features.......')
    else:
        print('Loading Last Features.......')
        x, y = loaddata(feature_csv, 100)
        print('End Loading Last Features.......')
    return x, y


def svm_model():
    estim = svm.SVC()
    estim.fit(x_train, y_train)
    print("f1score", f1_score(estim.predict(x_test), y_test))
    print("accuracy score", accuracy_score(estim.predict(x_test), y_test))


def svm_model_tpe():
    estim = HyperoptEstimator(classifier=svc('my_clf',
                                             kernels=['linear', 'sigmoid']),
                              preprocessing=[pca('my_pca')],
                              algo=tpe.suggest,
                              max_evals=150,
                              trial_timeout=60,
                              verbose=0)
    estim.fit(x_train, y_train)
    print("score", estim.score(x_test, y_test))
    print("accuracy score", accuracy_score(estim.predict(x_test), y_test))
    print(estim.best_model())


def knn_model():
    estim = KNeighborsClassifier(n_neighbors=3)
    estim.fit(x_train, y_train)
    print("f1score", f1_score(estim.predict(x_test), y_test))
    print("accuracy score", accuracy_score(estim.predict(x_test), y_test))


def knn_model_tpe():
    estim = HyperoptEstimator(classifier=knn('my_clf'),
                              preprocessing=[pca('my_pca')],
                              algo=tpe.suggest,
                              max_evals=150,
                              trial_timeout=60,
                              verbose=0)
    estim.fit(x_train, y_train)
    print("f1score", f1_score(estim.predict(x_test), y_test))
    print("accuracy score", accuracy_score(estim.predict(x_test), y_test))
    print(estim.best_model())


def randomforest_model():
    estim = RandomForestClassifier(max_depth=2, random_state=0)
    estim.fit(x_train, y_train)
    print("f1score", f1_score(estim.predict(x_test), y_test))
    print("accuracy score", accuracy_score(estim.predict(x_test), y_test))


def randomforst_model_tpe():
    estim = HyperoptEstimator(classifier=random_forest('my_clf'),
                              preprocessing=[pca('my_pca')],
                              algo=tpe.suggest,
                              max_evals=150,
                              trial_timeout=60,
                              verbose=0)
    estim.fit(x_train, y_train)
    print("f1score", f1_score(estim.predict(x_test), y_test))
    print("accuracy score", accuracy_score(estim.predict(x_test), y_test))
    print(estim.best_model())


def decisiontree_model():
    estim = DecisionTreeClassifier(random_state=0)
    estim.fit(x_train, y_train)
    print("f1score", f1_score(estim.predict(x_test), y_test))
    print("accuracy score", accuracy_score(estim.predict(x_test), y_test))


def decisiontree_model_tpe():
    estim = HyperoptEstimator(classifier=decision_tree('my_clf', min_samples_leaf=0.2, min_samples_split=0.5),
                              preprocessing=[pca('my_pca')],
                              algo=tpe.suggest,
                              max_evals=150,
                              trial_timeout=60,
                              verbose=0)
    estim.fit(x_train, y_train)
    print("f1score", f1_score(estim.predict(x_test), y_test))
    print("accuracy score", accuracy_score(estim.predict(x_test), y_test))
    print(estim.best_model())


def gaussian_nb_model():
    estim = GaussianNB()
    estim.fit(x_train, y_train)
    print("f1score", f1_score(estim.predict(x_test), y_test))
    print("accuracy score", accuracy_score(estim.predict(x_test), y_test))


def gaussian_nb_model_tpe():
    estim = HyperoptEstimator(classifier=gaussian_nb('my_clf'),
                              preprocessing=[pca('my_pca')],
                              algo=tpe.suggest,
                              max_evals=150,
                              trial_timeout=60,
                              verbose=0)
    estim.fit(x_train, y_train)
    print("f1score", f1_score(estim.predict(x_test), y_test))
    print("accuracy score", accuracy_score(estim.predict(x_test), y_test))
    print(estim.best_model())


def gaussian_nb_model():
    estim = GaussianNB()
    estim.fit(x_train, y_train)
    print("f1score", f1_score(estim.predict(x_test), y_test))
    print("accuracy score", accuracy_score(estim.predict(x_test), y_test))


def gaussian_nb_model_tpe():
    estim = HyperoptEstimator(classifier=gaussian_nb('my_clf'),
                              preprocessing=[pca('my_pca')],
                              algo=tpe.suggest,
                              max_evals=150,
                              trial_timeout=60,
                              verbose=0)
    estim.fit(x_train, y_train)
    print("f1score", f1_score(estim.predict(x_test), y_test))
    print("accuracy score", accuracy_score(estim.predict(x_test), y_test))
    print(estim.best_model())


def dbn():
    estim = SupervisedDBNClassification(hidden_layers_structure=[256, 256, 256, 256, 256, 256 ],
                                             learning_rate_rbm=0.05,
                                             learning_rate=0.1,
                                             n_epochs_rbm=10,
                                             n_iter_backprop=100,
                                             batch_size=32,
                                             activation_function='relu',
                                             dropout_p=0.2,
                                             verbose=0)
    estim.fit(x_train, y_train)
    print("f1score", f1_score(estim.predict(x_test), y_test))
    print("accuracy score", accuracy_score(estim.predict(x_test), y_test))
    return 0


def ensemble_group1_without_tpe():
    clf1 = DecisionTreeClassifier(random_state=0)
    clf2 = GaussianNB()
    clf3 = KNeighborsClassifier(n_neighbors=3)
    clf4 = RandomForestClassifier(max_depth=2, random_state=0)
    clf5 = svm.SVC(probability=True)
    estim = VotingClassifier(estimators=[('dt', clf1), ('GNB', clf2), ('KNN', clf3), ('RF', clf4), ('svm', clf5)],
                             voting='soft', weights=[97.98, 93.11, 99.05, 99.09, 99.09])
    estim.fit(x_train, y_train)
    print("f1score", f1_score(estim.predict(x_test), y_test))
    print("accuracy score", accuracy_score(estim.predict(x_test), y_test))


def ensemble_group1():
    clf1 = DecisionTreeClassifier(class_weight=None, criterion='entropy', max_depth=None,
                                    max_features='log2', max_leaf_nodes=None,
                                    min_impurity_decrease=0.0, min_impurity_split=None,
                                    min_samples_leaf=0.2, min_samples_split=0.5,
                                    min_weight_fraction_leaf=0.0, presort=False, random_state=2,
                                    splitter='random')
    clf2 = GaussianNB(priors=None)
    clf3 = KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='euclidean',
                               metric_params=None, n_jobs=1, n_neighbors=5, p=2,
                               weights='distance')
    clf4 = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='entropy',
                                    max_depth=None, max_features=0.6933792121972574,
                                    max_leaf_nodes=None, min_impurity_decrease=0.0,
                                    min_impurity_split=None, min_samples_leaf=18,
                                    min_samples_split=2, min_weight_fraction_leaf=0.0,
                                    n_estimators=2078, n_jobs=1, oob_score=False, random_state=1,
                                    verbose=False, warm_start=False)
    clf5 = svm.SVC(C=1045.8970220658168, cache_size=512, class_weight=None, coef0=0.0,
                                  decision_function_shape='ovr', degree=1, gamma='auto', kernel='linear',
                                  max_iter=14263117.0, random_state=3, shrinking=False, probability=True,
                                  tol=5.3658140645203695e-05, verbose=False)
    estim = VotingClassifier(estimators=[('dt', clf1), ('GNB', clf2), ('KNN', clf3), ('RF', clf4), ('svm', clf5)],
                            voting='soft', weights=[99.09, 99.05, 99.05, 99.09, 99.09])
    estim.fit(x_train, y_train)
    print("f1score", f1_score(estim.predict(x_test), y_test))
    print("accuracy score", accuracy_score(estim.predict(x_test), y_test))


def ensemble_group2_without_tpe():
    clf1 = DecisionTreeClassifier(random_state=0)
    clf2 = GaussianNB()
    clf3 = KNeighborsClassifier(n_neighbors=3)
    clf4 = RandomForestClassifier(max_depth=2, random_state=0)
    clf5 = svm.SVC(probability=True)
    estim = VotingClassifier(estimators=[('dt', clf1), ('GNB', clf2), ('KNN', clf3)],
                             voting='soft', weights=[97.98, 93.11, 99.05])
    estim.fit(x_train, y_train)
    print("f1score", f1_score(estim.predict(x_test), y_test))
    print("accuracy score", accuracy_score(estim.predict(x_test), y_test))


def ensemble_group2():
    clf1 = DecisionTreeClassifier(class_weight=None, criterion='entropy', max_depth=None,
                                    max_features='log2', max_leaf_nodes=None,
                                    min_impurity_decrease=0.0, min_impurity_split=None,
                                    min_samples_leaf=0.2, min_samples_split=0.5,
                                    min_weight_fraction_leaf=0.0, presort=False, random_state=2,
                                    splitter='random')
    clf2 = GaussianNB(priors=None)
    clf3 = KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='euclidean',
                               metric_params=None, n_jobs=1, n_neighbors=5, p=2,
                               weights='distance')
    clf4 = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='entropy',
                                    max_depth=None, max_features=0.6933792121972574,
                                    max_leaf_nodes=None, min_impurity_decrease=0.0,
                                    min_impurity_split=None, min_samples_leaf=18,
                                    min_samples_split=2, min_weight_fraction_leaf=0.0,
                                    n_estimators=2078, n_jobs=1, oob_score=False, random_state=1,
                                    verbose=False, warm_start=False)
    clf5 = svm.SVC(C=1045.8970220658168, cache_size=512, class_weight=None, coef0=0.0,
                                  decision_function_shape='ovr', degree=1, gamma='auto', kernel='linear',
                                  max_iter=14263117.0, random_state=3, shrinking=False, probability=True,
                                  tol=5.3658140645203695e-05, verbose=False)
    estim = VotingClassifier(estimators=[('dt', clf1), ('GNB', clf2), ('KNN', clf3)],
                            voting='soft', weights=[99.09, 99.05, 99.05])
    estim.fit(x_train, y_train)
    print("f1score", f1_score(estim.predict(x_test), y_test))
    print("accuracy score", accuracy_score(estim.predict(x_test), y_test))


def ensemble_group3_without_tpe():
    clf1 = DecisionTreeClassifier(random_state=0)
    clf2 = GaussianNB()
    clf3 = KNeighborsClassifier(n_neighbors=3)
    clf4 = RandomForestClassifier(max_depth=2, random_state=0)
    clf5 = svm.SVC(probability=True)
    estim = VotingClassifier(estimators=[('KNN', clf3), ('RF', clf4), ('svm', clf5)],
                             voting='soft', weights=[99.05, 99.09, 99.09])
    estim.fit(x_train, y_train)
    print("f1score", f1_score(estim.predict(x_test), y_test))
    print("accuracy score", accuracy_score(estim.predict(x_test), y_test))


def ensemble_group3():
    clf1 = DecisionTreeClassifier(class_weight=None, criterion='entropy', max_depth=None,
                                    max_features='log2', max_leaf_nodes=None,
                                    min_impurity_decrease=0.0, min_impurity_split=None,
                                    min_samples_leaf=0.2, min_samples_split=0.5,
                                    min_weight_fraction_leaf=0.0, presort=False, random_state=2,
                                    splitter='random')
    clf2 = GaussianNB(priors=None)
    clf3 = KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='euclidean',
                               metric_params=None, n_jobs=1, n_neighbors=5, p=2,
                               weights='distance')
    clf4 = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='entropy',
                                    max_depth=None, max_features=0.6933792121972574,
                                    max_leaf_nodes=None, min_impurity_decrease=0.0,
                                    min_impurity_split=None, min_samples_leaf=18,
                                    min_samples_split=2, min_weight_fraction_leaf=0.0,
                                    n_estimators=2078, n_jobs=1, oob_score=False, random_state=1,
                                    verbose=False, warm_start=False)
    clf5 = svm.SVC(C=1045.8970220658168, cache_size=512, class_weight=None, coef0=0.0,
                                  decision_function_shape='ovr', degree=1, gamma='auto', kernel='linear',
                                  max_iter=14263117.0, random_state=3, shrinking=False, probability=True,
                                  tol=5.3658140645203695e-05, verbose=False)
    estim = VotingClassifier(estimators=[('KNN', clf3), ('RF', clf4), ('svm', clf5)],
                            voting='soft', weights=[99.05, 99.09, 99.09])
    estim.fit(x_train, y_train)
    print("f1score", f1_score(estim.predict(x_test), y_test))
    print("accuracy score", accuracy_score(estim.predict(x_test), y_test))


def ensemble_group4_without_tpe():
    clf1 = DecisionTreeClassifier(random_state=0)
    clf2 = GaussianNB()
    clf3 = KNeighborsClassifier(n_neighbors=3)
    clf4 = RandomForestClassifier(max_depth=2, random_state=0)
    clf5 = svm.SVC(probability=True)
    estim = VotingClassifier(estimators=[('GNB', clf2), ('RF', clf4), ('svm', clf5)],
                             voting='soft', weights=[93.11, 99.09, 99.09])
    estim.fit(x_train, y_train)
    print("f1score", f1_score(estim.predict(x_test), y_test))
    print("accuracy score", accuracy_score(estim.predict(x_test), y_test))


def ensemble_group4():
    clf1 = DecisionTreeClassifier(class_weight=None, criterion='entropy', max_depth=None,
                                    max_features='log2', max_leaf_nodes=None,
                                    min_impurity_decrease=0.0, min_impurity_split=None,
                                    min_samples_leaf=0.2, min_samples_split=0.5,
                                    min_weight_fraction_leaf=0.0, presort=False, random_state=2,
                                    splitter='random')
    clf2 = GaussianNB(priors=None)
    clf3 = KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='euclidean',
                               metric_params=None, n_jobs=1, n_neighbors=5, p=2,
                               weights='distance')
    clf4 = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='entropy',
                                    max_depth=None, max_features=0.6933792121972574,
                                    max_leaf_nodes=None, min_impurity_decrease=0.0,
                                    min_impurity_split=None, min_samples_leaf=18,
                                    min_samples_split=2, min_weight_fraction_leaf=0.0,
                                    n_estimators=2078, n_jobs=1, oob_score=False, random_state=1,
                                    verbose=False, warm_start=False)
    clf5 = svm.SVC(C=1045.8970220658168, cache_size=512, class_weight=None, coef0=0.0,
                                  decision_function_shape='ovr', degree=1, gamma='auto', kernel='linear',
                                  max_iter=14263117.0, random_state=3, shrinking=False, probability=True,
                                  tol=5.3658140645203695e-05, verbose=False)
    estim = VotingClassifier(estimators=[('GNB', clf2), ('RF', clf4), ('svm', clf5)],
                            voting='soft', weights=[99.05, 99.09, 99.09])
    estim.fit(x_train, y_train)
    print("f1score", f1_score(estim.predict(x_test), y_test))
    print("accuracy score", accuracy_score(estim.predict(x_test), y_test))


def ensemble_group5_without_tpe():
    clf1 = DecisionTreeClassifier(random_state=0)
    clf2 = GaussianNB()
    clf3 = KNeighborsClassifier(n_neighbors=3)
    clf4 = RandomForestClassifier(max_depth=2, random_state=0)
    clf5 = svm.SVC(probability=True)
    estim = VotingClassifier(estimators=[('GNB', clf2), ('KNN', clf3), ('svm', clf5)],
                             voting='soft', weights=[93.11, 99.05, 99.09])
    estim.fit(x_train, y_train)
    print("f1score", f1_score(estim.predict(x_test), y_test))
    print("accuracy score", accuracy_score(estim.predict(x_test), y_test))


def ensemble_group5():
    clf1 = DecisionTreeClassifier(class_weight=None, criterion='entropy', max_depth=None,
                                    max_features='log2', max_leaf_nodes=None,
                                    min_impurity_decrease=0.0, min_impurity_split=None,
                                    min_samples_leaf=0.2, min_samples_split=0.5,
                                    min_weight_fraction_leaf=0.0, presort=False, random_state=2,
                                    splitter='random')
    clf2 = GaussianNB(priors=None)
    clf3 = KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='euclidean',
                               metric_params=None, n_jobs=1, n_neighbors=5, p=2,
                               weights='distance')
    clf4 = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='entropy',
                                    max_depth=None, max_features=0.6933792121972574,
                                    max_leaf_nodes=None, min_impurity_decrease=0.0,
                                    min_impurity_split=None, min_samples_leaf=18,
                                    min_samples_split=2, min_weight_fraction_leaf=0.0,
                                    n_estimators=2078, n_jobs=1, oob_score=False, random_state=1,
                                    verbose=False, warm_start=False)
    clf5 = svm.SVC(C=1045.8970220658168, cache_size=512, class_weight=None, coef0=0.0,
                                  decision_function_shape='ovr', degree=1, gamma='auto', kernel='linear',
                                  max_iter=14263117.0, random_state=3, shrinking=False, probability=True,
                                  tol=5.3658140645203695e-05, verbose=False)
    estim = VotingClassifier(estimators=[('GNB', clf2), ('KNN', clf3), ('svm', clf5)],
                            voting='soft', weights=[99.05, 99.05, 99.09])
    estim.fit(x_train, y_train)
    print("f1score", f1_score(estim.predict(x_test), y_test))
    print("accuracy score", accuracy_score(estim.predict(x_test), y_test))


if __name__ == '__main__':
    x_vectors, y_vectors = extract_features('D:\\My Source Codes\\Projects-Python'
                                            '\\TextBaseEmotionDetectionWithEnsembleMethod\\Dataset\\'
                                            'text_emotion_6class.csv',
                                            'D:\\My Source Codes\\Projects-Python'
                                            '\\TextBaseEmotionDetectionWithEnsembleMethod\\Dataset\\features2cl.csv')
    test_size = int(0.2 * len(y_vectors))
    np.random.seed(13)
    indices = np.random.permutation(len(x_vectors))
    x_train = x_vectors[indices[:-test_size]]
    y_train = y_vectors[indices[:-test_size]]
    x_test = x_vectors[indices[-test_size:]]
    y_test = y_vectors[indices[-test_size:]]

    estim = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 2), random_state=1)
    estim.fit(x_train, y_train)
    print("f1score", f1_score(estim.predict(x_test), y_test))
    print("accuracy score", accuracy_score(estim.predict(x_test), y_test))

