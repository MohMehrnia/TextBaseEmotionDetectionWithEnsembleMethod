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
        x, y = loaddata(feature_csv,100)
        print('End Loading Last Features.......')
    return x, y


def svm_model(x_tra, y_tra, x_tes, y_tes):
    estim = svm.SVC()
    estim.fit(x_train, y_train)
    print("score", estim.score(x_test, y_test))
    print("f1score", f1_score(estim.predict(x_test), y_test))
    print("accuracy score", accuracy_score(estim.predict(x_test), y_test))


def svm_model_tpe(x_tra, y_tra, x_tes, y_tes):
    estim = HyperoptEstimator(classifier=svc('my_clf',
                                             kernels=['linear', 'sigmoid']),
                              preprocessing=[pca('my_pca')],
                              algo=tpe.suggest,
                              max_evals=150,
                              trial_timeout=60,
                              verbose=0)
    estim.fit(x_train, y_train)
    print("score", estim.score(x_test, y_test))
    print("f1score", f1_score(estim.predict(x_test), y_test))
    print("accuracy score", accuracy_score(estim.predict(x_test), y_test))
    print(estim.best_model())


def knn_model(x_tra, y_tra, x_tes, y_tes):
    estim = KNeighborsClassifier(n_neighbors=3)
    estim.fit(x_train, y_train)
    print("score", estim.score(x_test, y_test))
    print("f1score", f1_score(estim.predict(x_test), y_test))
    print("accuracy score", accuracy_score(estim.predict(x_test), y_test))


def knn_model_tpe(x_tra, y_tra, x_tes, y_tes):
    estim = HyperoptEstimator(classifier=knn('my_clf'),
                              preprocessing=[pca('my_pca')],
                              algo=tpe.suggest,
                              max_evals=150,
                              trial_timeout=60,
                              verbose=0)
    estim.fit(x_train, y_train)
    print("score", estim.score(x_test, y_test))
    print("f1score", f1_score(estim.predict(x_test), y_test))
    print("accuracy score", accuracy_score(estim.predict(x_test), y_test))
    print(estim.best_model())


def randomforest_model(x_tra, y_tra, x_tes, y_tes):
    estim = RandomForestClassifier(max_depth=2, random_state=0)
    estim.fit(x_train, y_train)
    print("score", estim.score(x_test, y_test))
    print("f1score", f1_score(estim.predict(x_test), y_test))
    print("accuracy score", accuracy_score(estim.predict(x_test), y_test))


def randomforst_model_tpe(x_tra, y_tra, x_tes, y_tes):
    estim = HyperoptEstimator(classifier=random_forest('my_clf'),
                              preprocessing=[pca('my_pca')],
                              algo=tpe.suggest,
                              max_evals=150,
                              trial_timeout=60,
                              verbose=0)
    estim.fit(x_train, y_train)
    print("score", estim.score(x_test, y_test))
    print("f1score", f1_score(estim.predict(x_test), y_test))
    print("accuracy score", accuracy_score(estim.predict(x_test), y_test))
    print(estim.best_model())


def decisiontree_model(x_tra, y_tra, x_tes, y_tes):
    estim = DecisionTreeClassifier(random_state=0)
    estim.fit(x_train, y_train)
    print("score", estim.score(x_test, y_test))
    print("f1score", f1_score(estim.predict(x_test), y_test))
    print("accuracy score", accuracy_score(estim.predict(x_test), y_test))


def decisiontree_model_tpe(x_tra, y_tra, x_tes, y_tes):
    estim = HyperoptEstimator(classifier=decision_tree('my_clf', min_samples_leaf=0.2, min_samples_split=0.5),
                              preprocessing=[pca('my_pca')],
                              algo=tpe.suggest,
                              max_evals=150,
                              trial_timeout=60,
                              verbose=0)
    estim.fit(x_train, y_train)
    print("score", estim.score(x_test, y_test))
    print("f1score", f1_score(estim.predict(x_test), y_test))
    print("accuracy score", accuracy_score(estim.predict(x_test), y_test))
    print(estim.best_model())


def gaussian_nb_model(x_tra, y_tra, x_tes, y_tes):
    estim = GaussianNB()
    estim.fit(x_train, y_train)
    print("score", estim.score(x_test, y_test))
    print("f1score", f1_score(estim.predict(x_test), y_test))
    print("accuracy score", accuracy_score(estim.predict(x_test), y_test))


def gaussian_nb_model_tpe(x_tra, y_tra, x_tes, y_tes):
    estim = HyperoptEstimator(classifier=gaussian_nb('my_clf'),
                              preprocessing=[pca('my_pca')],
                              algo=tpe.suggest,
                              max_evals=150,
                              trial_timeout=60,
                              verbose=0)
    estim.fit(x_train, y_train)
    print("score", estim.score(x_test, y_test))
    print("f1score", f1_score(estim.predict(x_test), y_test))
    print("accuracy score", accuracy_score(estim.predict(x_test), y_test))
    print(estim.best_model())


if __name__ == '__main__':
    x_vectors, y_vectors = extract_features('D:\\My Source Codes\\Projects-Python'
                                            '\\TextBaseEmotionDetectionWithEnsembleMethod\\Dataset\\'
                                            'text_emotion_2class.csv',
                                            'D:\\My Source Codes\\Projects-Python'
                                            '\\TextBaseEmotionDetectionWithEnsembleMethod\\Dataset\\features.csv')
    test_size = int(0.2 * len(y_vectors))
    np.random.seed(13)
    indices = np.random.permutation(len(x_vectors))
    x_train = x_vectors[indices[:-test_size]]
    y_train = y_vectors[indices[:-test_size]]
    x_test = x_vectors[indices[-test_size:]]
    y_test = y_vectors[indices[-test_size:]]

    print('**********RBM*************')
    svm_model(x_train, y_train, x_test, y_test)
    print('******RBM TPE*************')
    svm_model_tpe(x_train, y_train, x_test, y_test)
