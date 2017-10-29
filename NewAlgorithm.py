import numpy as np
import csv
import pandas as pd
from sklearn import decomposition
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from keras.models import Sequential
from keras.layers import Dense, Dropout
import pickle as pk
from pandas import DataFrame
from random import sample
from nltk.corpus import stopwords
from nltk.stem.porter import *
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from collections import namedtuple
from gensim.models.doc2vec import Doc2Vec
from sklearn.metrics import classification_report


def loaddata(filename,instancecol):
    file_reader = csv.reader(open(filename,'r'),delimiter=',')
    x = []
    y = []
    for row in file_reader:
        x.append(row[0:instancecol])
        y.append(row[-1])
    return np.array(x[1:]).astype(np.float32), np.array(y[1:]).astype(np.int)


def readdata(train_set_path, y_value):
    x = []
    y = []
    stop_words = set(stopwords.words('english'))
    with open(train_set_path, encoding="utf8") as infile:
        for line in infile:
            stemmer = PorterStemmer()
            lemmatizer = WordNetLemmatizer()
            content = re.sub(r"(?:\@|https?\://)\S+", "", line)
            toker = RegexpTokenizer(r'((?<=[^\w\s])\w(?=[^\w\s])|(\W))+', gaps=True)
            word_tokens = toker.tokenize(content)
            filtered_sentence = [lemmatizer.lemmatize(w) for w in word_tokens if not w in stop_words and w.isalpha()]
            x.append(' '.join(filtered_sentence))
            y.append(y_value)

    x, y = np.array(x), np.array(y)
    return x, y


def create_docmodel(x, y, feature_count):
    docs = []
    dfs = []
    features_vectors = pd.DataFrame()
    analyzedDocument = namedtuple('AnalyzedDocument', 'words tags')
    for i, text in enumerate(x):
        words = text.lower().split()
        tags = [i]
        docs.append(analyzedDocument(words, tags))
    model = Doc2Vec(docs, size=feature_count, window=300, min_count=1, workers=4)
    for i in range(model.docvecs.__len__()):
        dfs.append(model.docvecs[i].transpose())

    features_vectors = pd.DataFrame(dfs)
    features_vectors['label'] = y
    return features_vectors, model


def db_model(modelname, x_train, y_train, x_test, y_test):
    model = Sequential()
    model.add(Dense(128, input_dim=100, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=10, batch_size=1, verbose=0)
    score = model.evaluate(x_test, y_test, batch_size=16)
    y_pred = model.predict_classes(x_test, batch_size=1)
    model.save(modelname)
    return f1_score(y_pred, y_test, average='micro'), \
           accuracy_score(y_pred, y_test), \
           1 - accuracy_score(y_pred, y_test)


def rf_model(modelname, x_train, y_train, x_test, y_test):
    estim = KNeighborsClassifier(n_neighbors=3)
    pip = Pipeline(steps=[('RF', estim)])
    pip.fit(x_train, y_train)
    with open(modelname, 'wb') as f:
        pk.dump(pip, f)
    return f1_score(estim.predict(x_test), y_test, average='micro'), \
           accuracy_score(estim.predict(x_test), y_test), \
           1-accuracy_score(estim.predict(x_test), y_test)


def dt_model(modelname, x_train, y_train, x_test, y_test):
    estim = RandomForestClassifier(n_estimators=50, max_depth=16, random_state=42)
    pip = Pipeline(steps=[('RF', estim)])
    pip.fit(x_train, y_train)
    with open(modelname, 'wb') as f:
        pk.dump(pip, f)
    return f1_score(estim.predict(x_test), y_test, average='micro'), \
           accuracy_score(estim.predict(x_test), y_test), \
           1-accuracy_score(estim.predict(x_test), y_test)


def mlp_model(modelname, x_train, y_train, x_test, y_test):
    estim = MLPClassifier(hidden_layer_sizes=(100, 50), random_state=42)
    pip = Pipeline(steps=[('SVM', estim)])
    pip.fit(x_train, y_train)
    with open(modelname, 'wb') as f:
        pk.dump(pip, f)
    return f1_score(estim.predict(x_test), y_test, average='micro'), \
           accuracy_score(estim.predict(x_test), y_test), \
           1-accuracy_score(estim.predict(x_test), y_test)


def create_model():
    print("Begin Classificaton....")
    feature_csv = 'D:\\My Source Codes\\Projects-Python' \
                  '\\TextBaseEmotionDetectionWithEnsembleMethod\\NewDataset\\features6cl.csv'
    RFmodel_save_csv = 'D:\\My Source Codes\\Projects-Python\\TextBaseEmotionDetectionWithEnsembleMethod\\Models\\RF\\'
    DTmodel_save_csv = 'D:\\My Source Codes\\Projects-Python\\TextBaseEmotionDetectionWithEnsembleMethod\\Models\\DT\\'
    MLPmodel_save_csv = 'D:\\My Source Codes\\Projects-Python\\TextBaseEmotionDetectionWithEnsembleMethod\\' \
                        'Models\\MLP\\'
    pd = DataFrame(columns=('ModelType', 'ModelName', 'Score', 'F1-Score', 'ErrorRate', 'Feature-Count', 'Train-Size'))
    x, y = loaddata(feature_csv, 100)
    for i in range(1, 500):
        np.random.seed(42)
        indices = sample(range(1, x.shape[0]), 5990)
        test_size = int(0.1 * len(indices))
        X_train = x[indices[:-test_size]]
        Y_train = y[indices[:-test_size]]
        X_test = x[indices[-test_size:]]
        Y_test = y[indices[-test_size:]]

        ModelName = "Model_KNN_" + str(i) + ".pkl"
        F1_Score, Score, ErrorRate = rf_model(RFmodel_save_csv + ModelName, X_train, Y_train
                                              , X_test, Y_test)
        pd.loc[len(pd)] = ["KNN ", ModelName , Score, F1_Score, ErrorRate, 0, 0]
        print(ModelName + ", Model Type=KNN , With Score Result " + str(Score) + " and Feature Count="
              + str(100))

        ModelName = "Model_RF_" + str(i) + ".pkl"
        F1_Score, Score, ErrorRate = dt_model(DTmodel_save_csv + ModelName, X_train, Y_train, X_test, Y_test)
        pd.loc[len(pd)] = ["Random Forest", ModelName, Score, F1_Score, ErrorRate, 0, 0]
        print(ModelName + ", Model Type=Random Forest , With Score Result " + str(Score) + " and Feature Count="
              + str(100))

        ModelName = "Model_MLP_" + str(i) + ".pkl"
        F1_Score, Score, ErrorRate = mlp_model(MLPmodel_save_csv + ModelName, X_train, Y_train, X_test, Y_test)
        pd.loc[len(pd)] = ["MLP Neural Network", ModelName, Score, F1_Score, ErrorRate, 0, 0]
        print(ModelName + ", Model Type=Neural Network , With Score Result " + str(Score) + " and Feature Count="
              + str(100))

    pd.to_csv("D:\\My Source Codes\\Projects-Python\\TextBaseEmotionDetectionWithEnsembleMethod\\Models\dataset.csv",
              mode='a', header=True, index=False)
    print("End Classification...")


def classification_methods():
    feature_csv = 'D:\\My Source Codes\\Projects-Python' \
                  '\\TextBaseEmotionDetectionWithEnsembleMethod\\NewDataset\\features6cl.csv'
    RFmodel_save_csv = 'D:\\My Source Codes\\Projects-Python\\TextBaseEmotionDetectionWithEnsembleMethod\\Models\\RF\\'
    DTmodel_save_csv = 'D:\\My Source Codes\\Projects-Python\\TextBaseEmotionDetectionWithEnsembleMethod\\Models\\DT\\'
    MLPmodel_save_csv = 'D:\\My Source Codes\\Projects-Python\\TextBaseEmotionDetectionWithEnsembleMethod\\' \
                        'Models\\MLP\\'
    x, y = loaddata(feature_csv, 100)
    Y_TEST = []
    Y_PRED = []
    for index in range(1, 2):
        np.random.seed(42)
        indices = sample(range(1, x.shape[0]), 1)
        test_size = int(1 * len(indices))
        X_test = x[indices[-test_size:]]
        Y_test = y[indices[-test_size:]]
        for i in range(0, 499):
            ModelName = RFmodel_save_csv + "Model_KNN_" + str(i) + ".pkl"
            with open(ModelName, 'rb') as f:
                model = pk.load(f)
                Y_TEST.append(np.asarray(Y_test))
                Y_PRED.append(np.asarray(model.predict(X_test)))
                print("KNN Model " + str(i) + ": " + str(Y_test) + "==>" + str(model.predict(X_test)))

            ModelName = DTmodel_save_csv + "Model_DT_" + str(i) + ".pkl"
            with open(ModelName, 'rb') as f:
                model = pk.load(f)
                Y_TEST.append(np.asarray(Y_test))
                Y_PRED.append(np.asarray(model.predict(X_test)))
                print("Decision Tree Model " + str(i) + ": " + str(Y_test) + "==>" + str(model.predict(X_test)))

            ModelName = MLPmodel_save_csv + "Model_MLP_" + str(i) + ".pkl"
            with open(ModelName, 'rb') as f:
                model = pk.load(f)
                Y_TEST.append(np.asarray(Y_test))
                Y_PRED.append(np.asarray(model.predict(X_test)))
                print("Neural Network Model " + str(i) + ": " + str(Y_test) + "==>" + str(model.predict(X_test)))

        print(accuracy_score(np.asarray(Y_PRED), np.asarray(Y_TEST)))


def feature_extraction():
    dataset_csv = "D:\\My Source Codes\\Projects-Python\\TextBaseEmotionDetectionWithEnsembleMethod" \
                  "\\NewDataset\\SURPRISE_Phrases_6.txt"
    feature_csv = "D:\\My Source Codes\\Projects-Python\\TextBaseEmotionDetectionWithEnsembleMethod" \
                  "\\NewDataset\\features6cl.csv"
    instancecol = 100
    x, y = readdata(dataset_csv, 6)
    features_vactors, model = create_docmodel(x, y, instancecol)
    features_vactors = features_vactors[1:1000]
    features_vactors.to_csv(feature_csv, mode='a', header=False, index=False)


def run_model():
    feature_csv = "D:\\My Source Codes\\Projects-Python\\TextBaseEmotionDetectionWithEnsembleMethod" \
                  "\\NewDataset\\features6cl.csv"
    x, y = loaddata(feature_csv, 100)
    np.random.seed(42)
    indices = sample(range(1, x.shape[0]), 5990)
    test_size = int(0.1 * len(indices))
    X_train = x[indices[:-test_size]]
    Y_train = y[indices[:-test_size]]

    X_test = x[indices[-test_size:]]
    Y_test = y[indices[-test_size:]]

    print(mlp_model("Model",X_train, Y_train, X_test, Y_test))


if __name__ == '__main__':
    create_model()
    print(0)
