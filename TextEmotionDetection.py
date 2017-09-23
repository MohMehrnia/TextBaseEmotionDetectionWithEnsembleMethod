import numpy as np
import pandas as pd
import io
from sklearn.preprocessing import LabelEncoder
from nltk.corpus import stopwords
from nltk.stem.porter import *
from nltk.tokenize import RegexpTokenizer
from gensim.models.doc2vec import Doc2Vec
from collections import namedtuple


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


def extract_features(dataset_csv, fearture_csv):
    print('Beginning Extract Features.......')
    x, y = readdata(dataset_csv)
    y = encode_label(y)
    features_vactors, model = create_model(x, y)
    features_vactors.to_csv(fearture_csv, mode='a', header=False, index=False)
    print('Ending Extract Features.......')
    return features_vactors


if __name__ == '__main__':
    features = extract_features('D:\\My Source Codes\\Projects-Python'
                    '\\TextBaseEmotionDetectionWithEnsembleMethod\\Dataset\\text_emotion.csv',
                     'D:\\My Source Codes\\Projects-Python' \
                     '\\TextBaseEmotionDetectionWithEnsembleMethod\\Dataset\\features.csv')