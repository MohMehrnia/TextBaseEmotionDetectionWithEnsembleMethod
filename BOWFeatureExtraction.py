import numpy as np
from sklearn.preprocessing import LabelEncoder
from nltk.corpus import stopwords
from nltk.stem.porter import *
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd

def encode_label(label):
    le = LabelEncoder()
    label_encoded = le.fit(label).transform(label)
    return label_encoded


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


if __name__ == '__main__':
    print("Begin Extract Features ....")
    dataset_csv = 'D:\\My Source Codes\\Projects-Python' \
                                            '\\TextBaseEmotionDetectionWithEnsembleMethod\\Dataset\\' \
                                            'text_emotion_6class.csv'
    feature_csv = 'D:\\My Source Codes\\Projects-Python' \
                                            '\\TextBaseEmotionDetectionWithEnsembleMethod\\Dataset\\' \
                                            'BOWfeature6cl.csv'
    x, y = readdata(dataset_csv)
    y = encode_label(y)
    features_vectors = pd.DataFrame()

    vectorizer = CountVectorizer()
    vectorizer.fit(x)
    x_bag_of_word = vectorizer.transform(x)
    features_vectors = pd.DataFrame(x_bag_of_word.toarray())
    features_vectors['label'] = y
    features_vectors.to_csv(feature_csv, mode='a', header=False, index=False)