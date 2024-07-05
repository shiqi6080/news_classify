from sklearn.naive_bayes import MultinomialNB
from collections import Counter
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import joblib


# 计算词频向量
def Text2Features(train_data_list, test_data_list, feature_words):
    def text_features(text, feature_words):
        word_counts = Counter(text)
        features = [word_counts[word] for word in feature_words]
        return features

    if train_data_list is not None:
        train_feature_list = [text_features(text, feature_words) for text in train_data_list]
        test_feature_list = [text_features(text, feature_words) for text in test_data_list]
        return train_feature_list, test_feature_list
    else:
        test_feature_list = [text_features(text, feature_words) for text in test_data_list]
        return test_feature_list


def PSbias(train_feature_list, test_feature_list, train_class_list, test_class_list):
    if train_feature_list is None:  # 如果不传入训练数据，说明在测试，就读取文件
        bias = joblib.load('models/bias.pkl')
    else:
        bias = MultinomialNB().fit(train_feature_list, train_class_list)
        joblib.dump(bias, 'models/bias.pkl')
    test_prob = bias.predict_proba(test_feature_list)
    test_accuracy = bias.score(test_feature_list, test_class_list)
    return test_prob, test_accuracy


def Logistic(train_feature_list, test_feature_list, train_class_list, test_class_list):
    if train_feature_list is None:
        logi = joblib.load('models/log.pkl')
    else:
        logi = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)
        logi.fit(train_feature_list, train_class_list)
        joblib.dump(logi, 'models/log.pkl')
    test_prob = logi.predict_proba(test_feature_list)
    test_pred = logi.predict(test_feature_list)
    test_accuracy = accuracy_score(test_class_list, test_pred)
    return test_prob, test_accuracy


def RanFor(train_feature_list, test_feature_list, train_class_list, test_class_list):
    if train_feature_list is None:
        raf = joblib.load('models/ranf.pkl')
    else:
        raf = RandomForestClassifier(n_estimators=20, random_state=42)
        raf.fit(train_feature_list, train_class_list)
        joblib.dump(raf, 'models/ranf.pkl')
    test_pred = raf.predict(test_feature_list)
    test_prob = raf.predict_proba(test_feature_list)
    test_accuracy = accuracy_score(test_class_list, test_pred)
    return test_prob, test_accuracy
