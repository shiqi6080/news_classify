import readlists
import pickle
import lstm
import classifier
import bpnn
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

acc_list = []

# 读入测试集（测试集新闻数量建议在一万条及以下，以防内存溢出）
test_data_list, test_label_list = readlists.TextProcessing('./cnews.test.txt', 0)

# 读入训练好的词典
with open('models/all_words.pkl', 'rb') as f:
    all_words_list = pickle.load(f)
f.close()

pre_result0, acc = lstm.lstm_test(all_words_list, test_data_list, test_label_list, None)
acc_list.append(acc)
print("LSTM acc:", acc)

feature_words = all_words_list[50:5000]  # 特征列表
test_feature_list = classifier.Text2Features(None, test_data_list, feature_words)

pre_result1, acc = classifier.PSbias(None, test_feature_list, None, test_label_list)
acc_list.append(acc)
print("bias acc:", acc)

pre_result2, acc = classifier.Logistic(None, test_feature_list, None, test_label_list)
acc_list.append(acc)
print("logi acc:", acc)

pre_result3, acc = classifier.RanFor(None, test_feature_list, None, test_label_list)
acc_list.append(acc)
print("ranf acc:", acc)

pre_result4, acc = bpnn.bpnn_test(None, test_feature_list, test_label_list)
acc_list.append(acc)
print("bpnn acc:", acc)

y_probs_all = np.stack([pre_result0, pre_result1, pre_result2, pre_result3, pre_result4], axis=0)
y_probs_all = np.mean(y_probs_all, axis=0)
y_pred = np.argmax(y_probs_all, axis=1)
acc = accuracy_score(test_label_list, y_pred)
acc_list.append(acc)

# 分类结果可视化
classheat = [[0] * 10 for _ in range(10)]
for i in range(len(test_label_list)):
    classheat[test_label_list[i]][y_pred[i]] += 1

plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文
x_ticks = ['体育', '财经', '房产', '家居', '教育', '科技', '时尚', '时政', '游戏', '娱乐']
y_ticks = ['体育', '财经', '房产', '家居', '教育', '科技', '时尚', '时政', '游戏', '娱乐']
ax = sns.heatmap(classheat, xticklabels=x_ticks, yticklabels=y_ticks, annot=True, fmt="d", cmap="YlGnBu", linewidths=.5)
ax.set_title('测试集分类结果')  # 图标题
ax.set_xlabel('分类结果')  # x轴标题
ax.set_ylabel('新闻类别')
plt.show()
figure = ax.get_figure()

print("五种模型的正确率和最终正确率分别为：", acc_list)
