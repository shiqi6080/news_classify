import numpy as np
from sklearn.metrics import accuracy_score
import classifier
import lstm
import bpnn
import readlists
import matplotlib.pyplot as plt
import seaborn as sns

acc_list = []

# 文本预处理
all_words_list, train_data_list, train_label_list = readlists.TextProcessing('./cnews.train.txt', 1)
val_data_list, val_label_list = readlists.TextProcessing('./cnews.val.txt', 0)

#  训练LSTM
pre_result0, acc = lstm.lstm_train(all_words_list, train_data_list, train_label_list, val_data_list, val_label_list)
acc_list.append(acc)
print("LSTM Finish")

# 获取特征词
feature_words = all_words_list[50:5000]  # 特征列表
train_feature_list, val_feature_list = classifier.Text2Features(train_data_list, val_data_list, feature_words)
# 训练贝叶斯
pre_result1, acc = classifier.PSbias(train_feature_list, val_feature_list, train_label_list, val_label_list)
acc_list.append(acc)
print('bias Finish')
# 训练逻辑回归
pre_result2, acc = classifier.Logistic(train_feature_list, val_feature_list, train_label_list, val_label_list)
acc_list.append(acc)
print('logi Finish')
# 训练随机森林
pre_result3, acc = classifier.RanFor(train_feature_list, val_feature_list, train_label_list, val_label_list)
acc_list.append(acc)
print('ranf Finish')
# 训练bp神经网络
pre_result4, acc = bpnn.bpnn_train(train_feature_list, val_feature_list, train_label_list, val_label_list)
acc_list.append(acc)
print('bp Finish')

# 软投票环节
y_probs_all = np.stack([pre_result0, pre_result1, pre_result2, pre_result3, pre_result4], axis=0)
y_probs_all = np.mean(y_probs_all, axis=0)
y_pred = np.argmax(y_probs_all, axis=1)
acc = accuracy_score(val_label_list, y_pred)
acc_list.append(acc)

# 验证集分类结果可视化
classheat = [[0] * 10 for _ in range(10)]
for i in range(len(val_label_list)):
    classheat[val_label_list[i]][y_pred[i]] += 1

plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文
x_ticks = ['体育', '财经', '房产', '家居', '教育', '科技', '时尚', '时政', '游戏', '娱乐']
y_ticks = ['体育', '财经', '房产', '家居', '教育', '科技', '时尚', '时政', '游戏', '娱乐']
ax = sns.heatmap(classheat, xticklabels=x_ticks, yticklabels=y_ticks, annot=True, fmt="d", cmap="YlGnBu", linewidths=.5)
ax.set_title('验证集分类结果')  # 图标题
ax.set_xlabel('分类结果')  # x轴标题
ax.set_ylabel('新闻类别')
plt.show()
figure = ax.get_figure()

print("验证集：各模型正确率和最终正确率", acc_list)
