import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


class BPNetModel(torch.nn.Module):
    def __init__(self, n_feature, n_output):
        super(BPNetModel, self).__init__()
        self.hiddden1 = torch.nn.Linear(n_feature, 50)  # 定义隐层网络
        self.out = torch.nn.Linear(50, n_output)  # 定义输出层网络

    def forward(self, x):
        x = F.relu(self.hiddden1(x))  # 隐层激活函数采用relu()函数
        out = F.softmax(self.out(x), dim=1)  # 输出层采用softmax函数
        return out


def bpnn_test(model, test_x, test_y):
    if model is None:
        model = BPNetModel(4950, 10)
        model.load_state_dict(torch.load('models/bpnn.pth'))
    test_x_tensor = torch.Tensor(test_x)
    output = model(test_x_tensor)
    output = np.stack(output.tolist(), axis=0)
    result = np.zeros(len(test_y))
    for i in range(len(test_y)):
        result[i] = output[i].argmax()
    test_accuracy = accuracy_score(test_y, result)
    return output, test_accuracy


def bpnn_train(train_x, test_x, train_y, test_y):
    train_n = len(train_x)
    train_x_tensor = torch.Tensor(train_x)
    train_y_np = np.zeros([train_n, 10])
    for i in range(train_n):
        train_y_np[i][train_y[i]] = 1

    model = BPNetModel(4950, 10)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    prev_loss = 1000
    EPOCH = 50

    TrainLoss = []
    TrainAcc = []
    ValAcc = []

    for epoch in range(EPOCH):
        print("epoch = ", epoch)
        output = model(train_x_tensor)
        temp = torch.from_numpy(train_y_np)
        temp = temp.float()
        loss = criterion(output, temp)
        output = np.stack(output.tolist(), axis=0)
        result = np.zeros(len(train_y))
        for i in range(len(train_y)):
            result[i] = output[i].argmax()
        test_accuracy = accuracy_score(train_y, result)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(loss.tolist(), test_accuracy)
        TrainLoss.append(loss.tolist())
        TrainAcc.append(test_accuracy)
        ValAcc.append(bpnn_test(model, test_x, test_y)[1])
        if loss < prev_loss:  # 如果损失函数更小就保存
            torch.save(model.state_dict(), 'models/bpnn.pth')
            prev_loss = loss

    # 训练数据可视化
    x = list(range(1, 51))
    fig, ax1 = plt.subplots()
    line1 = ax1.plot(x, TrainLoss, color='b', label='TrainLoss')
    ax1.set_ylabel('Loss')
    ax2 = ax1.twinx()
    line2 = ax2.plot(x, TrainAcc, color='r', label='TrainAcc')
    line3 = ax2.plot(x, ValAcc, color='g', label='ValAcc')
    ax2.set_ylabel('Acc')
    lines = line1 + line2 + line3
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left')
    plt.title('bpnn_train')
    plt.show()

    return bpnn_test(None, test_x, test_y)
