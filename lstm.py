import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import dataprocesser
import numpy as np
import matplotlib.pyplot as plt


class BiLSTMModel(nn.Module):
    def __init__(self, embedding_size, hidden_size, num_layers, num_directions, num_classes, vocab_size):
        super(BiLSTMModel, self).__init__()

        self.input_size = embedding_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_directions = num_directions

        self.embed = nn.Embedding(vocab_size + 2, embedding_size)
        self.lstm = nn.LSTM(embedding_size, hidden_size, num_layers=num_layers, bidirectional=True)
        self.attention_weights_layer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(inplace=True)
        )
        self.liner = nn.Linear(hidden_size, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, device):
        x = self.embed(x)
        x = x.permute(1, 0, 2)
        batch_size = x.size(1)
        # 设置最初的前项输出
        h_0 = torch.randn(self.num_layers * self.num_directions, batch_size, self.hidden_size).to(device)
        c_0 = torch.randn(self.num_layers * self.num_directions, batch_size, self.hidden_size).to(device)
        out, (h_n, c_n) = self.lstm(x, (h_0, c_0))
        # 拆分lstm的输出并将二者相加，形成最终输出
        (forward_out, backward_out) = torch.chunk(out, 2, dim=2)
        out = forward_out + backward_out
        out = out.permute(1, 0, 2)
        h_n = h_n.permute(1, 0, 2)
        h_n = torch.sum(h_n, dim=1)
        h_n = h_n.squeeze(dim=1)
        # 自注意力权重层
        aw = self.attention_weights_layer(h_n)
        aw = aw.unsqueeze(dim=1)
        ac = torch.bmm(aw, out.transpose(1, 2))
        softmax_w = F.softmax(ac, dim=-1)
        # 融合lstm的输出和自注意力的权重
        x = torch.bmm(softmax_w, out)
        x = x[:, -1, :]
        x = x.squeeze(dim=1)
        x = self.liner(x)
        x = self.softmax(x)
        return x


def lstm_test(all_words, test_data, test_label, model):
    processor = dataprocesser.DataProcessor()
    test_datasets = processor.get_datasets(20000, 500,
                                           all_words, test_data, test_label)
    batch_size = 100

    test_dataloader = DataLoader(test_datasets,
                                 batch_size=batch_size,
                                 shuffle=False,
                                 drop_last=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 根据运行设备选择更快的方式（本地无GPU，云端有GPU）

    if model is None:  # 如果没有模型传入，就加载模型
        model = BiLSTMModel(256, 128, 1, 2,
                            10, 20000)
        model = model.to(device)
        model.load_state_dict(torch.load('models/lstm.pth', device))

    model.eval()
    corrects = 0.0
    y_pred = []
    for datas, labels in test_dataloader:
        datas = datas.to(device)
        labels = labels.to(device)
        preds = model(datas, device)
        y_pred = y_pred + preds.detach().cpu().numpy().tolist()
        corrects += torch.sum(preds.argmax(dim=1) == labels.argmax(dim=1)).item()

    return np.stack(y_pred), corrects / len(test_datasets)


def lstm_train(all_words, train_data, train_label, test_data, test_label):
    torch.manual_seed(42)

    processor = dataprocesser.DataProcessor()
    train_datasets = processor.get_datasets(20000, 500,
                                            all_words, train_data, train_label)
    batch_size = 100
    train_dataloader = DataLoader(train_datasets,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  drop_last=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 根据运行设备选择更快的训练方式（本地无GPU，云端有GPU）

    lr = 1e-3
    EPOCH = 50
    best_loss = 100000
    filename = 'models/lstm.pth'
    model = BiLSTMModel(256, 128, 1, 2,
                        10, 20000)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_func = nn.CrossEntropyLoss()
    model.train()

    TrainLoss = []
    TrainAcc = []
    ValAcc = []
    for epoch in range(EPOCH):
        print("epoch =  ", epoch)
        all_loss = 0
        all_acc = 0
        for i, (datas, labels) in enumerate(train_dataloader):
            datas = datas.to(device)
            labels = labels.to(device)
            preds = model(datas, device)
            loss = loss_func(preds, labels)
            loss = (loss - 0.4).abs() + 0.4
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            all_loss += loss.item()  # 用于计算平均损失函数
            all_acc += torch.sum(preds.argmax(dim=1) == labels.argmax(dim=1)).item()  # 用于计算平均正确率
            if loss < best_loss:  # 如果损失函数更小，就保存
                best_loss = loss
                torch.save(model.state_dict(), filename)
        all_acc = all_acc / len(train_datasets)
        all_loss = all_loss / len(train_datasets)
        TrainAcc.append(all_acc)
        TrainLoss.append(all_loss)
        ValAcc.append(lstm_test(all_words, test_data, test_label, model)[1])
        print(all_acc, all_loss)

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
    plt.title('lstm_train')
    plt.show()

    return lstm_test(all_words, test_data, test_label, None)
