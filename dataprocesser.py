import pandas as pd
import torch
import torch.utils.data

class DataProcessor(object):
    def con2pd(self, datalist, labellist):
        df = pd.DataFrame({"cutword": datalist, "label": labellist})
        return df["cutword"], df["label"]

    def word_index(self, vocab_size, all_word_list):
        word2index = {}
        # 词表中未出现的词
        word2index["<unk>"] = 0
        # 句子添加的padding
        word2index["<pad>"] = 1
        # 词表的实际大小由词的数量和限定大小决定
        vocab_size = min(len(all_word_list), vocab_size)
        for i in range(vocab_size):
            word = all_word_list[i]
            word2index[word] = i + 2
        return word2index, vocab_size

    def get_datasets(self, vocab_size, max_len, all_words, data_list, label_list):
        datas, labels = self.con2pd(data_list, label_list)
        word2index, vocab_size = self.word_index(vocab_size, all_words)
        features = []
        for data_list in datas:
            feature = []
            for word in data_list:
                if word in word2index:
                    feature.append(word2index[word])
                else:
                    feature.append(word2index["<unk>"])  # 词表中未出现的词用<unk>代替
                if (len(feature) == max_len):  # 限制句子的最大长度，超出部分直接截断
                    break
            # 对未达到最大长度的句子添加padding
            feature = feature + [word2index["<pad>"]] * (max_len - len(feature))
            features.append(feature)

        pre = []
        for i in labels:
            temp = [0]*10
            temp[i] = 1
            pre.append(temp)

        features = torch.LongTensor(features)
        labels = torch.Tensor(pre)
        datasets = torch.utils.data.TensorDataset(features, labels)
        return datasets