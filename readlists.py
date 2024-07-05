import pandas as pd
import jieba


def TextProcessing(filename, istrain):
    # 生成stopwords_set
    stopwords_set = set()
    with open('./stopwords.txt', 'r', encoding='utf-8') as f:  # 打开文件
        for line in f.readlines():
            word = line.strip()  # 去回车
            if len(word) > 0:  # 有文本，则添加到words_set中
                stopwords_set.add(word)
    f.close()

    # 将label改为数字
    df = pd.read_csv(filename, sep='\t', header=None, names=['Label', 'Data'])
    df.loc[df['Label'] == '体育', 'Label'] = 0
    df.loc[df['Label'] == '财经', 'Label'] = 1
    df.loc[df['Label'] == '房产', 'Label'] = 2
    df.loc[df['Label'] == '家居', 'Label'] = 3
    df.loc[df['Label'] == '教育', 'Label'] = 4
    df.loc[df['Label'] == '科技', 'Label'] = 5
    df.loc[df['Label'] == '时尚', 'Label'] = 6
    df.loc[df['Label'] == '时政', 'Label'] = 7
    df.loc[df['Label'] == '游戏', 'Label'] = 8
    df.loc[df['Label'] == '娱乐', 'Label'] = 9

    data_list = []
    label_list = []
    for j in range(len(df)):
        temp = df.Data[j]
        temp = temp.lower()
        temp = jieba.lcut(temp, cut_all=True)
        temp1 = []
        for word in temp:
            if not word.isdigit() and word not in stopwords_set and 1 < len(word) < 5:
                temp1.append(word)
        data_list.append(temp1)
        label_list.append(df.Label[j])

    if istrain:  # 如果是训练集，就统计所有词作为词典
        all_words_dict = {}  # 统计训练集词频
        for word_list in data_list:
            for word in word_list:
                if word in all_words_dict.keys():
                    all_words_dict[word] += 1
                else:
                    all_words_dict[word] = 1

        # 根据词频的值倒序排序
        all_words_sorted_list = sorted(all_words_dict.items(), key=lambda f: f[1], reverse=True)
        all_words_list, all_words_nums = zip(*all_words_sorted_list)  # 解压缩
        all_words_list = list(all_words_list)  # 转换成列表
        return all_words_list, data_list, label_list

    else:
        return data_list, label_list
