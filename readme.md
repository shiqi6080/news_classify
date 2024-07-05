本课程设计将利用课内外学习的不同方法，对属于10个类别（体育, 财经, 房产, 家居, 教育, 科技, 时尚, 时政, 游戏, 娱乐）的新闻进行分类，训练集共50000条新闻数据，验证集5000条，测试集10000条。
./codes/main.py	训练程序，包含训练集训练、验证集测试
./codes/test.py	测试程序，包含测试集测试
./codes/lstm.py	基于LSTM神经网络的分类模型
./codes/dataprocesser.py	LSTM模型所需的文本编码操作
./codes/classifier.py	词频特征提取、朴素贝叶斯模型、逻辑回归模型、随机森林模型
./codes/bpnn.py	BP神经网络模型
./codes/readlists.py	数据预处理，包括读取数据、分词、去停用词
./codes/stopwords.txt	程序需要读取的停用词表
./codes/models/lstm.pth	训练好的LSTM模型文件
./codes/models/all_words.pkl	LSTM模型文本编码时所需词典，与LSTM模型文件对应
./codes/models/bias.pkl	训练好的朴素贝叶斯模型文件
./codes/models/log.pkl	训练好的逻辑回归模型文件
./codes/models/ranf.pkl	训练好的随机森林模型文件
./codes/models/bpnn.pth	训练好的BP神经网络模型文件
