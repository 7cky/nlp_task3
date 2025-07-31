import torch
import torch.nn as nn


class Input_Encoding(nn.Module):
  #将词 ID 序列转换为词向量，并通过双向 LSTM 提取上下文特征
    def __init__(self, len_feature, len_hidden, len_words,longest, weight=None, layer=1, batch_first=True, drop_out=0.5):
        super(Input_Encoding, self).__init__()
        self.len_feature = len_feature
        self.len_hidden = len_hidden
        self.len_words = len_words
        self.layer = layer
        self.longest=longest
        self.dropout = nn.Dropout(drop_out)
        #嵌入层
        if weight is None:
            x = nn.init.xavier_normal_(torch.Tensor(len_words, len_feature))
            self.embedding = nn.Embedding(num_embeddings=len_words, embedding_dim=len_feature, _weight=x).cuda()
        else:
            self.embedding = nn.Embedding(num_embeddings=len_words, embedding_dim=len_feature, _weight=weight).cuda()
        #双向LSTM
        self.lstm = nn.LSTM(input_size=len_feature, hidden_size=len_hidden, num_layers=layer, batch_first=batch_first,
                            bidirectional=True).cuda()

    def forward(self, x):
        x = torch.LongTensor(x).cuda()
        x = self.embedding(x)
        x = self.dropout(x)
        self.lstm.flatten_parameters()
        x, _ = self.lstm(x) #上下文编码
        return x


class Local_Inference_Modeling(nn.Module):
  #局部推理建模：通过注意力机制计算句子 A 和句子 B 之间的局部匹配关系（即句子 A 中的每个词与句子 B 中的每个词的关联
    def __init__(self):
        super(Local_Inference_Modeling, self).__init__()
        self.softmax_1 = nn.Softmax(dim=1).cuda()
        self.softmax_2 = nn.Softmax(dim=2).cuda()

    def forward(self, a_bar, b_bar):
      # 计算注意力权重矩阵 e
        e = torch.matmul(a_bar, b_bar.transpose(1, 2)).cuda()

        # 生成对齐向量
        a_tilde = self.softmax_2(e)
        a_tilde = a_tilde.bmm(b_bar)
        b_tilde = self.softmax_1(e)
        b_tilde = b_tilde.transpose(1, 2).bmm(a_bar)

        # 增强特征（原始+对齐+差异+乘积
        m_a = torch.cat([a_bar, a_tilde, a_bar - a_tilde, a_bar * a_tilde], dim=-1)
        m_b = torch.cat([b_bar, b_tilde, b_bar - b_tilde, b_bar * b_tilde], dim=-1)

        return m_a, m_b


class Inference_Composition(nn.Module):
  #推理合成，对局部推理特征（m_a 和 m_b）进行进一步编码，捕捉全局推理信息
    def __init__(self, len_feature, len_hidden_m, len_hidden, layer=1, batch_first=True, drop_out=0.5):
        super(Inference_Composition, self).__init__()
        self.linear = nn.Linear(len_hidden_m, len_feature).cuda() #调整维度
        self.lstm = nn.LSTM(input_size=len_feature, hidden_size=len_hidden, num_layers=layer, batch_first=batch_first,
                            bidirectional=True).cuda()
        self.dropout = nn.Dropout(drop_out).cuda()

    def forward(self, x):
        x = self.linear(x)
        x = self.dropout(x)
        self.lstm.flatten_parameters()
        x, _ = self.lstm(x)

        return x


class Prediction(nn.Module):
  #将编码后的特征映射到最终的逻辑关系类别
    def __init__(self, len_v, len_mid, type_num=4, drop_out=0.5):
        super(Prediction, self).__init__()
        self.mlp = nn.Sequential(nn.Dropout(drop_out), nn.Linear(len_v, len_mid), nn.Tanh(),
                                 nn.Linear(len_mid, type_num)).cuda() #多层感知机分类

    def forward(self, a,b):
      #池化

        v_a_avg=a.sum(1)/a.shape[1]
        v_a_max = a.max(1)[0]

        v_b_avg = b.sum(1) / b.shape[1]
        v_b_max = b.max(1)[0]
        #拼接特征和预测

        out_put = torch.cat((v_a_avg, v_a_max,v_b_avg,v_b_max), dim=-1)

        return self.mlp(out_put)


class ESIM(nn.Module):
  #上述模块串联组装
    def __init__(self, len_feature, len_hidden, len_words,longest, type_num=4, weight=None, layer=1, batch_first=True,
                 drop_out=0.5):
        super(ESIM, self).__init__()
        self.len_words=len_words
        self.longest=longest
        #输入编码
        self.input_encoding = Input_Encoding(len_feature, len_hidden, len_words,longest, weight=weight, layer=layer,
                                             batch_first=batch_first, drop_out=drop_out)
        #局部推理
        self.local_inference_modeling = Local_Inference_Modeling()
        #推理合成
        self.inference_composition = Inference_Composition(len_feature, 8 * len_hidden, len_hidden, layer=layer,
                                                           batch_first=batch_first, drop_out=drop_out)
        #预测
        self.prediction=Prediction(len_hidden*8,len_hidden,type_num=type_num,drop_out=drop_out)

    def forward(self,a,b):
        a_bar=self.input_encoding(a) 
        b_bar=self.input_encoding(b) #句子ab分别编码


        #局部推理
        m_a,m_b=self.local_inference_modeling(a_bar,b_bar)

        #句子全局特征
        v_a=self.inference_composition(m_a)
        v_b=self.inference_composition(m_b)

        out_put=self.prediction(v_a,v_b)

        return out_put
