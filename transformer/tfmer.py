import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

#模拟数据集
sentences = [
    #encoder_input            #decoder_input       #decoder_output
    ['ich mochte ein bier P', 'S i want a beer .', 'i want a beer . E'],
    ['ich mochte ein cola P', 'S i want a coke .', 'i want a coke . E']
]

#建立字典
src_vocb = {'P': 0, 'ich': 1, 'mochte': 2, 'ein': 3, 'bier': 4, 'cola': 5}
src_vocb_size = len(src_vocb)

tgt_vocb = {'P': 0, 'i': 1, 'want': 2, 'a': 3, 'beer': 4, 'coke': 5, 'S': 6, 'E': 7, '.': 8}
index2word = {i: w for w,i in tgt_vocb.items()}
tgt_vocb_size = len(tgt_vocb)

src_len = 5
tgt_len = 6

#将文字数据集转化为word index
def make_data(sentences):
    enc_inputs, dec_inputs, dec_outputs = [], [], []
    for i in range(len(sentences)):
        enc_input = [src_vocb[n] for n in sentences[i][0].split()]
        dec_input = [tgt_vocb[n] for n in sentences[i][1].split()]
        dec_output = [tgt_vocb[n] for n in sentences[i][2].split()]
        enc_inputs.append(enc_input)
        dec_inputs.append(dec_input)
        dec_outputs.append(dec_output)

    return torch.LongTensor(enc_inputs), torch.LongTensor(dec_inputs), torch.LongTensor(dec_outputs)

#构建训练数据集对象
class MyDataSet(data.Dataset):
    def __init__(self, enc_inputs, dec_inputs, dec_outputs):
        super(MyDataSet, self).__init__()
        self.enc_inputs = enc_inputs
        self.dec_inputs = dec_inputs
        self.dec_outputs = dec_outputs

    def __len__(self):
        return self.enc_inputs.shape[0]

    def __getitem__(self, index):
        return self.enc_inputs[index], self.dec_inputs[index], self.dec_outputs[index]

enc_inputs, dec_inputs, dec_outputs = make_data(sentences)
dataloader = data.DataLoader(MyDataSet(enc_inputs, dec_inputs, dec_outputs), 2, True)

#模型参数
emb_size = 512 #嵌入维度
ff_dim = 2048 #前馈神经网络隐藏单元数
K_dim, Q_dim, V_dim = 64, 64, 64 #K, Q, V矩阵维度
layers_num = 6 #encoder、decoder层数
heads_num = 8 #多头机制head数

#位置编码（Position Encoding）
def get_sinusoid_encoding_table(n_position, d_model):
    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_model)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_model)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    return torch.FloatTensor(sinusoid_table)

#Padding Mask
def get_attn_pad_mask(seq_q, seq_k):
    '''
    seq_q: [batch_size, seq_len]
    seq_k: [batch_size, seq_len]
    seq_len could be src_len or it could be tgt_len
    seq_len in seq_q and seq_len in seq_k maybe not equal
    '''
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    # eq(zero) is PAD token
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)  # [batch_size, 1, len_k], False is masked
    end = pad_attn_mask.expand(batch_size, len_q, len_k)  # [batch_size, len_q, len_k]
    return end

def get_attn_subsequence_mask(seq):
    '''
    seq: [batch_size, tgt_len]
    '''
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
    subsequence_mask = np.triu(np.ones(attn_shape), k=1) # Upper triangular matrix
    subsequence_mask = torch.from_numpy(subsequence_mask).byte()
    return subsequence_mask
a = torch.Tensor([[1,2,3,0],[1,2,3,0]])
print(get_attn_subsequence_mask(a))

#ScaledDotProductAttention
class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attention_mask):
        '''
        Q: [batch_size, n_heads, len_q, d_k]
        K: [batch_size, n_heads, len_k, d_k]
        V: [batch_size, n_heads, len_v(=len_k), d_v]
        attn_mask: [batch_size, n_heads, seq_len, seq_len]
        '''
        scores = torch.matmul(Q, K.transpose(-1, -2))/np.sqrt(K_dim)
        scores.masked_fill(mask=attention_mask, value=-1e9)

        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)

        return context, attn

#MultiHeadAttention
class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(emb_size, Q_dim*heads_num, bias=False)
        self.W_K = nn.Linear(emb_size, K_dim*heads_num, bias=False)
        self.W_V = nn.Linear(emb_size, V_dim*heads_num, bias=False)
        self.fc = nn.Linear(V_dim*heads_num, emb_size, bias=False)

    def forward(self, input_Q, input_K, input_V, attention_mask):
        '''
        input_Q: [batch_size, len_q, emb_size]
        input_K: [batch_size, len_k, emb_size]
        input_V: [batch_size, len_v(=len_k), emb_size]
        attn_mask: [batch_size, seq_len, seq_len]
        '''
        residual, batch_size = input_Q, input_Q.size(0)
        Q = self.W_Q(input_Q).view(batch_size, -1, heads_num, Q_dim).transpose(1, 2)
        K = self.W_K(input_K).view(batch_size, -1, heads_num, K_dim).transpose(1, 2)
        V = self.W_V(input_V).view(batch_size, -1, heads_num, V_dim).transpose(1, 2)

        attn_mask = attention_mask.unsqueeze(1).repeat(1, heads_num, 1, 1)
        context, attn = ScaledDotProductAttention()(Q, K, V, attn_mask)
        context = context.transpose(1, 2).reshape(batch_size, -1, heads_num * V_dim)
        outputs = self.fc(context)
        return nn.LayerNorm(emb_size)(outputs+input_Q), attn


class FeedForwardNet(nn.Module):
    def __init__(self):
        super(FeedForwardNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(emb_size, ff_dim, bias=False),
            nn.ReLU(),
            nn.Linear(ff_dim, emb_size, bias=False)
        )

    def forward(self, inputs):
        residual = inputs
        outputs = self.fc(inputs)
        return nn.LayerNorm(emb_size)(outputs+residual)

#Encoder Layer
class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.mulatten = MultiHeadAttention()
        self.ffn = FeedForwardNet()
    def forward(self, inputs, attention_mask):
        enc_outputs, attn = self.mulatten(inputs, inputs, inputs, attention_mask)
        enc_outputs = self.ffn(enc_outputs)
        return enc_inputs, attn

#Encoder
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.src_emb = nn.Embedding(src_vocb_size, emb_size)
        self.pos_emb = nn.Embedding.from_pretrained(get_sinusoid_encoding_table(src_vocb_size, emb_size), freeze=True)
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(layers_num)])

    def forward(self, enc_inputs):
        '''
        enc_inputs: [batch_size, src_len]
        '''
        word_emb = self.src_emb(enc_inputs)  # [batch_size, src_len, d_model]
        pos_emb = self.pos_emb(enc_inputs)  # [batch_size, src_len, d_model]
        enc_outputs = word_emb + pos_emb
        enc_self_attn_mask = get_attn_pad_mask(enc_inputs, enc_inputs)  # [batch_size, src_len, src_len]
        enc_self_attns = []
        for layer in self.layers:
            # enc_outputs: [batch_size, src_len, d_model], enc_self_attn: [batch_size, n_heads, src_len, src_len]
            enc_outputs, enc_self_attn = layer(enc_outputs, enc_self_attn_mask)
            enc_self_attns.append(enc_self_attn)
        return enc_outputs, enc_self_attns

#Decoder Layer
class DecoderLayer(nn.Module):
    def __init__(self):
        super(DecoderLayer, self).__init__()
        self.dec_self_mulattn = MultiHeadAttention()
        self.dec_enc_mulattn = MultiHeadAttention()
        self.ffn = FeedForwardNet()

    def forward(self, dec_inputs, enc_outputs, dec_self_mulattn_mask, dec_enc_mulattn_mask):
        """
        :param dec_inputs:[batch_size, tgt_len, emb_size]
        :param enc_outputs:[batch_size, src_len, emb_size]
        :param dec_self_mulattn_mask:[batch_size, tgt_len, tgt_len]
        :param dec_enc_mulattn_mask:[batch_size, tgt_len, src_len]
        :return:
        """
        # dec_outputs: [batch_size, tgt_len, d_model], dec_self_attn: [batch_size, n_heads, tgt_len, tgt_len]
        dec_outputs, dec_self_attn = self.dec_self_mulattn(dec_inputs, dec_inputs, dec_inputs, dec_self_mulattn_mask)
        # dec_outputs: [batch_size, tgt_len, d_model], dec_enc_attn: [batch_size, h_heads, tgt_len, src_len]
        dec_outputs, dec_enc_attn = self.dec_enc_mulattn(dec_outputs, enc_outputs, enc_outputs, dec_enc_mulattn_mask)
        dec_outputs = self.ffn(dec_outputs)  # [batch_size, tgt_len, d_model]
        return dec_outputs, dec_self_attn, dec_enc_attn

#Decoder
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.tgt_emb = nn.Embedding(tgt_vocb_size, emb_size)
        self.pos_emb = nn.Embedding.from_pretrained(get_sinusoid_encoding_table(tgt_vocb_size, emb_size),freeze=True)
        self.layers = nn.ModuleList([DecoderLayer() for _ in range(layers_num)])

    def forward(self, dec_inputs, enc_inputs, enc_outputs):
        '''
        dec_inputs: [batch_size, tgt_len]
        enc_intpus: [batch_size, src_len]
        enc_outputs: [batsh_size, src_len, d_model]
        '''
        word_emb = self.tgt_emb(dec_inputs) # [batch_size, tgt_len, d_model]
        pos_emb = self.pos_emb(dec_inputs) # [batch_size, tgt_len, d_model]
        dec_outputs = word_emb + pos_emb

        #mask掉pad和未来时间序列信息
        dec_self_attn_pad_mask = get_attn_pad_mask(dec_inputs, dec_inputs) # [batch_size, tgt_len, tgt_len]
        dec_self_attn_subsequent_mask = get_attn_subsequence_mask(dec_inputs) # [batch_size, tgt_len]
        dec_self_attn_mask = torch.gt((dec_self_attn_pad_mask + dec_self_attn_subsequent_mask), 0) # [batch_size, tgt_len, tgt_len]

        dec_enc_attn_mask = get_attn_pad_mask(dec_inputs, enc_inputs) # [batc_size, tgt_len, src_len]
        dec_self_attns, dec_enc_attns = [], []
        for layer in self.layers:
            # dec_outputs: [batch_size, tgt_len, d_model], dec_self_attn: [batch_size, n_heads, tgt_len, tgt_len], dec_enc_attn: [batch_size, h_heads, tgt_len, src_len]
            dec_outputs, dec_self_attn, dec_enc_attn = layer(dec_outputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask)
            dec_self_attns.append(dec_self_attn)
            dec_enc_attns.append(dec_enc_attn)
        return dec_outputs, dec_self_attns, dec_enc_attns

#Transformer
class Transformer(nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.projection = nn.Linear(emb_size, tgt_vocb_size, bias=False)

    def forward(self, enc_inputs, dec_inputs):
        '''
        enc_inputs: [batch_size, src_len]
        dec_inputs: [batch_size, tgt_len]
        '''
        # tensor to store decoder outputs
        # outputs = torch.zeros(batch_size, tgt_len, tgt_vocab_size).to(self.device)

        # enc_outputs: [batch_size, src_len, d_model], enc_self_attns: [n_layers, batch_size, n_heads, src_len, src_len]
        enc_outputs, enc_self_attns = self.encoder(enc_inputs)
        # dec_outpus: [batch_size, tgt_len, d_model], dec_self_attns: [n_layers, batch_size, n_heads, tgt_len, tgt_len], dec_enc_attn: [n_layers, batch_size, tgt_len, src_len]
        dec_outputs, dec_self_attns, dec_enc_attns = self.decoder(dec_inputs, enc_inputs, enc_outputs)
        dec_logits = self.projection(dec_outputs)  # dec_logits: [batch_size, tgt_len, tgt_vocab_size]
        return dec_logits.view(-1, dec_logits.size(-1)), enc_self_attns, dec_self_attns, dec_enc_attns

#模型&损失函数&优化器
model = Transformer()
criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.99)

#训练
for epoch in range(30):
    for enc_inputs, dec_inputs, dec_outputs in dataloader:
        '''
        enc_inputs: [batch_size, src_len]
        dec_inputs: [batch_size, tgt_len]
        dec_outputs: [batch_size, tgt_len]
        '''
        outputs, enc_self_attns, dec_self_attns, dec_enc_attns = model(enc_inputs, dec_inputs)
        loss = criterion(outputs, dec_outputs.view(-1))
        print('Epoch:', '%04d' % (epoch + 1), 'loss =', '{:.6f}'.format(loss))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

enc_inputs, dec_inputs, _ = next(iter(dataloader))
predict, _, _, _ = model(enc_inputs[0].view(1, -1), dec_inputs[0].view(1, -1)) # model(enc_inputs[0].view(1, -1), greedy_dec_input)
predict = predict.data.max(1, keepdim=True)[1]
print(enc_inputs[0], '->', [index2word[n.item()] for n in predict.squeeze()])