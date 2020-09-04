import torch
import numpy as np
import torch.nn as nn
import torch.utils.data as Data

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# S: Symbol that shows starting of decoding input
# E: Symbol that shows Ending of decoding output
# ?: Symbol that will fill in blank sequence if current batch data size is short than n_step

letter = [c for c in 'SE?abcdefghijklmnopqrstuvwxyz']
print(list(enumerate(letter)))
letter2idx = {l: index for index, l in enumerate(letter)}

seq_data = [['man', 'women'], ['black', 'white'], ['king', 'queen'], ['girl', 'boy'], ['up', 'down'], ['high', 'low']]

#模型参数
n_step = max([max(len(i), len(j)) for i, j in seq_data])
n_hidden = 128
n_class = len(letter2idx)
print(n_class)
batch_size = 3


#数据预处理
def make_data(seq_data):
    enc_input_all, dec_input_all, dec_output_all = [], [], []

    for seq in seq_data:
        for i in range(2):
            seq[i] = seq[i] + "?"*(n_step-len(seq[i]))

        enc_input = [letter2idx[n] for n in (seq[0] + "E")]
        dec_input = [letter2idx[n] for n in ("S" + seq[1])]
        dec_output = [letter2idx[n] for n in (seq[1] + "E")]

        enc_input_all.append(np.eye(n_class)[enc_input])
        dec_input_all.append(np.eye(n_class)[dec_input])
        dec_output_all.append(dec_output)
    #make tensor
    return torch.Tensor(enc_input_all), torch.Tensor(dec_input_all), torch.Tensor(dec_output_all)

'''
enc_input_all: [6, n_step+1 (because of 'E'), n_class]
dec_input_all: [6, n_step+1 (because of 'S'), n_class]
dec_output_all: [6, n_step+1 (because of 'E')]
'''
enc_input_all, dec_input_all, dec_output_all = make_data(seq_data)

#制作数据集
class TranslateDataset(Data.Dataset):

    def __init__(self, enc_input_all, dec_input_all, dec_output_all):
        self.enc_input_all = enc_input_all
        self.dec_input_all = dec_input_all
        self.dec_output_all = dec_output_all

    def __len__(self):
        return len(enc_input_all)

    def __getitem__(self, index):
        return enc_input_all[index], dec_input_all[index], dec_output_all[index]


loader = Data.DataLoader(TranslateDataset(enc_input_all, dec_input_all, dec_output_all), batch_size, True)

#模型定义
class Seq2Seq(nn.Module):

    def __init__(self):
        super(Seq2Seq, self).__init__()
        self.Encoder = nn.RNN(input_size=n_class, hidden_size=n_hidden, dropout=0.5)
        self.Decoder = nn.RNN(input_size=n_class, hidden_size=n_hidden, dropout=0.5)
        self.fc = nn.Linear(n_hidden, n_class)

    def forward(self, enc_input, enc_hidden, dec_input):
        # enc_input(=input_batch): [batch_size, n_step+1, n_class]
        # dec_input(=output_batch): [batch_size, n_step+1, n_class]
        enc_input = enc_input.transpose(0, 1) #[n_step+1, batch_size,  n_class]
        dec_input = dec_input.transpose(0, 1) #[n_step+1, batch_size,  n_class]

        # h_t : [num_layers(=1) * num_directions(=1), batch_size, n_hidden]
        _, h_t = self.Encoder(enc_input, enc_hidden)
        # outputs : [n_step+1, batch_size, num_directions(=1) * n_hidden(=128)]
        outputs, _ = self.Decoder(dec_input, h_t)

        return self.fc(outputs) # [n_step+1, batch_size, n_class]


model = Seq2Seq().to(device=device)
criterion = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

#训练模型
for epoch in range(5000):
    for enc_input_batch, dec_input_batch, dec_output_batch in loader:
        # 初始隐藏状态
        h_0 = torch.zeros(1, batch_size, n_hidden).to(device)

        # enc_input_batch : [batch_size, n_step+1, n_class]
        # dec_input_batch : [batch_size, n_step+1, n_class]
        # dec_output_batch : [batch_size, n_step+1], not one-hot
        (enc_input_batch, dec_input_batch, dec_output_batch) = (enc_input_batch.to(device), dec_input_batch.to(device),
                                                                dec_output_batch.to(device))

        # pred[n_step+1, batch_size, n_class]
        pred = model(enc_input_batch, h_0, dec_input_batch)
        # pred[batch_size, n_step+1, n_class]
        pred = pred.transpose(0,1)
        loss = 0
        for i in range(len(dec_output_batch)):
            # pred[i] : [n_step+1, n_class]
            # dec_output_batch[i] : [n_step+1]
            loss += criterion(pred[i] + dec_output_batch[i])

        if (epoch+1)%1000 == 0:
            print("Epoch:", "%04d" % (epoch+1), "loss=", "{:.6f}".format(loss))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# 测试
def translate(word):
    enc_input, dec_input, _ = make_data([[word, '?'*n_step]])
    enc_input, dec_input = enc_input.to(device), dec_input.to(device)
    hidden = torch.zeros(1, 1, n_hidden).to(device)
    output = model(enc_input, hidden, dec_input)
    # output:[n_step+1, batch_size, n_class]

    # 选择预测值最大的下标
    predict = output.data.max(2, keepdim=True)[1]
    decoded = [letter[i] for i in predict]

    translated = ''.join(decoded[:decoded.index('E')])
    return translated.replace('?', '')






