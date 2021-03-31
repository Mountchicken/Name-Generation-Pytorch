import torch
import torch.nn as nn
import string
import random
import sys
import unidecode
from torch.utils.tensorboard import SummaryWriter

#Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Get characters from string.printable
all_characters = string.printable
len_characters = len(all_characters)

#Read large text file 
name_file = unidecode.unidecode(open('data/names.txt',encoding='utf-8').read())
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embed = nn.Embedding(num_embeddings=input_size, embedding_dim=hidden_size)
        self.lstm = nn.LSTM(hidden_size,hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden, cell):
        out = self.embed(x)
        out, (hidden, cell) = self.lstm(out.unsqueeze(1), (hidden, cell))
        out = self.fc(out.reshape(out.shape[0],-1))
        return out, (hidden, cell)

    def init_hidden(self, batch_size):
        hidden = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        cell = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        return hidden, cell

class Generator():
    def __init__(self):
        self.chunk_len = 250 #一次从数据集取250个字母
        self.num_epochs = 200
        self.batch_size = 1
        self.print_every = 50 #print every 50 epochs
        self.hidden_size = 256
        self.num_layers = 2
        self.lr= 0.003

    def char_tensor(self, string): #将长度为chunk_len的字符串转换为其在字符表中的对应index,len(tensor)==len(string)
        tensor = torch.zeros(len(string)).long()
        for c in range(len(string)):
            tensor[c] = all_characters.index(string[c]) #ex :stirng="Hello" --> tensor = [15, 4, 10, 10 ,23]
        return tensor

    def get_random_batch(self):
        start_idx = random.randint(0, len(name_file) - self.chunk_len) #确保起始位置往后能至少有chunk_len个字符
        end_idx = start_idx + self.chunk_len + 1
        text_str = name_file[start_idx:end_idx] #chunk_len长的字符串
        text_input = torch.zeros(self.batch_size, self.chunk_len) #shape [batch_size, chunk_len]
        text_target = torch.zeros(self.batch_size, self.chunk_len) #shape [batch_size, chunk_len]

        for i in range(self.batch_size):
            text_input[i,:] = self.char_tensor(text_str[:-1]) #输入第一个词，希望能预测出下一个词，所以输入到最后一个字前截止
            text_target[i,:] = self.char_tensor(text_str[1:]) #输出从第二个词开始

        return text_input.long(), text_target.long()

    def generate(self, initial_str="A", predict_len=100, temperature=0.85):
        hidden, cell = self.rnn.init_hidden(batch_size=self.batch_size)
        inital_input = self.char_tensor(initial_str)
        predicted = initial_str

        for p in range(len(initial_str) - 1):
            _, (hidden, cell) = self.rnn(inital_input[p].view(1).to(device),hidden,cell)

        last_char = inital_input[-1]
        for p in range(predict_len):
            output, (hidden, cell) = self.rnn(last_char.view(1).to(device),hidden,cell)
            output_dist = output.data.view(-1).div(temperature).exp() #make more risky predictions
            tor_char = torch.multinomial(output_dist,1)[0]
            predicted_char = all_characters[tor_char]
            predicted += predicted_char
            last_char = self.char_tensor(predicted_char)

        return predicted

    def train(self):
        self.rnn = RNN(len_characters, self.hidden_size, self.num_layers, len_characters).to(device)
        optimizer = torch.optim.Adam(self.rnn.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()
        writer = SummaryWriter(f'runs/names0') #for tensorboard
        print("=> Starting training")

        for epoch in range(1, self.num_epochs +1 ):
            inp, target = self.get_random_batch()
            hidden, cell = self.rnn.init_hidden(batch_size=self.batch_size)

            self.rnn.zero_grad()
            loss = 0
            inp = inp.to(device)
            target = target.to(device)

            for c in range(self.chunk_len):
                output, (hidden,cell) = self.rnn(inp[:, c], hidden, cell)
                loss += criterion(output, target[:, c])

            loss.backward()
            optimizer.step()
            loss = loss.item() / self.chunk_len

            if epoch % self.print_every == 0 :
                print('loss: {}'.format(loss))
                print(self.generate())
            writer.add_scalar('Training loss', loss, global_step=epoch)
        torch.save(self.rnn.state_dict(),'weights/saved_weights')

gennames = Generator()
gennames.train()