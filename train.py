import torch
import torch.nn as nn 
from RNNmodel import RNN
from utils import get_random_batch, char_tensor
from generate import generate
import string
from torch.utils.tensorboard import SummaryWriter
import unidecode
from tqdm import tqdm
#Hyperparameters
chunk_len = 250 #一次从数据集取250个字母
num_epochs = 10000
batch_size = 1

learning_rate= 0.003
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Creat Model
load_model = False
save_model = True
print_every = 50 #print every 50 epochs
hidden_size = 256
num_layers = 2

all_characters = string.printable #Get characters from string.printable
len_characters = len(all_characters)
model = RNN(len_characters, hidden_size, num_layers, len_characters).to(device)
if load_model:
    model = torch.load('weights/saved_model')
model.train()

#optimizer and loss function and stufs
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()
writer = SummaryWriter(f'runs/names0') #for tensorboard
name_file = unidecode.unidecode(open('data/names.txt',encoding='utf-8').read())
print("=> Starting training")

#paramters when generate
initial_str='A'  # the first character of name that you expected
predict_len = 100 # the length of characters to generate
temperature=0.85

#train
for epoch in tqdm(range(1, num_epochs +1 )):
    inp, target = get_random_batch(name_file, chunk_len, batch_size, all_characters=all_characters)
    hidden, cell = model.init_hidden(batch_size=batch_size)

    model.zero_grad()
    loss = 0
    inp = inp.to(device)
    target = target.to(device)

    for c in range(chunk_len):
        output, (hidden,cell) =model(inp[:, c], hidden, cell)
        loss += criterion(output, target[:, c])

    loss.backward()
    optimizer.step()
    loss = loss.item() / chunk_len

    if epoch % print_every == 0 :
        print('loss: {}'.format(loss))
        #print(generate(model=model, initial_str='A', predict_len=100, temperature=0.85, all_characters=all_characters))
    writer.add_scalar('Training loss', loss, global_step=epoch)
    if epoch % 50 ==0 and save_model: #若保存模型，则每50个epoch保存一次
        torch.save(model,'weights/saved_model')




