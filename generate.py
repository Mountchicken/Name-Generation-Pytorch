import torch
from utils import char_tensor
from RNNmodel import RNN
import string
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def generate(model, all_characters, initial_str="A", predict_len=100, temperature=0.85):
    hidden, cell = model.init_hidden(batch_size=1)
    inital_input = char_tensor(initial_str, all_characters)
    predicted = initial_str
    for p in range(len(initial_str) - 1):
        _, (hidden, cell) = model(inital_input[p].view(1).to(device),hidden,cell)

    last_char = inital_input[-1]
    for p in range(predict_len):
        output, (hidden, cell) = model(last_char.view(1).to(device),hidden,cell)
        output_dist = output.data.view(-1).div(temperature).exp() #make more risky predictions
        tor_char = torch.multinomial(output_dist,1)[0]
        predicted_char = all_characters[tor_char]
        predicted += predicted_char
        last_char = char_tensor(predicted_char, all_characters)
    return predicted

if __name__=="__main__":
    model = torch.load('weights/saved_model')
    all_characters = string.printable #Get characters from string.printable
    generated_names = generate(model=model, initial_str='Rylon', predict_len=100, temperature=0.85, all_characters=all_characters)
    print(generated_names)

