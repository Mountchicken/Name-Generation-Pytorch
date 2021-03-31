import torch
import random
def char_tensor(string, all_characters): #将长度为chunk_len的字符串转换为其在字符表中的对应index,len(tensor)==len(string)
    '''
    all_characters: 所有可能字符
    string :输入的待转换字符串
    '''
    tensor = torch.zeros(len(string)).long()
    for c in range(len(string)):
        tensor[c] = all_characters.index(string[c]) #ex :stirng="Hello" --> tensor = [15, 4, 10, 10 ,23]
    return tensor

def get_random_batch(name_file, chunk_len, batch_size, all_characters):
    '''
    name_file: 存放名字的文件
    chunk_len: 需要返回的字符串长度
    batch_size:
    '''
    start_idx = random.randint(0, len(name_file) - chunk_len) #确保起始位置往后能至少有chunk_len个字符
    end_idx = start_idx + chunk_len + 1
    text_str = name_file[start_idx:end_idx] #chunk_len长的字符串
    text_input = torch.zeros(batch_size, chunk_len) #shape [batch_size, chunk_len]
    text_target = torch.zeros(batch_size, chunk_len) #shape [batch_size, chunk_len]

    for i in range(batch_size):
        text_input[i,:] = char_tensor(text_str[:-1], all_characters) #输入第一个词，希望能预测出下一个词，所以输入到最后一个字前截止
        text_target[i,:] = char_tensor(text_str[1:], all_characters) #输出从第二个词开始
    return text_input.long(), text_target.long()
