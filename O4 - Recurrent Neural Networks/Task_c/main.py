import torch
import torch.nn as nn

char_encodings = [
    [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],  # ' '
    [0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],  # 'a'
    [0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],  # 'c'
    [0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],  # 'f'
    [0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],  # 'h'
    [0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],  # 'l'
    [0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],  # 'm'
    [0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],  # 'n'
    [0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],  # 'o'
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],  # 'p'
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.],  # 'r'
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],  # 's'
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.]  # 't'
]

encoding_size = len(char_encodings)
index_to_char = [' ', 'a', 'c', 'f', 'h', 'l', 'm', 'n', 'o', 'p', 'r', 's', 't']


def encode_string(string):
    encoding = []

    for char in string:
        encoding.append(char_encodings[index_to_char.index(char)])
    return encoding


x_train = torch.tensor([
    encode_string('hat '),
    encode_string('rat '),
    encode_string('cat '),
    encode_string('flat'),
    encode_string('matt'),
    encode_string('cap '),
    encode_string('son '),
]).transpose(1, 0)

emoji_encodings = [
    [1., 0., 0., 0., 0., 0., 0.],  # 'Hat'  🤠
    [0., 1., 0., 0., 0., 0., 0.],  # 'rat'  🐀
    [0., 0., 1., 0., 0., 0., 0.],  # 'cat'  😼
    [0., 0., 0., 1., 0., 0., 0.],  # 'flat' 🏘
    [0., 0., 0., 0., 1., 0., 0.],  # 'matt' 🙋
    [0., 0., 0., 0., 0., 1., 0.],  # 'cap'  🧢
    [0., 0., 0., 0., 0., 0., 1.]  # 'son'   👶
]

emoji_size = len(emoji_encodings)
index_to_emoji = ['🤠', '🐀', '😼', '🏘', '🙋', '🧢', '👶']


def encode_emoji(emoji):
    return emoji_encodings[index_to_emoji.index(emoji)]


def decode_emoji(tensor):
    return index_to_emoji[tensor.argmax(1)]


y_train = torch.tensor(
    [encode_emoji('🤠'), encode_emoji('🐀'), encode_emoji('😼'), encode_emoji('🏘'),
     encode_emoji('🙋'), encode_emoji('🧢'), encode_emoji('👶')])


class LongShortTermMemoryModel(nn.Module):
    def __init__(self, in_size, out_size):
        super(LongShortTermMemoryModel, self).__init__()

        self.lstm = nn.LSTM(in_size, 128)  # 128 is the state size
        self.dense = nn.Linear(128, out_size)  # 128 is the state size

    def reset(self, batch_size):  # Reset states prior to new input sequence
        zero_state = torch.zeros(1, batch_size, 128)  # Shape: (number of layers, batch size, state size)
        self.hidden_state = zero_state
        self.cell_state = zero_state

    def logits(self, x):  # x shape: (sequence length, batch size, encoding size)
        out, (self.hidden_state, self.cell_state) = self.lstm(x, (self.hidden_state, self.cell_state))
        return self.dense(out[-1].reshape(-1, 128))  # uses last output

    def f(self, x):  # x shape: (sequence length, batch size, encoding size)
        return torch.softmax(self.logits(x), dim=1)

    def loss(self, x,
             y):  # x shape: (sequence length, batch size, encoding size), y shape: (sequence length, emoji size)
        return nn.functional.cross_entropy(self.logits(x), y.argmax(1))


model = LongShortTermMemoryModel(encoding_size, emoji_size)

learning_rate = 0.001
epochs = 500

optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)

for epoch in range(epochs):
    model.reset(x_train.size(1))
    model.loss(x_train, y_train).backward()
    optimizer.step()
    optimizer.zero_grad()

    if epoch % 10 == 9:
        model.reset(1)
        test_string = 'rt'
        print(decode_emoji(model.f(torch.tensor([encode_string(test_string)]).transpose(1, 0))))