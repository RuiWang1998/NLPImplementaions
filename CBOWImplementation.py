import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

torch.manual_seed(1)

CONTEXT_SIZE = 2  # 2 words to the left, 2 to the right
EMBED_DIM = 200

raw_text = """We are about to study the idea of a computational process.
Computational processes are abstract beings that inhabit computers.
As they evolve, processes manipulate other abstract things called data.
The evolution of a process is directed by a pattern of rules
called a program. People create programs to direct processes. In effect,
we conjure the spirits of the computer with our spells.""".split()

# By deriving a set from `raw_text`, we deduplicate the array
vocab = set(raw_text)
vocab_size = len(vocab)
EPOCH = 15

word_to_ix = {word: i for i, word in enumerate(vocab)}
data = []
for i in range(2, len(raw_text) - 2):
    context = [raw_text[i - 2], raw_text[i - 1],
               raw_text[i + 1], raw_text[i + 2]]
    target = raw_text[i]
    data.append((context, target))


class CBOW(nn.Module):


    def __init__(self):
        super(CBOW, self).__init__()
        self.embed_dim = EMBED_DIM
        self.context_size = CONTEXT_SIZE
        self.embeddings = nn.Embedding(vocab_size, self.embed_dim)
        self.lc = nn.Linear(self.embed_dim, vocab_size, bias=True)
        

    def forward(self, inputs):
        embeds = self.embeddings(inputs)
        output = embeds.sum(dim=0)
        output = self.lc(output)

        return output


# create your model and train.  here are some functions to help you make
# the data ready for use by your module


def make_context_vector(context, word_to_ix):
    idxs = [word_to_ix[w] for w in context]
    return torch.tensor(idxs, dtype=torch.long)


make_context_vector(data[0][0], word_to_ix)  # example

loss_func = nn.CrossEntropyLoss()
net = CBOW()
print(net)
optimizer = optim.SGD(net.parameters(), lr=0.01)

for context, target in data:
    context_inx = make_context_vector(context, word_to_ix)
    output = net(context_inx)

for epoch in range(EPOCH):
    loss_total = 0
    for context, target in data:
        context_inx = make_context_vector(context, word_to_ix)
        net.zero_grad()
        probs = net(context_inx)
        loss = nn.CrossEntropyLoss()
        loss = loss_func(probs.view(1,-1), torch.tensor([word_to_ix[target]]))
        print('The cross entropy loss value is {}'.format(loss))
        loss.backward()
        optimizer.step()

        loss_total += loss.data



def ix_to_word(ix):
    vocab_list = list(word_to_ix.keys())
    word_predicted = vocab_list[0]
    for word in word_to_ix:
        if word_to_ix[word] == ix:
            word_predicted = word

    return word_predicted


# testing
context = ['are','about','study', 'the']
context_vector = make_context_vector(context, word_to_ix)
a = net(context_vector).data.numpy()
print('Raw text: {}\n'.format(' '.join(raw_text)))
print('Context: {}\n'.format(context))
print('Prediction: {}'.format(ix_to_word(a.argmax())))

# output the parameters
for param in net.parameters():
    print(param)