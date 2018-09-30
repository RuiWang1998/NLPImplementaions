import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import time


torch.manual_seed(1)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
CONTEXT_SIZE = 5  # 2 words to the left, 2 to the right
EMBED_DIM = 100
EPOCH = 30

raw_text = """We are about to study the idea of a computational process.
Computational processes are abstract beings that inhabit computers.
As they evolve, processes manipulate other abstract things called data.
The evolution of a process is directed by a pattern of rules
called a program. People create programs to direct processes. In effect,
we conjure the spirits of the computer with our spells.""".split()


# By deriving a set from `raw_text`, we deduplicate the array
vocab = set(raw_text)
vocab_size = len(vocab)


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
        self.lc = nn.Linear(self.embed_dim, vocab_size, bias=False)
        

    def forward(self, inputs):
        embeds = self.embeddings(inputs.to(device))
        output = embeds.mean(dim=0)
        output = self.lc(output)

        return output


# create your model and train.  here are some functions to help you make
# the data ready for use by your module


def make_context_vector(context, word_to_ix):
    idxs = [word_to_ix[w] for w in context]
    context_vector = torch.tensor(idxs, dtype=torch.long)
    return context_vector


make_context_vector(data[0][0], word_to_ix)  # example

loss_func = nn.CrossEntropyLoss()
net = CBOW()
net.to(device)
optimizer = optim.Adam(net.parameters(), lr=0.016, betas=(0.99, 0.999), eps=1e-08, weight_decay=0, amsgrad=True)

#for param in net.parameters():
#    print(param)
start = time.time()

for epoch in range(EPOCH):
    loss_total = 0
    for context, target in data:
        context_inx = make_context_vector(context, word_to_ix)
        net.zero_grad()
        similarity = net(context_inx)
        loss = loss_func(similarity.view(1,-1), torch.tensor([word_to_ix[target]], device = device))
        # print('The cross entropy loss value is {}'.format(loss))
        loss.backward()
        optimizer.step()

        loss_total += loss.data
    print('Epoch: {} |The cross entropy loss value is {}'.format(epoch + 1, loss_total))


def ix_to_word(ix):
    vocab_list = list(word_to_ix.keys())
    word_predicted = vocab_list[0]
    for word in word_to_ix:
        if word_to_ix[word] == ix:
            word_predicted = word

    return word_predicted


# output the parameters
word2vec = torch.zeros(vocab_size, EMBED_DIM)
for param in net.parameters():
    word2vec = word2vec + torch.Tensor.cpu(param.detach())

word2numpy = word2vec.numpy()
np.savetxt("vectors.csv", word2numpy, delimiter=",")

print("The vacabulary size is {}.".format(vocab_size));

## testing
context = ['couple','even','to', 'stay']
context_vector = make_context_vector(context, word_to_ix)
a = torch.Tensor.cpu(net(context_vector)).data.numpy()
print('Context: {}\n'.format(context))
print('Prediction: {}'.format(ix_to_word(a.argmax())))
end = time.time()
print("Time elapsed: {}".format(end - start))
