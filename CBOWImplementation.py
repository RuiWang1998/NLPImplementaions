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

raw_text = """
the small town police chief caught up with his man at the local Burger King Hartville Ohio is a village of 3000 people stuck between Akron and Canton the 
fast-food restaurant squats on a parking lot spidered with cracks fronting a strip mall not far from the heart of the town on a Friday last April Hartville
Police Chief Larry Dordea marched through the door his eyes searching for Philip Snider Seventy-three-years old bespectacled a thinning crop of blond hair 
topping his narrow head Snider was at the Burger King like most mornings enjoying a cup of coffee the chief asked him how he was doing Peachy keen Snider 
cheerfully responded Dordea later recounted to Memphis’s News 10 Subscribe to the Post Most newsletter today’s most popular stories on the Washington Post 
three months earlier Snider’s wife of 53 years Roberta had vanished Police in two states were investigating Rivers had been searched polygraph tests administered 
but authorities struggled for answers Snider’s own account of the disappearance had skipped and sputtered like a warped record — a whirlwind of changing stories 
involving a pilgrimage to Elvis Presley’s home a mystery ambulance and a bloody sweatshirt but as Dordea stood in the Burger King he knew then Snider had made a 
crucial mistake he had spilled his secret to the wrong person his look changed when I told him he was under arrest Dordea told 10 News on Monday Snider pleaded 
guilty to aggravated murder tampering with evidence and gross abuse of a corpse Cleveland’s WKYC reported as part of his deal with prosecutors Snider has reportedly 
agreed to finally lead authorities to his wife’s body Although Snider’s plea signaled his cooperation with authorities the Ohio man’s confession was the result of 
complex police work at Monday’s hearing prosecutors revealed the breakthrough in the case came after an undercover officer won Snider’s trust and pulled the truth 
from the widower he’s now facing life in prison with at least 20 years inside before his eligibility for parole — a virtual death sentence for a man in his eighth 
decade this is one of the most ghastly crimes I have ever seen Judge Frank Forchione told the defendant on Monday according to a video of the proceeding the hottest
places in hell are reserved for people like you It all started with a trip to Graceland according to multiple media reports Snider told family members his 70-year-old 
wife was ill with cancer a lifelong Elvis Presley fan Roberta decided before her death she wanted to make a trip to Graceland the Colonial Revival mansion in Memphis 
that stands as the holy land for the rock ‘n roll pioneer’s fan base the couple even planned to stay in the famous Elvis-inspired Days Inn near the estate where the 
rooms are plastered with Presley photos and the building sits around a guitar-shaped swimming pool on Jan 4 Snider called up a buddy named John Lapp whom he regularly
met with others at the Hartville Burger King for morning coffee Snider said he and his wife were setting off on the 725-mile trip to Tennessee the Canton Repository 
reported he’d be back in a few days Lapp however would later tell police the phone call was odd Lapp had known Snider for decades It was only the second time they had 
ever talked on the phone Three days later Snider rang up a number of family members back in Ohio from the road this time he had devastating news Roberta had died 
in Memphis Snider claimed according to the Repository the husband explained his wife had choked on her phlegm as they arrived in town Snider claimed he approached an 
EMS truck in a parking lot for help he said that he flagged an ambulance down and they checked her and put her in an ambulance and left with her and he didn’t know 
where she went Tennessee’s Benton County Sheriff Kenny Christopher later told the Memphis Commercial Appeal the day after alerting his family about Roberta’s death 
— Jan 8 — Snider was back at the Burger King in Hartville for his morning coffee he repeated the story to the regulars a day later Roberta’s family contacted the 
Hartville police for help locating the dead woman in Tennessee but authorities down south were stumped EMS ambulance services and the Memphis Medical Examiner’s 
Office all had no record of Roberta’s body nor did they have a Jane Doe matching her details the Repository reported Investigators however did pull credit card data 
showing Snider had actually made the trip Police followed up by obtaining security footage from a motel in Kentucky the images captured Snider’s truck at the 
establishment again proving he had been on the road but he was alone Roberta wasn’t in the car on Jan 14 Hartville Police Chief Dordea and another officer 
confronted Snider with the Kentucky motel photos according to the Repository the husband’s story then changed Roberta had died on the road Snider now claimed 
but the death had occurred back in Ohio between Columbus and Cincinnati Snider allegedly told investigators he wrapped his dead wife’s corpse in two trash bags 
placed her in his truck’s bed and continued on to Tennessee so Roberta could have one last posthumous trip to Graceland after checking into the Days Inn Snider 
said he left the motel in the early morning and threw his wife’s body off a bridge on Interstate 40 into the Tennessee River Police in Ohio again reached out to 
their counterparts in Tennessee the police chief in Hartville contacted me and said they were getting a lot of conflicting stories from Mr Snider Sheriff Christopher 
told the Commercial Appeal Our Benton County rescue squad they went out and drug underneath the bridge and drove over the area searching with boats for about five to 
six miles and they haven’t found anything but within days Snider’s new account crumbled according to Ohiocom credit card receipts again led investigators to a gas 
station in Wooster Ohio where Snider had stopped in the first hours of his trip Security footage showed the husband was alone torpedoing his claim his wife died 
further south on the highway the same day as police executed a search warrant of Snider’s home a police dogged sniffed out a torn fragment of a sweatshirt With 
Hartville splashed across the front the clothing item was dabbed in blood Roberta’s family later told investigators it was her favorite shirt Lab tests later 
confirmed it was 99-percent probability it was the missing woman’s blood but police did not immediately get the opportunity to question Snider he attempted suicide 
in late January putting off the next confrontation with Chief Dordea When the Hartville lawman finally confronted Snider in his hospital bed with the gas station 
photos and bloody sweatshirt Snider related a third version of Roberta’s death Ohiocom reported his wife had died at home in Ohio Snider now claimed Still intent 
on giving Roberta a final trip to Graceland he loaded the body into the truck for Tennessee again he claimed he had driven to Memphis then dumped his wife into the 
Tennessee River Although police suspicion now had fastened firmly around Snider there was little they could do to advance a criminal casedown in Tennessee law 
enforcement had yet to turn up a body So after his release from the hospital Snider returned to Hartville and his morning rounds of coffee at the Burger King as 
prosecutors revealed at the hearing this week not long after his hospital stay Snider was befriended by a woman How the two met has not been revealed but the two 
began bonding over coffee at local fast food restaurants their conversations growing intimate the woman told Snider she was caring for her dying mother but the 
relationship was strained Eventually she confided in Snider that she wished her mother would die and even told him that she thought about killing her mother prosecutors
explained Monday according to the Repository the woman’s confession seemed to deepen the tie Snider proposed the two should get married so the woman could collect his 
pension In turn the woman told Snider they couldn’t continue their relationship unless he told her the truth about his wife’s death Ohiocom reported he relented
according to Ohiocom Snider told the woman he and Roberta had fought on the night of Jan 2 the next morning he went for his morning coffee at Burger King When he 
returned to the house Roberta was sleeping on a love seat in the living room Snider said he placed a cloth over her head and bashed her skull with a two-pound hammer 
Snider claimed he disposed of the body murder weapon and other blood-splattered pieces of evidence on the long ride to Graceland What Snider did not realize was the 
woman sharing his secrets over coffee was not actually another fast food lonely heart but an undercover police officer who had just obtained the evidentiary ammunition 
needed for a prosecution on April 20 Hartville Chief Dordea arrested Snider at his usual morning hangout on Monday as prosecutors laid out the final account of Roberta’s
death the defendant also briefly addressed the court I put her in the Tennessee River Snider said That’s where I put her Sitting down on PEOPLE’s Chatter talk show 
Tuesday Elfman 46 who has been a Scientologist for 28 years revealed how she handles the criticism of the religion Well I’ve been a Scientologist for 28 years and 
that’s a huge part of what helps us keep our communication going and our relationship said Elfman who shares two children — sons Story 11 and Easton 8 — with husband 
Bodhi Elfman to whom she has been married since 1995 We’ve never cheated on each other we’ve never broken up We hang in there she said Raising children maintaining my 
sanity in a crazy world Our world is crazy it’s getting crazier and Hollywood is the ne plus ultra of crazy Elfman went on to talk about some of the courses that 
Scientology offers to its members I think that anything that works tends to get attacked she said While she couldn’t pinpoint a specific misconception about the 
church the star said It’s been so workable for me I use it every single day of my life and it keeps me energized and vivacious and happy I like literally have so 
much going on Why am I going to go: ‘You know let me put some negativity in my life Let me go see who’s being a bigot why would I search for bigotry in this world 
when it’s the one thing that’s been this huge help in my life to keep me sane and to raise great kids? said Elfman who encouraged people to not believe everything 
they hear about Scientology Elfman is one of numerous Scientologists in Hollywood — including Tom Cruise Erika Christensen Elisabeth Moss and Danny Masterson — 
though the church has been criticized in recent years by some prominent former members Leah Remini famously left Scientology in 2013 and has since dedicated her 
career to exposing what she claims to be a system of lies and abuse within Scientology I’ve been given a second chance at life and so has my family It’s like a 
rebirth the actress told PEOPLE in November 2015 about leaving Scientology I am lucky and blessed Like Remini Paul Haggis also left Scientology Haggis made an 5
appearance on Remini’s A&E series Scientology and the Aftermath series last year where he discussed his departure from the church in 2009 after 35 years as a member
""".split()

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