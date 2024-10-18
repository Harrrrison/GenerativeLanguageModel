import torch
import torch.nn as nn
from torch.nn import functional as F
import mmap
import random 
import pickle
import argparse
'''
parser = argparse.ArgumentParser(description='Demo')

parser.add_argument('-batch_size', type=str, required=True, help='Please provide a batch_size')

args = parser.parse_args()

print(f'batch size: {args.batch_size}')
'''

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

# hyper params 
block_size = 64
batch_size = 128

max_iters = 3000
# eval_interval = 2500
learning_rate = 3e-4
eval_iters = 100
#eval_interval = 500
dropout = 0.2 # prevents over fitting 
n_embd = 384 # may be too big for PC --> this creates a vector for each word/char about its relevence
# take sad and happy sad may be [0.1, 0.8] say the first index is the positivity of the word
# and the second index is if its showing some sort of emotion, this helps us classify words ish
n_layer = 8 # each of these higher elarns more
n_head = 8
dropout = 0.2

chars = ''
with open('vocab.txt', 'r', encoding='utf-8') as f: # this is our vocab
    text = f.read()
    chars = sorted(list(set(text)))

vocab_size = len(chars)

string_to_int = { ch:i for i,ch in enumerate(chars) }
int_to_string = { i:ch for i,ch in enumerate(chars) }

encode = lambda s: [string_to_int[c] for c in s]
decode = lambda l: ''.join([int_to_string[i] for i in l])

def get_random_chunk(split):
    filename = "train_split.txt" if split == 'train' else "val_split.txt"
    with open(filename, 'rb') as f:
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            # Determine the file size and a random position to start reading
            file_size = len(mm)
            start_pos = random.randint(0, (file_size) - block_size*batch_size)

            # Seek to the random position and read the block of text
            mm.seek(start_pos)
            block = mm.read(block_size*batch_size-1)

            # Decode the block to a string, ignoring any invalid byte sequences
            decoded_block = block.decode('utf-8', errors='ignore').replace('\r', '')
            
            # Train and test splits
            data = torch.tensor(encode(decoded_block), dtype=torch.long)
            
    return data

def get_batch(split):
    data = get_random_chunk(split)
    ix = torch.randint(len(data) - block_size, (batch_size,))
   # print(ix)
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

class Head(nn.Module):

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # inpout of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        B,T,C = x.shape
        k = self.key(x)
        q = self.query(x)
        # compute attention scores
        wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5 # (B,T,hs) @ (B, hs, T) -> (B,T,T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B,T,T)
        wei = F.softmax(wei, dim=-1) # (B,T,T)
        wei = self.dropout(wei)
        # perform weighted aggregation of values
        v = self.value(x)
        out = wei @ v
        return out


class MultiHeadAttention(nn.Module):

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd) # adds in another learnable param
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1) # concats each head toghether along the last dimetion (B,T,F) -> (B,T,[h1, h1, h1, h1, h2, h2, h2, h2, ...])
        out = self.dropout(self.proj(out))
        return out


class FeedForward(nn.Module):

    def __init__ (self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(), # looks at a number and if <0 num =0 else num stays the same 
            nn.Linear(4* n_embd, n_embd), 
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):

    def __init__(self, n_embd, n_head):

        super().__init__()
        head_size = n_embd//n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        y = self.sa(x)
        x = self.ln1(x+y)
        y = self.ffwd(x)
        x = self.ln2(x+y)
        return x


class GPTLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embeding_table = nn.Embedding(vocab_size, n_embd)
        self.positonal_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)]) # howmany decoder blocks
        
        self.ln_f = nn.LayerNorm(n_embd) # can go and experiment with differnt norms
        self.lm_head = nn.Linear(n_embd, vocab_size) # makes it softMax workable

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            

    def forward(self, index, targets=None):
        ''' This is for the behind the scenes shit helping us understand what is going on under the hood --> also a lot easier to debug
        What are the logits:
        a bunch of normaized floating point numbers
        we sum the numbers and then div each number in the set buy the total (normalization) -> this gives us a prob dist of what we want to predict
        '''  
        B, T = index.shape

        tok_emb = self.token_embeding_table(index)
        pos_emb = self.positonal_embedding_table(torch.arange(T, device=device)) # T, C
        x = tok_emb + pos_emb # (B, T, C)
        x = self.blocks(x) # (B, T, C)
        x = self.ln_f(x) # (B, T, C)
        logits = self.lm_head(x) # (B, T, Vocab_size)
        
        if targets is None:
            loss = None
        else:
    # What does view do: al;lwos us to unpack with.shapoe and then pack back totgether
            B, T, C = logits.shape # Batch, Time, Channels
            logits = logits.view(B*T, C) #The batch and time arnt suepr important so we can blend them together
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets) # we use veiw to ensure that the shape that this funct expects is met -> it exprect b by c bt t 
            

        return logits, loss

    def generate(self, index, max_new_tokens):

        for _ in range(max_new_tokens):
            # get predictions
            logits, loss = self.forward(index)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes b, c
            # apply softmax tro get probs
            probs = F.softmax(logits, dim=-1) # b,c
            #sample from the distribution
            index_next = torch.multinomial(probs, num_samples=1) # b, 1
            # append the sampled index to the running sequence
            index = torch.cat((index, index_next), dim=1) # B, T+1
        return index

model = GPTLanguageModel(vocab_size)
'''
print('loading model parameters...')
with open('model-01.pkl', 'rb') as f:
    model = pickle.load(f)
print('loaded successfully!')
'''
m = model.to(device)

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):
    # print(iter)
    if iter % eval_iters == 0:
        losses = estimate_loss()
        print(f"step: {iter}, train loss: {losses['train']:.3f}, val loss: {losses['val']:.3f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model.forward(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
print(loss.item())

with open('model-01.pkl', 'wb') as f:
    pickle.dump(model,f)
print('model saved')

