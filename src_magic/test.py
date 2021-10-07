import torch
from transformers import *
from emotional_dataset import *
import argparse
from modeling_emotional_gpt import EmotionalGPT2LMHeadModel

tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
config = GPT2Config.from_pretrained('gpt2-medium')
model = EmotionalGPT2LMHeadModel.from_pretrained('gpt2-medium',from_tf=False,config=config)
print(sum([p.numel() for p in model.parameters()]))

#tokenizer.add_special_tokens({'pad_token':'<|endoftext|>'})
#tokenizer.add_tokens(['<BOS>', '<EOS>'])

'''
parser = argparse.ArgumentParser()
parser.add_argument('--cmt_len', default=20)
parser.add_argument('--num_cmts', default=6)
args = parser.parse_args()
d = ICEDataset(tokenizer, args, "/data/ICE/train/trainval_liwc_6x6_2011_NoBadImg.tsv", "/data/ICE/train/group_feat_training", "train")
d_sampler = RandomSampler(d)
train_dataloader = DataLoader(d, sampler=d_sampler, batch_size=1)
img, liwc, inputs, labels = next(iter(train_dataloader))

print(liwc.size())

inputs = inputs
#.cuda()
labels = labels
#.cuda()
img = img.unsqueeze(1)
#.cuda()
liwc = liwc.unsqueeze(1)
#.cuda()

print(inputs)
print(inputs.size())
print(labels)
print(labels.size())

#.cuda()

cmt_i=1

outputs = model((img, liwc[:,:,cmt_i,:].contiguous()), 
    inputs[:,:cmt_i*args.cmt_len].contiguous(), 
    inputs[:,cmt_i*args.cmt_len:(cmt_i+1)*args.cmt_len].contiguous(), 
    labels=labels[:,cmt_i*args.cmt_len:(cmt_i+1)*args.cmt_len].contiguous())

'''