# %%
# First step is to import the needed libraries
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
import random
import numpy as np
import pickle
from tqdm import tqdm
# %matplotlib inline
from sklearn.metrics import f1_score,accuracy_score
import math
import re
from torch.utils.data import Dataset, DataLoader

# %%
# in this section we define static values and variables for ease of access and testing
_fn="final" # file unique id for saving and loading models
bert_base='./bert-base-uncased/'
bert_large='./bert-large-uncased/'

snips_train="./dataset/snips_train.iob"
snips_test="./dataset/snips_test.iob"
atis_train="./dataset/atis-2.train.w-intent.iob"
atis_test="./dataset/atis-2.test.w-intent.iob"
#ENV variables directly affect the model's behaviour
ENV_DATASET_TRAIN=atis_train
ENV_DATASET_TEST=atis_test

ENV_BERT_ID_CLS=False # use cls token for id classification
ENV_EMBEDDING_SIZE=768# dimention of embbeding, bertbase=768,bertlarge&elmo=1024
ENV_BERT_ADDR=bert_base
ENV_SEED=1331
ENV_CNN_FILTERS=128
ENV_CNN_KERNELS=4
ENV_HIDDEN_SIZE=ENV_CNN_FILTERS*ENV_CNN_KERNELS

#these are related to training
BATCH_SIZE=16
LENGTH=60
STEP_SIZE=50

# you must use cuda to run this code. if this returns false, you can not proceed.
USE_CUDA = torch.cuda.is_available()
if USE_CUDA:
    print("You are using cuda. Good!")
else:
    print('You are NOT using cuda! Some problems may occur.')

torch.manual_seed(ENV_SEED)
random.seed(ENV_SEED)

# %% [markdown]
# implement dataloader

# %%

#this function converts tokens to ids and then to a tensor
def prepare_sequence(seq, to_ix):
    idxs = list(map(lambda w: to_ix[w] if w in to_ix.keys() else to_ix["<UNK>"], seq))
    tensor = Variable(torch.LongTensor(idxs)).cuda() if USE_CUDA else Variable(torch.LongTensor(idxs))
    return tensor
# this function turns class text to id
def prepare_intent(intent, to_ix):
    idxs = to_ix[intent] if intent in to_ix.keys() else to_ix["UNKNOWN"]
    return idxs
# converts numbers to <NUM> TAG
def number_to_tag(txt):
    return "<NUM>" if txt.isdecimal() else txt

# Here we remove multiple spaces and punctuation which cause errors in tokenization for bert & elmo.
def remove_punc(mlist):
    mlist = [re.sub(" +"," ",t.split("\t")[0][4:-4]) for t in mlist] # remove spaces down to 1
    temp_train_tokens = []
    # punct remove example:  play samuel-el jackson from 2009 - 2010 > play samuelel jackson from 2009 - 2010
    for row in mlist:
        tokens = row.split(" ")
        newtokens = []
        for token in tokens:
            newtoken = re.sub(r"[.,'\"\\/\-:&’—=–官方杂志¡…“”~%]",r"",token) # remove punc
            newtoken = re.sub(r"[楽園追放�]",r"A",newtoken)
            newtokens.append(newtoken if len(token)>1 else token)
        if newtokens[-1]=="":
            newtokens.pop(-1)
        if newtokens[0]=="":
            newtokens.pop(0)
        temp_train_tokens.append(" ".join(newtokens))
    return temp_train_tokens
# this function returns the main tokens so that we can apply tagging on them. see original paper.
def get_subtoken_mask(current_tokens,bert_tokenizer):
    temp_mask = []
    for i in current_tokens:
        temp_row_mask = []
        temp_row_mask.append(False) # for cls token
        temp = bert_tokenizer.tokenize(i)
        for j in temp:
            temp_row_mask.append(j[:2]!="##")
        while len(temp_row_mask)<LENGTH:
            temp_row_mask.append(False)
        temp_mask.append(temp_row_mask)
        if sum(temp_row_mask)!=len(i.split(" ")):
            print(f"inconsistent:{temp}")
            print(i)
            print(sum(temp_row_mask))
            print(len(i.split(" ")))
    return torch.tensor(temp_mask).cuda() if USE_CUDA else torch.tensor(temp_mask).cpu()

flatten = lambda l: [number_to_tag(item) for sublist in l for item in sublist]

# %% [markdown]
# # Data load and Preprocessing

# %%


# %%
def tokenize_dataset(dataset_address):
    # added tokenizer and tokens for
    bert_tokenizer = torch.hub.load(ENV_BERT_ADDR, 'tokenizer', ENV_BERT_ADDR,verbose=False,source="local")#38toks snips,52Atis
    ##open database and read line by line
    dataset = open(dataset_address,"r").readlines()
    print("example input:"+dataset[0])
    ##remove last character of lines -\n- in train file
    dataset = [t[:-1] for t in dataset]
    #converts string to array of tokens + array of tags + target intent [array with x=3 and y dynamic]
    dataset_tokens = remove_punc(dataset)
    dataset_subtoken_mask = get_subtoken_mask(dataset_tokens,bert_tokenizer)
    dataset_toks = bert_tokenizer.batch_encode_plus(dataset_tokens,max_length=LENGTH,add_special_tokens=True,return_tensors='pt'
                                                  ,return_attention_mask=True , padding='max_length',truncation=True)
    dataset = [[re.sub(" +"," ",t.split("\t")[0]).split(" "),t.split("\t")[1].split(" ")[:-1],t.split("\t")[1].split(" ")[-1]] for t in dataset]
    #removes BOS, EOS from array of tokens and tags
    dataset = [[t[0][1:-1],t[1][1:],t[2]] for t in dataset]
    return dataset, dataset_subtoken_mask,dataset_toks
train,train_subtoken_mask,train_toks = tokenize_dataset(ENV_DATASET_TRAIN)
test, test_subtoken_mask, test_toks = tokenize_dataset(ENV_DATASET_TEST)

# %%
#convert above array to separate lists
seq_in,seq_out, intent = list(zip(*train))
seq_in_test,seq_out_test, intent_test = list(zip(*test.copy()))

# %%
# Create Sets of unique tokens
vocab = set(flatten(seq_in))
slot_tag = set(flatten(seq_out))
intent_tag = set(intent)

# %%
# adds paddings
sin=[] #padded input tokens
sout=[] # padded output translated tags
sin_test=[] #padded input tokens
sout_test=[] # padded output translated tags
## adds padding inside input tokens
def add_paddings(seq_in,seq_out):
    sin=[]
    sout=[]
    for i in range(len(seq_in)):
        temp = seq_in[i]
        if len(temp)<LENGTH:
            while len(temp)<LENGTH:
                temp.append('<PAD>')
        else:
            temp = temp[:LENGTH]
        sin.append(temp)
        # add padding inside output tokens
        temp = seq_out[i]
        if len(temp)<LENGTH:
            while len(temp)<LENGTH:
                temp.append('<PAD>')
        else:
            temp = temp[:LENGTH]
        sout.append(temp)
    return sin,sout
sin,sout=add_paddings(seq_in,seq_out)
sin_test,sout_test=add_paddings(seq_in_test,seq_out_test)

# %%
# making dictionary (token:id), initial value
word2index = {'<PAD>': 0, '<UNK>':1,'<BOS>':2,'<EOS>':3,'<NUM>':4}
# add rest of token list to dictionary
for token in vocab:
    if token not in word2index.keys():
        word2index[token]=len(word2index)
#make id to token list ( reverse )
index2word = {v:k for k,v in word2index.items()}

# initial tag2index dictionary
tag2index = {'<PAD>' : 0,'<BOS>':2,'<UNK>':1,'<EOS>':3}
# add rest of tag tokens to list
for tag in slot_tag:
    if tag not in tag2index.keys():
        tag2index[tag] = len(tag2index)
# making index to tag
index2tag = {v:k for k,v in tag2index.items()}

#initialize intent to index
intent2index={'UNKNOWN':0}
for ii in intent_tag:
    if ii not in intent2index.keys():
        intent2index[ii] = len(intent2index)
index2intent = {v:k for k,v in intent2index.items()}

# %% [markdown]
# # Loading PreTrained Embeddings

# %%
#defining datasets.
def remove_values_from_list(the_list, val):
   return [value for value in the_list if value != val]

class NLUDataset(Dataset):
    def __init__(self, sin,sout,intent,input_ids,attention_mask,token_type_ids,subtoken_mask):
        self.sin = [prepare_sequence(temp,word2index) for temp in sin]
        self.sout = [prepare_sequence(temp,tag2index) for temp in sout]
        self.intent = Variable(torch.LongTensor([prepare_intent(temp,intent2index) for temp in intent])).cuda() if USE_CUDA else Variable(torch.LongTensor([prepare_intent(temp,intent2index) for temp in intent])).cpu()
        self.input_ids=input_ids.cuda() if USE_CUDA else input_ids.cpu()
        self.attention_mask=attention_mask.cuda() if USE_CUDA else attention_mask.cpu()
        self.token_type_ids=token_type_ids.cuda() if USE_CUDA else token_type_ids.cpu()
        self.subtoken_mask=subtoken_mask.cuda() if USE_CUDA else subtoken_mask.cpu()
        self.x_mask = [Variable(torch.BoolTensor(tuple(map(lambda s: s ==0, t )))).cuda() if USE_CUDA else Variable(torch.BoolTensor(tuple(map(lambda s: s ==0, t )))).cpu() for t in self.sin]
    def __len__(self):
        return len(self.intent)
    def __getitem__(self, idx):
        sample = self.sin[idx],self.sout[idx],self.intent[idx],self.input_ids[idx],self.attention_mask[idx],self.token_type_ids[idx],self.subtoken_mask[idx],self.x_mask[idx]
        return sample
#making single list
train_data=NLUDataset(sin,sout,intent,train_toks['input_ids'],train_toks['attention_mask'],train_toks['token_type_ids'],train_subtoken_mask)
test_data=NLUDataset(sin_test,sout_test,intent_test,test_toks['input_ids'],test_toks['attention_mask'],test_toks['token_type_ids'],test_subtoken_mask)
train_data = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
test_data = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

# %%
# we put all tags inside of the batch in a flat array for F1 measure.
# we use masking so that we only non PAD tokens are counted in f1 measurement
def mask_important_tags(predictions,tags,masks):
    result_tags=[]
    result_preds=[]
    for pred,tag,mask in zip(predictions.tolist(),tags.tolist(),masks.tolist()):
        #index [0] is to get the data
        for p,t,m in zip(pred,tag,mask):
            if not m:
                result_tags.append(p)
                result_preds.append(t)
        #result_tags.pop()
        #result_preds.pop()
    return result_preds,result_tags


# %% [markdown]
# # Modeling

# %%
# generates transformer mask
def generate_square_subsequent_mask(sz: int) :
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)
def generate_square_diagonal_mask(sz: int) :
    """Generates a matrix which there are zeros on diag and other indexes are -inf."""
    return torch.triu(torch.ones(sz,sz)-float('inf'), diagonal=1)+torch.tril(torch.ones(sz,sz)-float('inf'), diagonal=-1)
# positional embedding used in transformers
class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


#start of the shared encoder
class BertLayer(nn.Module):
    def __init__(self):
        super(BertLayer, self).__init__()
        self.bert_model = torch.hub.load(ENV_BERT_ADDR, 'model', ENV_BERT_ADDR,source="local")

    def forward(self, bert_info=None):
        (bert_tokens, bert_mask, bert_tok_typeid) = bert_info
        bert_encodings = self.bert_model(bert_tokens, bert_mask, bert_tok_typeid)
        bert_last_hidden = bert_encodings['last_hidden_state']
        bert_pooler_output = bert_encodings['pooler_output']
        return bert_last_hidden, bert_pooler_output


class Encoder(nn.Module):
    def __init__(self, p_dropout=0.5):
        super(Encoder, self).__init__()
        self.filter_number = ENV_CNN_FILTERS
        self.kernel_number = ENV_CNN_KERNELS  # tedad size haye filter : 2,3,5 = 3
        self.embedding_size = ENV_EMBEDDING_SIZE
        self.activation = nn.ReLU()
        self.p_dropout = p_dropout
        self.softmax = nn.Softmax(dim=1)
        self.conv1 = nn.Conv1d(in_channels=self.embedding_size, out_channels=self.filter_number, kernel_size=(2,),
                               padding="same", padding_mode="zeros")
        self.conv2 = nn.Conv1d(in_channels=self.embedding_size, out_channels=self.filter_number, kernel_size=(3,),
                               padding="same", padding_mode="zeros")
        self.conv3 = nn.Conv1d(in_channels=self.embedding_size, out_channels=self.filter_number, kernel_size=(5,),
                               padding="same", padding_mode="zeros")
        self.conv4 = nn.Conv1d(in_channels=self.embedding_size, out_channels=self.filter_number, kernel_size=(1,),
                               padding="same", padding_mode="zeros")

    def forward(self, bert_last_hidden):
        trans_embedded = torch.transpose(bert_last_hidden, dim0=1, dim1=2)
        convolve1 = self.activation(self.conv1(trans_embedded))
        convolve2 = self.activation(self.conv2(trans_embedded))
        convolve3 = self.activation(self.conv3(trans_embedded))
        convolve4 = self.activation(self.conv4(trans_embedded))
        convolve1 = torch.transpose(convolve1, dim0=1, dim1=2)
        convolve2 = torch.transpose(convolve2, dim0=1, dim1=2)
        convolve3 = torch.transpose(convolve3, dim0=1, dim1=2)
        convolve4 = torch.transpose(convolve4, dim0=1, dim1=2)
        output = torch.cat((convolve4, convolve1, convolve2, convolve3), dim=2)
        return output


# %%
#Middle
class Middle(nn.Module):
    def __init__(self ,p_dropout=0.5):
        super(Middle, self).__init__()
        self.activation = nn.ReLU()
        self.p_dropout = p_dropout
        self.softmax = nn.Softmax(dim=1)
        #Transformer
        nlayers = 2  # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
        self.pos_encoder = PositionalEncoding(ENV_HIDDEN_SIZE, dropout=0.1)
        encoder_layers = nn.TransformerEncoderLayer(ENV_HIDDEN_SIZE, nhead=2,batch_first=True, dim_feedforward=2048 ,activation="relu", dropout=0.1)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, nlayers,enable_nested_tensor=False)
        self.transformer_mask = generate_square_subsequent_mask(LENGTH).cuda() if USE_CUDA else generate_square_subsequent_mask(LENGTH).cpu()

    def forward(self, fromencoder,input_masking,training=True):
        src = fromencoder * math.sqrt(ENV_HIDDEN_SIZE)
        src = self.pos_encoder(src)
        output = (self.transformer_encoder(src,src_key_padding_mask=input_masking)) # outputs probably
        return output

# %%
#start of the decoder
class Decoder(nn.Module):

    def __init__(self,slot_size,intent_size,dropout_p=0.5):
        super(Decoder, self).__init__()
        self.slot_size = slot_size
        self.intent_size = intent_size
        self.dropout_p = dropout_p
        self.softmax= nn.Softmax(dim=1)
        # Define the layers
        self.embedding = nn.Embedding(self.slot_size, ENV_HIDDEN_SIZE)
        self.activation = nn.ReLU()
        self.dropout1 = nn.Dropout(self.dropout_p)
        self.dropout2 = nn.Dropout(self.dropout_p)
        self.dropout3 = nn.Dropout(self.dropout_p)
        self.slot_trans = nn.Linear(ENV_HIDDEN_SIZE, self.slot_size)
        self.intent_out = nn.Linear(ENV_HIDDEN_SIZE,self.intent_size)
        self.intent_out_cls = nn.Linear(ENV_EMBEDDING_SIZE,self.intent_size) # dim of bert
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=ENV_HIDDEN_SIZE, nhead=2,batch_first=True,dim_feedforward=300 ,activation="relu")
        self.transformer_decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=2)
        self.transformer_mask = generate_square_subsequent_mask(LENGTH).cuda() if USE_CUDA else generate_square_subsequent_mask(LENGTH).cpu()
        self.transformer_diagonal_mask = generate_square_diagonal_mask(LENGTH).cuda() if USE_CUDA else generate_square_diagonal_mask(LENGTH).cpu()
        self.pos_encoder = PositionalEncoding(ENV_HIDDEN_SIZE, dropout=0.1)
        self.self_attention = nn.MultiheadAttention(embed_dim=ENV_HIDDEN_SIZE
                                                    ,num_heads=8,dropout=0.1
                                                    ,batch_first=True)
        self.layer_norm = nn.LayerNorm(ENV_HIDDEN_SIZE)


    def forward(self, input,encoder_outputs,encoder_maskings,bert_subtoken_maskings=None,infer=False):
        # encoder outputs: BATCH,LENGTH,Dims (16,60,1024)
        batch_size = encoder_outputs.shape[0]
        length = encoder_outputs.size(1) #for every token in batches
        embedded = self.embedding(input)

        #print("NOT CLS")
        encoder_outputs2=encoder_outputs
        context,attn_weight = self.self_attention(encoder_outputs2,encoder_outputs2,encoder_outputs2
                                                  ,key_padding_mask=encoder_maskings)
        encoder_outputs2 = self.layer_norm(self.dropout2(context))+encoder_outputs2
        sum_mask = (~encoder_maskings).sum(1).unsqueeze(1)
        sum_encoder = ((((encoder_outputs2)))*((~encoder_maskings).unsqueeze(2))).sum(1)
        intent_score = self.intent_out(self.dropout1(sum_encoder/sum_mask)) # B,D


        newtensor = torch.cuda.FloatTensor(batch_size, length,ENV_HIDDEN_SIZE).fill_(0.) if USE_CUDA else torch.FloatTensor(batch_size, length,ENV_HIDDEN_SIZE).fill_(0.)  # size of newtensor same as original
        for i in range(batch_size): # per batch
            newtensor_index=0
            for j in range(length): # for each token
                if bert_subtoken_maskings[i][j].item()==1:
                    newtensor[i][newtensor_index] = encoder_outputs[i][j]
                    newtensor_index+=1

        if infer==False:
            embedded=embedded*math.sqrt(ENV_HIDDEN_SIZE)
            embedded = self.pos_encoder(embedded)
            zol = self.transformer_decoder(tgt=embedded,memory=newtensor
                                           ,memory_mask=self.transformer_diagonal_mask
                                           ,tgt_mask=self.transformer_mask)

            scores = self.slot_trans(self.dropout3(zol))
            slot_scores = F.log_softmax(scores,dim=2)
        else:
            bos = Variable(torch.LongTensor([[tag2index['<BOS>']]*batch_size])).cuda().transpose(1,0) if USE_CUDA else Variable(torch.LongTensor([[tag2index['<BOS>']]*batch_size])).cpu().transpose(1,0)
            bos = self.embedding(bos)
            tokens=bos
            for i in range(length):
                temp_embedded=tokens*math.sqrt(ENV_HIDDEN_SIZE)
                temp_embedded = self.pos_encoder(temp_embedded)
                zol = self.transformer_decoder(tgt=temp_embedded,
                                               memory=newtensor,
                                               tgt_mask=self.transformer_mask[:i+1,:i+1],
                                               memory_mask=self.transformer_diagonal_mask[:i+1,:]
                                               )
                scores = self.slot_trans(self.dropout3(zol))
                softmaxed = F.log_softmax(scores,dim=2)
                #the last token is apended to vectors
                _,input = torch.max(softmaxed,2)
                newtok = self.embedding(input)
                tokens=torch.cat((bos,newtok),dim=1)
            slot_scores = softmaxed

        return slot_scores.view(input.size(0)*length,-1), intent_score

# %% [markdown]
# # Training
# 
# 

# %%


# %%
bert_layer = BertLayer()
encoder = Encoder(len(word2index))
middle = Middle()
decoder = Decoder(len(tag2index),len(intent2index))
if USE_CUDA:
    encoder = encoder.cuda()
    decoder = decoder.cuda()
    middle = middle.cuda()
    bert_layer.cuda()

loss_function_1 = nn.CrossEntropyLoss(ignore_index=0)
loss_function_2 = nn.CrossEntropyLoss()
dec_optim = optim.AdamW(decoder.parameters(),lr=0.0001)
enc_optim = optim.AdamW(encoder.parameters(),lr=0.001)
ber_optim = optim.AdamW(bert_layer.parameters(),lr=0.0001)
mid_optim = optim.AdamW(middle.parameters(), lr=0.0001)
enc_scheduler = torch.optim.lr_scheduler.StepLR(enc_optim, 1, gamma=0.96)
dec_scheduler = torch.optim.lr_scheduler.StepLR(dec_optim, 1, gamma=0.96)
mid_scheduler = torch.optim.lr_scheduler.StepLR(mid_optim, 1, gamma=0.96)
ber_scheduler = torch.optim.lr_scheduler.StepLR(ber_optim, 1, gamma=0.96)

# %%
max_id_prec=0.
max_sf_f1=0.
max_id_prec_both=0.
max_sf_f1_both=0.

for step in tqdm(range(STEP_SIZE)):
    losses=[]
    id_precision=[]
    sf_f1=[]

    ### TRAIN
    encoder.train() # set to train mode
    middle.train()
    decoder.train()
    bert_layer.train()
    for i,(x,tag_target,intent_target,bert_tokens,bert_mask,bert_toktype,subtoken_mask,x_mask) in enumerate(train_data):
        batch_size=tag_target.size(0)
        bert_layer.zero_grad()
        encoder.zero_grad()
        middle.zero_grad()
        decoder.zero_grad()
        bert_hidden,bert_pooler = bert_layer(bert_info=(bert_tokens,bert_mask,bert_toktype))
        encoder_output = encoder(bert_last_hidden=bert_hidden)
        output = middle(encoder_output,bert_mask==0,training=True)
        start_decode = Variable(torch.LongTensor([[tag2index['<BOS>']]*batch_size])).cuda().transpose(1,0) if USE_CUDA else Variable(torch.LongTensor([[tag2index['<BOS>']]*batch_size])).cpu().transpose(1,0)
        start_decode = torch.cat((start_decode,tag_target[:,:-1]),dim=1)
        tag_score, intent_score = decoder(start_decode,output,bert_mask==0,bert_subtoken_maskings=subtoken_mask)
        loss_1 = loss_function_1(tag_score,tag_target.view(-1))
        loss_2 = loss_function_2(intent_score,intent_target)
        loss = loss_1+loss_2
        losses.append(loss.data.cpu().numpy() if USE_CUDA else loss.data.numpy())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(encoder.parameters(), 0.5)
        torch.nn.utils.clip_grad_norm_(middle.parameters(), 0.5)
        torch.nn.utils.clip_grad_norm_(decoder.parameters(), 0.5)
        torch.nn.utils.clip_grad_norm_(bert_layer.parameters(), 0.5)
        enc_optim.step()
        mid_optim.step()
        dec_optim.step()
        ber_optim.step()
        #print(bert_tokens[0])
        #print(tag_target[0])
        id_precision.append(accuracy_score(intent_target.detach().cpu(),torch.argmax(intent_score,dim=1).detach().cpu()))
        pred_list,target_list=mask_important_tags(torch.argmax(tag_score,dim=1).view(batch_size,LENGTH),tag_target,x_mask)
        sf_f1.append(f1_score(pred_list,target_list,average="micro",zero_division=0))
    #print report
    print("Step",step," batches",i," :")
    print("Train-")
    print(f"loss:{round(float(np.mean(losses)),4)}")
    print(f"SlotFilling F1:{round(float(np.mean(sf_f1)),3)}")
    print(f"IntentDet Prec:{round(float(np.mean(id_precision)),3)}")
    losses=[]
    sf_f1=[]
    id_precision=[]
    #scheduler.step()

    #### TEST
    encoder.eval() # set to test mode
    middle.eval()
    decoder.eval()
    bert_layer.eval()
    with torch.no_grad(): # to turn off gradients computation
        for i,(x,tag_target,intent_target,bert_tokens,bert_mask,bert_toktype,subtoken_mask,x_mask) in enumerate(test_data):
            batch_size=tag_target.size(0)
            encoder.zero_grad()
            middle.zero_grad()
            decoder.zero_grad()
            bert_layer.zero_grad()
            bert_hidden,bert_pooler = bert_layer(bert_info=(bert_tokens,bert_mask,bert_toktype))
            encoder_output = encoder(bert_last_hidden=bert_hidden)
            output = middle(encoder_output,bert_mask==0,training=True)
            start_decode = Variable(torch.LongTensor([[tag2index['<BOS>']]*batch_size])).cuda().transpose(1,0) if USE_CUDA else Variable(torch.LongTensor([[tag2index['<BOS>']]*batch_size])).cpu().transpose(1,0)
            tag_score, intent_score = decoder(start_decode,output,bert_mask==0,bert_subtoken_maskings=subtoken_mask,infer=True)
            loss_1 = loss_function_1(tag_score,tag_target.view(-1))
            loss_2 = loss_function_2(intent_score,intent_target)
            loss = loss_1+loss_2
            losses.append(loss.data.cpu().numpy() if USE_CUDA else loss.data.numpy()[0])
            id_precision.append(accuracy_score(intent_target.detach().cpu(),torch.argmax(intent_score,dim=1).detach().cpu()))
            pred_list,target_list=mask_important_tags(torch.argmax(tag_score,dim=1).view(batch_size,LENGTH),tag_target,x_mask)
            sf_f1.append(f1_score(pred_list,target_list,average="micro",zero_division=0))
    print("Test-")
    print(f"loss:{round(float(np.mean(losses)),4)}")
    print(f"SlotFilling F1:{round(float(np.mean(sf_f1)),4)}")
    print(f"IntentDet Prec:{round(float(np.mean(id_precision)),4)}")
    print("--------------")
    max_sf_f1 = max_sf_f1 if round(float(np.mean(sf_f1)),4)<=max_sf_f1 else round(float(np.mean(sf_f1)),4)
    max_id_prec = max_id_prec if round(float(np.mean(id_precision)),4)<=max_id_prec else round(float(np.mean(id_precision)),4)
    if max_sf_f1_both<=round(float(np.mean(sf_f1)),4) and max_id_prec_both<=round(float(np.mean(id_precision)),4):
        max_sf_f1_both=round(float(np.mean(sf_f1)),4)
        max_id_prec_both=round(float(np.mean(id_precision)),4)
        torch.save(bert_layer,f"models/ctran{_fn}-bertlayer.pkl")
        torch.save(encoder,f"models/ctran{_fn}-encoder.pkl")
        torch.save(middle,f"models/ctran{_fn}-middle.pkl")
        torch.save(decoder,f"models/ctran{_fn}-decoder.pkl")
    enc_scheduler.step()
    dec_scheduler.step()
    mid_scheduler.step()
    ber_scheduler.step()
print(f"max single SF F1: {max_sf_f1}")
print(f"max single ID PR: {max_id_prec}")
print(f"max mutual SF:{max_sf_f1_both}  PR: {max_id_prec_both}")


# %% [markdown]
# # Test
# 
# The following cells is for reviewing the performance of CTran.

# %%
# This cell reloads the best model during training from hard-drive.
bert_layer.load_state_dict(torch.load(f'models/ctran{_fn}-bertlayer.pkl').state_dict())
encoder.load_state_dict(torch.load(f'models/ctran{_fn}-encoder.pkl').state_dict())
middle.load_state_dict(torch.load(f'models/ctran{_fn}-middle.pkl').state_dict())
decoder.load_state_dict(torch.load(f'models/ctran{_fn}-decoder.pkl').state_dict())
if USE_CUDA:
    bert_layer = bert_layer.cuda()
    encoder = encoder.cuda()
    middle = middle.cuda()
    decoder = decoder.cuda()


# %%


# %%
global clipindex
clipindex=0
def removepads(toks,clip=False):
    global clipindex
    result = toks.copy()
    for i,t in enumerate(toks):
        if t=="<PAD>":
            result.remove(t)
        elif t=="<EOS>":
            result.remove(t)
            if not clip:
                clipindex=i
    if clip:
        result=result[:clipindex]
    return result

# %%
print("Example of model prediction on test dataset")
encoder.eval()
middle.eval()
decoder.eval()
bert_layer.eval()
with torch.no_grad():
    index = random.choice(range(len(test)))
    test_raw = test[index][0]
    bert_tokens = test_toks['input_ids'][index].unsqueeze(0).cuda() if USE_CUDA else test_toks['input_ids'][index].unsqueeze(0).cpu()
    bert_mask = test_toks['attention_mask'][index].unsqueeze(0).cuda() if USE_CUDA else test_toks['attention_mask'][index].unsqueeze(0).cpu()
    bert_toktype = test_toks['token_type_ids'][index].unsqueeze(0).cuda() if USE_CUDA else test_toks['token_type_ids'][index].unsqueeze(0).cpu()
    subtoken_mask = test_subtoken_mask[index].unsqueeze(0).cuda() if USE_CUDA else test_subtoken_mask[index].unsqueeze(0).cpu()
    test_in = prepare_sequence(test_raw,word2index)
    test_mask = Variable(torch.BoolTensor(tuple(map(lambda s: s ==0, test_in.data)))).cuda() if USE_CUDA else Variable(torch.ByteTensor(tuple(map(lambda s: s ==0, test_in.data)))).view(1,-1)
    start_decode = Variable(torch.LongTensor([[word2index['<BOS>']]*1])).cuda().transpose(1,0) if USE_CUDA else Variable(torch.LongTensor([[word2index['<BOS>']]*1])).transpose(1,0)
    test_raw = [removepads(test_raw)]
    bert_hidden,bert_pooler = bert_layer(bert_info=(bert_tokens,bert_mask,bert_toktype))
    encoder_output = encoder(bert_last_hidden=bert_hidden)
    output = middle(encoder_output,bert_mask==0)
    tag_score, intent_score = decoder(start_decode,output,bert_mask==0,bert_subtoken_maskings=subtoken_mask,infer=True)

    v,i = torch.max(tag_score,1)
    print("Sentence           : ",*test_raw[0])
    print("Tag Truth          : ", *test[index][1][:len(test_raw[0])])
    print("Tag Prediction     : ",*(list(map(lambda ii:index2tag[ii],i.data.tolist()))[:len(test_raw[0])]))
    v,i = torch.max(intent_score,1)
    print("Intent Truth       : ", test[index][2])
    print("Intent Prediction  : ",index2intent[i.data.tolist()[0]])

# %%
print("Instances where model predicted intent wrong")
encoder.eval()
middle.eval()
decoder.eval()
bert_layer.eval()
total_wrong_predicted_intents = 0
with torch.no_grad():
    for i in range(len(test)):
        index = i
        test_raw = test[index][0]
        bert_tokens = test_toks['input_ids'][index].unsqueeze(0).cuda() if USE_CUDA else test_toks['input_ids'][index].unsqueeze(0).cpu()
        bert_mask = test_toks['attention_mask'][index].unsqueeze(0).cuda() if USE_CUDA else test_toks['attention_mask'][index].unsqueeze(0).cpu()
        bert_toktype = test_toks['token_type_ids'][index].unsqueeze(0).cuda() if USE_CUDA else test_toks['token_type_ids'][index].unsqueeze(0).cpu()
        subtoken_mask = test_subtoken_mask[index].unsqueeze(0).cuda() if USE_CUDA else test_subtoken_mask[index].unsqueeze(0).cpu()
        test_in = prepare_sequence(test_raw,word2index)
        test_mask = Variable(torch.BoolTensor(tuple(map(lambda s: s ==0, test_in.data)))).cuda() if USE_CUDA else Variable(torch.ByteTensor(tuple(map(lambda s: s ==0, test_in.data)))).view(1,-1)
        # print(removepads(test_raw))
        start_decode = Variable(torch.LongTensor([[word2index['<BOS>']]*1])).cuda().transpose(1,0) if USE_CUDA else Variable(torch.LongTensor([[word2index['<BOS>']]*1])).transpose(1,0)
        test_raw = [removepads(test_raw)]
        bert_hidden,bert_pooler = bert_layer(bert_info=(bert_tokens,bert_mask,bert_toktype))
        encoder_output = encoder(bert_last_hidden=bert_hidden)
        output = middle(encoder_output,bert_mask==0)
        tag_score, intent_score = decoder(start_decode,output,bert_mask==0,bert_subtoken_maskings=subtoken_mask,infer=True)

        v,i = torch.max(intent_score,1)
        if test[index][2]!=index2intent[i.data.tolist()[0]]:
            v,i = torch.max(tag_score,1)
            print("Sentence           : ",*test_raw[0])
            print("Tag Truth          : ", *test[index][1][:len(test_raw[0])])
            print("Tag Prediction     : ",*list(map(lambda ii:index2tag[ii],i.data.tolist()))[:len(test_raw[0])])
            v,i = torch.max(intent_score,1)
            print("Intent Truth       : ", test[index][2])
            print("Intent Prediction  : ",index2intent[i.data.tolist()[0]])
            print("--------------------------------------")
            total_wrong_predicted_intents+=1

print("Total instances of wrong intent prediction is ",total_wrong_predicted_intents)

# %%



