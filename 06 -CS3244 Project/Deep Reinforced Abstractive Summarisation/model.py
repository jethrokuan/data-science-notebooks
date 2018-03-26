import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils.rnn import PackedSequence,pack_padded_sequence
from attention import Attention, IntraTempAttention
    
class Encoder(nn.Module):
    def __init__(self, input_size, embedding_size,hidden_size, n_layers=1,bidirec=False,dropout_p=0.5,use_cuda=False):
        super(Encoder, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = nn.Dropout(dropout_p)
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.use_cuda = use_cuda
        
        if bidirec:
            self.n_direction = 2 
            self.lstm = nn.LSTM(embedding_size, hidden_size, n_layers, batch_first=True,bidirectional=True)
        else:
            self.n_direction = 1
            self.lstm = nn.LSTM(embedding_size, hidden_size, n_layers, batch_first=True)
        
        self.init_weight()
        
    def cuda(self, device=None):
        self.use_cuda = True
        return self._apply(lambda t: t.cuda(device))

    def cpu(self):
        self.use_cuda = False
        return self._apply(lambda t: t.cpu())
    
    def init_weight(self):
        for name,param in self.named_parameters():
            if "weight" in name:
                param = nn.init.xavier_uniform(param)
            elif "bias" in name:
                param = nn.init.constant(param,0)
                
    def init_hidden(self,size):
        hidden = Variable(torch.zeros(self.n_layers*self.n_direction,size,self.hidden_size))
        cell = Variable(torch.zeros(self.n_layers*self.n_direction,size,self.hidden_size))
        if self.use_cuda:
            hidden = hidden.cuda()
            cell = cell.cuda()
        return hidden, cell
    
    def forward(self, inputs, input_lengths):
        """
        inputs : B,T (LongTensor)
        input_lengths : real lengths of input batch (list)
        """
        hidden = self.init_hidden(inputs.size(0))
        
        embedded = self.embedding(inputs)
        embedded = self.dropout(embedded)
        packed = pack_padded_sequence(embedded, input_lengths,batch_first=True)
        outputs, (hidden,cell) = self.lstm(packed, hidden)
        outputs, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(outputs,batch_first=True) # unpack (back to padded)
        
        if self.n_layers>1:
            if self.n_direction==2:
                hidden = hidden[-2:]
            else:
                hidden = hidden[-1]
        
        # B,T,D  / 1,B,D
        return outputs, torch.cat([h for h in hidden],1).unsqueeze(1).transpose(0,1)

    
class Decoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, n_layers=1,dropout_p=0.3,use_cuda=False):
        super(Decoder, self).__init__()
        
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        
        # Define the layers
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.dropout = nn.Dropout(dropout_p)
        
        self.lstm = nn.LSTM(embedding_size, hidden_size, n_layers,batch_first=True)
        self.linear = nn.Linear(hidden_size*3, input_size)
        self.dec_attention = Attention(hidden_size)
        self.enc_attention = IntraTempAttention(hidden_size)
        self.use_cuda = use_cuda
        self.init_weight()
        
    def cuda(self, device=None):
        self.use_cuda = True
        return self._apply(lambda t: t.cuda(device))

    def cpu(self):
        self.use_cuda = False
        return self._apply(lambda t: t.cpu())
    
    def init_weight(self):
        for name,param in self.named_parameters():
            if "weight" in name:
                param = nn.init.xavier_uniform(param)
            elif "bias" in name:
                param = nn.init.constant(param,0)
    
    def init_context(self,size):
        context = Variable(torch.zeros(self.n_layers,size,self.hidden_size))
        if self.use_cuda:
            context = context.cuda()
        return context
    
    
    def forward(self,inputs,hidden,max_length,encoder_outputs,encoder_lengths=None):
        """
        inputs : B,1 (LongTensor, START SYMBOL)
        hidden : B,1,D (FloatTensor, Last encoder hidden state)
        encoder_outputs : B,T,D
        encoder_lengths : B,T # list
        max_length : int, max length to decode
        """
        # Get the embedding of the current input word
        embedded = self.embedding(inputs)
        embedded = self.dropout(embedded)
        
        self.enc_attention.reset_step()
        cell = self.init_context(inputs.size(0))
        decoding_hiddens=[] # for Intra Decoding Attn
        decode=[]
        for i in range(max_length): 
            decoding_hiddens.append(hidden.transpose(0,1)) # B,1,D
            _, (hidden,cell) = self.lstm(embedded, (hidden,cell)) 
            
            # Intra Temporal Attention
            context_e = self.enc_attention(hidden.transpose(0,1), encoder_outputs, encoder_lengths)
            if i==0:
                context_d = self.init_context(inputs.size(0)).transpose(0,1)
            else:
                decoder_outputs = torch.cat(decoding_hiddens,1) # B,T_{t-1},D
                context_d = self.dec_attention(hidden.transpose(0,1),decoder_outputs)
                
            concat = torch.cat([hidden.transpose(0,1),context_e,context_d],2) # B,1,3D
            score = self.linear(concat.view(concat.size(0)*concat.size(1),-1)) # B,V
            decode.append(score)
            decoded = F.log_softmax(score,1)
            embedded = self.embedding(decoded.max(1)[1]).unsqueeze(1) # y_{t-1}
            embedded = self.dropout(embedded)
            
        #  column-wise concat, reshape!!
        scores = torch.cat(decode,1)
        return scores.view(inputs.size(0)*max_length,-1)


class Beam:
    def __init__(self,root,num_beam):
        """
        root : (score, hidden)
        """
        self.num_beam = num_beam
        
        score = F.log_softmax(root[0],1)
        s,i = score.topk(num_beam)
        s = s.data.tolist()[0]
        i = i.data.tolist()[0]
        i = [[ii] for ii in i]
        hiddens = [root[1] for _ in range(num_beam)]
        self.beams = list(zip(s,i,hiddens))
        self.beams = sorted(self.beams,key= lambda x:x[0], reverse=True)
        
    def select_k(self,siblings):
        """
        siblings : [score,hidden]
        """
        candits=[]
        for p_index,sibling in enumerate(siblings):
            parents = self.beams[p_index] # (cummulated score, list of sequence)
            score = F.log_softmax(sibling[0],1)
            s,i = score.topk(self.num_beam)
            scores = s.data.tolist()[0]
            #scores = [scores[i-1]+i for i in range(1,self.num_beam+1)] # penalize siblings
            indices = i.data.tolist()[0]

            candits.extend([(parents[0]+scores[i],parents[1]+[indices[i]],sibling[1]) for i in range(len(scores))])
        
        candits = sorted(candits,key= lambda x:x[0], reverse=True)

        self.beams = candits[:self.num_beam]
        
        # last_input, hidden
        return [[Variable(torch.LongTensor([b[1][-1]])).view(1,-1),b[2]] for b in self.beams]
        
    def get_best_seq(self):
        return self.beams[0][1]
    
    def get_next_nodes(self):
        return [[Variable(torch.LongTensor([b[1][-1]])).view(1,-1),b[2]] for b in self.beams]
    
