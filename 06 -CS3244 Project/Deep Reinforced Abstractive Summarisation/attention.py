import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, hidden_size, method='general'):
        super(Attention, self).__init__()
        
        self.method = method
        self.hidden_size = hidden_size
        
        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size, hidden_size)

        elif self.method == 'concat':
            self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
            self.v = nn.Parameter(torch.FloatTensor(1, hidden_size))

    def forward(self, hidden, encoder_outputs,encoder_lengths=None,return_weight=False):
        """
        hidden : query (previous hidden) B,1,D <FloatTensor>
        encoder_outputs : context (encoder outputs) B,T,D <FloatTensor>
        encoder_lengths : list[int]
        """
        q, c = hidden, encoder_outputs

        batch_size_q, n_q, dim_q = q.size()
        batch_size_c, n_c, dim_c = c.size()

        if not (batch_size_q == batch_size_c):
            msg = 'batch size mismatch (query: {}, context: {}, value: {})'
            raise ValueError(msg.format(q.size(), c.size()))

        batch_size = batch_size_q
        
        s = self.score(q,c)
        
        if encoder_lengths is not None:
            mask = s.data.new(batch_size, n_q, n_c)
            mask = self.fill_context_mask(mask, sizes=encoder_lengths, v_mask=float('-inf'), v_unmask=0)
            s = Variable(mask) + s
        
        w = F.softmax(s,2) # B,1,T
        
        # Combine
        z = w.bmm(c)
        if return_weight:
            return w, z
        return z
        
    
    def score(self, q, c):
        """
        q: B,1,D
        c: B,T,D
        """
        if self.method == 'dot':
            return q.bmm(c.transpose(1, 2)) # B,1,D * B,D,T => B,1,T
        
        elif self.method == 'general':
            energy = self.attn(c) # B,T,D => B,T,D
            return q.bmm(energy.transpose(1,2)) # B,1,D * B,D,T => B,1,T
                    
        elif self.method == 'concat':
            q = q.repeat(1,c.size(1),1) # B,T,D
            energy = self.attn(torch.cat([q, c], 2)) # B,T,2D => B,T,D
            v = self.v.repeat(c.size(1), 1).unsqueeze(1)  # B,1,D
            return v.bmm(energy.transpose(1,2)) # B,1,D * B,D,T => B,1,T 
            
    
    def fill_context_mask(self, mask, sizes, v_mask, v_unmask):
        """Fill attention mask inplace for a variable length context.
        Args
        ----
        mask: Tensor of size (B, T, D)
            Tensor to fill with mask values. 
        sizes: list[int]
            List giving the size of the context for each item in
            the batch. Positions beyond each size will be masked.
        v_mask: float
            Value to use for masked positions.
        v_unmask: float
            Value to use for unmasked positions.
        Returns
        -------
        mask:
            Filled with values in {v_mask, v_unmask}
        """
        mask.fill_(v_unmask)
        n_context = mask.size(2)
        for i, size in enumerate(sizes):
            if size < n_context:
                mask[i,:,size:] = v_mask
        return mask



class IntraTempAttention(nn.Module):
    def __init__(self, hidden_size, method='general'):
        super(IntraTempAttention, self).__init__()
        
        self.method = method
        self.hidden_size = hidden_size
        
        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size, hidden_size)

        elif self.method == 'concat':
            self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
            self.v = nn.Parameter(torch.FloatTensor(1, hidden_size))
        
        self.decoding_step = 0
        self.energies = []
        
    def reset_step(self):
        self.decoding_step = 0
        self.energies = []
        
    def forward(self, hidden, encoder_outputs,encoder_lengths=None,return_weight=False):
        """
        hidden : query (previous hidden) B,1,D <FloatTensor>
        encoder_outputs : context (encoder outputs) B,T,D <FloatTensor>
        encoder_lengths : list[int]
        """
        self.decoding_step+=1
        q, c = hidden, encoder_outputs

        batch_size_q, n_q, dim_q = q.size()
        batch_size_c, n_c, dim_c = c.size()

        if not (batch_size_q == batch_size_c):
            msg = 'batch size mismatch (query: {}, context: {}, value: {})'
            raise ValueError(msg.format(q.size(), c.size()))

        batch_size = batch_size_q
        
        s = self.score(q,c) # B,1,T
        
        self.energies.append(s)
        
        # timestep-wise normalize
        if self.decoding_step==1:
            s = torch.exp(s)
        else:
            s = F.softmax(torch.cat(self.energies,1),1)[:,-1].unsqueeze(1) # B,1,T
        
        if encoder_lengths is not None:
            mask = s.data.new(batch_size, n_q, n_c)
            mask = self.fill_context_mask(mask, sizes=encoder_lengths, v_mask=float('-inf'), v_unmask=0)
            s = Variable(mask) + s

        w = F.softmax(s,2) # B,1,T
        
        # Combine
        z = w.bmm(c)
        if return_weight:
            return w, z
        return z
        
    
    def score(self, q, c):
        """
        q: B,1,D
        c: B,T,D
        """
        if self.method == 'dot':
            return q.bmm(c.transpose(1, 2)) # B,1,D * B,D,T => B,1,T
        
        elif self.method == 'general':
            energy = self.attn(c) # B,T,D => B,T,D
            return q.bmm(energy.transpose(1,2)) # B,1,D * B,D,T => B,1,T
                    
        elif self.method == 'concat':
            q = q.repeat(1,c.size(1),1) # B,T,D
            energy = self.attn(torch.cat([q, c], 2)) # B,T,2D => B,T,D
            v = self.v.repeat(c.size(1), 1).unsqueeze(1)  # B,1,D
            return v.bmm(energy.transpose(1,2)) # B,1,D * B,D,T => B,1,T 
            
    
    def fill_context_mask(self, mask, sizes, v_mask, v_unmask):
        """Fill attention mask inplace for a variable length context.
        Args
        ----
        mask: Tensor of size (B, T, D)
            Tensor to fill with mask values. 
        sizes: list[int]
            List giving the size of the context for each item in
            the batch. Positions beyond each size will be masked.
        v_mask: float
            Value to use for masked positions.
        v_unmask: float
            Value to use for unmasked positions.
        Returns
        -------
        mask:
            Filled with values in {v_mask, v_unmask}
        """
        mask.fill_(v_unmask)
        n_context = mask.size(2)
        for i, size in enumerate(sizes):
            if size < n_context:
                mask[i,:,size:] = v_mask
        return mask
