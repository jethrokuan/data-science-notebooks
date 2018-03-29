import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

class BeamSearch:
    """
    last_hidden    : last LSTM hidden output
    last_cell      : last LSTM cell state
    last_token     : vocabulary index of last token fed into decoder
    encoder_outputs: all the outputs of the encoder (for attention)
    decoder        : decoder model
    vocabulary     : vocabulary generated from dataset
    beam_width     : width of beam search (number of candidates per level to consider)
    beam_depth     : depth of beam search (length of sequence to generate)
    """
    def get_words(self, last_hidden, last_cell, last_token, encoder_outputs, input_lengths, decoder, vocabulary, beam_width, beam_depth):
        was_decoder_training = decoder.training
        decoder = decoder.eval()

        candidates = [{"generated_tokens": [vocabulary.itos[last_token],],
                       "hidden": last_hidden.clone(),
                       "cell": last_cell.clone(),
                       "last_token": last_token,
                       "probability": 1.0}]

        depth = 0
        while depth < beam_depth:
            new_candidates = []
            # For each of the current candidates, generate all of its possible children and choose the top few
            for candidate in candidates:
                last_token = Variable(torch.LongTensor([candidate["last_token"],]).unsqueeze(0))
                hidden, cell, scores = self.pass_through_decoder(candidate["hidden"], candidate["cell"], last_token, encoder_outputs, input_lengths, decoder)
                top_children = self.get_n_highest_indices(scores, beam_width)
                new_candidates += [{"generated_tokens": candidate["generated_tokens"] + [vocabulary.itos[child],],
                                    "hidden": hidden.clone(),
                                    "cell": cell.clone(),
                                    "last_token": child,
                                    "probability": float(candidate["probability"] * scores[0][child].data)} for child in top_children]
            new_candidate_probabilities = [candidate["probability"] for candidate in new_candidates]
            new_candidate_indices = self.get_n_highest_indices(new_candidate_probabilities, beam_width)
            candidates = [new_candidates[i] for i in new_candidate_indices]

            depth += 1

        if was_decoder_training:
            decoder = decoder.train()

        top_candidate_sequences = [candidate["generated_tokens"] for candidate in candidates]

        candidate_probabilities = [candidate["probability"] for candidate in candidates]
        best_token_sequence = candidates[np.argmax(candidate_probabilities)]["generated_tokens"]

        return best_token_sequence, top_candidate_sequences

    def get_n_highest_indices(self, values, n):
        if type(values) is torch.autograd.variable.Variable:
            values = values.clone().cpu().data.numpy()[0]
        else:
            values = values[:]
        max_indices = []
        minimum = -100000 # To take a chosen value out of the running
        for i in range(n):
            max_idx = int(np.argmax(values)) # Ensure that it'll fit in a LongTensor
            max_indices.append(max_idx)
            values[max_idx] = minimum
        return max_indices

    def pass_through_decoder(self, last_hidden, last_cell, last_token, encoder_outputs, input_lengths, decoder):

        # Get embedding of input
        if decoder.use_cuda:
            last_token = last_token.cuda()
        embedded = decoder.dropout(decoder.embedding(last_token))

        # Prepare for attention
        decoder.enc_attention.reset_step()

        # Forward pass
        _, (hidden, cell) = decoder.lstm(embedded, (last_hidden, last_cell))

        # Intra Temporal Attention
        context_e = decoder.enc_attention(hidden.transpose(0,1), encoder_outputs, input_lengths)
        context_d = decoder.init_context(1).transpose(0,1) # 1 replaces batch size

        concat = torch.cat([hidden.transpose(0,1),context_e,context_d],2) # B,1,3D
        scores = decoder.linear(concat.view(concat.size(0)*concat.size(1),-1)) # B,V

        scores = nn.Softmax(1)(scores)
        return (hidden, cell, scores)
