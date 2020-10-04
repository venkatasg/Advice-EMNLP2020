import torch
import numpy as np


class ClassificationModel(torch.nn.Module):

    def __init__(self, transformer, dropout, num_labels, activation='tanh',
                 pooling='cls', heads=3):
        '''
        Setup the modules in the model - a transformer, followed by a GRU for
        the CLS hidden states/taking the mean of all tokens, followed by Linear
        layers that outputs one number, followed by softmax
        '''
        super(ClassificationModel, self).__init__()

        # Setup the transformer and the GRU layer on top of the CLS tokens
        self.transformer = transformer

        # Setup the linear layers on top of the GRU final hidden state
        self.tr_output_size = self.transformer.config.hidden_size
        self.num_labels = num_labels

        self.classifier = torch.nn.Linear(self.tr_output_size, self.num_labels)
        self._activation = activation
        self._pooling = pooling
        self._dropout = torch.nn.Dropout(p=dropout)
        self._logsoftmax = torch.nn.LogSoftmax(dim=2)
        self.loss_fn = torch.nn.CrossEntropyLoss()

        if self._pooling == 'attn':
            self.attention = torch.nn.MultiheadAttention(
                                        embed_dim=self.tr_output_size,
                                        num_heads=heads)
            self.attention_map = None

    def _nonlinearity(self, x):
        '''Applies relu or tanh activation on tensor.'''

        if self._activation == 'relu':
            return torch.nn.functional.relu(x)
        elif self._activation == 'tanh':
            return torch.tanh(x)

    def find_len(self, a):
        '''
        Finds the length of the sequence beyond which padding is necessary
        (in the batch of cls tokens)
        a: tensor of shape max_len

        returns an integer (which is length for that sequence - this is
        equivalent to number of sentences in a reply)
        '''

        try:
            len = ((a != 0).nonzero())[-1].item() + 1
        except IndexError:
            len = 1

        return len


    def forward(self, input_ids, token_type_ids, attention_mask, labels=None):
        '''
        Runs forward pass on neural network

        Arguments:
        ---------
        input_ids: the tokenized, bert wordpiece IDs. (batch_size, MAX_LEN)
        input_masks: the masking to be done on input_ids due to padding.
        (batch_size, MAX_LEN)
        labels: target against which to computer the loss. DEFAULT: None
        max_seq_len: The length to which to pad the output of the rnn

        Returns:
        -------

        Object of type Tuple of form (loss, logits)

        loss: Cross Entropy loss calculated in loss_fn which implements masking
        logits: logsoftmaxed probabilities of classifier output

        '''


        # Forward pass through transformer
        # other values returned are pooler_output, hidden_states, and attentions
        outputs = self.transformer(input_ids,
                                   token_type_ids=None,
                                   attention_mask=attention_mask)

        last_hidden_states = outputs[0]

        if self._pooling == 'avg':
            extended_mask = \
                    input_mask.unsqueeze(-1).expand(last_hidden_states.size())
            last_hidden_states[extended_mask==0] = 0
            sent_hidden_states = torch.mean(last_hidden_states, dim=1)
            # Now normalize for the different lengths of sentences
            lengths = torch.min(input_mask, dim=1)[1].to(dtype=torch.float)
            lengths[lengths==0] = input_mask.shape[1]
            normalized_lengths = \
                (lengths/input_mask.shape[1]).unsqueeze(-1).expand(
                                                      sent_hidden_states.size()
                                                      )
            sent_hidden_states /= normalized_lengths

        elif self._pooling == 'cls':
            sent_hidden_states = last_hidden_states[:,0,:]

        elif self._pooling == 'max':
            extended_mask = \
                    input_mask.unsqueeze(-1).expand(last_hidden_states.size())
            last_hidden_states[extended_mask==0]= -1e9
            sent_hidden_states = torch.max(last_hidden_states, dim=1)[0]

        elif self._pooling == 'attn':
            # MultiheadAttention doesn't support batch first, so convert to
            # appropriate shape
            last_hidden_states = last_hidden_states.view(-1,
                                                         input_ids.shape[0],
                                                         self.tr_output_size)
            query = last_hidden_states[0,:,:].unsqueeze(0)
            key = last_hidden_states
            value = last_hidden_states

            output, weights = self.attention(query=query,
                                             key=key,
                                             value=value,
                                             key_padding_mask=~attention_mask.bool()
                                             )
            sent_hidden_states = output.squeeze()

        else:
            sys.exit("Invalid pooling - either cls, avg, max or attn only")


        # Then run it through linear layers
        x = self._nonlinearity(sent_hidden_states)
        x = self._dropout(x)
        logits = self.classifier(x)

        if len(logits.shape) < 2:
            logits = logits.unsqueeze(0)
        outputs  = (logits,)

        if labels is not None:
            loss = self.loss_fn(logits, labels)
            outputs = (loss,) + outputs

        return outputs