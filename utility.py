from sklearn.metrics import f1_score as f1, accuracy_score as acc, \
precision_score as prec, recall_score as rec, matthews_corrcoef as mcc
import numpy as np
import os
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
import re

def get_metrics(targets, preds, average='micro', mask=None):
    '''
    Return accuracy, mcc, f1 score, precision and recall in that order
    '''

    acc_score = acc(targets, preds, sample_weight=mask)
    num_labels = len(np.unique(targets))

    if num_labels > 2:
        mcc_score = 0.00
    else:
        average = 'binary'
        mcc_score = mcc(targets, preds, sample_weight=mask)

    f1_score = f1(targets, preds, sample_weight=mask, average=average)
    prec_score = prec(targets, preds, sample_weight=mask, average=average)
    rec_score = rec(targets, preds, sample_weight=mask, average=average)

    return acc_score, mcc_score, f1_score, prec_score, rec_score


def write_preds_to_file(preds, df, model_name):
    '''
    Write predictions of model to a csv file with extra columns
    '''

    # Then add the preds to the dataframe
    df.loc[:, 'Pred_Label'] = preds
    df.to_csv(('../preds/test/' + model_name +'.csv'), sep='\t', index=False)


def BIO_convert(sequence):
    '''
    Convert a sequence of 1s and 0s to BIO(Beginning-Inside-Outside) format
    '''

    bio_to_int_map = {'O': 0, 'B': 1, 'I': 2}
    int_to_bio_map = {0: 'O', 1: 'B', 2: 'I'}

    bio = ['O' for i in range(len(sequence))]
    if 1 in sequence:
        bio[sequence.index(1)] = 'B'
        for k in range(sequence.index(1)+1, len(sequence)):
            if sequence[k] == 1 and bio[k-1] in ['B','I']:
                bio[k] = 'I'
            elif sequence[k] == 0:
                bio[k] = 'O'
            elif sequence[k] == 1 and sequence[k-1] == 0:
                bio[k] = 'B'
    bio = [bio_to_int_map[a] for a in bio]
    return bio

def get_questions(path):
    '''
        Read in all the questions with their ids from a specific path
    '''

    questions = {}
    for root, dirs, files in os.walk(path):
        for file in files:
            if file=='context.txt':
                with open(os.path.join(root,file)) as f:
                    q_full = f.read().split('<br>')
                    q_text = q_full[2]
                    q_title = q_full[0]
                    q_context = ' '.join([x for x in re.split("[?.!;]", q_text) if x!=""][-3:])
                    # Strip off newline and tab characters
                    q_context = q_context.replace('\n', '').replace('\t', '')
                    q_context = q_context + " " + q_title
                    questions[root.split('/')[-1]] = q_context

    return questions

def find_context(ind, df, pad, sep):
    '''
    Return a reliable context for every sentence, instead of one of 3 possible
    things which might confuse the downstream model
    '''

    split_ind = ind.split('-')

    context_ind =  "-".join(split_ind[:2]) + '-' + str(int(split_ind[-1]) - 1)

    if df[df['ID'] == context_ind].empty:
        context = pad
    else:
        context = df[df['ID'] == context_ind]['Sentence'].values[0]

    return context

def find_context_both(ind, df):
    '''
    Return a reliable context for every sentence, instead of one of 3 possible
    things which might confuse the downstream model
    '''

    split_ind = ind.split('-')

    context_ind_left = \
                    "-".join(split_ind[:2]) + '-' + str(int(split_ind[-1]) - 1)

    if df[df['ID'] == context_ind_left].empty:
        context_left = ""
    else:
        context_left = df[df['ID'] == context_ind_left]['Sentence'].values[0]

    context_ind_right = \
                    "-".join(split_ind[:2]) + '-' + str(int(split_ind[-1]) + 1)
    if df[df['ID'] == context_ind_right].empty:
        context_right = ""
    else:
        context_right = df[df['ID'] == context_ind_right]['Sentence'].values[0]

    context = context_left + " " + context_right
    return context


def find_context_full(ind, df):
    '''
    Return a reliable context for every sentence, instead of one of 3 possible
    things which might confuse the downstream model
    '''

    split_ind = ind.split('-')
    context = df.loc[(df['Post.ID']==split_ind[0]) & \
                     (df['Reply.ID']==split_ind[1]) & \
                     (df['Sent.Num']!=split_ind[2]), 'Sentence'].values.tolist()

    context = " ".join(context)

    return context


def EncodeAndLoad(tokenizer, sequences, labels, batch_size, MAX_LEN):
    '''
    Spits out a Pytorch DataLoader object that one can efficiently iterate over
    sequences: Sentences or sentence pairs
    tokenizer: PreTrainedTokenizer object
    labels: Labels on which to train
    '''

    # Batch tokenize data, and create attention masks and token type ids
    encoded_dict = tokenizer.batch_encode_plus(sequences,
                                               add_special_tokens=True,
                                               max_length=MAX_LEN,
                                               pad_to_max_length=True,
                                               return_attention_mask=True,
                                               return_token_type_ids = True,
                                               return_tensors='pt')
    token_type_ids = encoded_dict['token_type_ids']
    input_ids = encoded_dict['input_ids']
    attention_mask = encoded_dict['attention_mask']

    # Create an iterator of our data with torch DataLoader. This helps save \
    # on memory during training because, unlike a for loop, with an iterator \
    # the entire dataset does not need to be loaded into memory
    data = TensorDataset(input_ids, attention_mask, labels, token_type_ids)
    dataloader = DataLoader(data,batch_size=batch_size)

    return dataloader