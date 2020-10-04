import pandas as pd
from tqdm import tqdm
import numpy as np
from collections import Counter
import torch
from transformers import AutoTokenizer, AutoModel, AdamW,\
get_linear_schedule_with_warmup
from ClassificationModel import ClassificationModel
import random
import argparse
import sys
import os
from sklearn.metrics import classification_report, f1_score
from utility import write_preds_to_file, get_questions, find_context, \
EncodeAndLoad, find_context_full


if __name__ == "__main__":
    # initialize argument parser
    description = 'Script to classify sentences as advice or non-advice'
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument('--savedmodel',
                        type=str,
                        default=None,
                        help='the model to load weights from')
    parser.add_argument('--data',
                        type=str,
                        default='askparents',
                        help='the dataset to use to train - askparents or \
                              needadvice')
    parser.add_argument('--model',
                        type=str,
                        default='bert',
                        help='the transformer model to use - bert, roberta, \
                              albert or xlnet')
    parser.add_argument('--labels',
                        type=str,
                        default='ds',
                        help='the annotation labels to use as targets - \
                              ds (David-Skeene) or maj (majority vote)')
    parser.add_argument('--dropout',
                        type=float,
                        default=0.1,
                        help='config.hidden_dropout_prob in classifier of \
                              transformer')
    parser.add_argument('--lr_tr',
                        type=float,
                        default=0.00001,
                        help='Learning rate for training transformer')
    parser.add_argument('--lr_cl',
                        type=float,
                        default=0.00001,
                        help='Learning rate for training classifier')
    parser.add_argument('--weight_decay',
                        type=float,
                        default=0,
                        help='Weight decay')
    parser.add_argument('--job',
                        action='store_true',
                        help='Turn off tqdm and some print for a job')
    parser.add_argument('--multigpu',
                        action='store_true',
                        help='Distribute the model')
    parser.add_argument('--test',
                        action='store_true',
                        help='Predict on test using saved model')
    parser.add_argument('--dev',
                        action='store_true',
                        help='Predict on dev using saved model')
    parser.add_argument('--noft',
                        action='store_true',
                        help='Dont finetune')
    parser.add_argument('--query',
                        action='store_true',
                        help='Append query')
    parser.add_argument('--context',
                        action='store_true',
                        help='Append query')
    parser.add_argument('--batch',
                        type=int,
                        default=32,
                        help='Batch size for training')
    parser.add_argument('--heads',
                        type=int,
                        default=3,
                        help='Number of Heads in multihead attention module')
    parser.add_argument('--seed',
                        type=int,
                        default=1,
                        help='Random seed')
    parser.add_argument('--frac',
                        type=float,
                        default=1,
                        help='Fraction for transfer learning')
    parser.add_argument('--act',
                        type=str,
                        default='tanh',
                        help='Activation function to use in classification model')
    parser.add_argument('--pooling',
                        type=str,
                        default='cls',
                        help='How to pool embeddings')

    args = parser.parse_args()


    ## Set random seeds for reproducibility on a specific machine
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    np.random.RandomState(args.seed)


    ## Set some global parameters
    sigdig = 4

    ## Preliminary file reading and setup
    filename = '../annotated_data/' + args.data

    train = pd.read_csv(filename + '_train.tsv', sep='\t', header=0)
    # For transfer learning replace 1 with ratio of choice to make size similar
    train = train.sample(frac=args.frac)
    train_sentences = train['Sentence'].tolist()
    train_labels_DS = train['DS_Label'].values
    train_labels_Maj = train['Majority_label'].values
    train['Post.ID'] = train['ID'].apply(lambda x: x.split('-')[0])
    train['Reply.ID'] = train['ID'].apply(lambda x: x.split('-')[1])
    train['Sent.Num'] = train['ID'].apply(lambda x: x.split('-')[2])


    dev = pd.read_csv(filename + '_dev.tsv', sep='\t', header=0)
    dev = dev.sample(frac=1)
    dev_sentences = dev['Sentence'].tolist()
    dev_labels_DS = dev['DS_Label'].values
    dev_labels_Maj = dev['Majority_label'].values
    dev['Post.ID'] = dev['ID'].apply(lambda x: x.split('-')[0])
    dev['Reply.ID'] = dev['ID'].apply(lambda x: x.split('-')[1])
    dev['Sent.Num'] = dev['ID'].apply(lambda x: x.split('-')[2])

    test = pd.read_csv(filename + '_test.tsv', sep='\t', header=0)
    test = test.sample(frac=1)
    test_sentences = test['Sentence'].tolist()
    test_labels_DS = test['DS_Label'].values
    test_labels_Maj = test['Majority_label'].values
    test['Post.ID'] = test['ID'].apply(lambda x: x.split('-')[0])
    test['Reply.ID'] = test['ID'].apply(lambda x: x.split('-')[1])
    test['Sent.Num'] = test['ID'].apply(lambda x: x.split('-')[2])


    ## Setup some parameters for the model itself

    MODELS = {'bert': 'bert-base-cased',
              'xlnet': 'xlnet-base-cased',
              'roberta': 'roberta-base',
              'albert': 'albert-base-v2',
              'electra': 'google/electra-base-discriminator',
              'bart': 'bart-large'
              }

    if args.model == 'electra':
        do_lower_case = True
    else:
        do_lower_case = False

    MODEL_CONFIG= MODELS[args.model]

    # Select a batch size for training. For fine-tuning BERT on a specific \
    # task, the authors recommend a batch size of 16 or 32
    batch_size = args.batch


    # How many labels in your problem?
    num_labels = np.unique(train_labels_DS).shape[0]

    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    ## Load sequences for training

    pad_token = AutoTokenizer.from_pretrained(MODEL_CONFIG)._pad_token
    sep_token = AutoTokenizer.from_pretrained(MODEL_CONFIG)._sep_token

    if args.query:
        questions = get_questions('../rawdata/' + args.data)
        train['Question'] = train['Post.ID'].apply(lambda x: questions[x])
        dev['Question'] = dev['Post.ID'].apply(lambda x: questions[x])
        test['Question'] = test['Post.ID'].apply(lambda x: questions[x])

        train_questions = train['Question'].tolist()
        dev_questions = dev['Question'].tolist()
        test_questions = test['Question'].tolist()

        # Set the maximum length of sequence - just the longest length sentence
        # from train, test and dev
        MAX_LEN = 256

        train_sequences = list(zip(train_sentences, train_questions))
        dev_sequences = list(zip(dev_sentences, dev_questions))
        test_sequences = list(zip(test_sentences, test_questions))

    elif args.context:
        train['Context'] = train['ID'].apply(lambda x: find_context_full(x, train))
        dev['Context'] = dev['ID'].apply(lambda x: find_context_full(x, dev))
        test['Context'] = test['ID'].apply(lambda x: find_context_full(x, test))

        train_context = train['Context'].tolist()
        dev_context = dev['Context'].tolist()
        test_context = test['Context'].tolist()

        train_sequences = list(zip(train_sentences, train_context))
        dev_sequences = list(zip(dev_sentences, dev_context))
        test_sequences = list(zip(test_sentences, test_context))

        MAX_LEN = 256

    else:
        MAX_LEN = max(max([len(a.split()) for a in train_sentences]),
                      max([len(a.split()) for a in dev_sentences]),
                      max([len(a.split()) for a in test_sentences]))
        train_sequences = train_sentences
        dev_sequences = dev_sentences
        test_sequences = test_sentences

    ## Model initialisation

    config = {'num_labels':num_labels}
    transformer_model = AutoModel.from_pretrained(MODEL_CONFIG,
                                                  config=config)

    model = ClassificationModel(transformer=transformer_model,
                                num_labels=num_labels,
                                dropout=args.dropout,
                                pooling=args.pooling,
                                activation=args.act,
                                heads=args.heads)
    model = model.to(device)

    if torch.cuda.device_count() > 1 and args.multigpu:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        device_ids=[i for i in range(torch.cuda.device_count())]
        model = torch.nn.DataParallel(model, device_ids=device_ids)


    ######### TRAINING ##########
    if not args.savedmodel:
        # Load tokenizer for the pretrained args.model
        tokenizer = AutoTokenizer.from_pretrained(MODEL_CONFIG,
                                                  do_lower_case=do_lower_case)

        ## Encode sequences and load into efficient Pytorch Tensor Dataloaders
        train_dataloader = EncodeAndLoad(tokenizer=tokenizer,
                                         sequences=train_sequences,
                                         labels=torch.tensor(train_labels_DS),
                                         batch_size=batch_size,
                                         MAX_LEN=MAX_LEN)

        dev_dataloader = EncodeAndLoad(tokenizer=tokenizer,
                                       sequences=dev_sequences,
                                       labels=torch.tensor(dev_labels_DS),
                                       batch_size=batch_size,
                                       MAX_LEN=MAX_LEN)

        ## Setup all the training parameters

        # Parameters:
        adam_epsilon = 1e-8

        # Number of training epochs (authors recommend between 2 and 4)
        epochs = 6

        num_warmup_steps = 1
        num_training_steps = len(train_dataloader)*epochs

        # Set different learning rates for transformer and classifier
        # Also set no weight decay for bias and layernorm
        no_decay = ["bias", "LayerNorm.weight"]
        classifier_parameters = ["classifier", "sequence_summary",
                                 "pooler", "logits_proj"]
        transformer_parameters = [args.model]

        if not args.noft:
            parameters = [
                    {
                        "params": [p for n, p in model.named_parameters() if \
                                   (not any(nd in n for nd in no_decay) and not \
                                   any(nd in n for nd in classifier_parameters))],
                        "weight_decay": args.weight_decay,
                        "lr": args.lr_tr
                    },
                    {
                        "params": [p for n, p in model.named_parameters() if \
                                   (not any(nd in n for nd in no_decay) and \
                                   any(nd in n for nd in classifier_parameters))],
                        "weight_decay": args.weight_decay,
                        "lr": args.lr_cl
                    },
                    {
                        "params": [p for n, p in model.named_parameters() if \
                                   (any(nd in n for nd in no_decay) and not \
                                   any(nd in n for nd in classifier_parameters))],
                        "weight_decay": 0.0,
                        "lr": args.lr_tr
                    },
                    {
                        "params": [p for n, p in model.named_parameters() if \
                                   (any(nd in n for nd in no_decay) and \
                                   any(nd in n for nd in classifier_parameters))],
                        "weight_decay": 0.0,
                        "lr": args.lr_cl
                    },
                ]
        else:
            # Set learning rate to 0 for all bert layers except classification head
            parameters = [
                    {
                        "params": [p for n, p in model.named_parameters() if \
                                   any(nd in n for nd in transformer_parameters) \
                                   and not \
                                   any(nd in n for nd in classifier_parameters) \
                                   and (not any(nd in n for nd in no_decay))],
                        "weight_decay": args.weight_decay,
                        "lr":0
                    },
                    {
                        "params": [p for n, p in model.named_parameters() if \
                                   any(nd in n for nd in transformer_parameters) \
                                   and not \
                                   any(nd in n for nd in classifier_parameters) \
                                   and any(nd in n for nd in no_decay)],
                        "weight_decay": 0.0,
                        "lr":0
                    },
                    {
                        "params": [p for n, p in model.named_parameters() if \
                                   (not any(nd in n for nd in no_decay) and \
                                   any(nd in n for nd in classifier_parameters))],
                        "weight_decay": args.weight_decay,
                        "lr": args.lr_cl
                    },
                    {
                        "params": [p for n, p in model.named_parameters() if \
                                   (any(nd in n for nd in no_decay) and \
                                   any(nd in n for nd in classifier_parameters))],
                        "weight_decay": 0.0,
                        "lr": args.lr_cl
                    },
                ]

        ### Instantiate optimizer and schedules
        # To reproduce BertAdam specific behavior set correct_bias=False
        optimizer = AdamW(parameters, lr=args.lr_tr,eps=adam_epsilon,
                          correct_bias=False)
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                          num_warmup_steps=num_warmup_steps,
                                          num_training_steps=num_training_steps)


        ## Training/Fine-tuning step


        # Store our loss and accuracy for plotting
        train_loss_set = []
        learning_rate = []

        # Store the f1s in a list for early stopping
        dev_loss_list = [10000]
        dev_f1_list = [0]
        early_stopping_threshold = 0.0001

        # Gradients gets accumulated by default
        model.zero_grad()

        for epoch in range(1,epochs+1):
            print("<" + "="*40 + F" Epoch {epoch} "+ "="*40 + ">")

            # Calculate total loss for this epoch
            total_train_loss = 0

            for step, batch in enumerate(tqdm(train_dataloader, disable=args.job)):
                # Set our model to training mode (as opposed to evaluation mode)
                model.train()

                # Unpack the inputs from our dataloader
                b_input_ids, b_input_mask, b_labels, b_token_type_ids = \
                                                                           batch

                # truncate the batch to maximum length for a speedup
                max_length = (b_input_mask != 0).max(0)[0].nonzero()[-1].item()

                if max_length < MAX_LEN:
                    b_input_ids = b_input_ids[:, :max_length].to(device)
                    b_input_mask = b_input_mask[:, :max_length].to(device)
                    b_token_type_ids = \
                                    b_token_type_ids[:, :max_length].to(device)
                else:
                    b_input_ids = b_input_ids.to(device)
                    b_input_mask = b_input_mask.to(device)
                    b_token_type_ids = b_token_type_ids.to(device)

                b_labels = b_labels.to(device)

                if not args.query:
                    # Ignore this when there is only one sequence (no question)
                    b_token_type_ids = None

                # Forward pass
                outputs = model(input_ids=b_input_ids,
                                token_type_ids=b_token_type_ids,
                                attention_mask=b_input_mask,
                                labels=b_labels)
                loss = outputs[0]

                # Backward pass
                loss.mean().backward()

                # Clip the norm of the gradients to 1.0
                # Gradient clipping is not in AdamW anymore
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                # Update parameters and take a step using the computed gradient
                optimizer.step()

                # Update learning rate schedule
                scheduler.step()

                # Clear the previous accumulated gradients
                optimizer.zero_grad()

                # Update tracking variables
                total_train_loss += loss.mean().item()

            # Calculate the average loss over the training data.
            avg_train_loss = total_train_loss / len(train_dataloader)

            #store the current learning rate
            for param_group in optimizer.param_groups:
                # print("\n\tCurrent Learning rate: ",param_group['lr'])
                learning_rate.append(param_group['lr'])

            train_loss_set.append(avg_train_loss)
            print(F'\n\t\tAverage Training loss: {avg_train_loss}')

            print("\n\tRunning Validation...")
            # Put model in evaluation mode to evaluate loss on the validation set
            model.eval()

            # Tracking variables
            pred_labels = np.array([])
            target_labels = np.array([])
            total_eval_loss = 0
            for batch in dev_dataloader:

                b_input_ids, b_input_mask, b_labels, b_token_type_ids = batch

                # truncate the batch to maximum length for a speedup
                max_length = (b_input_mask != 0).max(0)[0].nonzero()[-1].item()

                if max_length < MAX_LEN:
                    b_input_ids = b_input_ids[:, :max_length].to(device)
                    b_input_mask = b_input_mask[:, :max_length].to(device)
                    b_token_type_ids = \
                                    b_token_type_ids[:, :max_length].to(device)
                else:
                    b_input_ids = b_input_ids.to(device)
                    b_input_mask = b_input_mask.to(device)
                    b_token_type_ids = b_token_type_ids.to(device)

                b_labels = b_labels.to(device)

                if not args.query or not args.context:
                    # Ignore this when there is only one sequence (no question)
                    b_token_type_ids = None

                with torch.no_grad():
                    # Forward pass, calculate logit predictions
                    (loss,logits) = model(b_input_ids,
                                          token_type_ids=b_token_type_ids,
                                          attention_mask=b_input_mask,
                                          labels=b_labels)

                # Move logits and labels to CPU
                logits = logits.to('cpu').numpy()
                label_ids = b_labels.to('cpu').numpy()
                total_eval_loss += loss.mean().item()

                pred_flat = np.argmax(logits, axis=1).flatten()
                pred_labels = np.append(pred_labels, pred_flat)
                labels_flat = label_ids.flatten()
                target_labels = np.append(target_labels, labels_flat)

            dev_loss = total_eval_loss / len(dev_dataloader)

            print("\n\t\t Validation Loss:", dev_loss)
            dev_f1 = f1_score(target_labels, pred_labels)

            if (dev_f1 - dev_f1_list[-1] < early_stopping_threshold) and epoch>=2:
                sys.exit("\n\n\t\tEarly stopping threshold reached!")
            else:
                dev_f1_list.append(dev_f1)

                labels = [0, 1]
                target_names = ['0', '1']
                print(classification_report(target_labels, pred_labels,
                                            labels=labels,
                                            target_names=target_names,
                                            digits=sigdig))

                name_of_model = ('classifier' + '_' + args.data + '_' + \
                                 args.model + '_dropout:' + str(args.dropout) +
                                 '_lr_tr:' + str(args.lr_tr) +\
                                 '_lr_cl:' + str(args.lr_cl) +\
                                 '_wd:' + str(args.weight_decay) +\
                                 '_batch:' + str(args.batch) +\
                                 "_finetune:" + str(not args.noft) +\
                                 "_query:" + str(args.query) +\
                                 "_context:" + str(args.context)+\
                                 "_seed:" + str(args.seed)+\
                                 "_multigpu:" + str(args.multigpu)+\
                                 "_labels:" + (args.labels)+\
                                 "_frac:" + str(args.frac)+'/')
                save_path = '../saved_pretrained/' + name_of_model
                if not os.path.exists(save_path):
                    os.mkdir(save_path)
                # Take care of distributed/parallel training
                model_to_save = model.module if hasattr(model, "module") \
                                             else model
                torch.save(model_to_save.state_dict(), save_path + 'pytorch_model.bin')
                tokenizer.save_pretrained(save_path)
        # Exit training
        sys.exit(0)

    ########## EVALUTATION ##########

    ## Load the saved finetuned tokenizer and model
    saved_model = args.savedmodel
    name_of_model = saved_model.split('/')[-2]
    tokenizer = AutoTokenizer.from_pretrained(saved_model)
    model.load_state_dict(torch.load(saved_model + 'pytorch_model.bin'))

    ## Choose and load dataset to evaluate
    if args.test:
        dataloader = EncodeAndLoad(tokenizer=tokenizer,
                                   sequences=test_sequences,
                                   labels=torch.tensor(test_labels_DS),
                                   batch_size=batch_size,
                                   MAX_LEN=MAX_LEN)
        name_of_model += '_TEST'
        df = test
    else:
        dataloader = EncodeAndLoad(tokenizer=tokenizer,
                                   sequences=dev_sequences,
                                   labels=torch.tensor(dev_labels_DS),
                                   batch_size=batch_size,
                                   MAX_LEN=MAX_LEN)
        name_of_model += '_DEV'
        df = dev
    pred_labels = np.array([])
    target_labels = np.array([])

    model.eval()

    print("\n\n\t\tRunning evaluation")
    for batch in dataloader:
        # Unpack the inputs from our dataloader
        b_input_ids, b_input_mask, b_labels, b_token_type_ids = batch

        # truncate the batch to maximum length for a speedup
        max_length = (b_input_mask != 0).max(0)[0].nonzero()[-1].item()

        if max_length < MAX_LEN:
            b_input_ids = b_input_ids[:, :max_length].to(device)
            b_input_mask = b_input_mask[:, :max_length].to(device)
            b_token_type_ids = b_token_type_ids[:, :max_length].to(device)
        else:
            b_input_ids = b_input_ids.to(device)
            b_input_mask = b_input_mask.to(device)
            b_token_type_ids = b_token_type_ids.to(device)

        b_labels = b_labels.to(device)

        if not args.query or not args.context:
            b_token_type_ids = None

        with torch.no_grad():
            # Forward pass, calculate logit predictions
            (loss,logits) = model(b_input_ids, token_type_ids=b_token_type_ids,
                                  labels=b_labels, attention_mask=b_input_mask)

        # Move logits and labels to CPU
        logits = logits.to('cpu').numpy()
        label_ids = b_labels.to('cpu').numpy()

        pred_flat = np.argmax(logits, axis=1).flatten()
        pred_labels = np.append(pred_labels, pred_flat)
        labels_flat = label_ids.flatten()
        target_labels = np.append(target_labels, labels_flat)


    labels = [0, 1]
    target_names = ['0', '1']
    print(classification_report(target_labels, pred_labels,labels=labels,
                                target_names=target_names, digits=sigdig))

    write_preds_to_file(preds=pred_labels, df=df, model_name=name_of_model)
