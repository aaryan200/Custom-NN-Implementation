import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

from model import Transformer

from config import get_weights_file_path, get_config

from torch.utils.tensorboard import SummaryWriter

from dataset import BilingualDataest, causal_mask
from model import build_transformer

from pathlib import Path
import warnings

def write_val_log(msg, val_log_file):
    # Append the log
    with open(val_log_file, 'a') as f:
        f.write(msg + '\n')

def greedy_decode(model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device):
    sos_idx = tokenizer_tgt.token_to_id('[SOS]')
    eos_idx = tokenizer_tgt.token_to_id('[EOS]')

    # Precompute the encoder output and reuse it for every token we get from the decoder
    encoder_output = model.encode(source, source_mask)
    # Initialize the decoder input with the SOS token
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device) # type_as sets the dtype of the tensor to the dtype of the source tensor

    # decoder_input is of shape (batch_size, )

    while True:
        if decoder_input.shape[1] == max_len:
            break

        # Build mask for the target (decoder input)
        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)

        # Calculate the output of the decoder
        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)

        # Get the next token, find the projection of the last token
        prob = model.project(out[:, -1])
        # Select the token with maximum probability (greedy search)
        _, next_word = torch.max(prob, dim = 1)

        decoder_input = torch.cat([decoder_input, torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device)], dim = 1)

        if next_word == eos_idx:
            break
    return decoder_input.squeeze(0)


def run_validation(model: Transformer, validation_ds, tokenizer_src, tokenizer_tgt, max_len, device, print_msg, global_state, writer, num_examples = 2):
    model.eval()
    count = 0

    # source_texts = []
    # expected = []
    # predicted = []

    # Size of the control window
    console_width = 10

    with torch.no_grad():
        for batch in validation_ds:
            count += 1
            encoder_input = batch['encoder_input'].to(device)
            encoder_mask = batch['encoder_mask'].to(device)

            assert encoder_input.shape[0] == 1, "Batch size should be 1 for validation"

            model_out = greedy_decode(model, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, max_len, device)

            src_text = batch['src_text'][0]
            tgt_text = batch['tgt_text'][0]
            # source_texts.append(src_text)
            # expected.append(tgt_text)
            model_out_text = tokenizer_tgt.decode(model_out.detach().cpu().numpy())
            # detach creates a tensor that shares storage with model_out but does not require gradient computation. In other words, it detaches the output tensor from the computation graph
            # predicted.append(model_out_text)

            # Print to console
            print_msg('='*console_width)
            print_msg(f'SOURCE: {src_text}')
            print_msg(f'TARGET: {tgt_text}')
            print_msg(f'PREDICTED: {model_out_text}')

            if count == num_examples:
                break

    # if writer:
    #     # Evaluate the character error rate
    #     # Compute the char error rate 
    #     metric = torchmetrics.CharErrorRate()
    #     cer = metric(predicted, expected)
    #     writer.add_scalar('validation cer', cer, global_step)
    #     writer.flush()

    #     # Compute the word error rate
    #     metric = torchmetrics.WordErrorRate()
    #     wer = metric(predicted, expected)
    #     writer.add_scalar('validation wer', wer, global_step)
    #     writer.flush()

    #     # Compute the BLEU metric
    #     metric = torchmetrics.BLEUScore()
    #     bleu = metric(predicted, expected)
    #     writer.add_scalar('validation BLEU', bleu, global_step)
    #     writer.flush()

def get_all_sentences(ds, lang):
    for item in ds:
        yield item['translation'][lang]

def get_or_build_tokenizer(config, ds, lang):
    """
    config: Model configurations
    ds: dataset
    lang: language for which tokenizer is to be build
    """
    # For example, if config['tokenizer_file'] is 'tokenizer_{}.txt' and lang is 'english', then config['tokenizer_file'].format(lang) will return 'tokenizer_english.txt'
    tokenizer_path = Path(config['tokenizer_file'].format(lang))
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token='[UNK]'))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens = ["[UNK]", "[PAD]", "[SOS]", "[EOS]"],
                                   min_frequency = 2) # For a word to appear in the vocab, it should have a minimum of 2 frequency
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer = trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))

    return tokenizer

def get_ds(config):
    ds_raw = load_dataset('cfilt/iitb-english-hindi', split='train')

    # Keep 15k examples for training and 1k examples for validation
    num_examples = 31_000
    ds_raw = ds_raw.select(range(num_examples))

    # Build tokenizers
    tokenizer_src = get_or_build_tokenizer(config, ds_raw, config['lang_src'])
    tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, config['lang_tgt'])

    train_size = 30_000
    val_size = num_examples - train_size

    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_size, val_size])
    
    train_ds = BilingualDataest(train_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])

    val_ds = BilingualDataest(val_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])

    # Just checking the max number of tokens in the source language as well as the target language

    max_len_src = 0
    max_len_tgt = 0

    for item in ds_raw:
        src_ids = tokenizer_src.encode(item['translation'][config['lang_src']]).ids
        tgt_ids = tokenizer_tgt.encode(item['translation'][config['lang_tgt']]).ids

        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))

    print(f'Max length of source sentence: {max_len_src}')
    print(f'Max length of target sentence: {max_len_tgt}')

    train_dataloader = DataLoader(train_ds, batch_size = config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size = 1, shuffle=False)

    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt


def get_model(config, src_vocab_size, tgt_vocab_size):
    model = build_transformer(src_vocab_size, tgt_vocab_size, config['seq_len'], config['seq_len'], config['d_model'])
    return model

def train_model(config):
    # Define the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f'Using device {device}')

    Path(config['model_folder']).mkdir(parents=True, exist_ok=True)

    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)

    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)
    # Tensorboard
    writer = SummaryWriter(config['experiment_name'])

    optimizer = torch.optim.Adam(model.parameters(), lr = config['lr'], eps = 1e-9)

    initial_epoch = 0
    global_step = 0
    if config['preload']:
        model_filename = get_weights_file_path(config, config['preload'])
        print(f'Preloading model {model_filename}')

        state = torch.load(model_filename)
        model.load_state_dict(state['model_state_dict'])
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']

    # Instruct the loss function to ignore the padding tokens while calculating the loss
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id('[PAD]'),
                                  label_smoothing=0.1).to(device)
    # label_smoothing means that for every highest probability token, take 0.1 percent of the score and distribute it to all the other tokens
    # This makes model less confident about the predictions and reduces overfitting

    for epoch in range(initial_epoch, config['num_epochs']):
        torch.cuda.empty_cache()
        model.train()
        batch_iterator = tqdm(train_dataloader, desc = f'Processing Epoch {epoch:02d}') # :02d means that the number should be printed with atleast 2 digits and if the number is less than 2 digits, then pad it with 0 in the beginning

        for batch in batch_iterator:
            encoder_input = batch['encoder_input'].to(device) # (batch_size, seq_len)
            decoder_input = batch['decoder_input'].to(device) # (batch_size, seq_len)
            encoder_mask = batch['encoder_mask'].to(device) # (batch_size, 1, 1, seq_len)
            decoder_mask = batch['decoder_mask'].to(device) # (batch_size, 1, seq_len, seq_len)

            # Run the tensors through the transformer
            encoder_output = model.encode(encoder_input, encoder_mask) # (batch_size, seq_len, d_model)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask) # (batch_size, seq_len, d_model)
            proj_output = model.project(decoder_output) # (batch_size, seq_len, tgt_vocab_size)

            label = batch['label'].to(device) # (batch_size, seq_len)

            # .view(-1, tgt_vocab_size) converts (batch_size, seq_len, tgt_vocab_size) to (batch_size * seq_len, tgt_vocab_size)
            loss = loss_fn(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
            # label.view(-1) converts (batch_size, seq_len) to (batch_size * seq_len)
            # So proj_output contains logits i.e., the probabilities of each token in the vocab
            # CrossEntropyLoss picks the highest probability token and compares it with the actual token in the label

            batch_iterator.set_postfix({'loss': f"{loss.item():6.3f}"}) # 6.3f means that the loss should be printed with 6 digits in total and 3 digits after the decimal point

            # Log the loss
            writer.add_scalar('train loss', loss.item(), global_step)
            writer.flush()

            # Backpropagation
            loss.backward()

            # Update the weights
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            # run_validation(model, val_dataloader, tokenizer_src, tokenizer_tgt, config['seq_len'], device, lambda msg: batch_iterator.write(msg), global_step, writer, num_examples = 2)

            global_step += 1

            if global_step % config['val_log_interval'] == 0:
                run_validation(model, val_dataloader, tokenizer_src, tokenizer_tgt, config['seq_len'], device, lambda msg: write_val_log(msg, config['val_log_file']), global_step, writer, num_examples = 2)
                # Set the model back to training mode
                model.train()

        write_val_log(f'Epoch {epoch:02d} completed', config['val_log_file'])
        run_validation(model, val_dataloader, tokenizer_src, tokenizer_tgt, config['seq_len'], device, lambda msg: write_val_log(msg, config['val_log_file']), global_step, writer, num_examples = 2)


        # Save the model at the end of every epoch
        model_filename = get_weights_file_path(config, f'{epoch:02d}')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step
        }, model_filename)

if __name__ == '__main__':
    # warnings.filterwarnings('ignore')

    print("Running....")

    config = get_config()
    train_model(config)