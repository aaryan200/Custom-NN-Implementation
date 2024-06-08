from typing import Any
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from tokenizers import Tokenizer

class BilingualDataest(Dataset):

    def __init__(self, ds,
                 tokenizer_src: Tokenizer,
                 tokenizer_tgt: Tokenizer,
                 src_lang, tgt_lang, seq_len) -> None:
        super().__init__()
        self.seq_len = seq_len

        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

        # Both src as well as tgt contains [SOS], [EOS], [PAD] and [UNK] tokens, so use any of them
        # to convert these tokens to tensors
        # Since vocab_size can be more than 32-bit long, so use torch.int64
        self.sos_token = torch.tensor([tokenizer_src.token_to_id('[SOS]')], dtype = torch.int64)
        self.eos_token = torch.tensor([tokenizer_src.token_to_id('[EOS]')], dtype = torch.int64)
        self.pad_token = torch.tensor([tokenizer_src.token_to_id('[PAD]')], dtype = torch.int64)

    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, index) -> Any:
        src_tgt_pair = self.ds[index]
        src_text = src_tgt_pair['translation'][self.src_lang]
        tgt_text = src_tgt_pair['translation'][self.tgt_lang]

        encoder_input_tokens = self.tokenizer_src.encode(src_text).ids
        decoder_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids

        # Number of padding tokens required
        enc_num_pad_tokens = self.seq_len - len(encoder_input_tokens) - 2 # -2 because [SOS] and [EOS] are always present in the input to the encoder
        dec_num_pad_tokens = self.seq_len - len(decoder_input_tokens) - 1 # -1 because only [SOS] is added to the decoder input

        if enc_num_pad_tokens < 0 or dec_num_pad_tokens < 0:
            raise ValueError("Sentence is too long")
        
        # Concatenate the tokens

        # Input to be sent to the encoder
        encoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(encoder_input_tokens, dtype = torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * enc_num_pad_tokens, dtype = torch.int64)
            ],
            dim = 0
        )

        # Input to be sent to the decoder
        decoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(decoder_input_tokens, dtype = torch.int64),
                torch.tensor([self.pad_token] * dec_num_pad_tokens, dtype = torch.int64)
            ],
            dim = 0
        )

        # Expected output from the deocder aka label
        label = torch.cat(
            [
                torch.tensor(decoder_input_tokens, dtype = torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * dec_num_pad_tokens, dtype = torch.int64)
            ],
            dim = 0
        )

        assert encoder_input.shape[0] == self.seq_len
        assert decoder_input.shape[0] == self.seq_len
        assert label.shape[0] == self.seq_len

        return {
            "encoder_input": encoder_input, # (seq_len, )
            "decoder_input": decoder_input, # (seq_len, )
            # Wherever mask will be 0, it will be replaced by -inf in attention layer
            # .unsqueeze(0) is used add a new 0th dimension
            "encoder_mask": (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(), # (1,1, seq_len), here 1 and 1 are batch dim and seq dim
            # seq dim will be used to broadcast this mask to shape (batch_size, seq_len, seq_len)
            # In decoder mask, make sure that no padding token interact with any other word and also, no word should interact with the future words
            "decoder_mask": (decoder_input != self.pad_token).unsqueeze(0).int() & causal_mask(self.seq_len), # (1, seq_len) & (1, seq_len, seq_len) = (1, seq_len, seq_len)
            "label" : label, # (seq_len, )
            "src_text": src_text,
            "tgt_text": tgt_text
        }


def causal_mask(seq_len):
    # Everything below and on the diagonal will become 0 and everything above the diagonal will become 1
    mask = torch.triu(torch.ones(1, seq_len, seq_len), diagonal=1).type(torch.int)
    # But we want the opposite, so:
    return mask == 0