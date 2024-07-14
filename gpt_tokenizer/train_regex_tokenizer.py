"""
The basic difference between a basic tokenizer and regex tokenizer is that the text is first splitted into multiple chunks using a regex pattern.
The most frequently occuring pair will be found across all the chunks.
Then, that pair will be replaced by a placeholder token in all the chunks.
Split pattern is used so that tokens like "dog", "dog.", "dog?" are not formed.
To do that, we split "dog" or " dog" or "dog " as a text chunk and avoid punctuations and other things to be present in the chunk.
After that, each of the chunk is treated separately and the pairs of these chunks are combined during training of the tokenizer.

Therefore, while encoding, the text is splitted using the same regex pattern and then, the text chunks are encoded separately.
Why is there a need to chunk them before encoding?
This is because while training they were splitted and then tokenizer was trained to learn the pairs among the matched pattern chunks.
Therefore, the tokenizer learnt to combine the tokens in the matches rather than the complete text.
"""

import regex as re

GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

sample_text = """Hello, my name is Aaryan. आपका नाम क्या है?"""

class RegexTokenizer:

    def __init__(self, pattern = None) -> None:
        self.pattern = GPT4_SPLIT_PATTERN if pattern is None else pattern
        self.compiled_pattern = re.compile(self.pattern)

    def train(self, text: str, vocab_size: int, verbose: bool = False):
        self.vocab_size = vocab_size

        num_merges = vocab_size - 256
        
        assert num_merges >= 0, "vocab_size should be more than equal to 256"

        text_chunks: list[str] = re.findall(self.compiled_pattern, text)
        
        # List of list of integers in range 0..255
        ids = [list(chunk.encode("utf-8")) for chunk in text_chunks]

        merges = dict()

        vocab = {idx: bytes([idx]) for idx in range(256)}

        for i in range(num_merges):
            pair_freq = dict()

            for chunk_ids in ids:
                self._get_pair_freq(chunk_ids, pair_freq)
            
            top_pair = max(pair_freq, key=pair_freq.get)
            idx = 256 + i
            ids = [self._merge(chunk_ids, top_pair, idx)
                   for chunk_ids in ids]

            p0, p1 = top_pair

            vocab[idx] = vocab[p0] + vocab[p1]

            if verbose:
                print(f"merge {i+1}/{num_merges}: {top_pair} -> {idx} ({vocab[idx]}) had {pair_freq[top_pair]} occurrences")

            merges[top_pair] = idx

        self.merges  = merges

        self.vocab = vocab # Map index of the token to the bytes

        return

    def _encode_chunk(self, text: str):
        """
        The approach is to encode the basic tokens and then follow up to build more complex tokens.
        Justification:
        If we don't follow this approach and say the text is "your's sincerely", and the tokens are ["y", "o", "u", "r", "'s" "you", "your", "your's"], how would you find the word "your's" directly?
        The better appraoch is to go step by step. Merge the basic pairs first, i.e., with lower index and then merge the complex pairs.
        """
        tokens = list(text.encode("utf-8"))

        while len(tokens) >= 2:
            pair_freq = dict()
            self._get_pair_freq(tokens, pair_freq)
            # Find the token with the least index in merges because given the current tokens, it is only possible to find a pair at the lower indices of merges rather than higher indices. The higher indices merges would be performed later.
            merge_pair = min(pair_freq, key = lambda x: self.merges.get(x, float("inf")))
            if merge_pair not in self.merges:
                break # No further merging possible

            idx = self.merges[merge_pair]
            tokens = self._merge(tokens, merge_pair, idx)
        return tokens
    
    def encode(self, text: str):
        text_chunks = re.findall(self.compiled_pattern, text)

        tokens = []

        for chunk in text_chunks:
            tokens.extend(self._encode_chunk(chunk))

        return tokens
    
    def decode(self, ids: list[int]):
        tokens: bytearray = b"".join([self.vocab[idx] for idx in ids])
        text: str = tokens.decode("utf-8", errors = "replace")
        return text
    
    def save(self, vocab_file_name):
        # First save the token_character -> idx mapping from 0...255
        # Then save the merge -> idx mapping from merges dictionary
        with open(vocab_file_name, "w") as f:
            for i in range(256):
                ch = bytes([i]).decode("utf-8", errors="replace")
                f.write(f"[{ch}] -> {i}\n")
            
            for (p0, p1), idx in self.merges.items():
                ch0 = self.vocab[p0].decode("utf-8", errors = "replace")
                ch1 = self.vocab[p1].decode("utf-8", errors = "replace")
                f.write(f"[{ch0}][{ch1}] -> [{ch0 + ch1}] {idx}\n")

        return

    def _get_pair_freq(self, ids: list[int], count_dict: dict):
        for pair in zip(ids, ids[1:]):
            count_dict[pair] = count_dict.get(pair, 0) + 1
        return count_dict
    
    def _merge(self, ids, pair, idx):
        """
        Iterate through the list of ids and replace the pair with `idx` if found
        """
        newids = []
        i = 0
        while i < len(ids):
            if (i < len(ids) - 1) and ((ids[i], ids[i+1]) == pair):
                newids.append(idx)
                i += 2
            else:
                newids.append(ids[i])
                i += 1
        return newids
    

def main():
    file_name = "taylorswift.txt"
    with open(file_name, "r") as f:
        text = f.read()

    tokenizer = RegexTokenizer()
    tokenizer.train(text, vocab_size=768, verbose=True)

    vocab_file_name = "regex_tokenizer.vocab"
    tokenizer.save(vocab_file_name)

    print(f"\nSuccessfully saved tokenizer at {vocab_file_name}\n")

    print(f"Sample text: \'{sample_text}\'\n")

    ids = tokenizer.encode(sample_text)
    print(f"Encoded ids: {ids}\n")

    encoded_tokens = [tokenizer.decode([idx]) for idx in ids]
    print(f"Encoded tokens: {encoded_tokens}\n")

    decoded_text = tokenizer.decode(ids)
    print(f"Decoded text: \'{decoded_text}\'\n")

    if sample_text == decoded_text:
        print("Success! The decoded text is same as the original text.")
    else:
        print("Failed! The decoded text is different from the original text.")
    
    return

if __name__ == '__main__':
    main()