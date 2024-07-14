class BasicTokenizer:

    def __init__(self) -> None:
        pass

    def train(self, text: str, vocab_size: int, verbose: bool = False):
        self.vocab_size = vocab_size

        num_merges = vocab_size - 256
        
        assert num_merges >= 0, "vocab_size should be more than equal to 256"

        text_bytes = text.encode("utf-8")
        ids = list(text_bytes) # List of integers in range 0..255

        merges = dict()

        vocab = {idx: bytes([idx]) for idx in range(256)}

        for i in range(num_merges):
            pair_freq = self._get_pair_freq(ids)
            top_pair = max(pair_freq, key=pair_freq.get)
            idx = 256 + i
            ids = self._merge(ids, top_pair, idx)

            p0, p1 = top_pair

            vocab[idx] = vocab[p0] + vocab[p1]

            if verbose:
                print(f"merge {i+1}/{num_merges}: {top_pair} -> {idx} ({vocab[idx]}) had {pair_freq[top_pair]} occurrences")

            merges[top_pair] = idx

        self.merges  = merges

        self.vocab = vocab # Map index of the token to the bytes

        return
    
    def encode(self, text: str):
        """
        The approach is to encode the basic tokens and then follow up to build more complex tokens.
        Justification:
        If we don't follow this approach and say the text is "your's sincerely", and the tokens are ["y", "o", "u", "r", "'s" "you", "your", "your's"], how would you find the word "your's" directly?
        The better appraoch is to go step by step. Merge the basic pairs first, i.e., with lower index and then merge the complex pairs.
        """
        tokens = list(text.encode("utf-8"))

        while len(tokens) >= 2:
            pair_freq = self._get_pair_freq(tokens)
            # Find the token with the least index in merges because given the current tokens, it is only possible to find a pair at the lower indices of merges rather than higher indices. The higher indices merges would be performed later.
            merge_pair = min(pair_freq, key = lambda x: self.merges.get(x, float("inf")))
            if merge_pair not in self.merges:
                break # No further merging possible

            idx = self.merges[merge_pair]
            tokens = self._merge(tokens, merge_pair, idx)
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

    def _get_pair_freq(self, ids):
        count = dict()
        for pair in zip(ids, ids[1:]):
            count[pair] = count.get(pair, 0) + 1
        return count
    
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

    tokenizer = BasicTokenizer()
    tokenizer.train(text, vocab_size=512, verbose=True)

    vocab_file_name = "basic_tokenizer.vocab"
    tokenizer.save(vocab_file_name)

    print(f"\nSuccessfully saved tokenizer at {vocab_file_name}\n")

    sample_text = "Hello, my name is Taylor Swift. I am a singer and songwriter."
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