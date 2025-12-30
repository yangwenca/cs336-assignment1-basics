from collections.abc import Iterable, Iterator
import os
import regex as re

from cs336_basics.train_bpe import PAT
import pickle

class Tokenizer:
    def __init__(
            self,
            vocab: dict[int, bytes],
            merges: list[tuple[bytes, bytes]],
            special_tokens: list[str] | None = None,
        ):
        self.vocab = vocab
        self.token_id = {value : key for key, value in vocab.items()}
        self.merges = merges
        self.special_tokens = sorted(special_tokens, key=len, reverse=True) if special_tokens else []


    @classmethod
    def from_files(
            cls,
            vocab_filepath: str,
            merges_filepath: str,
            special_tokens: list[str] | None = None,
        ):
        try:
            with open(vocab_filepath, "rb") as f:
                vocab = pickle.load(f)
            with open(merges_filepath, "rb") as f:
                merges = pickle.load(f)
        except (FileNotFoundError, pickle.UnpicklingError) as e:
            print("Failed to load pickle:", e)
        return cls(vocab, merges, special_tokens)


    def encode(self, text: str) -> list[int]:
        escaped_tokens = [re.escape(t) for t in self.special_tokens]
        pattern = "|".join(escaped_tokens)
        # keep the dropped pattern
        pattern = f"({pattern})"
        if not self.special_tokens:
            chunks = [text]
        else:
            chunks = [c for c in re.compile(pattern).split(text) if c]
        ans = []
        def encode_id(txt):
            if txt in set(self.special_tokens):
                ans.append(self.token_id[txt.encode('utf-8')])
                return
            for tmp_str in re.finditer(PAT, txt):
                byte_str = tmp_str.group().encode('utf-8')
                if byte_str in self.vocab:
                    ans.append(self.vocab[byte_str])
                else:
                    byte_array = [bytes([b]) for b in byte_str]
                    for merge in self.merges:
                        if len(byte_array) == 1:
                            break
                        idx = 0
                        tmp = []
                        while idx < len(byte_array):
                            if byte_array[idx] == merge[0] and \
                            (idx + 1) < len(byte_array) and \
                            byte_array[idx+1] == merge[1]:
                                tmp.append(merge[0] + merge[1])
                                idx += 2
                            else:
                                tmp.append(byte_array[idx])
                                idx += 1
                        byte_array = tmp
                    for cur_byte in byte_array:
                        ans.append(self.token_id[cur_byte])
        for chunk in chunks:
            encode_id(chunk)
        return ans
        

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for chunk in iterable:
            yield from self.encode(chunk)


    def decode(self, ids: list[int]) -> str:
        byte_list = []
        for id in ids:
            byte_list.append(self.vocab[id])
        combined_bytes = b"".join(byte_list)
        # replace the malformed bytes with the
        # official Unicode replacement character U+FFFD
        return combined_bytes.decode("utf-8", errors='replace')

# vocab = {0: b' ', 1: b'a', 2: b'c', 3: b'e', 4: b'h',
#          5: b't', 6: b'th', 7: b' c', 8: b' a', 9: b'the', 10: b' at', 11: b'<|endoftext|>'}
# merges = [(b't', b'h'), (b' ', b'c'), (b' ', 'a'), (b'th', b'e'), (b' a', b't')]
# special_token = ["<|endoftext|>"]

# tt = Tokenizer(vocab, merges, special_token)
# inputs = 'the cat ate <|endoftext|> the'
# print(tt.encode(inputs))

# inputs = 'the cat ate  the'
# print(tt.encode(inputs))
