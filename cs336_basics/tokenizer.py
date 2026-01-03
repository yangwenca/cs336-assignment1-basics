from collections.abc import Iterable, Iterator
import regex as re

from cs336_basics.bpe import PAT
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
        self.merges = set(merges)
        self.special_tokens = sorted(special_tokens, key=len, reverse=True) if special_tokens else []


    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] | None = None,
    ) -> "Tokenizer":
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
                    while True:
                        pair_idx = float('inf')
                        merge = ()
                        # check whether this is any matches
                        for idx in range(len(byte_array) - 1):
                            tmp = (byte_array[idx], byte_array[idx+1])
                            token = byte_array[idx] + byte_array[idx+1]
                            if tmp in self.merges and self.token_id[token] < pair_idx:
                                pair_idx = self.token_id[token]
                                merge = tmp
                        if pair_idx == float('inf'):
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
# assert tt.encode(inputs) == [9, 7, 1, 5, 0, 1, 5, 3, 0, 11, 0, 9]

# inputs = 'the cat ate  the'
# assert tt.encode(inputs) == [9, 7, 1, 5, 0, 1, 5, 3, 0, 0, 9]


'''
Tokenizer experiments
(a) & (b)
TinyStories
number of bytes is  22502601
number of tokens is  5448321
ratio is  4.130190016337143


OWT

own text

number of bytes is  13968
number of tinystory tokens is  10277
number of owt tokens 7092
tinystory ratio is  1.3591515033570107
owt ratio is  1.9695431472081217

(c)
bytes/sec: 1567558.77
MB/sec: 1.57
825GB: 825 * 10e9 / (1.57 * 1e6) = 5254777 sec ~ 60 days

(d)
it must be unsigned instead of signed because ID is always non-negative
8 bit is only 2**8 = 256.
16 bit is 2**16 = 65536
vocab size 10k to 32k. thus, it needs 16 bit
'''


# def tokenizer_experiments():
#     save_file = False
#     prefix = "/Users/YangWen/Documents/Code/github/assignment1-basics/data/"

#     TinyStoriesTokenzier = Tokenizer.from_files(
#         vocab_filepath=prefix + "TinyStoriesV2-GPT4-train_vocab.pkl",
#         merges_filepath=prefix + "TinyStoriesV2-GPT4-train_merge.pkl",
#         special_tokens=["<|endoftext|>"],
#     )

#     OWTTokenizer = Tokenizer.from_files(
#         vocab_filepath=prefix + "owt_train_vocab.pkl",
#         merges_filepath=prefix + "owt_train_merge.pkl",
#         special_tokens=["<|endoftext|>"],
#     )

#     if save_file:
#         file_name = "owt_train"

#         with open(prefix + file_name + ".txt") as f:
#             ids = []
#             for _id in OWTTokenizer.encode_iterable(f):
#                 ids.append(_id)

#         import numpy as np
#         arr = np.array(ids, dtype=np.uint16)
#         np.save(prefix + file_name + "-id.npy", arr)
#     else:

#         with open("/Users/YangWen/Downloads/failed_xplanner_arm64_fp16_build_log.txt", "r", encoding="utf-8") as f:
#             text = f.read()

#         total_bytes = len(text.encode('utf-8'))
#         total_tokens = len(TinyStoriesTokenzier.encode(text))
#         total_tokens_owt = len(OWTTokenizer.encode(text))
#         print("number of bytes is ", total_bytes)
#         print("number of tinystory tokens is ", total_tokens)
#         print("number of owt tokens", total_tokens_owt)
#         print("tinystory ratio is ", total_bytes / total_tokens)
#         print("owt ratio is ", total_bytes / total_tokens_owt)

#         import time

#         diff = 0
#         counter = 10
#         for _ in range(counter):
#             start = time.perf_counter()
#             TinyStoriesTokenzier.encode(text)
#             end = time.perf_counter()
#             diff += end - start

#         throughput = total_bytes / (diff / counter)
#         print(f"bytes/sec: {throughput:.2f}")
#         print(f"MB/sec: {throughput/1e6:.2f}")


# tokenizer_experiments()
