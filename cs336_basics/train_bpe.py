from __future__ import annotations

import os
from collections import defaultdict
from collections.abc import Iterable
from typing import IO, Any, BinaryIO

import numpy.typing as npt
import regex as re
import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor
from cs336_basics.pretokenization_example import find_chunk_boundaries


'''
Problem (unicode1):

(a) chr(0) return null character
(b) printed is nothing. in repr is '\\x00' in ascii
(c) it does not prit it out, but it is existed inside string

Problem (unicode2)
(a): UTF-8 smallest, the most common encoding on the web, byte-oriented and variable length (perfect for BPE),
encodes ASCII characters as 1 byte, which is extremely important, avoids endianness issues, fully reversible
(b): for non ascii characters, it needs to represented by more than 1 byte.
decoding these bytes together or separately produces different different results.
(c): 0xC0 0xAF
0xC0 starts a 2-byte UTF-8 sequence
But any sequence starting with 0xC0 or 0xC1 is forbidden (they are “overlong” encodings)
0xC0 0xAF is an invalid overlong form of '/'
Decoders must reject it according to the UTF-8 spec
'''

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def implement_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """
    # initialize a byte vocabulary with all 256 possible byte
    default_size = 256
    vocab = {idx: bytes([idx]) for idx in range(default_size)}
    cur_idx = default_size

    # add special tokens to the vocab
    for special_token in special_tokens:
        vocab[cur_idx] = special_token.encode('utf-8')
        cur_idx += 1
    # string to count
    string_count = defaultdict(int)
    tmp_special = set(special_tokens)
    with open(input_path, "rb") as f:
        num_processes = 1
        boundaries = find_chunk_boundaries(f, num_processes, special_tokens[0].encode('utf-8'))
        # pre-tokenize
        # todo, optimize with parallel
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            # removing special tokens before pre-tokenization
            escaped_tokens = [re.escape(t) for t in special_tokens]
            panew_tokenern = "|".join(escaped_tokens)
            for doc in re.split(panew_tokenern, chunk):
                for tmp_str in re.finditer(PAT, doc):
                    tmp_str = tmp_str.group()
                    if tmp_str in tmp_special:
                        continue
                    string_count[tmp_str] += 1
    special_count = len(special_tokens)
    left_count = vocab_size - default_size - special_count
    merges = []
    if left_count == 0:
        return vocab, merges
    # count to token
    count_token = defaultdict(set)
    # token to count
    token_count = defaultdict(int)
    # token to string
    token_string = defaultdict(set)
    # string to token
    string_token = defaultdict(list)

    for string, count in string_count.items():
        bytearray = [bytes([b]) for b in string.encode("utf-8")]
        for i in range(len(bytearray) - 1):
            token = (bytearray[i], bytearray[i+1])
            token_count[token] += count
            token_string[token].add(string)
            string_token[string].append(token)
    for token, count in token_count.items():
        count_token[count].add(token)


    for i in range(left_count):
        if len(count_token) == 0:
            break
        count = max(count_token.keys())
        if count == 0:
            break
        tokens = count_token[count]
        token = max(tokens)
        merges.append(token)
        vocab[cur_idx] = token[0] + token[1]
        cur_idx += 1
        if i == left_count - 1:
            break
        ## token
        # count_token
        count_token[count].remove(token)
        if len(count_token[count]) == 0:
            del count_token[count]
        # token_count
        del token_count[token]
        # token_string
        all_string = token_string[token]
        del token_string[token]

        remove_token_count = defaultdict(int)
        remove_token_string = defaultdict(set)
        add_token_count = defaultdict(int)
        add_token_string = defaultdict(set)
        for cur_string in all_string:
            tmp = []
            length = len(string_token[cur_string])
            cur_count = string_count[cur_string]
            for token_idx, cur_token in enumerate(string_token[cur_string]):
                if cur_token == token:
                    continue
                elif (((token_idx < length - 1) and (string_token[cur_string][token_idx + 1] == token)) or 
                    ((token_idx > 0) and (string_token[cur_string][token_idx - 1] == token))):
                    ## cur_token
                    remove_token_count[cur_token] += cur_count
                    remove_token_string[cur_token].add(cur_string)

                    ## new_token
                    if (token_idx < length - 1) and (string_token[cur_string][token_idx + 1] == token):
                        new_token = (cur_token[0], token[0] + token[1])
                    else:
                        new_token = (token[0] + token[1], cur_token[1])
                    tmp.append(new_token)
                    add_token_count[new_token] += cur_count
                    add_token_string[new_token].add(cur_string)
                else:
                    tmp.append(cur_token)
            ## token, cur_token, new_token
            # string_token
            string_token[cur_string] = tmp

        ## remove old token
        # token_count
        # count_token
        for token, reduce_count in remove_token_count.items():
            old_count = token_count[token]
            new_count = old_count - reduce_count
            token_count[token] = new_count
            if new_count == 0:
                del token_count[token]
            count_token[old_count].discard(token)
            if len(count_token[old_count]) == 0:
                del count_token[old_count]
            if new_count != 0:
                count_token[new_count].add(token)
        # token_string
        for token, strings in remove_token_string.items():
            token_string[token].difference_update(strings)
            if len(token_string[token]) == 0:
                del token_string[token]
        
        ## add new token
        # token_count
        # count_token
        for token, count in add_token_count.items():
            token_count[token] += count
            count_token[count].add(token)
        # token_string
        for token, strings in add_token_string.items():
            token_string[token].update(strings)
    return (vocab, merges)


# def test_implement_train_bpe():
#     input_path = "/Users/YangWen/Documents/Code/github/assignment1-basics/cs336_basics/train_bpe_file.txt"
#     with open(input_path, "w") as f:
#         f.write(" low low low low low lower lower widest widest widest newest newest newest newest newest newest")
#     vocab, merges = implement_train_bpe(input_path, 300, ["<|endoftext|>"])
#     print(merges)
#     # print(vocab, merges)
# test_implement_train_bpe()