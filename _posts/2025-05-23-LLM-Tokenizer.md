---
title: 'Language Models 1: Tokenization'
date: 2025-04-01 09:52:23
tags:
- tokenization
categories:
- language models
pin: false
math: true
---

Raw text is generally represented as Unicode strings. For example, in python we can do following thing:

```python
string = "Hello, 🌍! 你好!"
```

A language model places a probability distribution over sequences of tokens (usually represented by integer indices).

```python
indices = [15496, 11, 995, 0]
```

So we need a procedure that *encodes* strings into tokens. We also need a procedure that *decodes* tokens back into strings. A  [Tokenizer](https://stanford-cs336.github.io/spring2025-lectures/?trace=var%2Ftraces%2Flecture_01.json&step=357#) is a class that implements the encode and decode methods. The **vocabulary size** is number of possible tokens (integers).

## Tokenization Examples

To get a feel for how tokenizers work, you can play with this  [interactive site](https://tiktokenizer.vercel.app/?encoder=gpt2) which contains the implementation of the tokenizer of modern model implementation.

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@master/uPic/image-20250525111633838.png" alt="image-20250525111633838" style="zoom:50%;" />

We can get following observations:

- A word and its preceding space are part of the same token (e.g., " world").
- A word at the beginning and in the middle are represented differently (e.g., "hello hello").
- Numbers are tokenized into every few digits.

Let's see the GPT-2 tokenizer from OpenAI (tiktoken) in action. You can find more details here: [tiktoken](https://github.com/openai/tiktoken)

```python
tokenizer = get_gpt2_tokenizer()
string = "Hello, 🌍! 你好!"  
# Check that encode() and decode() roundtrip:
indices = tokenizer.encode(string)  
reconstructed_string = tokenizer.decode(indices)  
compression_ratio = get_compression_ratio(string, indices)
```

The result shows as follows:

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@master/uPic/image-20250525112142635.png" alt="image-20250525112142635" style="zoom:50%;" />

## Character-based tokenization

The string is representing as unicode.A unicode string is a sequence of Unicode characters. In python, each character can be converted into a code point (integer) via `ord`:

```python
assert ord("a") == 97
assert ord("🌍") == 127757
```

It can be converted back via `chr`:

```python
assert chr(97) == "a"
assert chr(127757) == "🌍"
```

We can use `ord` and `chr` to build a simple character-based tokenizer.

```python
class CharacterTokenizer(Tokenizer):
    """Represent a string as a sequence of Unicode code points."""
    def encode(self, string: str) -> list[int]:
        return list(map(ord, string))
    def decode(self, indices: list[int]) -> str:
        return "".join(map(chr, indices))

tokenizer = CharacterTokenizer()
string = "Hello, 🌍! 你好!"  
indices = tokenizer.encode(string)  
reconstructed_string = tokenizer.decode(indices) 
```

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@master/uPic/image-20250525112937535.png" alt="image-20250525112937535" style="zoom:50%;" />

However, there are approximately 150K Unicode characters.  [[Wikipedia\]](https://en.wikipedia.org/wiki/List_of_Unicode_characters). So this is **a very large vocabulary**. Besides, in current vocabulary, **many characters are quite rare** (e.g., 🌍), which is inefficient use of the vocabulary. Often it's not practical to use character-based tokenizer.

## Byte-based tokenization

Unicode strings can be represented as a sequence of bytes, which can be represented by integers between 0 and 255. The most common Unicode encoding is  [UTF-8](https://en.wikipedia.org/wiki/UTF-8). Some Unicode characters are represented by one byte:

```python
assert bytes("a", encoding="utf-8") == b"a"
```

Others take multiple bytes:

```python
assert bytes("🌍", encoding="utf-8") == b"\xf0\x9f\x8c\x8d"
```

Now let's build a byte-based  `Tokenizer` and make sure it round-trips:

```python
class ByteTokenizer(Tokenizer):
    """Represent a string as a sequence of bytes."""
    def encode(self, string: str) -> list[int]:
        string_bytes = string.encode("utf-8")  
        indices = list(map(int, string_bytes)) 
        return indices
    def decode(self, indices: list[int]) -> str:
        string_bytes = bytes(indices)  
        string = string_bytes.decode("utf-8")  
        return string

tokenizer = ByteTokenizer()
string = "Hello, 🌍! 你好!"  
indices = tokenizer.encode(string) 
reconstructed_string = tokenizer.decode(indices) 
assert string == reconstructed_string
vocabulary_size = 256  
compression_ratio = get_compression_ratio(string, indices) 
```

The result is:

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@master/uPic/image-20250525121935702.png" alt="image-20250525121935702" style="zoom:50%;" />

However, **the compression ratio of byte-based tokenizer is terrible**, which means the sequences will be too long.Given that the context length of a Transformer is limited (since attention is quadratic), this is not looking great.

## Word-based tokenization

Another approach (closer to what was done classically in NLP) is to split strings into words. for example:

```python
string = "I'll say supercalifragilisticexpialidocious!"

# This regular expression keeps all alphanumeric characters together (words).
segments = regex.findall(r"\w+|.", string)

```

The result is:

<img src="https://cdn.jsdelivr.net/gh/xjh42/oss@master/uPic/image-20250525123315473.png" alt="image-20250525123315473" style="zoom: 50%;" />

To turn this into a `Tokenizer`, we need to map these segments into integers. Then, we can build a mapping from each segment into an integer. But there are problems:

- The number of words is huge (like for Unicode characters).

- Many words are rare and the model won't learn much about them.

- This doesn't obviously provide a fixed vocabulary size.

New words we haven't seen during training get a special UNK token, which is ugly and can mess up perplexity calculations. But the compression ratio is quite good.

```python
vocabulary_size = "Number of distinct segments in the training data"
compression_ratio = get_compression_ratio(string, segments)  # which is 8.8
```

## [Byte Pair Encoding (BPE)](https://en.wikipedia.org/wiki/Byte-pair_encoding)

Finally, we'll see a practical tokenizer--BPE Tokenizer. The BPE algorithm was introduced by Philip Gage in 1994 for data compression[[article\]](http://www.pennelynn.com/Documents/CUJ/HTML/94HTML/19940045.HTM). It was adapted to NLP for neural machine translation[[Sennrich+ 2015\]](https://arxiv.org/abs/1508.07909) . BPE was then used by GPT-2[[Radford+ 2019\]](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf).

The basic idea is to *train* the tokenizer on raw text to automatically determine the vocabulary. It makes common sequences of characters are represented by a single token, rare sequences are represented by many tokens.

### Vanilla BPE

The simple BPE algorithm is to start with each byte as a token, and successively merge the most common pair of adjacent tokens.

- train bpe to get bpe parameters:

  ```python
  from abc import ABC
  from collections import defaultdict
  from dataclasses import dataclass
  
  @dataclass(frozen=True)
  class BPETokenizerParams:
      """
      All you need to specify BPE tokenizer.
      """
      vocab: dict[int, bytes] # int->bytes
      merges: dict[tuple[int, int], int] # index1, index2 -> new_index
  
  class Tokenizer(ABC):
      """Abstract base class for tokenizer classes."""
      def encode(self, text: str) -> list[int]:
          raise NotImplementedError
      def decode(self, indices: list[int]) -> str:
          raise NotImplementedError
  
  def train_bpe(string: str, num_merges: int) -> BPETokenizerParams:
      indices = list(map(int, string.encode(encoding="utf-8")))
      merges: dict[tuple[int, int], int] = {}
      vocab: dict[int, bytes] = {x : bytes([x]) for x in range(256)}
  
      for i in range(num_merges):
          # get counts
          counts = defaultdict(int)
          for (index1, index2) in zip(indices, indices[1:]):
              counts[(index1, index2)] += 1
          # get max count
          pairs = max(counts, key=counts.get)
          index1, index2 = pairs
  
          # merge counts
          new_index = 256 + i
          merges[pairs] = new_index
          vocab[new_index] = vocab[index1] + vocab[index2]
          indices = merge(indices, pairs, new_index)
      return BPETokenizerParams(vocab, merges)
  
  def merge(indices: list[int], pairs: tuple[int,int], new_index: int) -> list[int]:
      new_indices = []
      i = 0
      while i < len(indices):
          if i + 1 < len(indices) and indices[i] == pairs[0] and indices[i + 1] == pairs[1]:
              new_indices.append(new_index)
              i += 2
          else:
              new_indices.append(indices[i])
              i += 1
      return new_indices
  ```

  - `BPETokenizer` to implement `encode` and `decode` method:

  ```python
  class BPETokenizer(Tokenizer):
      def __init__(self, params: BPETokenizerParams):
          self.params = params
  
      def encode(self, text: str) -> list[int]:
          indices = list(map(int, text.encode(encoding="utf-8")))
          for pair, new_index in self.params.merges.items():
              indices = merge(indices, pair, new_index)
          return indices
  
      def decode(self, indices: list[int]) -> str:
          byte_lists = list(map(self.params.vocab.get, indices))
          return b"".join(byte_lists).decode("utf-8")
  ```

  

- Test examples:

```python
if __name__ == "__main__":
    string = "the cat in the hat"  # @inspect string
    params = train_bpe(string, num_merges=3)
    tokenizer = BPETokenizer(params)
    string = "the quick brown fox"  # @inspect string
    indices = tokenizer.encode(string)  # @inspect indices
    reconstructed_string = tokenizer.decode(indices)  # @inspect reconstructed_string
    assert string == reconstructed_string
```
