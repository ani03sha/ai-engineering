"""
This class has similar logic to CharacterBytePairEncoding (defined in character_level_byte_pair_encoding.py).
The only difference is that it processes actual bytes instead of characters, rest of the logic remains same.

I am duplicating the logic here so that this class is complete in itself.
In production, we will create a common class for common logic and have these two classes refer to it.
"""

from collections import Counter
from typing import Any


class ByteBytePairEncoding:
    def __init__(self):
        pass

    def fit_corpus(self, corpus: list[str]) -> dict[tuple[int | str, ...], int]:
        """
        Convert list of words into a dict mapping symbol tuples -> frequencies.
        """
        word_frequencies = Counter(corpus)
        processed = {}
        for word, frequency in word_frequencies.items():
            # Convert each word to bytes first
            word_bytes = word.encode("utf-8")
            # Convert these bytes to stringified bytes
            symbols = list(word_bytes) + ["</w>"]  # Bytes + end marker
            processed[tuple(symbols)] = frequency

        return processed

    def get_pair_frequencies(
        self, corpus: dict[tuple[int | str, ...], int]
    ) -> dict[tuple[int | str | tuple, ...], int]:
        """
        Compute frequency of adjacent symbol pairs across the corpus.

        corpus: dict where
            key = tuple of symbols (e.g. (108, 111, 119, "</w>"))
            value = word frequency (int)

        returns: dict mapping (symbol1, symbol2) -> frequency count
        """
        pair_frequencies = Counter()

        for word_symbols, frequency in corpus.items():
            # Skip 1-byte symbols, since they don't have a pair
            if len(word_symbols) < 2:
                continue

            # Sliding window: count each adjacent pair once per occurrence of the word
            for i in range(len(word_symbols) - 1):
                pair = (word_symbols[i], word_symbols[i + 1])
                pair_frequencies[pair] += frequency

        return dict(pair_frequencies)

    def merge_most_frequent_pair(
        self, corpus: dict[tuple[Any, ...], int], target_pair: tuple[Any, Any]
    ) -> dict[tuple[Any, ...], int]:
        updated_corpus: dict[tuple[int | str | tuple, ...], int] = {}
        for word_symbols, frequency in corpus.items():
            symbols = list(word_symbols)
            i = 0
            while i < len(symbols) - 1:
                if symbols[i + 1] == "</w>":
                    i += 1
                    continue

                # Join the symbols to form the pair
                pair = (symbols[i], symbols[i + 1])
                if pair == target_pair:
                    merged_symbol = tuple(self._flatten(pair))
                    symbols[i : i + 2] = [merged_symbol]
                else:
                    i += 1
            # Add new tuple to the updated corpus
            updated_corpus[tuple(symbols)] = frequency

        return updated_corpus

    def _flatten(self, symbol):
        if isinstance(symbol, int):
            return [symbol]
        elif isinstance(symbol, tuple):
            result = []
            for s in symbol:
                result.extend(self._flatten(s))
            return result
        else:
            return []  # Skip "</w>"
        
    def train(self, corpus: dict[tuple[Any, ...], int], num_merges: int) -> dict[tuple[Any, ...], int]:
        """
        Learn BPE merges from a preprocessed corpus.
        corpus: dict of symbol sequences -> frequency
        num_merges: number of merge operations to perform
        """
        self.merges: list[tuple[Any, Any]] = []
        
        # Train for num_merges iteration
        for i in range(num_merges):
            # Get pair frequencies
            pair_frequencies = self.get_pair_frequencies(corpus)
            # Stop when no adjacent pairs left
            if not pair_frequencies:
                print("Stopping: no adjacent pairs left")
                break
            
            # Pick the most frequent pair
            most_frequent_pair = max(pair_frequencies, key=pair_frequencies.get)
            print(f"Merge: {i + 1}: {most_frequent_pair} (frequency={pair_frequencies[most_frequent_pair]})")
            
            # Merge the pair in the corpus
            corpus = self.merge_most_frequent_pair(corpus, most_frequent_pair)
            
            # Record the merge
            self.merges.append(most_frequent_pair)
        
        print("\nLearned merges:")
        for index, m in enumerate(self.merges):
            print(f"{index + 1}: {m}")
        
        return corpus
    
    def encode(self, word: str) -> list[Any]:
        """
        Encode a single word using learned BPE merges
        """
        symbols = list(word.encode("utf-8")) + ["</w>"]
        
        for pair in self.merges:
            i = 0
            while i < len(symbols) - 1:
                if (symbols[i], symbols[i + 1]) == pair:
                    merged_symbol = tuple(self._flatten(pair))
                    symbols[i:i+2] = [merged_symbol]
                else:
                    i += 1
        
        return symbols

    def decode(self, tokens: list[Any]) -> str:
        """
        Decode tokens back to the string (removes </w> markers).
        """
        byte_sequence = []
        for token in tokens:
            if token == "</w>":
                continue
            
            byte_sequence.extend(self._flatten(token))
        
        return bytes(byte_sequence).decode("utf-8", errors="replace")
    
    
def main():
    bbpe = ByteBytePairEncoding()
    words = [
        "low",
        "lower",
        "newest",
        "newest",
        "newest",
        "widest"
    ]
    corpus = bbpe.fit_corpus(words)

    # Train BPE
    final_corpus = bbpe.train(corpus, num_merges=10)
    print("Final Corpus: ", final_corpus)
    
    # Encode a word
    encoded = bbpe.encode("lowest")
    print("Encoded:", encoded)
    
    # Decode back
    decoded = bbpe.decode(encoded)
    print("Decoded:", decoded)
    
    # Test new word
    encoded = bbpe.encode("slowest")
    print("Encoded new word:", encoded)
    
    decoded = bbpe.decode(encoded)
    print("Decoded new word:", decoded)
    

if __name__ == "__main__":
    main()
