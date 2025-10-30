from collections import Counter


class BytePairEncoding:
    def __init__(self):
        pass
    
    
    def fit_corpus(self, corpus: list[str]) -> dict[tuple[str, ...], int]:
        """
        Convert list of words into a dict mapping symbol tuples -> frequencies.
        """
        word_frequencies = Counter(corpus)
        processed = {}
        for word, frequency in word_frequencies.items():
            symbols = list(word) + ["</w>"] # Split into characters + end marker
            processed[tuple(symbols)] = frequency
            
        return processed
    
    
    def get_pair_frequencies(self, corpus: dict[tuple[str, ...], int]) -> dict[tuple[str, str], int]:
        """
        Compute frequency of adjacent symbol pairs across the corpus.

        corpus: dict where
            key = tuple of symbols (e.g. ("l", "o", "w", "</w>"))
            value = word frequency (int)

        returns: dict mapping (symbol1, symbol2) -> frequency count
        """
        pair_frequencies = Counter()
        
        for word_symbols, frequency in corpus.items():
            # Skip 1-character words, since they have no pairs
            if len(word_symbols) < 2:
                continue
            
            # Sliding window: count each adjacent pair once per occurrence of the word
            for i in range(len(word_symbols) - 1):
                pair = (word_symbols[i], word_symbols[i + 1])
                pair_frequencies[pair] += frequency # Weighed by word frequency
                
        return dict(pair_frequencies)
    
    
    def merge_most_frequent_pair(self, corpus: dict[tuple[str, ...], int], target_pair: tuple[str, str]) -> dict[tuple[str, ...], int]:
        updated_corpus: dict[tuple[str, ...], int] = {}
        for word_symbols, count in corpus.items():
            symbols = list(word_symbols)
            i = 0
            while i < len(symbols) - 1:
                # Join the symbols to form the pair
                pair = (symbols[i], symbols[i + 1])
                if pair == target_pair:
                    merged_symbol = "".join(pair)
                    symbols[i:i+2] = [merged_symbol] # Replace two elements with one merged element
                    i += 1
                    continue
                else:
                    i += 1
            # Add new tuple to the updated corpus
            updated_corpus[tuple(symbols)] = count

        return updated_corpus

    def train(self, corpus: dict[tuple[str, ...], int], num_merges: int) -> dict[tuple[str, ...], int]:
        """
        Learn BPE merges from a preprocessed corpus.
        corpus: dict of symbol sequences -> frequency
        num_merges: number of merge operations to perform
        """
        self.merges: list[tuple[str, str]] = []
        
        # Train for num_merges iterations
        for i in range(num_merges):
            # Get pair frequencies
            pair_frequencies = self.get_pair_frequencies(corpus)
            # Stop when no adjacent pairs left
            if not pair_frequencies:
                print(f"Stopping early: no pairs left to merge.")
                break
            
            # Pick the most frequent pair
            most_frequent_pair = max(pair_frequencies, key=lambda p: (pair_frequencies[p], p))
            print(f"Merge: {i + 1}: {most_frequent_pair} (frequency={pair_frequencies[most_frequent_pair]})")
            
            # Merge the pair in the corpus
            corpus = self.merge_most_frequent_pair(corpus, most_frequent_pair)
            
            # Record the merge
            self.merges.append(most_frequent_pair)
            
        print("\nLearned merges:")
        for index, m in enumerate(self.merges):
            print(f"{index + 1}: {m}")
        
        return corpus
    
    def encode(self, word: str) -> list[str]:
        """
        Encode a single word using learned BPE merges
        """
        symbols = list(word) + ["</w>"]
        
        for pair in self.merges:
            i = 0
            while i < len(symbols) - 1:
                if (symbols[i], symbols[i + 1]) == pair:
                    merged = "".join(pair)
                    symbols[i:i+2] = [merged]
                else:
                    i += 1
        
        return symbols
    
    def decode(self, tokens: list[str]) -> str:
        """
        Decode tokens back to the string (removes </w> markers).
        """
        text = "".join(tokens)
        return text.replace("</w>", "").strip()
        
        
        
def main():
    bpe = BytePairEncoding()
    words = [
        "low",
        "lower",
        "newest",
        "newest",
        "newest",
        "widest"
    ]
    corpus = bpe.fit_corpus(words)

    # Train BPE
    final_corpus = bpe.train(corpus, num_merges=10)
    print("Final Corpus: ", final_corpus)
    
    # Encode a word
    encoded = bpe.encode("lowest")
    print("Encoded:", encoded)
    
    # Decode back
    decoded = bpe.decode(encoded)
    print("Decoded:", decoded)
    
    # Test new word
    encoded = bpe.encode("slowest")
    print("Encoded new word:", encoded)
    
    decoded = bpe.decode(encoded)
    print("Decoded new word:", decoded)
    

if __name__ == "__main__":
    main()
