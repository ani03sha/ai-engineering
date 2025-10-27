from collections import Counter
from typing import Iterable, List, Optional


class WordLevelEncoding:
    """
    Word-level tokenizer with:
    - fit(corpus): builds vocabulary and frequency counts
    - encode(text, unknown_policy): encodes text according to policy
    - decode(tokens): decodes token ids to text
    - exposes stats: vocab_size, average_tokens_per_sentence, last_oov_count/fraction
    """
    def __init__(self, add_unk_token: bool = True):
        # Core mappings
        self.mappings: dict[str, int] = {}
        self.reverse_mappings: dict[int, str] = {}
        
        # Frequency + indexing
        self.frequencies: Counter[str] = Counter()
        self.index: int = 0
        
        # Corpus stats
        self.total_sentences: int = 0
        self.total_words: int = 0
        
        # Last encode OOV diagnostics
        self.last_oov_count: int = 0
        self.last_oov_fraction: float = 0.0
        
        # UNK token handling
        self.add_unk_token = add_unk_token
        self.unk_token = "<UNK>"
        self.unk_id: Optional[int] = None
        if self.add_unk_token:
            # Reserve id for UNK token
            self.unk_id = self._reserve_token(self.unk_token)
            
            
    # -------------- Internal Helpers -------------
    def _reserve_token(self, token: str) -> int:
        """Reserve token id if not already present and return its id."""
        if token not in self.mappings:
            token_id = self.index
            self.mappings[token] = token_id
            self.reverse_mappings[token_id] = token
            self.index += 1
            return token_id
        return self.mappings[token]
        
    
    # -------------- Public API -------------    
    def fit(self, corpus: list[str], deterministic_by_frequency: bool = True) -> None:
        """
        Builds vocabulary from the corpus (list of sentences)
        If deterministic_by_frequency = True, assign ids by descending frequency (tie-break: lexicographic)
        This makes runs reproducible
        """
        # Update corpus level counters
        self.total_sentences += len(corpus)
        # Process all sentences in the corpus
        for text in corpus:
            words: list[str] = text.split()
            self.total_words += len(words)
            self.frequencies.update(words)
        
        # Assign ids for tokens we don't already have
        # (We keep any pre-reserved tokens like UNK unchanged)
        existing_tokens = set(self.mappings.keys())
        
        # Decide ordering for new tokens
        new_tokens = [word for word in self.frequencies.keys() if word not in existing_tokens]
        if deterministic_by_frequency:
            # Sort by descending frequency and then lexicographically for tie-break
            new_tokens.sort(key=lambda word: (-self.frequencies[word], word))
        else:
            new_tokens.sort() # Fallback deterministic ordering
        
        for word in new_tokens:
            # Assign id only once per token
            if word not in self.mappings:
                token_id = self.index
                self.mappings[word] = token_id
                self.reverse_mappings[token_id] = word
                self.index += 1
        
                
    def encode(self, text: str, unknown_policy: str = "raise") -> List[int]:
        """
        Encode a sentence into token ids
        Unknown policy: "raise" | "add" | "unk"
            - "raise": raise KeyError if unknown word seen
            - "add": add unknown word to vocab (mutates vocab)
            - "unk": map unknown words to UNK id (requires add_unk_token=True)
        """
        if unknown_policy not in {"raise", "add", "unk"}:
            raise ValueError("Unknown policy must be 'raise', 'add', or 'unk'")
        
        words = text.split()
        if len(words) == 0:
            self.last_oov_count = 0
            self.last_oov_fraction = 0.0
            return []
        
        encoded: List[int] = []
        oov_count = 0
        
        for word in words:
            if word in self.mappings:
                encoded.append(self.mappings[word])
                continue
            
            # Word is unknown
            oov_count += 1
            if unknown_policy == "raise":
                raise KeyError(f"Unknown word during encode: {word!r}")
            elif unknown_policy == "add":
                # Add to vocabulary and return its new id
                token_id = self.index
                self.mappings[word] = token_id
                self.reverse_mappings[token_id] = word
                self.index += 1
                encoded.append(token_id)
                # Also update frequency table so future fits/ordering include this
                self.frequencies[word] += 1
            else: # "unk"
                if self.unk_id is None:
                    raise ValueError("UNK policy requested but UNK token is not enabled.")
                encoded.append(self.unk_id)
                
        # Update last oov diagnostics
        self.last_oov_count = oov_count
        self.last_oov_fraction = oov_count / len(words)
        return encoded
    
    
    def decode(self, encoded: Iterable[int], unknown_token_str: str = "<INVALID>") -> str:
        """
        Decode list of token ids back to a sentence string
        If an id is unknown, replace it with unknown_token_str
        """
        words: List[str] = []
        for token in encoded:
            if token in self.reverse_mappings:
                words.append(self.reverse_mappings[token])
            else:
                words.append(unknown_token_str)
        return " ".join(words)
    
    
    # -------------- Stats/helpers -------------
    def vocabulary_size(self) -> int:
        return len(self.mappings)
    
    def average_tokens_per_sentence(self) -> float:
        if self.total_sentences == 0:
            return 0.0
        return self.total_words / self.total_sentences
    
    def last_oov_stats(self):
        return self.last_oov_count, self.last_oov_fraction
    
    def most_common(self, n: int = 20):
        """
        Return the n most common words in the fitted corpus
        """
        return self.frequencies.most_common(n)
        

def main():
    # Training corpus
    corpus = [
        "the cat sat on the mat",
        "the dog sat on the rug"
    ]

    # Initialize the tokenizer
    tokenizer = WordLevelEncoding(add_unk_token=True)

    # Fit vocabulary on corpus
    tokenizer.fit(corpus)
    print("=== Vocabulary Summary ===")
    print("Vocabulary size:", tokenizer.vocabulary_size())
    print("Average tokens per sentence:", tokenizer.average_tokens_per_sentence())
    print("Most common words:", tokenizer.most_common())

    # Define a held-out sentence
    test_sentence = "the cow sat under the tree"
    print("\n=== Held-out sentence ===")
    print("Input:", repr(test_sentence))

    # Policy: raise (strict)
    print("\n-- unknown_policy='raise' --")
    try:
        encoded = tokenizer.encode(test_sentence, unknown_policy="raise")
        print("Encoded:", encoded)
        print("Decoded:", tokenizer.decode(encoded))
    except KeyError as e:
        print("Error:", e)

    # Policy: add (dynamically add unknown words)
    print("\n-- unknown_policy='add' --")
    encoded = tokenizer.encode(test_sentence, unknown_policy="add")
    print("Encoded:", encoded)
    print("Decoded:", tokenizer.decode(encoded))
    print("New vocabulary size after 'add':", tokenizer.vocabulary_size())
    count, frac = tokenizer.last_oov_stats()
    print(f"OOVs encountered: {count} ({frac*100:.1f}%)")

    # Policy: unk (map unknowns to <UNK>)
    print("\n-- unknown_policy='unk' --")
    encoded = tokenizer.encode(test_sentence, unknown_policy="unk")
    print("Encoded:", encoded)
    print("Decoded:", tokenizer.decode(encoded))
    count, frac = tokenizer.last_oov_stats()
    print(f"OOVs encountered: {count} ({frac*100:.1f}%)")

    # Final summary
    print("\n=== Final Vocabulary ===")
    print(tokenizer.mappings)


if __name__ == "__main__":
    main()
    