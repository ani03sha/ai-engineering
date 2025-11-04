import random
import numpy as np
from collections import Counter


# ---------------------- Tokenizer ---------------------- #
class Tokenizer:
    """
    Handles text preprocessing:
    - Tokenization
    - Vocabulary building
    - Word <-> index mapping
    - Corpus integer encoding
    """

    def __init__(self, corpus: str):
        self.corpus: str = corpus
        self.vocabulary: list[str] = None
        self.word_to_index: dict[str, int] = None
        self.index_to_word: dict[int, str] = None
        self.encoded_corpus: list[int] = None

    def build_vocabulary(self, min_frequency: int = 1):
        # Flatten sentences into tokens
        tokens = []
        for sentence in self.corpus:
            tokens.extend(sentence.lower().split())

        # Count frequency and filter
        word_frequency = Counter(tokens)
        vocabulary = [
            word for word, count in word_frequency.items() if count >= min_frequency
        ]
        vocabulary = sorted(vocabulary)

        self.vocabulary = vocabulary
        self.word_to_index = {word: index for index, word in enumerate(vocabulary)}
        self.index_to_word = {index: word for word, index in self.word_to_index.items()}

        # Convert entire corpus into integer ids
        self.encoded_corpus = [self.word_to_index[word] for word in tokens]

        print(f"Vocabulary built. Size: {len(self.vocabulary)} words")

        return self

    def encode(self, word: str) -> int | None:
        """Return integer id for word"""
        return self.word_to_index.get(word, None)

    def decode(self, index: int) -> str | None:
        """Return word for integer id"""
        return self.index_to_word.get(index, None)

    def get_encoded_corpus(self) -> list[int]:
        return self.encoded_corpus


# ---------------------- Dataset ---------------------- #


class Dataset:
    """
    Generates skip-gram (center, context) pairs and negative samples.
    """

    def __init__(
        self,
        encoded_corpus: list[int],
        vocabulary_size: int,
        window_size: int = 2,
        negative_samples: int = 5,
    ):
        self.encoded_corpus = encoded_corpus
        self.vocabulary_size = vocabulary_size
        self.window_size = window_size
        self.negative_samples = negative_samples
        
        # Generate all positive pairs once
        self.pairs: list[tuple[int, int]] = self._generate_skipgram_pairs()
        # Build a unigram distribution for negative sampling
        self.word_probabilities = self._build_unigram_table()
        
    def _generate_skipgram_pairs(self) -> list[tuple[str, str]]:
        pairs: list[str] = []
        for center_position, center_word in enumerate(self.encoded_corpus):
            for w in range(-self.window_size, self.window_size + 1):
                context_position = center_position + w
                if (context_position < 0 or context_position >= len(self.encoded_corpus) or context_position == center_position):
                    continue
                
                context_word = self.encoded_corpus[context_position]
                pairs.append((center_word, context_word))
        
        return pairs
    
    def _build_unigram_table(self) -> list[float]:
        """
        Word2Vec uses unigram distribution raised to 3/4 power for sampling.
        """
        word_frequency = Counter(self.encoded_corpus)
        
        # Unigram distribution raised to 3/4
        probabilities = np.zeros(self.vocabulary_size)
        for i in range(self.vocabulary_size):
            probabilities[i] = word_frequency.get(i, 0) ** 0.75
        
        probabilities = probabilities / np.sum(probabilities)
        
        return probabilities
    
    def get_batch(self, batch_size: int = 8):
        """
        Samples a mini-batch of (center, context) pairs and corresponding negatives.
        """
        batch = random.sample(self.pairs, batch_size)
        centers, contexts = zip(*batch)
        
        # Sample negatives for each center word
        negative_samples = np.random.choice(
            np.arange(self.vocabulary_size),
            size=(batch_size, self.negative_samples),
            p=self.word_probabilities
        )
        
        return np.array(centers), np.array(contexts), negative_samples


def main():
    random.seed(42)
    np.random.seed(42) # To make runs reproducible
    
    corpus: list[str] = [
        "the player kicked the ball",
        "the team won the match",
        "the coach praised the player",
        "the match was exciting",
        "the player scored a goal",
        "the goal won the game",
    ]

    tokenizer = Tokenizer(corpus).build_vocabulary()
    print(tokenizer.vocabulary)
    print(tokenizer.get_encoded_corpus())
    
    dataset = Dataset(tokenizer.get_encoded_corpus(), vocabulary_size=len(tokenizer.vocabulary))
    centers, contexts, negatives = dataset.get_batch(batch_size=4)
    
    print("Centers:", [tokenizer.decode(c) for c in centers])
    print("Contexts:", [tokenizer.decode(c) for c in contexts])
    print("Negatives:", [[tokenizer.decode(n) for n in row] for row in negatives])


if __name__ == "__main__":
    main()
