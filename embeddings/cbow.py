import random
import time
from typing import Self
import numpy as np
from collections import Counter
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import wikipedia


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

    def build_vocabulary(self, min_frequency: int = 1) -> Self:
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
        self.index_to_word = {index: word for index, word in self.word_to_index.items()}

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
    Generates (context, target) pairs for CBOW training.
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
        self.word_probabilities = self._build_unigram_table()

    def _build_unigram_table(self) -> list[float]:
        word_frequencies = Counter(self.encoded_corpus)
        probabilities = np.zeros(self.vocabulary_size)
        for i in range(self.vocabulary_size):
            probabilities[i] = word_frequencies.get(i, 0) * 0.75
        probabilities /= sum(probabilities)
        return probabilities

    def get_batch(self, batch_size: int = 0):
        contexts, targets, negatives = [], [], []
        for _ in range(batch_size):
            center_position = np.random.randint(
                self.window_size, len(self.encoded_corpus) - self.window_size
            )
            context = (
                self.encoded_corpus[
                    center_position - self.window_size : center_position
                ]
                + self.encoded_corpus[
                    center_position + 1 : center_position + 1 + self.window_size
                ]
            )
            target = self.encoded_corpus[center_position]
            contexts.append(context)
            targets.append(target)
            # Negative sampling
            negative = np.random.choice(
                np.arange(self.vocabulary_size),
                size=self.negative_samples,
                p=self.word_probabilities,
            )
            negatives.append(negative)

        return np.array(contexts), np.array(targets), np.array(negatives)


# ---------------------- CBOW ---------------------- #
class CBOW:
    """
    Continuous Bag of Words model with Negative Sampling (NumPy implementation)
    """

    def __init__(
        self,
        vocabulary_size: int,
        embedding_dimensions: int = 10,
        learning_rate: float = 0.05,
    ):
        self.vocabulary_size = vocabulary_size
        self.embedding_dimensions = embedding_dimensions
        self.learning_rate = learning_rate
        self.W_in = np.random.uniform(
            -0.5 / embedding_dimensions,
            0.5 / embedding_dimensions,
            (vocabulary_size, embedding_dimensions),
        )
        self.W_out = np.random.uniform(
            -0.5 / embedding_dimensions,
            0.5 / embedding_dimensions,
            (vocabulary_size, embedding_dimensions),
        )

    @staticmethod
    def _sigmoid(x):
        return 1 / (1 + np.exp(-np.clip(x, -10, 10)))

    def forward_backward(
        self, contexts: np.ndarray, targets: np.ndarray, negatives: np.ndarray
    ):
        """
        contexts: (B, 2*window) array of word IDs
        targets : (B,) array of target word IDs
        negatives: (B, K) array of negative word IDs
        """
        batch_size, context_size = contexts.shape
        negative_size = negatives.shape[1]

        # 1. Average context embeddings
        v_context = self.W_in[contexts]  # (B, C, D)
        h = np.mean(v_context, axis=1)

        # 2. Positive score
        u_target = self.W_out[targets]  # (B, D)
        positive_dot = np.sum(h * u_target, axis=1)  # (B, )
        positive_sigmoid = self._sigmoid(positive_dot)
        loss_positive = -np.log(positive_sigmoid + 1e-10)

        # 3. Negative score
        u_negative = self.W_out[negatives]  # (B, K, D)
        negative_dot = np.einsum("bd, bkd->bk", h, u_negative)
        negative_sigmoid = self._sigmoid(-negative_dot)
        loss_negative = -np.sum(np.log(negative_sigmoid + 1e-10), axis=1)

        loss = np.mean(loss_positive + loss_negative)

        # 4. Backpropagation
        gradient_h = (positive_sigmoid - 1)[:, np.newaxis] * u_target + np.sum(
            (1 - negative_sigmoid)[:, :, np.newaxis] * (-u_negative), axis=1
        )
        # Distribute gradient_h equally to all context embeddings
        gradient_v_context = gradient_h[:, np.newaxis, :] / context_size
        np.add.at(self.W_in, contexts, -self.learning_rate * gradient_v_context)

        # Update output embeddings
        gradient_u_target = (positive_sigmoid - 1)[:, np.newaxis] * h
        self.W_out[targets] -= self.learning_rate * gradient_u_target

        gradient_u_negative = (1 - negative_sigmoid)[:, :, np.newaxis] * (
            -h[:, np.newaxis, :]
        )
        np.add.at(self.W_out, negatives, self.learning_rate * gradient_u_negative)

        return loss


class Trainer:
    """
    Handles the training loop for Word2Vec.
    """

    def __init__(self, model: CBOW, dataset: Dataset, tokenizer: Tokenizer):
        self.model = model
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.loss_history = []

    def train(
        self,
        epochs: int = 5,
        batch_size: int = 8,
        log_interval: int = 10,
        steps_per_epoch: int = 50,
    ):
        print("Training started...")
        start_time = time.time()

        for epoch in range(epochs):
            epoch_loss = []
            for step in range(steps_per_epoch):
                centers, contexts, negatives = self.dataset.get_batch(batch_size)
                loss = self.model.forward_backward(centers, contexts, negatives)
                epoch_loss.append(loss)

                # Logging
                if step % log_interval == 0:
                    print(f"Epoch {epoch} | Step {step:04d} | Loss = {loss:.4f}")

            average_loss = np.mean(epoch_loss)
            self.loss_history.append(average_loss)
            print(f"✅ Epoch {epoch} complete. Avg Loss = {average_loss:.4f}\n")

        print(f"Training finished in {time.time() - start_time:.2f}s\n")

    def save_embeddings(self, path: str = "embeddings.npy"):
        np.save(path, self.model.W_in)
        print(f"Embeddings saved to {path}")

    def most_similar(self, word: str, topK: int = 5):
        """Find top-k most similar words to the given word."""
        if word not in self.tokenizer.word_to_index:
            print("Word is not present in vocabulary")
            return []

        target_id = self.tokenizer.word_to_index[word]
        target_vector = self.model.W_in[target_id]

        similarities = (
            self.model.W_in
            @ target_vector
            / (
                np.linalg.norm(self.model.W_in, axis=1) * np.linalg.norm(target_vector)
                + 1e-10
            )
        )

        # Ensure indices are valid Python ints within vocab size
        best_ids = np.argsort(-similarities)[: topK + 1]
        best_ids = [
            int(i)
            for i in best_ids
            if i < len(self.tokenizer.vocabulary) and i != target_id
        ]

        results = []
        for i in best_ids[:topK]:
            if i in self.tokenizer.index_to_word:
                results.append(
                    (self.tokenizer.index_to_word[i], float(similarities[i]))
                )

        return results


# ---------------------- Visualizer ---------------------- #
class Visualizer:
    """
    Projects learned embeddings into 2D and visualizes them.
    """

    def __init__(self, model: CBOW, tokenizer: Tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def plot_embeddings(self, topK: int = 15):
        """
        Plots the top k most frequent words in 2D using PCA.
        """
        vocabulary_size = len(self.tokenizer.vocabulary)
        n = min(topK, vocabulary_size)

        # Extract embeddings
        words = list(self.tokenizer.vocabulary)[:n]
        vectors = self.model.W_in

        # Reduce to 2D
        pca = PCA(n_components=2)
        reduced = pca.fit_transform(vectors)

        # Plot
        plt.figure(figsize=(8, 6))
        plt.scatter(reduced[:, 0] + 0.02, reduced[:, 1], c="steelblue", s=50)

        for i, word in enumerate(words):
            plt.text(reduced[i, 0], reduced[i, 1] + 0.02, word, fontsize=12)

        plt.title("CBOW Projection (PCA)")
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.grid(True)
        plt.show()


# ---------------------- Wikipedia ---------------------- #
def load_wikipedia_corpus(query="Machine Learning", n_articles=5):
    """
    Fetches summary paragraphs from a few Wikipedia articles.
    """
    sentences = []
    topics = [
        query,
        "Artificial intelligence",
        "Deep learning",
        "Neural network",
        "Data science",
    ]
    for topic in topics[:n_articles]:
        try:
            text = wikipedia.page(topic).content
            paragraphs = text.split("\n")
            for paragraph in paragraphs:
                if len(paragraph.split()) > 5:  # Skip short lines
                    sentences.append(paragraph.lower())
        except Exception as e:
            print(f"Skipping {topic}: {e}")

    print(f"✅ Loaded {len(sentences)} Wikipedia sentences.")
    return sentences


def main():
    random.seed(42)
    np.random.seed(42)  # To make runs reproducible

    # Load wikipedia corpus
    corpus = load_wikipedia_corpus("Machine Learning", n_articles=5)

    tokenizer = Tokenizer(corpus).build_vocabulary()

    dataset = Dataset(
        tokenizer.get_encoded_corpus(),
        vocabulary_size=len(tokenizer.vocabulary),
        window_size=5,
        negative_samples=10,
    )

    model = CBOW(
        vocabulary_size=len(tokenizer.vocabulary),
        embedding_dimensions=50,
        learning_rate=0.01,
    )

    trainer = Trainer(model, dataset, tokenizer)
    trainer.train(epochs=10, batch_size=128, log_interval=5, steps_per_epoch=50)

    print(trainer.most_similar("intelligence"))
    print(trainer.most_similar("data"))
    visualizer = Visualizer(model, tokenizer)
    visualizer.plot_embeddings(topK=50)


if __name__ == "__main__":
    main()
