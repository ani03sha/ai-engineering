import random
import time
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
                if (
                    context_position < 0
                    or context_position >= len(self.encoded_corpus)
                    or context_position == center_position
                ):
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
            p=self.word_probabilities,
        )

        return np.array(centers), np.array(contexts), negative_samples


# ---------------------- Word2Vec ---------------------- #
class Word2Vec:
    """
    Skip-gram model with negative sampling (NumPy version).
    Learns two embedding matrices:
    - W_in  : center (input) word embeddings
    - W_out : context (output) word embeddings
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

        # Initialize embeddings randomly (small values for stability)
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
        return 1 / (1 + np.exp(-x))

    def forward_backward(
        self, centers: np.ndarray, contexts: np.ndarray, negatives: np.ndarray
    ) -> float:
        """
        Perform forward & backward pass on one mini-batch.

        centers  : (B,) array of center word IDs
        contexts : (B,) array of true context word IDs
        negatives: (B, K) array of negative word IDs
        """

        # 1. Lookup embeddings
        v_c = self.W_in[centers]
        u_o = self.W_out[contexts]
        u_k = self.W_out[negatives]

        # 2. Positive scores and loss
        positive_dot = np.sum(v_c * u_o, axis=1)
        positive_sigmoid = self._sigmoid(positive_dot)
        loss_positive = -np.log(positive_sigmoid + 1e-10)

        # 3. Negative scores and loss
        negative_dot = np.einsum("bd,bkd->bk", v_c, u_k)
        negative_sigmoid = self._sigmoid(-negative_dot)
        loss_negative = -np.sum(np.log(negative_sigmoid + 1e-10), axis=1)

        loss = np.mean(loss_positive + loss_negative)

        # 4. Backpropagation
        # Gradients w.r.t positive pair
        gradient_positive = (positive_sigmoid - 1)[:, np.newaxis] * u_o
        # Gradients w.r.t negative pairs
        gradient_negative = (1 - negative_sigmoid)[:, :, np.newaxis] * (-u_k)
        gradient_v = gradient_positive + np.sum(gradient_negative, axis=1)

        # 5. Update embeddings
        # Update W_in
        self.W_in[centers] -= self.learning_rate * gradient_v
        # Update W_out for positive contexts
        grad_u_o = (positive_sigmoid - 1)[:, np.newaxis] * v_c
        self.W_out[contexts] -= self.learning_rate * grad_u_o
        # Update W_out for negatives
        grad_u_k = (1 - negative_sigmoid)[:, :, np.newaxis] * (-v_c[:, np.newaxis, :])
        np.add.at(self.W_out, negatives, self.learning_rate * grad_u_k)

        return loss


# ---------------------- Trainer ---------------------- #
class Trainer:
    """
    Handles the training loop for Word2Vec.
    """

    def __init__(self, model: Word2Vec, dataset: Dataset, tokenizer: Tokenizer):
        self.model = model
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.loss_history = []

    def train(self, epochs: int = 5, batch_size: int = 8, log_interval: int = 10):
        print("Training started...")
        start_time = time.time()

        for epoch in range(epochs):
            epoch_loss = []
            for step in range(len(self.dataset.pairs) // batch_size):
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

        # Computer cosine similarities
        similarities = (
            self.model.W_in
            @ target_vector
            / (
                np.linalg.norm(self.model.W_in, axis=1) * np.linalg.norm(target_vector)
                + 1e-10
            )
        )

        best_ids = np.argsort(-similarities)[1 : topK + 1]

        return [
            (self.tokenizer.index_to_word[i], float(similarities[i])) for i in best_ids
        ]


# ---------------------- Trainer ---------------------- #
class Visualizer:
    """
    Projects learned embeddings into 2D and visualizes them.
    """

    def __init__(self, model: Word2Vec, tokenizer: Tokenizer):
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
        indices = [self.tokenizer.word_to_index[word] for word in words]
        vectors = self.model.W_in

        # Reduce to 2D
        pca = PCA(n_components=2)
        reduced = pca.fit_transform(vectors)

        # Plot
        plt.figure(figsize=(8, 6))
        plt.scatter(reduced[:, 0] + 0.02, reduced[:, 1], c="steelblue", s=50)

        for i, word in enumerate(words):
            plt.text(reduced[i, 0], reduced[i, 1] + 0.02, word, fontsize=12)

        plt.title("Word Embeddings Projection (PCA)")
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
        negative_samples=10
    )

    model = Word2Vec(
        vocabulary_size=len(tokenizer.vocabulary),
        embedding_dimensions=50,
        learning_rate=0.01
    )

    trainer = Trainer(model, dataset, tokenizer)
    trainer.train(epochs=10, batch_size=128, log_interval=50)

    print(trainer.most_similar("intelligence"))
    print(trainer.most_similar("data"))
    visualizer = Visualizer(model, tokenizer)
    visualizer.plot_embeddings(topK=50)


if __name__ == "__main__":
    main()
