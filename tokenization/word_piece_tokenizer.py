from collections import Counter
import math


class WordPieceTokenizer:
    def __init__(self, target_vocab_size: int = 100):
        self.target_vocab_size = target_vocab_size
        # Set to store the vocabulary
        self.vocab: set[str] = set()

    def fit(self, corpus: list[str]) -> None:
        # Step 1: tokenize corpus as list of char lists
        tokenized: list[list[str]] = [list(word) + ["</w>"] for word in corpus]

        # Step 2: Fill vocab with all unique characters in corpus
        self.vocab = set(ch for word in tokenized for ch in word)

        while len(self.vocab) < self.target_vocab_size:
            # Get frequencies of individual tokens and pairs
            token_frequencies = self._get_token_frequencies(tokenized)
            pair_frequencies = self._get_pair_frequencies(tokenized)

            if not pair_frequencies:
                break

            # Get the best pair based on PMI and frequencies
            best_pair = self._get_best_pair(token_frequencies, pair_frequencies)

            tokenized = self._merge_pair(tokenized, best_pair)
            self.vocab.add("".join(best_pair))
            print(f"Merged pair: {best_pair}")
            print(f"Vocab size: {len(self.vocab)} | New token added: {''.join(best_pair)}")
            if len(self.vocab) >= self.target_vocab_size:
                break

    def _get_token_frequencies(self, corpus: list[list[str]]) -> dict[str, int]:
        frequencies = Counter()
        for word in corpus:
            frequencies.update(word)

        return dict(frequencies)

    def _get_pair_frequencies(self, corpus: list[list[str]]) -> dict[str, int]:
        pair_frequencies = Counter()
        for word in corpus:
            for i in range(len(word) - 1):
                pair_frequencies[(word[i], word[i + 1])] += 1

        return dict(pair_frequencies)

    def _get_best_pair(
        self, token_frequencies: dict[str, int], pair_frequencies: dict[str, int]
    ) -> tuple[str, str]:
        scores: dict[tuple, float] = {}

        for (x, y), f_xy in pair_frequencies.items():
            f_x = token_frequencies.get(x, 1)
            f_y = token_frequencies.get(y, 1)

            epsilon = 1e-10
            score = f_xy * (math.log(f_xy + epsilon) - math.log(f_x + epsilon) - math.log(f_y + epsilon))
            scores[(x, y)] = score

        best_pair = max(scores, key=scores.get)

        return best_pair

    def _merge_pair(self, corpus: list[list[str]], target_pair: tuple[str, str]) -> list[list[str]]:
        merged_corpus: list[list[str]] = []

        for word in corpus:
            new_word = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and (word[i], word[i + 1]) == target_pair:
                    merged_token = word[i] + word[i + 1]
                    # If merge not at start of word (after start or previous </w>)
                    merged_token = word[i] + word[i + 1]
                    new_word.append(merged_token)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            merged_corpus.append(new_word)
        return merged_corpus

    def encode(self, text: str) -> list[str]:
        tokens: list[str] = []

        for word in text.split():
            start = 0
            while start < len(word):
                for end in range(len(word), start, -1):
                    sub_word = word[start:end]
                    if (sub_word if start == 0 else "##" + sub_word) in self.vocab:
                        tokens.append(sub_word if start == 0 else "##" + sub_word)
                        start = end
                        break
                else:
                    tokens.append("[UNK]")
                    break
        return tokens

    def decode(self, tokens: list[str]) -> str:
        text: str = ""
        for token in tokens:
            if token.startswith("##"):
                text += token[2:]
            else:
                text += " " + token

        return text.strip()


def main():
    corpus = ["playing", "player", "played"]
    tokenizer = WordPieceTokenizer(target_vocab_size=20)
    tokenizer.fit(corpus)

    print(tokenizer.vocab)
    print(tokenizer.encode("playing"))
    print(tokenizer.decode(["play", "##ing"]))


if __name__ == "__main__":
    main()
