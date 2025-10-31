from collections import Counter, defaultdict
import math


class UnigramLMTokenizer:
    def __init__(self, target_vocab_size: int = 10):
        self.target_vocab_size = target_vocab_size

    def init_vocab(
        self, corpus: list[str], max_token_length: int = 6, min_count: int = 1
    ) -> list[str]:
        """
        Return candidate tokens: all substrings of words up to max_token_length,
        whose raw frequency across the corpus >= min_count.
        Sorted by frequency desc, then length, then lexicographically.

        Sorting gives us:
            1. Deterministic, reproducible order (critical for consistent vocab builds).
            2. Frequency prioritization (tokens that matter more appear first).
            3. Length preference (favor longer units: closer to “words”).
            4. Simplifies pruning: we can take the top-N items instead of random ones.
        """
        print("here")
        substring_frequencies = Counter()
        for word in corpus:
            L = len(word)
            for i in range(L):
                # Length from 1 upto max_token_length (but not exceeding word boundary)
                for l in range(1, min(max_token_length, L - i) + 1):
                    substring = word[i : i + l]
                    substring_frequencies[substring] += 1

        # Keep only substrings that have frequencies at least min_count
        candidates = [s for s, c in substring_frequencies.items() if c >= min_count]

        # Sort deterministically: prefer higher frequency, then longer substrings, then lexicographic
        candidates.sort(key=lambda s: (-substring_frequencies[s], -len(s), s))

        return candidates

    def build_token_mappings(
        self, vocab: list[str]
    ) -> tuple[dict[str, int], list[str]]:
        """
        Build forward and reverse maps for token ids
        """
        token_to_id = {t: i for i, t in enumerate(vocab)}
        id_to_token = vocab[:]  # Copy list

        return token_to_id, id_to_token

    def precompute_matches_for_word(
        self, word: str, token_to_id: dict[str, int], max_token_length: int
    ) -> list[list[tuple[int, int]]]:
        """
        For a given word, return for each start position a list of (token_id, token_length)
        where token matches the substring starting there.
        """
        L = len(word)
        matches = [[] for _ in range(L)]

        for i in range(L):
            # Check all substrings starting at position i
            for l in range(1, min(max_token_length, L - i) + 1):
                substring = word[i : i + l]
                if substring in token_to_id:
                    token_id = token_to_id[substring]
                    matches[i].append((token_id, l))

        return matches

    def precompute_matches_for_corpus(
        self, corpus: list[str], token_to_id: dict[str, int], max_token_length: int
    ):
        """
        Compute match lists for each word in corpus.
        Returns a list of matches arrays, one per word.
        """
        return [
            self.precompute_matches_for_word(word, token_to_id, max_token_length)
            for word in corpus
        ]

    def forward_backward_expected_counts(
        self, matches: list[list[tuple[int, int]]], logP: list[float]
    ) -> tuple[dict[int, float], float]:
        """
        Computes expected count for one word
        Returns:
            expected_counts[token_id] -> float,
            logZ (log-likelihood for this word)
        """
        L = len(matches)
        # Forward probabilities α
        log_alpha = [-math.inf] * (L + 1)
        log_alpha[0] = 0.0  # log(1)

        for i in range(L):
            if log_alpha[i] == -math.inf:
                continue
            for tid, l in matches[i]:
                j = i + l
                log_alpha[j] = self._log_sum_exp(log_alpha[j], log_alpha[i] + logP[tid])
        logZ = log_alpha[L]

        # Backward probabilities β
        log_beta = [-math.inf] * (L + 1)
        log_beta[L] = 0.0

        for i in range(L - 1, -1, -1):
            for tid, l in matches[i]:
                j = i + l
                log_beta[i] = self._log_sum_exp(log_beta[i], logP[tid] + log_beta[j])

        # Expected counts
        expected = defaultdict(float)
        if logZ == -math.inf:
            return expected, -math.inf

        for i in range(L):
            for tid, l in matches[i]:
                j = i + l
                log_contribution = log_alpha[i] + logP[tid] + log_beta[j] - logZ
                contribution = math.exp(log_contribution)

                if contribution > 0.0:
                    expected[tid] += contribution

        return expected, logZ

    def em_iteration(
        self,
        corpus: list[str],
        precomputed_matches: list,
        logP: list[float],
        word_counts: dict[str, int],
    ) -> tuple[dict[int, float], float]:
        """
        Runs the E-step across the entire corpus.
        Returns: (expected_counts, total_log_likelihood)
        """
        total_expected = defaultdict(float)
        total_log_likelihood = 0.0

        for i, word in enumerate(corpus):
            matches = precomputed_matches[i]
            expected, logZ = self.forward_backward_expected_counts(matches, logP)
            if logZ == -math.inf:
                # Word cannot be segmented. Skip of treat it as unknown
                continue

            count = word_counts[word]
            for tid, c in expected.items():
                total_expected[tid] += c * count
            total_log_likelihood += logZ * count

        return total_expected, total_log_likelihood

    def m_step_update(
        self, expected_counts: dict[int, float], vocab_size: int
    ) -> tuple[list[float], list[float]]:
        """
        Updates token probabilities from expected counts.
        Ensure the returned P and logP have length == vocab_size and indices align with token ids.
        """
        total = sum(expected_counts.values()) + 1e-20
        # Build P of full vocab size, using 0.0 for tokens with no expected counts
        P = [expected_counts.get(i, 0.0) / total for i in range(vocab_size)]
        logP = [math.log(p + 1e-20) for p in P]
        return P, logP

    def prune_vocabulary(
        self,
        expected_counts: dict[int, float],
        id_to_token: list[str],
        target_vocabulary_size: int,
    ) -> tuple[dict[str, int], list[str], list[int]]:
        """
        Keep top-K tokens by expected count. Rebuilds mappings and returns kept indices.
        """
        ranked = sorted(expected_counts.items(), key=lambda x: -x[1])
        top = ranked[:target_vocabulary_size]
        kept_ids = [tid for tid, _ in top]

        new_vocabulary = [id_to_token[tid] for tid in kept_ids]
        token_to_id = {t: i for i, t in enumerate(new_vocabulary)}

        return token_to_id, new_vocabulary, kept_ids

    def fit(
        self,
        corpus: list[str],
        max_token_length: int = 6,
        min_count: int = 1,
        max_iterations: int = 30,
    ):
        """
        Full Unigram training loop
        """
        # Count word frequencies
        word_counts = Counter(corpus)

        # Initialize vocabulary candidates
        vocabulary = self.init_vocab(
            corpus=corpus, max_token_length=max_token_length, min_count=min_count
        )
        token_to_id, id_to_token = self.build_token_mappings(vocabulary)

        # Initialize probabilities uniformly
        n = len(vocabulary)
        P = [1.0 / n] * n
        logP = [math.log(p) for p in P]

        for iteration in range(max_iterations):
            # Precompute matches
            precomputed = [
                self.precompute_matches_for_word(word, token_to_id, max_token_length)
                for word in corpus
            ]

            # E-step
            expected, total_log_likelihood = self.em_iteration(
                corpus, precomputed, logP, word_counts
            )

            # M-step
            P, logP = self.m_step_update(expected, vocab_size=len(id_to_token))

            print(
                f"Iteration {iteration+1:02d} | Vocab size: {len(id_to_token)} | Log-likelihood: {total_log_likelihood:.4f}"
            )

            # Stop if vocabulary is already at target
            if len(id_to_token) <= self.target_vocab_size:
                break

            # Prune vocabulary
            token_to_id, id_to_token, kept_indices = self.prune_vocabulary(
                expected, id_to_token, self.target_vocab_size
            )

            # Filter P and logP to match the new vocab
            P = [P[i] for i in kept_indices]
            logP = [math.log(p + 1e-10) for p in P]

        self.vocabulary = id_to_token
        self.probability = P
        self.token_to_id = token_to_id
        return self

    def encode(self, word: str) -> list[str]:
        """
        Segment a word into sub-words using Viterbi decoding (most probable segmentation)
        """
        L = len(word)
        matches = self.precompute_matches_for_word(
            word, self.token_to_id, max_token_length=10
        )
        logP_dict = {tid: math.log(p + 1e-10) for tid, p in enumerate(self.probability)}

        cost = [math.inf] * (L + 1)
        previous = [None] * (L + 1)
        cost[0] = 0.0

        for i in range(L):
            if cost[i] == math.inf:
                continue

            for tid, l in matches[i]:
                j = i + l
                new_cost = cost[i] - logP_dict[tid]
                if new_cost < cost[j]:
                    cost[j] = new_cost
                    previous[j] = (i, tid)

        # Reconstruct best path
        if previous[L] is None:
            return ["[UNK]"]

        tokens = []
        position = L
        while position > 0:
            i, tid = previous[position]
            tokens.append(self.vocabulary[tid])
            position = i

        tokens.reverse()
        return tokens

    def decode(self, tokens: list[str]) -> str:
        return "".join(tokens)

    def _log_sum_exp(self, a: float, b: float) -> float:
        """
        Stable log-sum-exp for two log-values.
        """
        if a == -math.inf:
            return b

        if b == -math.inf:
            return a

        if a > b:
            return a + math.log1p(math.exp(b - a))
        else:
            return b + math.log1p(math.exp(a - b))


def main():
    corpus = ["playing", "player", "played"]
    tokenizer = UnigramLMTokenizer(target_vocab_size=20)
    tokenizer.fit(corpus, max_token_length=5, min_count=1, max_iterations=10)

    print("\nFinal vocab:")
    for t, p in zip(tokenizer.vocabulary, tokenizer.probability):
        print(f"{t:10s} -> {p:.4f}")

    print("\nEncoding examples:")
    for w in ["playing", "played", "player"]:
        tokens = tokenizer.encode(w)
        print(f"{w:10s} -> {tokens} -> {tokenizer.decode(tokens)}")


if __name__ == "__main__":
    main()
