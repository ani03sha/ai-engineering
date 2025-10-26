"""
Imagine your text corpus is just this one sentence: "the cat sat on the mat"

Let's start with character-level tokenization — i.e., map each unique character to an integer.

How would you go about building such a mapping in code (you can describe it or outline pseudocode)?
"""

from typing import Iterable


class CharacterTokenization:
    def __init__(self):
        # Dictionary to hold character to integer mappings
        self.mappings: dict[str, int] = {}
        # Dictionary to hold integer to character mappings
        self.reverse_mappings: dict[int, str] = {}
        # Index to keep track for next integer to assign
        self.index: int = 0


    def fit(self, corpus: list[str]) -> None:
        # Traverse through each character in the text
        for text in corpus:
            for char in text:
                if char not in self.mappings:
                    self.mappings[char] = self.index
                    self.reverse_mappings[self.index] = char
                    self.index += 1


    def encode(self, text: str) -> list[int]:
        # List to hold the encoded integers
        encoded: list[int] = []
        for char in text:
            if char in self.mappings:
                encoded.append(self.mappings[char])
            else:
                raise KeyError(f"Unknown character: {char!r}")
        return encoded


    def decode(self, encoded: Iterable[int]) -> str:
        # String to hold decoded word
        decoded: str = ""
        for token in encoded:
            decoded += self.reverse_mappings[token]
        return decoded


def main():
    corpus = ["the cat sat on the mat"]
    character_tokenization = CharacterTokenization()
    # Tokenize the corpus
    character_tokenization.fit(corpus)
    
    # Encode
    encoded = character_tokenization.encode("the cat")
    print("Encoded:", encoded)

    # Decode the same tokens
    decoded = character_tokenization.decode(encoded=encoded)
    print("Decoded:", decoded)


if __name__ == "__main__":
    main()