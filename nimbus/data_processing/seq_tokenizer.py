from typing import List, Union


class SeqTokenizer:
    """
    A class for tokenizing (index to amino acids) and encoding (amino acid to
    index) sequences of amino acids.
    """
    def __init__(self):
        self.res_tokens = ['G', 'A', 'V', 'L', 'I', 'F', 'W', 'Y', 'D', 'N',
                           'E', 'K', 'Q', 'M', 'S', 'T', 'C', 'P', 'H', 'R']
        self.padding_token = ['-']
        self.tokens = self.padding_token + self.res_tokens
        self.padding_index = self.encode(self.padding_token)[0]

    def get_token(self, index: int) -> str:
        """Return the token for the given index."""
        assert isinstance(index, int), \
            f"Index must be an integer, but got `{index}` with type {type(index)}."
        assert 0 <= index < len(self.tokens), \
            f"Index {index} out of bounds. Must be in the range [0, {len(self.tokens)})."
        return self.tokens[index]

    def get_index(self, token: str) -> int:
        """Return the index for the given token."""
        try:
            return self.tokens.index(token.upper())
        except ValueError:
            raise ValueError(f"Token {token} not found in the vocabulary.")

    def tokenize(self, seq: Union[tuple[int, ...], list[int]]) -> List[str]:
        """Tokenize the sequence of indices into a sequence of tokens."""
        return [self.get_token(i) for i in seq]

    def encode(self, seq: Union[str, List[str], tuple[str, ...]]) -> List[int]:
        """Encode the sequence of tokens into a sequence of indices."""
        return [self.get_index(s) for s in seq]
