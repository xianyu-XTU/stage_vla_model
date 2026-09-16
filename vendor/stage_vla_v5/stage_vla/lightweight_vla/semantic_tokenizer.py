class SemanticTokenizer:
    """Project semantic action tokenizer.

    This is intentionally lightweight. It converts structured instructions into
    semantic tokens before action decoding. It is not a replacement for Llama.
    """
    def __init__(self):
        self.vocab = {
            "pick": "<PICK>",
            "place": "<PLACE>",
            "red": "<RED>",
            "blue": "<BLUE>",
        }

    def encode(self, instruction: str):
        tokens = []
        text = instruction.lower()
        for key, value in self.vocab.items():
            if key in text:
                tokens.append(value)
        return tokens
