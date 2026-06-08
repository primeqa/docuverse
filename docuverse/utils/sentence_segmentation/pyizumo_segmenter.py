from .sentence_segmenter import ParsedText, Sentence, SentenceSegmenter, Token


class PyizumoSegmenter(SentenceSegmenter):
    def __init__(self, language_code: str = "en"):
        super().__init__(language_code)
        try:
            import pyizumo
        except ImportError as e:
            raise ImportError(
                "You need to install the pyizumo package to use the pyizumo sentence segmenter."
            ) from e

        try:
            self._nlp = pyizumo.load(language_code, parsers=["token", "sentence"])
        except Exception as e:
            raise ImportError(
                f"Problem loading the pyizumo package: {e}. "
                f"If you're having trouble, maybe turn off sentence-based text splitting "
                f"(--split_on_sentences=False) or pick a different sentence_segmenter backend."
            ) from e

    def parse(self, text: str) -> ParsedText:
        doc = self._nlp(text)
        sentences = []
        for sent in doc.sentences:
            tokens = [Token(begin=tok.begin) for tok in sent.tokens]
            sentences.append(Sentence(begin=sent.begin, text=sent.text, tokens=tokens))
        return ParsedText(sentences=sentences)
