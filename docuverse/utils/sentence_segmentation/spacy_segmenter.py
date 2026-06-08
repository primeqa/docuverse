from .sentence_segmenter import ParsedText, Sentence, SentenceSegmenter, Token

DEFAULT_MODEL = "xx_sent_ud_sm"


class SpacySegmenter(SentenceSegmenter):
    def __init__(self, language_code: str = "en", model: str = DEFAULT_MODEL):
        super().__init__(language_code)
        try:
            import spacy
        except ImportError as e:
            raise ImportError(
                "You need to install the spacy package to use the spacy sentence segmenter."
            ) from e

        self._model_name = model
        try:
            self._nlp = spacy.load(model)
        except OSError as e:
            raise ImportError(
                f"spaCy model '{model}' is not installed. "
                f"Run: python -m spacy download {model}"
            ) from e

    def parse(self, text: str) -> ParsedText:
        doc = self._nlp(text)
        sentences = []
        for sent in doc.sents:
            tokens = [Token(begin=tok.idx) for tok in sent]
            sentences.append(
                Sentence(begin=sent.start_char, text=sent.text, tokens=tokens)
            )
        return ParsedText(sentences=sentences)
