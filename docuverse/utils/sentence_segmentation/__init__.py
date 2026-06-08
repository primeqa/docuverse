from .sentence_segmenter import ParsedText, Sentence, SentenceSegmenter, Token


def create_segmenter(backend: str = "pyizumo",
                     language_code: str = "en",
                     **kwargs) -> SentenceSegmenter:
    if backend == "pyizumo":
        from .pyizumo_segmenter import PyizumoSegmenter
        return PyizumoSegmenter(language_code=language_code, **kwargs)
    elif backend == "spacy":
        from .spacy_segmenter import SpacySegmenter
        return SpacySegmenter(language_code=language_code, **kwargs)
    raise ValueError(f"Unknown sentence segmenter backend: {backend!r}")


__all__ = [
    "ParsedText",
    "Sentence",
    "SentenceSegmenter",
    "Token",
    "create_segmenter",
]
