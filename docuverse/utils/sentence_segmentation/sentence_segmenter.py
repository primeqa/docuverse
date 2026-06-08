from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List


@dataclass
class Token:
    begin: int


@dataclass
class Sentence:
    begin: int
    text: str
    tokens: List[Token] = field(default_factory=list)


@dataclass
class ParsedText:
    sentences: List[Sentence] = field(default_factory=list)


class SentenceSegmenter(ABC):
    def __init__(self, language_code: str = "en"):
        self.language_code = language_code

    @abstractmethod
    def parse(self, text: str) -> ParsedText:
        ...

    def __call__(self, text: str) -> ParsedText:
        return self.parse(text)
