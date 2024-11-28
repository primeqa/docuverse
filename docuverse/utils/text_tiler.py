import os.path
import unicodedata
import re
from typing import List, Any, Dict, Union
from collections import deque

from transformers import (
    AutoTokenizer,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast
)


class TextTiler:
    """
    TextTiler is a class that helps splits a long string into segments that are
    of a specified length (in LLM tokenizer units), and can follow sentence boundaries
    or not.
    """
    url_re = r'https?://(?:www\.)?(?:[-a-zA-Z0-9@:%._\+~#=]{1,256})\.(:?[a-zA-Z0-9()]{1,6})(?:[-a-zA-Z0-9()@:%_\+.~#?&/=]*)*\b'
    COUNT_TYPE_TOKEN = 0
    COUNT_TYPE_CHAR = 1

    def __init__(self, max_doc_size: int, stride: int,
                 tokenizer: Union[str, PreTrainedTokenizer, None],
                 aligned_on_sentences: bool = True,
                 count_type='token'):
        """

        Initialize the class instance.

        :param max_doc_size: Maximum size of the document in terms of tokens/characters.
        :param stride: Overlap between consecutive windows (yes, it's misnamed).
        :param tokenizer: Tokenizer object or name of the tokenizer model.
        :param aligned_on_sentences: Flag indicating whether window alignment should be performed on sentences.
        :param count_type: Type of count to be performed, either 'tokens' or 'characters'.

        :raises RuntimeError: If the tokenizer argument is neither a string nor a PreTrainedTokenizer class.
        """
        self.max_doc_size = max_doc_size
        self.stride = stride
        if count_type == 'token':
            self.count_type = self.COUNT_TYPE_TOKEN
        elif count_type == 'char':
            self.count_type = self.COUNT_TYPE_CHAR
        else:
            raise RuntimeError('count_type must be either "token" or "chars"')
        if self.count_type == self.COUNT_TYPE_CHAR:
            self.tokenizer = None
        else:
            if isinstance(tokenizer, str):
                self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
            elif isinstance(tokenizer, PreTrainedTokenizer | PreTrainedTokenizerFast):
                self.tokenizer = tokenizer
            else:
                raise RuntimeError("The tokenizer argument must be either a string or a PreTrainedTokenizer class.")
            # Force the tokenizer to not complain.
            self.tokenizer.deprecation_warnings['sequence-length-is-longer-than-the-specified-maximum'] = True
            self.tokenizer_num_special_tokens = self.tokenizer.num_special_tokens_to_add()
            self.max_doc_size -= self.tokenizer_num_special_tokens
        self.product_counts = {}
        self.aligned_on_sentences = aligned_on_sentences
        self.nlp = None

    def create_tiles(self,
                     id_: str,
                     text: str,
                     title: str = "",
                     max_doc_size: int = None,
                     stride: int = None,
                     remove_url: bool = True,
                     normalize_text: bool = False,
                     title_handling: str = "all",
                     template=None,
                     ):
        """
        Converts a given document or passage (from 'output.json') to a dictionary, splitting the text as necessary.
        :param id_: str - the prefix of the id of the resulting piece/pieces
        :param title: str - the title of the new piece
        :param text: the input text to be split
        :param max_doc_size: int - the maximum size (in word pieces) of the resulting sub-document/sub-passage texts
        :param stride: int - the stride/overlap for consecutive pieces
        :param remove_url: Boolean - if true, URL in the input text will be replaced with "URL"
        :param normalize_text: boolean - if true, normalize the text for UTF-8 encoding.
        :param title_handling:str - can be ['all', 'first', 'none'] - defines how the title will be added to the text pieces:
           * 'all': the title is added to each of the created tiles
           * 'first': the title is added to only the first tile
           * 'none': the title is not added to any of the resulting tiles
        :param template: Dict[str, str] - the template for each json entry; each entry will be appended an 'id' and 'text'
                         keys with the split text.
        :return - a list of indexable items, each containing a title, id, text, and url.
        """
        if template is None:
            template = {}
        if max_doc_size is None:
            max_doc_size = self.max_doc_size
        if self.count_type == self.COUNT_TYPE_CHAR:
            max_doc_size -= 1

        if stride is None:
            stride = self.stride
        itm = template
        if text is None:
            return []
        else:
            text = text.replace(r'\n+', '\n').replace(r' +', ' ')
        pieces = []

        if text.find("With this app") >= 0 or text.find("App ID") >= 0:
            itm['app_name'] = title
        title_in_text = False
        if 0 <= text.find(title) <= 2:
            expanded_text = text
            title_in_text = True
        else:
            expanded_text = f"{title}\n{text}"

        if remove_url:
            text = self.cleanup_url(text, normalize_text)

        if self.tokenizer is not None:
            merged_length = self.get_tokenized_length(text=expanded_text)
            if merged_length <= max_doc_size:
                itm.update({'id': f"{id_}-0-{len(text)}", 'text': expanded_text})
                pieces.append(itm.copy())
            else:
                maxl = max_doc_size  # - title_len
                psgs, inds, added_titles = \
                    self.split_text(text=text, max_length=maxl, title=title,
                                    stride=stride, tokenizer=self.tokenizer, title_handling=title_handling,
                                    title_in_text=title_in_text)
                for pi, (p, index, added_title) in enumerate(zip(psgs, inds, added_titles)):
                    itm.update({
                        'id': f"{id_}-{index[0]}-{index[1]}",
                        'text': (f"{title}\n{p}"
                                 if added_title
                                 else p)
                    })
                    pieces.append(itm.copy())
        else:
            itm.update({'id': id_, 'text': expanded_text})
            pieces.append(itm.copy())
        return pieces

    @staticmethod
    def cleanup_url(text: str, normalize_text=True):
        # The normalization below deals with some issue in the re library - it would get stuck
        # if the URL has some strange chars, like `\xa0`.
        if normalize_text:
            text = re.sub(TextTiler.url_re, 'URL', unicodedata.normalize("NFKC", text))
        else:
            text = re.sub(TextTiler.url_re, 'URL', text)
        return text

    def get_tokenized_length(self, text, exclude_special_tokens=True, forced_tok=False):
        """
        Returns the size of the <text> (in tokens) after being tokenized by <tokenizer>
        :param text: str - the input text
        :return the length (in word pieces) of the tokenized text.
        """
        if not forced_tok and self.count_type == self.COUNT_TYPE_CHAR:
            return len(text)
        else:
            if self.tokenizer is not None:
                toks = self.tokenizer(text)
                return len(toks['input_ids']) - (self.tokenizer_num_special_tokens if not exclude_special_tokens else 0)
            else:
                return -1

    def split_text(self, text: str, title: str = "",
                   max_length: int = -1, stride: int = -1,
                   language_code='en', title_handling: str = 'all',
                   title_in_text=False) \
            -> tuple[list[str], list[list[int | Any]], list[bool]]:
        """
        Method to split a text into pieces that are of a specified <max_length> length, with the
        <stride> overlap, using an HF tokenizer.
        :param text: str - the text to split
        :param tokenizer: HF Tokenizer
           - the tokenizer to do the work of splitting the words into word pieces
        :param title: str - the title of the document
        :param max_length: int - the maximum length of the resulting sequence
        :param stride: int - the overlap between windows
        :param language_code: str - the 2-letter language code of the text (default, 'en').
        :param title_handling: str - can be ['all', 'first', 'none'] - defines how the title will be added to the text pieces:
           * 'all': the title is added to each of the created tiles
           * 'first': the title is added to only the first tile
           * 'none': the title is not added to any of the resulting tiles
        :param title_in_text: bool - true if the title is part of the text.
        :return: Tuple[list[str], list[list[int | Any]]] - returns a pair of tile string list and a position list (each entry
                 is a list of the start/end position of the corresponding tile in the first returned argument).
        """
        added_titles = []
        title_length = self.get_tokenized_length(title)

        def get_expanded_text(text: str, title: str, pos: int = 0,
                              title_handling: str = "all", title_in_text: bool = False):
            if self._need_to_add_title(pos, title_handling, title_in_text):
                return f"{title}\n{text}"
            else:
                return text

        if max_length == -1:
            max_length = self.max_doc_size
        if stride == -1:
            stride = self.stride
        text = re.sub(r' {2,}', ' ', text, flags=re.MULTILINE)  # remove multiple spaces.
        pos = 0
        texts = []
        positions = []
        added_titles = []
        if max_length is not None:
            tok_len = self.get_tokenized_length(get_expanded_text(text=text, title=title,
                                                                  title_handling=title_handling,
                                                                  title_in_text=title_in_text)
                                                )
            if tok_len <= max_length:
                return [text], [[0, len(text)]], [self._need_to_add_title(0, title_handling, title_in_text)]
            else:
                if title and title_handling == "all":  # make space for the title in each split text.
                    ltitle = self.get_tokenized_length(title)
                    max_length -= ltitle
                    ind = text.find(title)
                    if ind == 0:
                        text = text[ind + len(title):]

                if self.aligned_on_sentences:
                    try:
                        import pyizumo
                    except:
                        raise ImportError(f"You need to install the pyzimo package before using this method.")

                    if not self.nlp:
                        self.nlp = pyizumo.load(language_code, parsers=['token', 'sentence'])
                    parsed_text = self.nlp(text)

                    tsizes = []
                    begins = []
                    ends = []
                    _begins = []
                    _ends = []
                    sents = list(parsed_text.sentences)
                    for i in range(len(sents)):
                        _begins.append(sents[i].begin)
                        _ends.append(sents[i + 1].begin if i < len(sents) - 1 else len(text))

                    num_sents = len(list(parsed_text.sentences))
                    for i, sent in enumerate(parsed_text.sentences):
                        stext = sent.text
                        begin = _begins[i]
                        end = _begins[i + 1] if i < num_sents - 1 else len(text)
                        slen = self.get_tokenized_length(text[begin:end])
                        if slen > max_length:
                            tokens = list(sent.tokens)
                            too_long = [[tokens[k].begin, tokens[k + 1].begin] for k in range(len(tokens) - 1)]
                            too_long.append([tokens[-1].begin, end])
                            q = deque()
                            q.append(too_long)
                            while len(q) > 0:
                                head = q.pop()
                                ll = self.get_tokenized_length(text[head[0][0]:head[-1][1]])
                                if ll <= max_length:
                                    tsizes.append(ll)
                                    begins.append(head[0][0])
                                    ends.append(head[-1][1])
                                else:
                                    if len(head) > 1:
                                        mid = int(len(head) / 2)
                                        q.extend([head[mid:], head[:mid]])
                                    else:
                                        pass
                        else:
                            tsizes.append(slen)
                            begins.append(sent.begin)
                            ends.append(end)
                    first_length = max_length
                    if title_handling in ['all', 'first']:
                        first_length = max_length - title_length if not title_in_text else max_length
                    elif title_handling in ['none']:
                        first_length = max_length

                    intervals = TextTiler.compute_intervals(segment_lengths=tsizes,
                                                            max_length=max_length,
                                                            first_length=first_length,
                                                            stride=stride)

                    positions = [[begins[p[0]], ends[p[1]]] for p in intervals]
                    texts = [text[p[0]:p[1]] for p in positions]
                    added_titles.extend([True if title_handling == 'all' else False for _ in positions])
                    added_titles[0] = False if title_in_text or title_handling == 'none' else True
                else:  # Not aligned on sentences
                    if self.count_type == self.COUNT_TYPE_TOKEN:
                        if max_length is None:
                            max_length = self.max_doc_size
                        res = self.tokenizer(text=text, max_length=max_length, stride=stride,
                                             return_overflowing_tokens=True, truncation=True,
                                             return_offsets_mapping=True)
                        texts = []
                        positions = []
                        end_token = re.compile(f' ?{re.escape(self.tokenizer.sep_token)}$')
                        start_token = re.compile(f'{re.escape(self.tokenizer.cls_token)} ?')
                        # init_pos = 0
                        for split_passage, offsets in zip(res['input_ids'], res['offset_mapping']):
                            tt = self.remove_start_end_tokens(self.tokenizer, split_passage, start_token, end_token)
                            texts.append(tt)
                            id = 0
                            while split_passage[id] == self.tokenizer.cls_token_id:
                                id += 1
                            init_pos = offsets[id][0]
                            id = -1
                            while split_passage[id] == self.tokenizer.sep_token_id:
                                id -= 1
                            end_pos = offsets[id][1]
                            positions.append([init_pos, end_pos])
                            # init_pos += stride
                        added_titles = [False for _ in positions]
                    elif self.count_type == self.COUNT_TYPE_CHAR:
                        init_pos = 0
                        end_pos = max_length
                        texts = []
                        positions = []
                        added_titles = []
                        title_length = len(title)
                        max_len = len(text)
                        idx = 0
                        while init_pos < max_length:
                            clen = max_length
                            if self._need_to_add_title(idx, title_handling):
                                clen -= title_length + 1
                            end_pos = min(init_pos + clen, max_len)
                            while not text[end_pos].isspace() and end_pos >= init_pos:
                                end_pos -= 1
                            texts.append(get_expanded_text(text[init_pos:end_pos], title,
                                                           pos, title_handling, title_in_text))
                            positions.append([init_pos, end_pos])
                            added_titles.append(self._need_to_add_title(idx, title_handling, title_in_text))
                            init_pos = end_pos - stride
                            init_pos1 = init_pos
                            while not text[init_pos].isspace() and init_pos < end_pos:
                                init_pos += 1
                            # If there are only non-space chars till the end of the current chunk, then break forcesully
                            # in the middle of text.
                            if init_pos < end_pos:
                                init_pos += 1
                            else:
                                init_pos = init_pos1
                            idx += 1
                            if idx > 100:
                                raise RuntimeError(f"Data too big: {idx} fragments.")

                return texts, positions, added_titles

    @staticmethod
    def remove_start_end_tokens(tokenizer, split_passage, start_token, end_token):
        tt = end_token.sub(
            "",
            start_token.sub("",
                            tokenizer.decode(split_passage)
                            )
        )
        return tt

    MAX_TRIED = 10000

    @staticmethod
    def compute_intervals(segment_lengths: List[int],
                          max_length: int,
                          first_length: int,
                          stride: int) -> List[List[int | Any]]:
        """
        Compute Intervals Method
        This method computes intervals from a list of segment lengths based on a maximum length and a stride. It returns a list of interval ranges.

        Parameters:
        - segment_lengths (List[int]): A list of segment lengths.
        - max_length (int): The maximum length for each interval.
        - first_length (int): The length of the first sentences, if different from the rest (it will happen when using
                              title_handling="first" and the title is not in text.).
        - stride (int): The stride value for overlapping intervals.

        Returns:
        - List[List[int | Any]]: A list of interval ranges.

        """

        def check_for_cycling(segments_: List[List[int | Any]], prev_start_index_: int) -> None:
            if len(segments_) > 0 and segments_[-1][0] == prev_start_index_:
                raise RuntimeError(f"You have a problem with the splitting - it's cycling!: {segments_[-3:]}")

        def check_excessive_tries(number_of_tries_: int) -> bool:
            if number_of_tries_ > TextTiler.MAX_TRIED:
                print(f"Too many tried - probably something is wrong with the document.")
                return True
            else:
                return False

        current_index = 1
        current_total_length = segment_lengths[0]
        prev_start_index = 0
        segments = []
        number_of_tries = 0
        this_max_length = first_length

        while current_index < len(segment_lengths):
            if current_total_length + segment_lengths[current_index] > this_max_length:
                check_for_cycling(segments, prev_start_index)
                if check_excessive_tries(number_of_tries):
                    return segments
                if segments is None:
                    break
                number_of_tries += 1
                segments.append([prev_start_index, current_index - 1])
                this_max_length = max_length

                overlap = 0
                if current_index > 1 and \
                        segment_lengths[current_index - 1] + segment_lengths[current_index] <= this_max_length:
                    overlap_index = current_index - 1
                    max_length_tmp = this_max_length - segment_lengths[current_index]
                    while overlap_index > 0:
                        overlap += segment_lengths[overlap_index]
                        if overlap_index > prev_start_index + 1 and \
                                overlap < stride and \
                                overlap + segment_lengths[overlap_index - 1] <= max_length_tmp:
                            overlap_index -= 1
                        else:
                            break
                    current_index = overlap_index
                current_total_length = 0
                prev_start_index = current_index
                # current_total_length = overlap
            else:
                current_total_length += segment_lengths[current_index]
                current_index += 1
                number_of_tries = 0  # reset the failure count
        segments.append([prev_start_index, len(segment_lengths) - 1])

        return segments

    def print_product_histogram(self):
        """
        Prints the product ID histogram.

        This method iterates through the product_counts dictionary and prints the product ID along with its count in descending order of the count.

        :param self: The current instance of the class.
        """
        print(f"Product ID histogram:")
        for k in sorted(self.product_counts.keys(), key=lambda x: self.product_counts[x], reverse=True):
            print(f" {k}\t{self.product_counts[k]}")

    @staticmethod
    def _need_to_add_title(idx: int, title_handling: str, title_in_text: bool = False):
        if idx == 0:
            return not title_in_text and title_handling in ['first', 'all']
        else:
            return title_handling == 'all'
