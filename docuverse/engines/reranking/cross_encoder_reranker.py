import torch
from cv2.gapi.onnx.ep import OpenVINO

from docuverse.utils import detect_device, get_param
from tqdm import tqdm

from .reranker import Reranker
from docuverse.engines.search_engine_config_params import RerankerConfig as RerankerConfig
from sentence_transformers.cross_encoder import CrossEncoder

class CrossEncoderModel:
    """
    A model for cross-encoder tasks, designed to compute similarity scores
    for input pairs using a pre-trained transformer model.

    Provides methods to initialize a cross-encoder model with a specified
    pre-trained transformer model and compute similarity scores between
    pairs of inputs. The model uses AutoModelForSequenceClassification for
    sequence classification tasks and supports tokenization for input pairs.

    Attributes:
    model : Pre-trained transformer model for sequence classification.
    tokenizer : Tokenizer associated with the model for input processing.

    """
    def __init__(self, model_name_or_path, device=None,
                 attn_implementation=None, **kwargs):
        """
        Initializes the object with a language model and its associated tokenizer. The model is configured
        for sequence classification tasks and is set up to run on the specified device.

        Parameters:
        model_name : str
            The name or path of the pre-trained model to load.
        device : str, optional
            The device on which the model will run, such as 'cuda' or 'cpu'.
        """
        # self.model = AutoModel.from_pretrained(model_name)
        if device is None:
            device = detect_device()
        model_kwargs = {}
        if attn_implementation is not None:
            model_kwargs['attn_implementation'] = attn_implementation
            if attn_implementation.find("flash") >= 0:
                import torch
                model_kwargs["torch_dtype"] = torch.bfloat16
        backend = get_param(kwargs, 'reranker_backend', None)
        if backend is not None:
            model_kwargs['backend'] = backend

        self.model = CrossEncoder(model_name_or_path, device=device,
                                  tokenizer_kwargs={'model_max_length': 512},
                                  model_kwargs=model_kwargs)
        # self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        # self.model.to(device)
        # self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def predict(self, pairs, device='cuda'):
        """
        Predicts confidence scores for given data pairs using a pre-trained model.

        This method tokenizes the input data pairs, processes them through a pre-trained model,
        and extracts prediction scores. The function utilizes the specified device for
        computation and ensures that gradients are not calculated during inference.

        Parameters:
        pairs: list
            A list of input data pairs to be tokenized and processed by the model.
        device: str, optional
            The device to use for computation, either 'cuda' or 'cpu'. Default is 'cuda'.

        Returns:
        list
            A list of confidence scores predicted by the model for the input data pairs.
        """
        scores = None
        with torch.no_grad():
            # enc = self.tokenizer(pairs, padding=True, truncation=True, return_tensors='pt').to(device)
            # res = self.model(**enc, return_dict=True)
            # scores = res.logits.view(-1, ).float()
            scores = self.model.predict(pairs, convert_to_numpy=True)

            # scores = res.last_hidden_state[:,0,0].float()
            # scores = res.last_hidden_state[:,0,:].softmax(dim=1)[:,0].tolist()
            # pred_ids = scores.argsort(descending=True)
        # return scores.exp().tolist()
        return scores

class CrossEncoderReranker(Reranker):
    """
    CrossEncoderReranker is a class designed to perform re-ranking of candidate answers
    based on their semantic similarity to the given question and other candidates.

    This class inherits from the Reranker base class and utilizes a cross-encoder model
    for comparing question-answer pairs. It is designed for use in contexts where improving
    the ranking quality of retrieved answers is essential, particularly when computational
    resources allow for the application of cross-encoding techniques.

    Attributes:
        model (CrossEncoderModel): The model used to compute semantic similarity
        scores for re-ranking candidate answers.

    Methods:
        __init__: Initializes the CrossEncoderReranker with specified configurations.
        _rerank: Re-ranks a list of candidate answers based on semantic similarity.
    """
    def __init__(self, reranking_config: RerankerConfig|dict, **kwargs):
        super().__init__(reranking_config, **kwargs)
        self.model = CrossEncoderModel(reranking_config.reranker_model,
                                       attn_implementation=reranking_config.reranker_attn_implementation,
                                       backend=reranking_config.reranker_backend)

    def _rerank(self, answer_list, show_progress):
        """
        Reranks a list of answers based on similarity scores obtained from a model.

        Uses a cross-encoding model to compute similarity scores between the question
        and corresponding answers in the provided list. Answers are then reranked
        appropriately using the similarity scores. A progress bar is displayed if
        enabled.

        Parameters:
        answer_list : list
            A list of answers to be reranked. Each answer in the list contains the
            question text and associated answers.
        show_progress : bool
            Indicates whether or not to display a progress bar during reranking.

        Returns:
        list
            A list of reranked answers, where the similarity scores were applied to
            reorder the answers.

        Raises:
        None
        """
        num_docs = len(answer_list)
        output = []
        for answer in tqdm(answer_list, desc="Computing cross-encodings",
                                total=num_docs, disable=not show_progress):
            self.tm.mark()
            similarity_scores = self.model.predict([[answer.question.text, t.text] for t in answer.top_k(self.top_k)])
            self.tm.add_timing("similarity_computation")
            # sorted_similarities = sorted(zip(answer, similarity_scores),
            #                              key=lambda pair: pair[1], reverse=True)
            output.append(self._build_sorted_list(answer, similarity_scores))
            # self.tm.add_timing("cosine::reorder")
        return output

