import requests
import numpy as np

from docuverse.utils.embeddings.embedding_function import EmbeddingFunction


# Create a wrapper class to make it compatible with SentenceTransformer interface
class OllamaSentenceTransformer:
    _dim = None

    def __init__(self, model_name: str, base_url: str = "http://localhost:11434"):
        self.model_name = model_name
        self.base_url = base_url

        # Test connection to Ollama
        try:
            response = requests.get(f"{self.base_url}/api/tags")
            if response.status_code != 200:
                raise ConnectionError(f"Cannot connect to Ollama at {self.base_url}")

            # Check if model is available
            models = response.json().get('models', [])
            model_names = [m['name'] for m in models]
            if not any(model_name in name for name in model_names):
                print(f"⚠ Warning: Model '{model_name}' not found in Ollama")
                print(f"Available models: {', '.join(model_names)}")
                print(f"You may need to run: ollama pull {model_name}")
        except requests.exceptions.ConnectionError:
            raise ConnectionError(
                f"Cannot connect to Ollama at {self.base_url}. "
                "Make sure Ollama is running (ollama serve)"
            )

    def encode(self, sentences, batch_size: int = 32, **kwargs):
        """Encode sentences using Ollama model."""
        single_vector=False
        if isinstance(sentences, str):
            sentences = [sentences]
            single_vector = True

        embeddings = []

        # Process in batches
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i + batch_size]

            for text in batch:
                try:
                    response = requests.post(
                        f"{self.base_url}/api/embeddings",
                        json={
                            "model": self.model_name,
                            "prompt": text,
                        },
                        timeout=30
                    )

                    if response.status_code == 200:
                        embedding = response.json()["embedding"]
                        embeddings.append(embedding)
                    else:
                        raise RuntimeError(
                            f"Ollama API error: {response.status_code} - {response.text}"
                        )
                except requests.exceptions.Timeout:
                    raise TimeoutError(f"Ollama request timeout for text: {text[:50]}...")

        if single_vector:
            embeddings = embeddings[0]

        return np.array(embeddings)

    def get_sentence_embedding_dimension(self):
        """Get embedding dimension by encoding a test sentence."""
        if self._dim is None:
            enc = self.encode(["Test"])
            self._dim = len(enc[0])
        return self._dim

    def __repr__(self):
        return f"OllamaSentenceTransformer(model_name='{self.model_name}', base_url='{self.base_url}')"

class OllamaEmbeddingFunction(EmbeddingFunction):
    _dim = None

    def __init__(self, model_name: str, base_url: str = "http://localhost:11434"):
        self.model_name = model_name
        self.base_url = base_url
        try:
            response = requests.get(f"{self.base_url}/api/tags")
            if response.status_code != 200:
                raise ConnectionError(f"Cannot connect to Ollama at {self.base_url}")
            models = response.json().get('models', [])
            if not any(model_name in name for name in models):
                print(f"⚠ Warning: Model '{model_name}' not found in Ollama")
                print(f"Available models: {', '.join(models)}")
                print(f"You may need to run: ollama pull {model_name}")
        except requests.exceptions.ConnectionError:
            raise ConnectionError(
                f"Cannot connect to Ollama at {self.base_url}. "
                "Make sure Ollama is running (ollama serve)"
            )

    def __repr__(self):
        return f"OllamaEmbeddingFunction(model_name='{self.model_name}', base_url='{self.base_url}')"

    def __call__(self, sentences, batch_size: int = 32, **kwargs):
        return self.encode(sentences, batch_size=batch_size, **kwargs)

    def encode(self, sentences, batch_size: int = 32, **kwargs):
        """Encode sentences using Ollama model."""
        if isinstance(sentences, str):
            sentences = [sentences]

        embeddings = []

        # Process in batches
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i + batch_size]

            for text in batch:
                try:
                    response = requests.post(
                        f"{self.base_url}/api/embeddings",
                        json={
                            "model": self.model_name,
                            "prompt": text,
                        },
                        timeout=30
                    )

                    if response.status_code == 200:
                        embedding = response.json()["embedding"]
                        embeddings.append(embedding)
                    else:
                        raise RuntimeError(
                            f"Ollama API error: {response.status_code} - {response.text}"
                        )
                except requests.exceptions.Timeout:
                    raise TimeoutError(f"Ollama request timeout for text: {text[:50]}...")

        return np.array(embeddings)

    def get_sentence_embedding_dimension(self):
        """Get embedding dimension by encoding a test sentence."""
        if self._dim is None:
            enc = self.encode(["Test"])
            self._dim = len(enc[0])
        return self._dim

    