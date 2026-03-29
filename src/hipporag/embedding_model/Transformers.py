from typing import List, Optional
import json

import torch
import numpy as np
from tqdm import tqdm

from .base import BaseEmbeddingModel
from ..utils.config_utils import BaseConfig
from ..prompts.linking import get_query_instruction
from sentence_transformers import SentenceTransformer

class TransformersEmbeddingModel(BaseEmbeddingModel):
    """
    To select this implementation you can initialise HippoRAG with:
        embedding_model_name starts with "Transformers/"
    """
    def __init__(self, global_config: Optional[BaseConfig] = None, embedding_model_name: Optional[str] = None) -> None:
        super().__init__(global_config=global_config)

        self.model_id = embedding_model_name[len("Transformers/"):]
        self.embedding_type = 'float'
        self.batch_size = getattr(global_config, 'embedding_batch_size', 64) if global_config else 64

        self.model = SentenceTransformer(self.model_id, device = "cuda" if torch.cuda.is_available() else "cpu", trust_remote_code=True)
        # Disable KV cache to avoid DynamicCache.get_usable_length() error with transformers>=4.56
        self.model._first_module().auto_model.config.use_cache = False

        self.search_query_instr = set([
            get_query_instruction('query_to_fact'),
            get_query_instruction('query_to_passage')
        ])

    def encode(self, texts: List[str], prompt: str = None) -> None:
        try:
            encode_kwargs = {"batch_size": self.batch_size}
            if prompt:
                encode_kwargs["prompt"] = prompt
            response = self.model.encode(texts, **encode_kwargs)
        except Exception as err:
            raise Exception(f"An error occurred: {err}")
        return np.array(response)

    def batch_encode(self, texts: List[str], **kwargs) -> None:
        # Build instruction prompt for instruction-tuned models (e.g. GTE-Qwen2-7B-instruct)
        prompt = None
        instruction = kwargs.get("instruction", "")
        if instruction:
            prompt = f"Instruct: {instruction}\nQuery: "

        if len(texts) < self.batch_size:
            return self.encode(texts, prompt=prompt)

        results = []
        batch_indexes = list(range(0, len(texts), self.batch_size))
        for i in tqdm(batch_indexes, desc="Batch Encoding"):
            results.append(self.encode(texts[i:i + self.batch_size], prompt=prompt))
        return np.concatenate(results)
