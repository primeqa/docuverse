"""
SentenceTransformers module for the Matryoshka residual adapter.

This file is **self-contained** — it depends only on ``torch`` and
``sentence_transformers``.  When a model is exported, this file is copied
into the model directory so that any environment can load it with::

    model = SentenceTransformer("path/to/model", trust_remote_code=True)

Architecture
------------
    adapted = sentence_embedding + MLP(sentence_embedding)

The skip (residual) connection is not representable with built-in ST modules,
hence the need for a custom module + ``trust_remote_code``.
"""

import json
import os
from collections import OrderedDict
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from sentence_transformers.models import Module as STModule
except ImportError:
    # Minimal shim so the file can at least be imported without ST installed
    from abc import ABC, abstractmethod

    class STModule(ABC, nn.Module):  # type: ignore[no-redef]
        config_keys: list = []
        config_file_name: str = "config.json"

        @abstractmethod
        def forward(self, features, **kwargs):
            ...

        @abstractmethod
        def save(self, output_path, *args, **kwargs):
            ...


class MatryoshkaAdaptorModule(STModule):
    """Residual MLP adapter for Matryoshka embeddings.

    ``forward()`` reads ``"sentence_embedding"`` from the ST features dict,
    applies ``emb + MLP(emb)``, optionally L2-normalizes, and writes it back.

    Parameters
    ----------
    in_features : int
        Embedding dimensionality (must match the upstream model).
    hidden_dims : list[int], optional
        Hidden-layer sizes for the residual MLP.  Default ``[512]``.
    normalize : bool, optional
        L2-normalize the output.  Default ``True``.
    """

    config_keys = ["in_features", "hidden_dims", "normalize"]

    def __init__(
        self,
        in_features: int,
        hidden_dims: Optional[List[int]] = None,
        normalize: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.hidden_dims = hidden_dims or [512]
        self.normalize = normalize

        # d → h1 → ReLU → … → d   (last layer near-zero so adapter starts ≈ identity)
        layers: list = []
        dim_in = in_features
        for h in self.hidden_dims:
            layers.append(nn.Linear(dim_in, h))
            layers.append(nn.ReLU())
            dim_in = h
        layers.append(nn.Linear(dim_in, in_features))
        self.mlp = nn.Sequential(*layers)

        nn.init.zeros_(self.mlp[-1].bias)
        nn.init.normal_(self.mlp[-1].weight, std=0.01)

    # ---- ST pipeline interface ------------------------------------------------

    def forward(
        self, features: Dict[str, torch.Tensor], **kwargs
    ) -> Dict[str, torch.Tensor]:
        emb = features["sentence_embedding"]
        adapted = emb + self.mlp(emb)
        if self.normalize:
            adapted = F.normalize(adapted, p=2, dim=-1)
        features["sentence_embedding"] = adapted
        return features

    def get_sentence_embedding_dimension(self) -> int:
        return self.in_features

    # ---- Persistence ----------------------------------------------------------

    def get_config_dict(self) -> dict:
        return {
            "in_features": self.in_features,
            "hidden_dims": self.hidden_dims,
            "normalize": self.normalize,
        }

    def save(
        self, output_path: str, *args, safe_serialization: bool = True, **kwargs
    ):
        os.makedirs(output_path, exist_ok=True)
        self.save_config(output_path)
        self.save_torch_weights(output_path, safe_serialization=safe_serialization)

    @classmethod
    def load(cls, model_name_or_path, subfolder="", **kwargs):
        config = cls.load_config(model_name_or_path, subfolder=subfolder)
        model = cls(**config)
        model = cls.load_torch_weights(
            model_name_or_path, subfolder=subfolder, model=model
        )
        return model

    # ---- Factory: load from a trainer checkpoint ------------------------------

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        normalize: bool = True,
        device: str = "cpu",
    ) -> "MatryoshkaAdaptorModule":
        """Create from a ``best_model.pt`` saved by the Matryoshka trainer.

        The trainer's ``AdaptorMLP`` uses the same ``self.mlp`` layout, so
        the ``state_dict`` keys (``mlp.0.weight``, …) match directly.
        """
        state = torch.load(checkpoint_path, map_location=device, weights_only=False)
        config = state["config"]
        module = cls(
            in_features=config.embedding_dim,
            hidden_dims=config.hidden_dims,
            normalize=normalize,
        )
        if "model_state_dict" in state:
            module.load_state_dict(state["model_state_dict"])
        return module


# ---------------------------------------------------------------------------
# Export helper
# ---------------------------------------------------------------------------

def export_to_sentence_transformer(
    base_model_name: str,
    checkpoint_path: str,
    output_dir: str,
    normalize: bool = True,
    device: str = "cpu",
):
    """Export a trained adapter as a portable SentenceTransformer model.

    The saved model directory will contain a copy of this file so it can be
    loaded in **any** environment (no ``docuverse`` required)::

        model = SentenceTransformer("path", trust_remote_code=True)

    Parameters
    ----------
    base_model_name : str
        HuggingFace model name or local path for the base embedding model.
    checkpoint_path : str
        Path to the trainer's ``best_model.pt``.
    output_dir : str
        Destination directory for the combined model.
    normalize : bool
        Whether the adapter should L2-normalize its output.
    device : str
        Torch device for loading weights.
    """
    import shutil
    from sentence_transformers import SentenceTransformer
    from sentence_transformers.models import Normalize

    print(f"Loading base model: {base_model_name}")
    base_model = SentenceTransformer(base_model_name, device=device)

    print(f"Loading adapter from: {checkpoint_path}")
    adapter = MatryoshkaAdaptorModule.from_checkpoint(
        checkpoint_path, normalize=normalize, device=device
    )

    # Strip existing Normalize — the adapter handles normalization
    new_modules = OrderedDict()
    for name, mod in base_model.named_children():
        if not isinstance(mod, Normalize):
            new_modules[name] = mod

    adapter_idx = len(new_modules)
    new_modules[str(adapter_idx)] = adapter

    combined = SentenceTransformer(modules=list(new_modules.values()), device=device)
    print(f"Saving combined model to: {output_dir}")
    combined.save(output_dir)

    # Copy this source file into the adapter subfolder AND the model root.
    # SentenceTransformer's get_class_from_dynamic_module looks for the file
    # in the model root directory, so both locations are needed.
    adapter_subfolder = f"{adapter_idx}_MatryoshkaAdaptorModule"
    adapter_dir = os.path.join(output_dir, adapter_subfolder)
    src_file = os.path.abspath(__file__)
    dst_file = os.path.join(adapter_dir, "st_module.py")
    shutil.copy2(src_file, dst_file)
    print(f"  Copied module source to {dst_file}")
    root_dst_file = os.path.join(output_dir, "st_module.py")
    shutil.copy2(src_file, root_dst_file)
    print(f"  Copied module source to {root_dst_file}")

    # Patch modules.json: type → "st_module.MatryoshkaAdaptorModule"
    modules_json_path = os.path.join(output_dir, "modules.json")
    with open(modules_json_path, "r") as f:
        modules_config = json.load(f)

    for entry in modules_config:
        if entry["idx"] == adapter_idx:
            entry["type"] = "st_module.MatryoshkaAdaptorModule"
            break

    with open(modules_json_path, "w") as f:
        json.dump(modules_config, f, indent=2)

    print(f"\nExported to: {output_dir}")
    print(f"Load with:   SentenceTransformer('{output_dir}', trust_remote_code=True)")
    return combined


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Export a trained Matryoshka adapter as a SentenceTransformer model"
    )
    parser.add_argument("--base-model", required=True,
                        help="Base embedding model (HF name or local path)")
    parser.add_argument("--checkpoint", required=True,
                        help="Path to best_model.pt from training")
    parser.add_argument("--output-dir", required=True,
                        help="Where to save the combined ST model")
    parser.add_argument("--no-normalize", action="store_true",
                        help="Skip L2 normalization after adaptation")
    parser.add_argument("--device", default="cpu")

    args = parser.parse_args()
    export_to_sentence_transformer(
        base_model_name=args.base_model,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        normalize=not args.no_normalize,
        device=args.device,
    )
