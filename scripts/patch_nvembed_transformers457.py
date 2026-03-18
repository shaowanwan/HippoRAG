#!/usr/bin/env python3
"""
Patch NV-Embed-v2 model files for compatibility with transformers >= 4.57.

transformers 4.57 introduced three breaking changes affecting NV-Embed-v2:
1. Removed MISTRAL_INPUTS_DOCSTRING from transformers.models.mistral.modeling_mistral
2. Moved rotary embedding computation from MistralDecoderLayer to MistralModel level
3. MistralDecoderLayer.forward() now returns a single Tensor instead of a tuple

This script patches all three locations where NV-Embed-v2 code lives:
  A) Local model files (e.g., /data/.../hf_models/NV-Embed-v2/)
  B) HF cache (e.g., /data/.../cache/huggingface/modules/transformers_modules/NV_hyphen_Embed_hyphen_v2/)
  C) pip-installed nv_embed_v2 fixed fork (e.g., site-packages/nv_embed_v2/)

Usage:
    python patch_nvembed_transformers457.py /path/to/modeling_nvembed.py [--dry-run]

Or patch all three locations at once:
    python patch_nvembed_transformers457.py --all \
        --model-dir /data/s4303873/hf_models/NV-Embed-v2 \
        --cache-dir /data/s4303873/cache/huggingface/modules/transformers_modules/NV_hyphen_Embed_hyphen_v2 \
        --fork-dir /data/s4303873/envs/hipporag/lib/python3.10/site-packages/nv_embed_v2
"""

import argparse
import re
import shutil
from pathlib import Path


def patch_file(filepath: Path, dry_run: bool = False) -> list[str]:
    """Apply all necessary patches to a modeling_nvembed.py file.

    Returns list of patches applied.
    """
    text = filepath.read_text()
    original = text
    patches = []

    # Patch 1: Replace MISTRAL_INPUTS_DOCSTRING import with empty string
    old_import = "from transformers.models.mistral.modeling_mistral import MISTRAL_INPUTS_DOCSTRING"
    if old_import in text:
        text = text.replace(old_import, 'MISTRAL_INPUTS_DOCSTRING = ""')
        patches.append("Replaced MISTRAL_INPUTS_DOCSTRING import with empty string")

    # Patch 2: Fix layer_outputs[0] → handle both Tensor and tuple returns
    # Pattern: hidden_states = layer_outputs[0]
    old_layer_output = "hidden_states = layer_outputs[0]"
    new_layer_output = "hidden_states = layer_outputs if isinstance(layer_outputs, torch.Tensor) else layer_outputs[0]"
    if old_layer_output in text and new_layer_output not in text:
        text = text.replace(old_layer_output, new_layer_output)
        patches.append("Fixed layer_outputs[0] to handle Tensor return type")

    # Patch 3: Add position_embeddings computation before decoder layer loop
    # Only needed for files that don't already have it (AutoModel path)
    if "position_embeddings = self.rotary_emb" not in text:
        # Look for the decoder layer loop pattern
        pattern = r"(for idx, decoder_layer in enumerate\(self\.layers\):)"
        if re.search(pattern, text):
            replacement = (
                "position_embeddings = self.rotary_emb(hidden_states, position_ids)\n"
                "\n"
                "        \\1"
            )
            text = re.sub(pattern, replacement, text)
            patches.append("Added position_embeddings computation before decoder loop")

    # Patch 4: Pass position_embeddings to decoder_layer call
    # Look for decoder_layer calls missing position_embeddings kwarg
    if "position_embeddings=position_embeddings" not in text and "position_embeddings" in text:
        # This is context-dependent; only add if position_embeddings was computed
        pattern = r"(decoder_layer\([^)]*?)(,?\s*\))"
        # Only patch if there's a decoder_layer call without position_embeddings
        for match in re.finditer(r"decoder_layer\(([^)]*)\)", text):
            if "position_embeddings" not in match.group(1):
                old_call = match.group(0)
                new_call = old_call[:-1] + ",\n                    position_embeddings=position_embeddings,\n                )"
                text = text.replace(old_call, new_call, 1)
                patches.append("Added position_embeddings to decoder_layer call")
                break

    if text != original:
        if not dry_run:
            backup = filepath.with_suffix(".py.bak")
            shutil.copy2(filepath, backup)
            filepath.write_text(text)
            print(f"  Patched {filepath} ({len(patches)} changes, backup: {backup.name})")
        else:
            print(f"  [DRY RUN] Would patch {filepath} ({len(patches)} changes)")
        for p in patches:
            print(f"    - {p}")
    else:
        print(f"  {filepath}: already patched or no changes needed")

    return patches


def main():
    parser = argparse.ArgumentParser(description="Patch NV-Embed-v2 for transformers >= 4.57")
    parser.add_argument("file", nargs="?", help="Single modeling_nvembed.py to patch")
    parser.add_argument("--all", action="store_true", help="Patch all three locations")
    parser.add_argument("--model-dir", help="Path to local model directory")
    parser.add_argument("--cache-dir", help="Path to HF cache transformers_modules directory")
    parser.add_argument("--fork-dir", help="Path to pip-installed nv_embed_v2 package directory")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be changed")
    args = parser.parse_args()

    if args.file:
        path = Path(args.file)
        if not path.exists():
            print(f"Error: {path} not found")
            return 1
        patch_file(path, dry_run=args.dry_run)
    elif args.all:
        dirs = {
            "model": args.model_dir,
            "cache": args.cache_dir,
            "fork": args.fork_dir,
        }
        for name, d in dirs.items():
            if not d:
                print(f"Skipping {name} (no path provided)")
                continue
            path = Path(d) / "modeling_nvembed.py"
            if not path.exists():
                print(f"Warning: {path} not found, skipping")
                continue
            print(f"\nPatching {name}: {path}")
            patch_file(path, dry_run=args.dry_run)
    else:
        parser.print_help()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
