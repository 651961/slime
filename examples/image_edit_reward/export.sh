#!/usr/bin/env bash
# Export a Megatron ``torch`` checkpoint produced by run_qwen38_image_edit.sh
# to a Transformers/Hugging Face checkpoint.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." &>/dev/null && pwd)"
cd "${REPO_ROOT}"

# Override these paths when the checkpoint or the original HF model is stored
# elsewhere.  The original model supplies config/tokenizer/vision assets that
# are not part of a Megatron checkpoint.
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${REPO_ROOT}/checkpoints/Qwen3.8-27B_slime/iter_0000024}"
HF_CHECKPOINT="${HF_CHECKPOINT:-/models/Qwen3.8-27B}"
OUTPUT_DIR="${OUTPUT_DIR:-${CHECKPOINT_DIR}_transformers}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
MEGATRON_ROOT="${MEGATRON_ROOT:-/root/Megatron-LM}"
CHECKPOINT_ROOT="$(dirname -- "${CHECKPOINT_DIR}")"
CHECKPOINT_ITERATION="${CHECKPOINT_ITERATION:-${CHECKPOINT_DIR##*_}}"

[[ -d "${CHECKPOINT_DIR}" ]] || { echo "Checkpoint directory does not exist: ${CHECKPOINT_DIR}" >&2; exit 1; }
[[ -d "${HF_CHECKPOINT}" ]] || { echo "HF checkpoint directory does not exist: ${HF_CHECKPOINT}" >&2; exit 1; }

source "${REPO_ROOT}/scripts/models/qwen3.5-27B.sh"
# Qwen3.8 image-edit training uses the native Qwen3.5-VL model provider.
MODEL_ARGS[1]="slime_plugins.models.qwen3_5_vl"
MODEL_ARGS[2]="get_qwen3_5_vl_model_provider"

export PYTHONPATH="${MEGATRON_ROOT}:${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
# Megatron's legacy ``torch`` checkpoints contain the training args object;
# allow PyTorch to unpickle this trusted local checkpoint.
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

torchrun --standalone --nproc-per-node="${NPROC_PER_NODE}" tools/convert_to_hf.py \
   "${MODEL_ARGS[@]}" \
   --hf-checkpoint "${HF_CHECKPOINT}" \
   --load "${CHECKPOINT_ROOT}" \
   --ref-load "${CHECKPOINT_ROOT}" \
   --ckpt-step "${CHECKPOINT_ITERATION#0}" \
   --ckpt-format torch \
   --num-rollout 0 \
   --rollout-batch-size 1 \
   --global-batch-size 1 \
   --micro-batch-size 1 \
   --tensor-model-parallel-size 4 \
   --sequence-parallel \
   --pipeline-model-parallel-size 2 \
   --use-tp-pp-dp-mapping \
   --decoder-last-pipeline-num-layers 30 \
   --context-parallel-size 1 \
   --expert-model-parallel-size 1 \
   --output-dir "${OUTPUT_DIR}"

# The generic HF saver writes parameter-sized shards and the Megatron model
# provider does not instantiate Qwen's MTP head.  Repack the converted tensors
# using the original HF index, filling those unchanged MTP tensors from the
# base model.  This keeps the exported directory layout identical to the base
# Transformers checkpoint.
python - "${OUTPUT_DIR}" "${HF_CHECKPOINT}" <<'PY'
import json
import os
import shutil
import sys
from contextlib import ExitStack
from pathlib import Path

import safetensors.torch
from safetensors import safe_open

output_dir = Path(sys.argv[1])
origin_dir = Path(sys.argv[2])
origin_index_path = origin_dir / "model.safetensors.index.json"
converted_index_path = output_dir / "model.safetensors.index.json"
origin_index = json.loads(origin_index_path.read_text())
converted_index = json.loads(converted_index_path.read_text())
origin_map = origin_index["weight_map"]
converted_map = converted_index["weight_map"]

missing = set(origin_map) - set(converted_map)
unexpected = set(converted_map) - set(origin_map)
if unexpected:
    raise RuntimeError(f"Converted checkpoint has unexpected tensors: {sorted(unexpected)}")
if missing:
    print(f"Copying {len(missing)} tensors from the origin HF checkpoint: {sorted(missing)}")

target_files = set(origin_map.values())
for target_file in sorted(target_files):
    target_names = [name for name, filename in origin_map.items() if filename == target_file]
    tensors = {}
    with ExitStack() as stack:
        source_handles = {}
        for name in target_names:
            if name in converted_map:
                source_dir, source_file = output_dir, converted_map[name]
            else:
                source_dir, source_file = origin_dir, origin_map[name]
            key = (source_dir, source_file)
            if key not in source_handles:
                source_handles[key] = stack.enter_context(
                    safe_open(str(source_dir / source_file), framework="pt", device="cpu")
                )
            tensors[name] = source_handles[key].get_tensor(name)

    temporary_file = output_dir / f".{target_file}.tmp"
    safetensors.torch.save_file(tensors, str(temporary_file), metadata={"format": "pt"})
    os.replace(temporary_file, output_dir / target_file)

for path in output_dir.glob("*.safetensors"):
    if path.name not in target_files:
        path.unlink()
shutil.copy2(origin_index_path, output_dir / origin_index_path.name)
print(f"Repacked {len(target_files)} safetensors shards to match {origin_dir}")
PY
