#!/usr/bin/env bash
# Qwen3.8 image-edit reward training on an existing Ray cluster.
#
# Prerequisites:
#   - A Ray cluster is already running; execute this script on its head node.
#   - The cluster has enough GPUs for the actor configuration below.
#   - The model, datasets, and checkpoint directory are visible on every node.

set -ex
export PYTHONUNBUFFERED=1

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." &>/dev/null && pwd)"
cd "${REPO_ROOT}"

BASE_FOLDER="${BASE_FOLDER:-${REPO_ROOT}/checkpoints}"
ACTOR_NUM_NODES=${ACTOR_NUM_NODES:-2}

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

# Qwen3.8 uses the Qwen3.5-27B model configuration with the VLM provider.
source "${REPO_ROOT}/scripts/models/qwen3.5-27B.sh"
MODEL_ARGS[1]="slime_plugins.models.qwen3_5_vl"
MODEL_ARGS[2]="get_qwen3_5_vl_model_provider"

CKPT_ARGS=(
   --hf-checkpoint "/models/Qwen3.8-27B"
   --ref-load "/models/Qwen3.8-27B"
   --load "/models/Qwen3.8-27B"
   --save "${BASE_FOLDER}/Qwen3.8-27B_slime/"
   --no-save-optim
   --save-interval 5
   --ckpt-format torch
)

# Default train/eval JSONL image blocks set max_pixels=4194304.
# Dataset loading resizes images before both rollout and training consume them.
ROLLOUT_ARGS=(
   --save-debug-rollout-data "${BASE_FOLDER}/Qwen3.8-27B_slime/rollout_details/{rollout_id}.pt"
   --prompt-data "${PROMPT_DATA:-${SCRIPT_DIR}/image_edit_reward.jsonl}"
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle
   --custom-rm-path slime_plugins.rewards.qwen3_8_image_edit.compute_reward
   --num-rollout 30000
   --rollout-batch-size 8
   --n-samples-per-prompt 16
   --rollout-max-response-len 28000
   --rollout-temperature 1.0
   --global-batch-size 128
   --balance-data
)

EVAL_ARGS=(
   --eval-interval 5
   --eval-prompt-data reward_test /datasets/codes_zsqiao/slime/examples/image_edit_reward/eval_image_edit_reward.jsonl
   --eval-input-key prompt
   --eval-label-key label
   --n-samples-per-eval-prompt 3
   --eval-max-response-len 28000
)

PERF_ARGS=(
   --distributed-timeout-minutes 60
   --tensor-model-parallel-size 4
   --sequence-parallel
   --pipeline-model-parallel-size 2
   --use-tp-pp-dp-mapping
   --decoder-last-pipeline-num-layers 30
   --context-parallel-size 1
   --expert-model-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   --micro-batch-size 1
   --calculate-per-token-loss
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --kl-coef 0.00
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --clip-grad 0.5
   --lr-decay-style constant
   --weight-decay 0.1
   --freeze-params-name-list '^model\.visual\.'
   --adam-beta1 0.9
   --adam-beta2 0.98
   --optimizer-cpu-offload
   --overlap-cpu-optimizer-d2h-h2d
   --use-precision-aware-optimizer
)

WANDB_ARGS=(
   # --use-wandb
   # --wandb-project slime-dev
   # --wandb-group qwen3.5-27B-32k
   # --wandb-key ${WANDB_KEY}
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 2
   --sglang-mem-fraction-static 0.7
   --sglang-speculative-algorithm EAGLE
   --sglang-speculative-num-steps 3
   --sglang-speculative-eagle-topk 1
   --sglang-speculative-num-draft-tokens 4
   --sglang-mamba-scheduler-strategy extra_buffer
)

MISC_ARGS=(
   --use-tensorboard
   --tb-project-name image_edit_reward
   --tb-experiment-name "$(date +%Y%m%d_%H%M%S)"
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
)

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/Megatron-LM/:${REPO_ROOT}\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\"
  }
}"

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train.py \
   --actor-num-nodes "${ACTOR_NUM_NODES}" \
   --actor-num-gpus-per-node 8 \
   --colocate \
   "${MODEL_ARGS[@]}" \
   "${CKPT_ARGS[@]}" \
   "${ROLLOUT_ARGS[@]}" \
   "${OPTIMIZER_ARGS[@]}" \
   "${GRPO_ARGS[@]}" \
   "${WANDB_ARGS[@]}" \
   "${PERF_ARGS[@]}" \
   "${EVAL_ARGS[@]}" \
   "${SGLANG_ARGS[@]}" \
   "${MISC_ARGS[@]}"
