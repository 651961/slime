import os

import torch.distributed as dist

from slime.backends.megatron_utils.hf_checkpoint_saver import save_hf_model_to_path
from slime.backends.megatron_utils.initialize import init
from slime.backends.megatron_utils.model import initialize_model_and_optimizer
from slime.utils import accelerator
from slime.utils.arguments import parse_args
from slime.utils.distributed_utils import init_gloo_group


def add_checkpoint_args(parser):
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save the converted HF model.",
    )
    parser.add_argument(
        "--check-same",
        action="store_true",
        default=False,
        help="Check if the converted model is the same as the original model.",
    )
    return parser


def main(args):
    world_size = int(os.environ.get("WORLD_SIZE") or os.environ.get("SLURM_NTASKS") or 1)
    local_rank = int(os.environ.get("LOCAL_RANK") or os.environ.get("SLURM_LOCALID") or 0)
    global_rank = int(os.environ.get("RANK") or os.environ.get("SLURM_PROCID") or 0)
    accelerator.set_device(local_rank)
    os.environ.setdefault("WORLD_SIZE", str(world_size))
    os.environ.setdefault("RANK", str(global_rank))
    os.environ.setdefault("LOCAL_RANK", str(local_rank))
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "12355")
    dist.init_process_group(
        backend=accelerator.process_group_backend(),
        world_size=world_size,
        rank=global_rank,
        device_id=accelerator.distributed_device_id(local_rank),
    )
    init_gloo_group()
    init(args)

    # Loading an existing checkpoint is all that is needed for conversion.  In
    # particular, do not construct an optimizer: the exporter only reads model
    # parameters and old ``torch`` checkpoints may have been saved without one.
    args.no_load_optim = True
    args.no_load_rng = True
    model, _, _, _ = initialize_model_and_optimizer(args)

    if args.output_dir is None:
        raise ValueError("--output-dir is required when converting a checkpoint")
    if args.check_same:
        raise NotImplementedError("--check-same is not supported by the current HF checkpoint saver")
    save_hf_model_to_path(args, args.output_dir, model)
    dist.destroy_process_group()


if __name__ == "__main__":
    args = parse_args(add_custom_arguments=add_checkpoint_args)
    main(args)
