import os
import time
from math import ceil

import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
from mlperf_logging import mllog
from mlperf_logging.mllog import constants

from data_loading.data_loader.unet3d_data_loader import get_data_loaders
from model.losses import DiceCELoss, DiceScore
from model.unet3d import Unet3D
from runtime.arguments import PARSER
from runtime.callbacks import get_callbacks
from runtime.distributed.distributed_utils import (
    get_device,
    get_rank,
    get_world_size,
    init_distributed,
    is_main_process,
    seed_everything,
    setup_seeds,
)
from runtime.inference import evaluate
from runtime.logging import (
    get_dllogger,
    mllog_end,
    mllog_event,
    mllog_start,
    mlperf_run_param_log,
    mlperf_submission_log,
)
from runtime.trainer.trainer_factory import get_trainer

DATASET_SIZE = 168


def xla_main(local_rank, flags):
    main(local_rank, flags)
    xm.rendezvous("exit")


def main(local_rank, flags):
    is_distributed = init_distributed(flags)

    mllog.config(filename=os.path.join(os.path.dirname(os.path.abspath(__file__)), "unet3d.log"))
    mllogger = mllog.get_mllogger()
    mllogger.logger.propagate = False
    mllog_start(key=constants.INIT_START)

    dllogger = get_dllogger(flags)
    device = get_device(local_rank)
    world_size = get_world_size()
    local_rank = get_rank()
    worker_seeds, shuffling_seeds = setup_seeds(
        master_seed=flags.seed, epochs=flags.epochs, device=device
    )
    worker_seed = worker_seeds[local_rank]
    seed_everything(worker_seed)
    mllog_event(
        key=constants.SEED,
        value=flags.seed if flags.seed != -1 else worker_seed,
        sync=False,
    )

    if is_main_process():
        mlperf_submission_log()
        mlperf_run_param_log(flags)

    callbacks = get_callbacks(flags, dllogger, local_rank, world_size)
    flags.seed = worker_seed
    model = Unet3D(1, 3, normalization=flags.normalization, activation=flags.activation)

    if flags.use_fsdp:
        import torch_xla.core.xla_model as xm
        from pprint import pprint
        from torch_xla.distributed.fsdp import XlaFullyShardedDataParallel as FSDP, checkpoint_module

        import torch
        if flags.use_bf16:
            fsdp_wrap = lambda m: FSDP(m),shard_param_on_dim_0 = True,  compute_dtype = torch.bfloat16)
        else:
            fsdp_wrap = lambda m: FSDP(m),shard_param_on_dim_0 = True)

        # A wrapper over each transformer block with inner FSDP
        nested_fsdp_wrap = fsdp_wrap if flags.use_nested_fsdp else (lambda m: m)
        # A wrapper over each transformer block with gradient checkpointing
        grad_ckpt_wrap = checkpoint_module if flags.use_grad_ckpt else (lambda m: m)
        
        if flags.use_nested_fsdp or flags.use_grad_ckpt:
            # applying the wrappers above to the FSDP layers
            for i in range(len(model.downsample)):
                model.downsample[i] = nested_fsdp_wrap(grad_ckpt_wrap(model.downsample[i]))
            for i in range(len(model.upsample)):
                model.upsample[i] = nested_fsdp_wrap(grad_ckpt_wrap(model.upsample[i]))
            
            model.input_block = nested_fsdp_wrap(grad_ckpt_wrap(model.input_block))

            model.output = nested_fsdp_wrap(grad_ckpt_wrap(model.output))

        pprint(vars(model))
        # Wrap the base model with an outer FSDP wrapper
        # Also, copy the signature of the original model's forward method -- otherwise
        # Hugging Face datasets drops the columns not appearing in the forward method's argument
        # in its `_remove_unused_columns` in trainer.py
        import inspect
        forward_signature = inspect.signature(model.forward.__func__)
        model = fsdp_wrap(model)
        model.forward.__func__.__signature__ = forward_signature

        # Patch `xm.optimizer_step` not to reduce gradients in this case,
        # as FSDP does not need gradient reduction over sharded parameters.
        # Note: this ultimately should be something to be implemented in the Hugging Face trainer
        # to directly call `optimizer.step()` when the model is an FSDP instance,
        # but we chose to patch it here to get a standalone example without changing the Hugging Face trainer
        def patched_optimizer_step(optimizer, barrier=False, optimizer_args={}):
            loss = optimizer.step(**optimizer_args)
            if barrier:
                xm.mark_step()
            return loss

        xm.optimizer_step = patched_optimizer_step

    model = model.to(device)

    param_nums = sum(p.numel() for p in model.parameters().values())

    mllog_event(key="per-TPU (sharded) parameter num", value=param_nums)

    mllog_end(key=constants.INIT_STOP, sync=True)
    mllog_start(key=constants.RUN_START, sync=True)
    mllog_event(key="training_params", value=str(flags), sync=True)
    train_loader, val_loader = get_data_loaders(
        flags=flags,
        num_shards=world_size,
        global_rank=local_rank,
        device=device,
    )
    samples_per_epoch = world_size * len(train_loader) * flags.batch_size
    mllog_event(key="samples_per_epoch", value=samples_per_epoch, sync=False)
    flags.evaluate_every = flags.evaluate_every or ceil(20 * DATASET_SIZE / samples_per_epoch)
    flags.start_eval_at = flags.start_eval_at or ceil(1000 * DATASET_SIZE / samples_per_epoch)

    mllog_event(
        key=constants.GLOBAL_BATCH_SIZE,
        value=flags.batch_size * world_size * flags.ga_steps,
        sync=False,
    )
    mllog_event(key=constants.GRADIENT_ACCUMULATION_STEPS, value=flags.ga_steps)
    loss_fn = DiceCELoss(
        to_onehot_y=True,
        use_softmax=True,
        layout=flags.layout,
        include_background=flags.include_background,
    )
    score_fn = DiceScore(
        to_onehot_y=True,
        use_argmax=True,
        layout=flags.layout,
        include_background=flags.include_background,
    )

    if flags.exec_mode == "train":
        trainer = get_trainer(
            flags, model, train_loader, val_loader, loss_fn, score_fn, device, callbacks
        )
        trainer.train()

    elif flags.exec_mode == "evaluate":
        eval_metrics = evaluate(
            flags,
            model,
            val_loader,
            loss_fn,
            score_fn,
            device=device,
            is_distributed=is_distributed,
        )
        if local_rank == 0:
            for key in eval_metrics.keys():
                print(key, eval_metrics[key])
    else:
        print("Invalid exec_mode.")
        pass


if __name__ == "__main__":
    flags = PARSER.parse_args()
    # record the program start time, which is later used for
    # calculating the training start-up time
    flags.program_start_time = time.time()

    if flags.device == "xla":
        xmp.spawn(xla_main, args=(flags,))
    elif flags.device == "cuda":
        assert "LOCAL_RANK" in os.environ, (
            "Please use torchrun to launch the program\n"
            "See https://pytorch.org/docs/stable/distributed.html#launch-utility"
            "for further instructions"
        )
        local_rank = int(os.environ["LOCAL_RANK"])
        main(local_rank, flags)
    else:
        raise ValueError(f"Device {flags.device} unknown. Valid devices are: cuda, xla")
