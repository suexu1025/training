import contextlib
from typing import Union

import torch
import torch_xla.amp
import torch_xla.amp.grad_scaler
import torch_xla.amp.syncfree
import torch_xla.core.xla_model as xm
import torch_xla.debug.profiler as xp
import torch_xla.distributed.parallel_loader as pl
from runtime.trainer.unet3d_trainer import UNet3DTrainer
from datetime import datetime
import torch_xla.test.test_utils as test_utils

class XLATrainer(UNet3DTrainer):
    """Trains UNet3D in PyTorch/XLA"""

    def __init__(
        self,
        flags,
        model,
        train_loader,
        val_loader,
        loss_fn,
        score_fn,
        device,
        callbacks,
    ) -> None:
        super().__init__(
            flags, model, train_loader, val_loader, loss_fn, score_fn, device, callbacks
        )
        assert isinstance(self.train_loader, pl.MpDeviceLoader), (
            "training data loader for XLATrainer must be an instance of "
            "torch_xla.distributed.parallel_loader.MpDeviceLoader"
        )
        assert isinstance(self.val_loader, pl.MpDeviceLoader), (
            "validation data loader for XLATrainer must be an instance of "
            "torch_xla.distributed.parallel_loader.MpDeviceLoader"
        )

        # Setup grad scaler and autocast
        self.scaler: torch_xla.amp.GradScaler = torch_xla.amp.grad_scaler.GradScaler()

        # Setup train sampler
        self.train_sampler = self.train_loader._loader.sampler

        # Get hardware type
        self.hw_type = xm.xla_device_hw(self.device)

        # Start and persist the profiler server
        if self.flags.profile_port:
            self.profile_server = xp.start_server(self.flags.profile_port)
        
        if flags.tb_dir !="":
            dataset_name = "kitts/" if not flags.use_brats else ""
            self.summary_dir = flags.tb_dir + dataset_name + datetime.now().strftime("%Y%m%d-%H%M%S")
            self.summary_interval = 100
            if xm.is_master_ordinal():
                self.summary_writer = test_utils.get_summary_writer(
                    self.summary_dir) if self.summary_interval else None

            test_utils.test_write_to_summary(
            self.summary_writer, 
            dict_to_write = {
                'flags': str(flags),
                'tb dir': self.summary_dir})
            xm.master_print('tb_summery_dir is {}'.format(self.summary_dir))

    def __del__(self):
        if self.summary_writer:
            test_utils.close_summary_writer(self.summary_writer)

    def train_step(
        self, iteration: int, images: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        """Overrides UNet3DTrainer.train_step"""
        if self.flags.profile_port:
            trace_context = xp.Trace("build_graph")
        else:
            trace_context = contextlib.nullcontext()
        # trace the forward and backward pass
        with trace_context:
            loss_value = self._forward(images=images, labels=labels)
            self._backward(loss_value=loss_value)

        self._optimizer_step(iteration=iteration)

        return loss_value

    def get_step_trace_context(self) -> Union[xp.StepTrace, contextlib.nullcontext]:
        """Overrides UNet3DTrainer.get_step_trace_context"""
        if self.flags.profile_port:
            return xp.StepTrace("train_unet3d")
        else:
            return contextlib.nullcontext()

    @staticmethod
    def get_optimizer(params, flags):
        """Overrides UNet3DTrainer.get_optimizer"""
        if flags.amp:
            if flags.optimizer == "adam":
                optim = torch_xla.amp.syncfree.Adam(
                    params, lr=flags.learning_rate, weight_decay=flags.weight_decay
                )
            elif flags.optimizer == "sgd":
                optim = torch_xla.amp.syncfree.SGD(
                    params,
                    lr=flags.learning_rate,
                    momentum=flags.momentum,
                    nesterov=True,
                    weight_decay=flags.weight_decay,
                )
            else:
                raise ValueError(f"Optimizer {flags.optimizer} is not supported for torch-xla amp")
            return optim
        else:
            return UNet3DTrainer.get_optimizer(params, flags)

    def _forward(self, images: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # Run the model forward pass
        with torch_xla.amp.autocast(enabled=self.flags.amp):
            output = self.model(images)
            if self.hw_type == "GPU":
                # currently, running torch-xla on GPU requires sandwiching
                # the loss value computation in between mark_step, otherwise
                # the CUDA memory will be corrupted
                # this is a temporary solution, and these two mark_step will
                # be removed once this memory corruption bug is resolved in torch-xla
                xm.mark_step()
            loss_value = self.loss_fn(output, labels)
            if self.hw_type == "GPU":
                xm.mark_step()
            loss_value /= self.flags.ga_steps

        return loss_value

    def _backward(self, loss_value: torch.Tensor):
        # Run the model backward pass
        if self.flags.amp:
            self.scaler.scale(loss_value).backward()
        else:
            loss_value.backward()

    def _optimizer_step(self, iteration: int):
        # Run the model weights update
        if (iteration + 1) % self.flags.ga_steps == 0:
            if self.flags.amp:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                xm.optimizer_step(self.optimizer)
