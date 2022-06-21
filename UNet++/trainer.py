#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Jun 18 12:18:52 2022

@author: Nacriema

Refs:

"""
import torch.cuda

from torch.utils.data import DataLoader
import numpy as np
import yaml
import shutil
import time
import argparse

# TODO: Is there a better way of reducing these import prefix src. ?
from src.utils import coerce_to_path_and_check_exist, coerce_to_path_and_create_dir
from src.utils.metrics import AverageMeter, RunningMetrics
from src.datasets import get_dataset
from src.models import get_model
from src.models.tools import safe_model_state_dict, count_parameters, EarlyStopping
from src.optimizers import get_optimizer
from src.schedulers import get_scheduler
from src.loss import get_loss
from src.utils.path import CONFIGS_PATH, MODELS_PATH, MODEL_FILE
from src.utils.logger import get_logger, print_info

PRINT_TRAIN_STAT_FMT = "Epoch [{}/{}], Iter [{}/{}]: train_loss = {:.4f}, time/img = {:.4f}s"
PRINT_VAL_STAT_FMT = "Epoch [{}/{}], Iter [{}/{}]: val_loss = {:.4f}"
PRINT_LR_UPD_FMT = "Epoch [{}/{}], Iter [{}/{}]: LR updated, lr = {}"

TRAIN_METRICS_FILE = "train_metrics.tsv"
VAL_METRICS_FILE = "val_metrics.tsv"


class Trainer:
    """Pipeline to train a NN model using a certain dataset, both specified by an YML config."""

    def __init__(self, config_path, run_dir):
        self.config_path = coerce_to_path_and_check_exist(config_path)

        # Create the run dir directory, this is inside the /models folder
        self.run_dir = coerce_to_path_and_create_dir(run_dir)
        self.logger = get_logger(log_dir=self.run_dir, name="trainer")
        self.print_and_log_info("Trainer initialization: run directory is {}".format(run_dir))

        shutil.copy(src=self.config_path, dst=self.run_dir)
        self.print_and_log_info("Config {} copied to run directory.".format(self.config_path))

        with open(self.config_path) as fp:
            cfg = yaml.load(fp, Loader=yaml.FullLoader)

        if torch.cuda.is_available():
            type_device = "cuda"
            nb_device = torch.cuda.device_count()

            # TODO XXX: set to False when input image size are not fixed
            torch.backends.cudnn.benchmark = cfg["training"].get("cudnn_benchmark", True)

        else:
            type_device = "cpu"
            nb_device = None

        self.device = torch.device(type_device)
        self.print_and_log_info("Using {} device, nb_device is {}".format(type_device, nb_device))

        # Datasets and data loaders
        self.dataset_kwargs = cfg["dataset"]
        self.dataset_name = self.dataset_kwargs.pop("name")

        # Get train and val dataset
        # TODO: Design and read the dataset normalization
        train_dataset = get_dataset(self.dataset_name)(split="train", **self.dataset_kwargs)

        val_dataset = get_dataset(self.dataset_name)(split="val", **self.dataset_kwargs)

        self.restricted_labels = sorted(self.dataset_kwargs["restricted_labels"])

        self.n_classes = len(self.restricted_labels) + 1  # I wanted 1 class output to apply the BCELoss
        self.is_val_empty = len(val_dataset) == 0

        self.print_and_log_info("Dataset {} instantiated with {}".format(self.dataset_name, self.dataset_kwargs))
        self.print_and_log_info(
            "Found {} classes, {} train samples, {} val samples".format(self.n_classes, len(train_dataset),
                                                                        len(val_dataset)))

        self.batch_size = cfg["training"]["batch_size"]
        self.n_workers = cfg["training"]["n_workers"]

        # Patience times to wait for a new better validation performance
        self.patience = cfg["training"]["patience"]

        # Use Dataloader to create batch sample for training
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size,
                                       num_workers=self.n_workers, shuffle=True)

        self.val_loader = DataLoader(val_dataset, batch_size=self.batch_size, num_workers=self.n_workers)
        self.print_and_log_info("Dataloaders instantiated with batch_size={} and n_worker={}".format(self.batch_size, self.n_workers))

        self.n_batches = len(self.train_loader)
        self.n_iterations, self.n_epoches = cfg["training"].get("n_iterations"), cfg["training"].get("n_epoches")

        # In my case, I prefer given epochs in the config file
        if self.n_iterations is not None:
            self.n_epoches = max(self.n_iterations // self.n_batches, 1)
        else:
            self.n_iterations = self.n_epoches * len(self.train_loader)

        # Model
        self.model_kwargs = cfg["model"]
        self.model_name = self.model_kwargs.pop("name")
        self.save_path = self.run_dir / MODEL_FILE

        # Enable the EarlyStopping
        self.early_stopping = EarlyStopping(patience=self.patience, verbose=True, path=self.save_path,
                                            trace_func=self.print_and_log_info)

        model = get_model(self.model_name)(self.n_classes, **self.model_kwargs).to(self.device)
        self.model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
        self.print_and_log_info("Using model {} with kwargs {}".format(self.model_name, self.model_kwargs))
        self.print_and_log_info('Number of trainable parameters: {}'.format(f'{count_parameters(self.model):,}'))

        # Optimizer
        optimizer_params = cfg["training"]["optimizer"] or {}
        optimizer_name = optimizer_params.pop("name", None)
        self.optimizer = get_optimizer(optimizer_name)(model.parameters(), **optimizer_params)
        self.print_and_log_info("Using optimizer {} with kwargs {}".format(optimizer_name, optimizer_params))

        # Scheduler
        scheduler_params = cfg["training"].get("scheduler", {}) or {}
        scheduler_name = scheduler_params.pop("name", None)

        # Scheduler update range unit
        self.scheduler_update_range = scheduler_params.pop("update_range", "epoch")
        assert self.scheduler_update_range in ["epoch", "batch"]
        if scheduler_name == "multi_step" and isinstance(scheduler_params["milestones"][0], float):
            n_tot = self.n_epoches if self.scheduler_update_range == "epoch" else self.n_iterations
            scheduler_params["milestones"] = [round(m * n_tot) for m in scheduler_params["milestones"]]
        # print(f"TEST {scheduler_params['milestones']}")  # [30, 60, 80]

        self.scheduler = get_scheduler(scheduler_name)(self.optimizer, **scheduler_params)
        self.cur_lr = -1
        self.print_and_log_info("Using scheduler {} with parameters {}".format(scheduler_name, scheduler_params))

        # Loss
        # TODO: Currently this part of code is not applicable for all loss functions
        loss_name = cfg["training"]["loss"]["name"]
        loss_weight = cfg["training"]["loss"]["weight"]

        if loss_weight is not None:
            assert len(loss_weight) == self.n_classes, f"Length between number of classes: {self.n_classes} and class " \
                                                       f"weight: {len(loss_weight)} miss match !!!"
        self.criterion = get_loss(loss_name)(weight=torch.tensor(loss_weight).to(self.device))
        self.print_and_log_info("Using loss {} with class weight {}".format(self.criterion, loss_weight))

        # Pretrained / Resume
        # Determine whether we are in the Resume mode or not
        checkpoint_path = cfg["training"].get("pretrained")
        checkpoint_path_resume = cfg["training"].get("resume")

        # checkpoint_path and checkpoint_path_resume can't be not None at same time !
        assert not (checkpoint_path is not None and checkpoint_path_resume is not None)

        # Each load_from_tag change the self.start_epoch and self.start_batch value
        if checkpoint_path is not None:
            self.load_from_tag(checkpoint_path)
        elif checkpoint_path_resume is not None:
            self.load_from_tag(checkpoint_path_resume, resume=True)
        else:
            self.start_epoch, self.start_batch = 1, 1

        # Train metrics
        train_iter_interval = cfg["training"].get("train_stat_interval", self.n_epoches * self.n_batches // 200)
        self.train_stat_interval = train_iter_interval
        self.train_time = AverageMeter()
        self.train_loss = AverageMeter()

        self.train_metric_path = self.run_dir / TRAIN_METRICS_FILE
        with open(self.train_metric_path, mode='w') as f:
            f.write("iteration\tepoch\tbatch\ttrain_loss\ttrain_time_per_img\n")

        # Val metrics, currently not available
        val_iter_interval = cfg["training"].get("val_stat_interval", self.n_epoches * self.n_batches // 100)
        self.val_stat_interval = val_iter_interval
        self.val_loss = AverageMeter()
        self.val_metrics = RunningMetrics(self.restricted_labels)
        self.val_current_score = None
        self.val_metrics_path = self.run_dir / VAL_METRICS_FILE
        with open(self.val_metrics_path, mode="w") as f:
            f.write("iteration\tepoch\tbatch\tval_loss\t" + "\t".join(self.val_metrics.names) + "\n")

    @property
    def score_name(self):
        return self.val_metrics.score_name

    # TODO: Currently this function is not used
    def print_memory_usage(self, prefix):
        usage = {}
        for attr in ["memory_allocated", "max_memory_allocated", "memory_cached", "max_memory_cached"]:
            usage[attr] = getattr(torch.cuda, attr)() * 0.000001
        self.print_and_log_info("{}:\t{}".format(
            prefix, " / ".join(["{}: {:.0f}MiB".format(k, v) for k, v in usage.items()])))

    def print_and_log_info(self, string):
        """This function handler writing Terminal Log and File Log"""
        print_info(string)
        self.logger.info(string)

    def run(self):
        """Code part for train model"""
        self.model.train()
        cur_iter = (self.start_epoch - 1) * self.n_batches + self.start_batch - 1
        prev_train_stat_iter, prev_val_stat_iter = cur_iter, cur_iter

        for epoch in range(self.start_epoch, self.n_epoches + 1):
            batch_start = self.start_batch if epoch == self.start_epoch else 1

            for batch, (images, labels) in enumerate(self.train_loader, start=1):
                if batch < batch_start:
                    continue
                cur_iter += 1
                if cur_iter > self.n_iterations:
                    break
                if self.scheduler_update_range == "batch":
                    self.update_scheduler(epoch, batch=batch)

                self.single_train_batch_run(images, labels)

                # LOGING FOR THE TRAIN METRIC (Include Train Loss and Train Time for each image)
                if (cur_iter - prev_train_stat_iter) >= self.train_stat_interval:
                    prev_train_stat_iter = cur_iter
                    # log_train_metrics is reset the loss, and time each train_stat interval
                    self.log_train_metrics(cur_iter, epoch, batch)

                # LOGING FOR THE VALIDATION METRIC (Include Validation Loss and the Validation Metrics)
                if (cur_iter - prev_val_stat_iter) >= self.val_stat_interval:
                    prev_val_stat_iter = cur_iter
                    self.run_val()
                    # Using pytorchtools to customize the saving process. Use Early stopping !!!
                    # After self.run_val(), self.val_loss contains the data we need
                    self.log_val_metric(cur_iter, epoch, batch)

            # UserWarning: Detected call of `lr_scheduler.step()` before `optimizer.step()`. In PyTorch 1.1.0 and
            # later, you should call them in the opposite order: `optimizer.step()` before `lr_scheduler.step()`.
            # Failure to do this will result in PyTorch skipping the first value of the learning rate schedule. See
            # more details at https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate
            # "https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate", UserWarning)

            if self.scheduler_update_range == "epoch":
                # When start with new epoch, call the update_scheduler to check whether the scheduler update the weight
                if batch_start == 1:
                    self.update_scheduler(epoch, batch=batch_start)

        self.print_and_log_info("Training run is over !")

    # Function for the validation process during training
    def run_val(self):
        self.model.eval()
        with torch.no_grad():
            for images, labels in self.val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                pred = outputs.data.max(1)[1].cpu().numpy()
                if images.size() == labels.size():
                    gt = labels.data.max(1)[1].cpu().numpy()
                else:
                    gt = labels.cpu().numpy()

                self.val_metrics.update(gt, pred)
                self.val_loss.update(loss.clone().item())
        self.model.train()

    def update_scheduler(self, epoch, batch):
        self.scheduler.step()
        # UserWarning: To get the last learning rate computed by the scheduler, please use `get_last_lr()`.
        #   "please use `get_last_lr()`.", UserWarning)
        lr = self.scheduler.get_last_lr()
        if lr != self.cur_lr:
            self.cur_lr = lr
            msg = PRINT_LR_UPD_FMT.format(epoch, self.n_epoches, batch, self.n_batches, lr)
            self.print_and_log_info(msg)

    def single_train_batch_run(self, images, labels):
        start_time = time.time()

        # Weight for each batch seem to slow down training process
        images, labels = images.to(self.device), labels.to(self.device)

        self.optimizer.zero_grad()
        loss = self.criterion(self.model(images), labels)
        loss.backward()
        self.optimizer.step()

        self.train_loss.update(loss.item())
        self.train_time.update((time.time() - start_time) / self.batch_size)

    def log_train_metrics(self, cur_iter, epoch, batch):
        stat = PRINT_TRAIN_STAT_FMT.format(epoch, self.n_epoches, batch, self.n_batches,
                                           self.train_loss.avg, self.train_time.avg)
        self.print_and_log_info(stat)

        with open(self.train_metric_path, mode="a") as f:
            f.write("{}\t{}\t{}\t{:.4f}\t{:.4f}\n"
                    .format(cur_iter, epoch, batch, self.train_loss.avg, self.train_time.avg))

        self.train_loss.reset()
        self.train_time.reset()

    def log_val_metric(self, cur_iter, epoch, batch):
        stat = PRINT_VAL_STAT_FMT.format(epoch, self.n_epoches, batch, self.n_batches, self.val_loss.avg)
        self.print_and_log_info(stat)

        metrics = self.val_metrics.get()
        self.print_and_log_info("Val metrics: " + ", ".join(["{} = {:.4f}".format(k, v) for k, v in metrics.items()]))

        with open(self.val_metrics_path, mode="a") as f:
            f.write("{}\t{}\t{}\t{:.4f}\t".format(cur_iter, epoch, batch, self.val_loss.avg) +
                    "\t".join(map("{:.4f}".format, metrics.values())) + "\n")

        # Due to the @property declare above, use self.score_name instead of self.val_metrics.score_name
        self.val_current_score = metrics[self.score_name]

        # Use EarlyStopping class to handle whether to save the model or not
        self.early_stopping(val_loss=self.val_loss.avg, state_dict=self.prepare_state_dict(epoch, batch))
        if self.early_stopping.early_stop:
            print("Early stopping")
            raise SystemExit

        self.val_loss.reset()
        self.val_metrics.reset()

    # TODO: Combine this save function with the pytorchtools EarlyStopping class
    def prepare_state_dict(self, epoch, batch):
        state = {
            "epoch": epoch,
            "batch": batch,
            "model_name": self.model_name,
            "model_kwargs": self.model_kwargs,
            "model_state": self.model.state_dict(),
            "n_classes": self.n_classes,
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "score": self.val_current_score,
            "train_resolution": self.dataset_kwargs["img_size"],
            "restricted_labels": self.dataset_kwargs["restricted_labels"],
        }
        return state

    # Load from tag
    def load_from_tag(self, tag, resume=False):
        self.print_and_log_info("Loading model from run {}".format(tag))
        path = coerce_to_path_and_check_exist(MODELS_PATH / tag / MODEL_FILE)
        checkpoint = torch.load(path, map_location=self.device)
        try:
            self.model.load_state_dict(checkpoint["model_state"])
        except RuntimeError:
            state = safe_model_state_dict(checkpoint["model_state"])
            self.model.module.load_state_dict(state)

        self.start_epoch, self.start_batch = 1, 1

        if resume:
            self.start_epoch, self.start_batch = checkpoint["epoch"], checkpoint.get("batch", 0) + 1
            self.optimizer.load_state_dict(checkpoint["optimizer_state"])
            self.scheduler.load_state_dict(checkpoint["scheduler_state"])
            self.cur_lr = self.scheduler.get_last_lr()
        self.print_and_log_info("Checkpoint loaded at epoch {}, batch {}".format(self.start_epoch, self.start_batch))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Pipeline to train a Neural Network model specified by a YML config")
    parser.add_argument("-t", "--tag", nargs="?", type=str, help="Model tag of the experiment, this will create a "
                                                                 "folder models/TAG", required=True)
    parser.add_argument("-c", "--config", nargs="?", type=str, help="Config file name", required=True)
    args = parser.parse_args()

    config = coerce_to_path_and_check_exist(CONFIGS_PATH / args.config)
    run_dir = MODELS_PATH / args.tag

    trainer = Trainer(config, run_dir=run_dir)
    trainer.run()
