#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Jun 17 18:57:50 2022

@author: Nacriema

Refs:

"""
import argparse
from PIL import Image
import yaml

import torch
from tqdm import tqdm

from src.utils.logger import print_info
from src.utils import coerce_to_path_and_check_exist, coerce_to_path_and_create_dir
from src.datasets import get_dataset
from src.models import load_model_from_path
from src.utils.metrics import RunningMetrics
from src.utils.image import LabeledArray2Image, resize
from src.utils.path import MODELS_PATH, MODEL_FILE


class Tester:
    """Pipeline to test a given trained NN model on the test split of a specific dataset."""

    def __init__(self, output_dir, model_path, dataset_name, dataset_wkargs=None, save_annotations=True):
        print_info("Tester initialized for model {} and dataset {}".format(model_path, dataset_name))

        # Output directory
        self.output_dir = coerce_to_path_and_create_dir(output_dir)
        self.save_annotations = save_annotations
        print_info("Output dir is {}".format(self.output_dir))

        # Dataset
        self.dataset_kwargs = dataset_wkargs or {}
        self.dataset = get_dataset(dataset_name)(split="test", **self.dataset_kwargs)
        print_info("Dataset {} loaded with kwargs {}: {} samples".format(dataset_name,
                                                                         self.dataset_kwargs, len(self.dataset)))

        # Model
        torch.backends.cudnn.benchmark = False  # XXX. At interference, input images are usually not fixed !
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = load_model_from_path(model_path, device=self.device)
        self.model.eval()  # This line may be redundant
        print_info("Model {} created and checkpoint state loaded".format(self.model.name))

        # Metrics
        if self.dataset.label_files is not None:
            # TODO: self.dataset.metric_labels may be not FOUND !!! Instead we just pass the first argument
            # self.metrics = RunningMetrics(self.dataset.restricted_labels, self.dataset.metric_labels)
            self.metrics = RunningMetrics(self.dataset.restricted_labels)
            print_info("Labels found, metrics instantiated")
        else:
            self.metrics = None
            print_info("No labels found, performance metrics won't be computed")

        # Outputs
        # TODO: Saving probability maps takes a lot of space, remove comment if needed
        # self.prob_dir = coerce_to_path_and_create_dir(self.output_dir / "prob_map")
        self.prob_maps, self.seg_maps = [], []
        if self.save_annotations:
            self.seg_dir = coerce_to_path_and_create_dir(self.output_dir / "seg_map")
            self.blend_dir = coerce_to_path_and_create_dir(self.output_dir / "blend")

    def run(self):
        for i, (image, label) in enumerate(tqdm(self.dataset)):
            self.single_run(image, label)
        print_info("Probabilities and segmentation maps computed")

        if self.metrics is not None:
            self.save_metrics()

        if self.save_annotations:
            self.save_prob_and_seg_maps()

        print_info("Run is over")

    @torch.no_grad()
    def single_run(self, image, label=None):
        image = image.to(self.device)
        prob = self.model(image.reshape(1, *image.shape))[0]
        pred = prob.max(0)[1].cpu().numpy()
        self.prob_maps.append(prob.cpu().numpy())
        self.seg_maps.append(pred)

        if label is not None:
            gt = label.cpu().numpy()
            self.metrics.update(gt, pred)

    def save_metrics(self):
        with open(self.output_dir / "test_metrics.tsv", mode="w") as f:
            f.write("\t".join(self.metrics.names) + "\n")
            f.write("\t".join(map("{:.4f}".format, self.metrics.get().values())) + "\n")

        print_info("Metrics saved")

    def save_prob_and_seg_maps(self):
        for k in range(len(self.dataset)):
            name = self.dataset.input_files[k].stem  # The final file name without suffix
            # Saving probability maps take a lot of space, remove component if needed
            # np.save(self.prob_dir / "{}.npy".format(name), self.prob_maps[k])
            pred = self.seg_maps[k]
            pred_img = LabeledArray2Image.convert(pred, label_color_mapping=self.dataset.label_idx_color_mapping)
            pred_img.save(self.seg_dir / "{}.png".format(name))

            img = resize(Image.open(self.dataset.input_files[k]).convert("RGB"), pred_img.size, keep_aspect_ratio=True)
            blend_img = Image.blend(img, pred_img, alpha=0.4)
            blend_img.convert("RGB").save(self.blend_dir / "{}.jpg".format(name))

        print_info("Probabilities and segmentation map saved")


if __name__ == '__main__':
    # run_dir = coerce_to_path_and_check_exist(MODELS_PATH / "ABC")
    run_dir = coerce_to_path_and_check_exist(MODELS_PATH / "FirstModel")
    # output_dir = run_dir / "test_{}".format("TTPLA_Test")
    output_dir = run_dir / "test_{}".format("RealData")
    config_path = list(run_dir.glob("*.yml"))[0]
    with open(config_path) as fp:
        cfg = yaml.load(fp, Loader=yaml.FullLoader)
    dataset_kwargs = cfg["dataset"]
    dataset_kwargs.pop("name")

    # tester = Tester(output_dir, run_dir / MODEL_FILE, "TTPLA_Test", dataset_kwargs)
    tester = Tester(output_dir, run_dir / MODEL_FILE, "RealData", dataset_kwargs)
    tester.run()
