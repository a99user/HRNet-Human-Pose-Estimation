from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os

import torch
import torchvision.transforms as transforms

import _init_paths
from config import cfg
from config import update_config
from core.function import AverageMeter
from utils.utils import create_logger

import dataset
import models

import numpy as np
import time
from utils.vis import save_debug_images

from openvino.inference_engine import IECore


def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('-m', '--model', type=str, default='model.xml',
                        help='Path to IR model')
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)

    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    parser.add_argument('--modelDir',
                        help='model directory',
                        type=str,
                        default='')
    parser.add_argument('--logDir',
                        help='log directory',
                        type=str,
                        default='')
    parser.add_argument('--dataDir',
                        help='data directory',
                        type=str,
                        default='')
    parser.add_argument('--prevModelDir',
                        help='prev Model directory',
                        type=str,
                        default='')
    parser.add_argument('--convert_onnx', action="store_true", default=False)

    args = parser.parse_args()
    return args


def load_to_IE(model):
    # Getting the *.bin file location
    model_bin = model[:-3] + "bin"
    # Loading the Inference Engine API
    ie = IECore()

    # Loading IR files
    net = ie.read_network(model=model, weights=model_bin)

    # Loading the network to the inference engine
    exec_net = ie.load_network(network=net, device_name="CPU")
    print("IR successfully loaded into Inference Engine.")

    return exec_net


def sync_inference(exec_net, image):
    input_blob = next(iter(exec_net.inputs))
    return exec_net.infer({input_blob: image})


def main():
    args = parse_args()
    update_config(cfg, args)

    logger, final_output_dir, tb_log_dir = create_logger(
        cfg, args.cfg, 'valid')

    exec_net = load_to_IE(args.model)

    valid_dataset = eval('dataset.'+cfg.DATASET.DATASET)(
        cfg, cfg.DATASET.ROOT, cfg.DATASET.TEST_SET, False,
        transforms.Compose([
            transforms.ToTensor()
        ])
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=cfg.WORKERS,
        pin_memory=True
    )

    process_time = AverageMeter()

    with torch.no_grad():
        for i, (input, target, target_weight, meta) in enumerate(valid_loader):
            start_time = time.time()

            # compute output
            output = sync_inference(exec_net, image=np.expand_dims(input[0].numpy(), 0))

            # measure elapsed time
            process_time.update(time.time() - start_time)

            batch_heatmaps = output['Conv_746']
            coords = output['Mul_772']

            prefix = '{}_{}'.format(
                os.path.join(final_output_dir, 'val'), i
            )
            save_debug_images(cfg, input, meta, target, coords * 4, torch.from_numpy(batch_heatmaps),
                              prefix)

            if i == 100:
                break

        logger.info(f'OpenVINO IE: Average processing time of model with merged pre- and post-processing:{process_time.avg}')


if __name__ == '__main__':
    main()
