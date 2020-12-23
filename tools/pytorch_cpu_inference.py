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
from core.inference import get_max_preds
from utils.vis import save_debug_images


def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
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


def main():
    args = parse_args()
    update_config(cfg, args)

    logger, final_output_dir, tb_log_dir = create_logger(
        cfg, args.cfg, 'valid')

    model = eval('models.'+cfg.MODEL.NAME+'.get_pose_net')(
        cfg, is_train=False
    )

    logger.info('=> loading model from {}'.format(cfg.TEST.MODEL_FILE))
    model.load_state_dict(torch.load(cfg.TEST.MODEL_FILE, map_location=torch.device('cpu')), strict=False)

    # Data loading code
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    valid_dataset = eval('dataset.'+cfg.DATASET.DATASET)(
        cfg, cfg.DATASET.ROOT, cfg.DATASET.TEST_SET, False,
        transforms.Compose([
            transforms.ToTensor(),
            normalize,
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

    # switch to evaluate mode
    model.eval()

    if args.convert_onnx:
        x_tensor = torch.rand(1, 3, 256, 192)
        torch.onnx.export(model.cpu(), x_tensor.cpu(), 'model.onnx', export_params=True,
                          operator_export_type=torch.onnx.OperatorExportTypes.ONNX,
                          opset_version=9,
                          verbose=False)
        logger.info('Model is converted to ONNX')

    with torch.no_grad():
        for i, (input, target, target_weight, meta) in enumerate(valid_loader):
            start_time = time.time()
            # compute output
            output = model(input)

            batch_heatmaps = output.clone().cpu().numpy()
            coords, maxvals = get_max_preds(batch_heatmaps)

            # measure elapsed time
            process_time.update(time.time() - start_time)

            prefix = '{}_{}'.format(
                os.path.join(final_output_dir, 'val'), i
            )
            save_debug_images(cfg, input, meta, target, coords * 4, output,
                              prefix)

            if i == 100:
                break

        logger.info(f'PyTorch: Inference EngineAverage processing time of model:{process_time.avg}')


if __name__ == '__main__':
    main()
