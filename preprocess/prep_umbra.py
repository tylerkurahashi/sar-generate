import logging
import os
from glob import glob

import tifffile
from omegaconf import OmegaConf


cfg = OmegaConf.load("/workspace/preprocess/prep_config.yaml")

logging.basicConfig(
    filename=cfg.log_path,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


assert not os.path.exists(cfg.output_dir), "output directory already exists."


def sliding_window(cfg, img_path):
    img = tifffile.imread(img_path)
    window_size = cfg.window_size
    step_size = cfg.step_size

    for y in range(0, img.shape[0] - window_size + 1, step_size):
        top = y
        bottom = y + window_size

        if bottom > img.shape[0]:
            top = img.shape[0] + 1 - window_size
            bottom = img.shape[0] + 1

        for x in range(0, img.shape[1] - window_size + 1, step_size):
            left = x
            right = x + window_size

            if right > img.shape[1]:
                left = img.shape[1] + 1 - window_size
                right = img.shape[1] + 1

            window = img[top:bottom, left:right]
            fn = f"{img_path.split('/')[-1].split('_')[-4]}_{top}-{bottom}-{left}-{right}.tif"
            tifffile.imwrite(f"{cfg.output_dir}/{fn}", window)
            logger.info(f"{fn} created and saved.")


os.makedirs(cfg.output_dir, exist_ok=True)
paths = glob(f"{cfg.data_dir}/SkyFi_*.tif")

for path in paths:
    sliding_window(cfg, path)
