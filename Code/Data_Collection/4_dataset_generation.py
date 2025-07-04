import pandas as pd
from pathlib import Path
import shutil
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from PIL import Image
import logging
from typing import Tuple
import random

# Configure logging
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# Configuration settings
config = {
    "input_csv": Path(' '),
    "image_source_dir": Path(' '),
    "dataset_dir": Path(' '),
    "split_ratios": {
        "train": 0.8,
        "val": 0.1,
        "test": 0.1
    },
    "flow_rate_column": "flow_rate",
    "crop_params": {
        "center_x": 610,
        "center_y": 220,
        "crop_size": 224
    },
    "image_format": "jpeg",
    "augmentation_probability": 0.5  # 50% chance of horizontal flip augmentation
}


def load_data(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    df = pd.read_csv(csv_path)
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df


def split_data(df: pd.DataFrame, config: dict) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_size = config['split_ratios']['train']
    val_size = config['split_ratios']['val']
    test_size = config['split_ratios']['test']

    train_df, temp_df = train_test_split(
        df,
        test_size=(1 - train_size),
        shuffle=True,
        stratify=df[config['flow_rate_column']]
    )

    val_ratio = val_size / (val_size + test_size)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=(1 - val_ratio),
        shuffle=True,
        stratify=temp_df[config['flow_rate_column']]
    )

    return train_df, val_df, test_df


def create_dataset_dirs(dataset_dir: Path, flow_rates) -> None:
    subsets = ['train', 'val', 'test']
    for subset in subsets:
        for flow_rate in flow_rates:
            (dataset_dir / subset / str(flow_rate)).mkdir(parents=True, exist_ok=True)


def crop_image(img: Image.Image, center_x: int, center_y: int, crop_size: int) -> Image.Image:
    half_size = crop_size // 2
    left = center_x - half_size
    top = center_y - half_size
    right = center_x + half_size
    bottom = center_y + half_size
    return img.crop((left, top, right, bottom))


def copy_and_crop_images(df: pd.DataFrame, subset: str, config: dict) -> None:
    image_source_dir = config['image_source_dir']
    dataset_dir = config['dataset_dir']
    flow_rate_column = config['flow_rate_column']
    crop_params = config['crop_params']
    image_format = config['image_format']
    augmentation_probability = config['augmentation_probability']

    for _, row in tqdm(df.iterrows(), total=df.shape[0], desc=f"Processing {subset} images", unit="img"):
        img_relative_path = Path(row['img_path'])
        flow_rate = row[flow_rate_column]

        src = image_source_dir / img_relative_path
        dst_dir = dataset_dir / subset / str(flow_rate)
        dst = dst_dir / img_relative_path.name

        if src.exists():
            try:
                with Image.open(src) as img:
                    img = img.convert("RGB")
                    img_cropped = crop_image(
                        img,
                        crop_params['center_x'],
                        crop_params['center_y'],
                        crop_params['crop_size']
                    )

                    # Data augmentation: random horizontal flip
                    if random.random() < augmentation_probability:
                        img_cropped = img_cropped.transpose(Image.FLIP_LEFT_RIGHT)

                    img_cropped.save(dst, format=image_format)
            except Exception as e:
                logger.error(f"Error processing image {src}: {e}")
        else:
            logger.warning(f"Warning: Source image {src} not found, skipping.")


def main(config: dict) -> None:
    df = load_data(config['input_csv'])
    train_df, val_df, test_df = split_data(df, config)
    flow_rates = df[config['flow_rate_column']].unique()
    create_dataset_dirs(config['dataset_dir'], flow_rates)
    copy_and_crop_images(train_df, 'train', config)
    copy_and_crop_images(val_df, 'val', config)
    copy_and_crop_images(test_df, 'test', config)
    logger.info("Image processing completed successfully!")


if __name__ == "__main__":
    main(config)