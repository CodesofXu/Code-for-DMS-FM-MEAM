# evaluate.py

import os
import yaml
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import matplotlib  # Add this line
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pytorch_lightning as pl
from tqdm import tqdm  # Import tqdm library

# Import custom modules
from main_v3 import BaseRegressionModel, CIFAR10RegressionDataset, load_config




def load_config_evaluate(config_path: str) -> dict:

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file {config_path} does not exist. Please verify the path.")

    with open(config_path, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
    return config


def load_model_from_checkpoint(checkpoint_path: str, device: torch.device) -> BaseRegressionModel:

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file {checkpoint_path} does not exist. Please verify the path.")

    model = BaseRegressionModel.load_from_checkpoint(checkpoint_path, map_location=device)
    model.to(device)
    model.eval()
    return model


def load_model_from_state_dict(model_architecture: str, state_dict_path: str,
                               device: torch.device) -> BaseRegressionModel:

    if not os.path.exists(state_dict_path):
        raise FileNotFoundError(f"State dict file {state_dict_path} does not exist. Please verify the path.")


    model = BaseRegressionModel(model_architecture=model_architecture)

    state_dict = torch.load(state_dict_path, map_location=device)
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model


def get_test_loader(data_dir: str, batch_size: int, num_workers: int,
                    transform: transforms.Compose, raw: bool) -> DataLoader:

    test_dataset = CIFAR10RegressionDataset(root_dir=data_dir, transform=transform, raw=raw)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=True
    )
    return test_loader


def evaluate_model(model: BaseRegressionModel, test_loader: DataLoader, device: torch.device) -> (list, list):

    all_preds = []
    all_labels = []

    total_batches = len(test_loader)
    with torch.no_grad():

        for batch in tqdm(test_loader, total=total_batches, desc="Evaluating model"):
            images, labels = batch
            images = images.to(device)  # Move input data to target device
            preds = model(images)
            all_preds.extend(preds.cpu().numpy())  # Move predictions to CPU for processing
            all_labels.extend(labels.cpu().numpy())  # Move labels to CPU for processing

    mse = mean_squared_error(all_labels, all_preds)
    mae = mean_absolute_error(all_labels, all_preds)
    r2 = r2_score(all_labels, all_preds)

    print(f"Test MSE: {mse:.4f}")
    print(f"Test MAE: {mae:.4f}")
    print(f"Test R²: {r2:.4f}")

    return all_labels, all_preds


def plot_predictions(all_labels: list, all_preds: list):

    plt.figure(figsize=(8, 8))
    plt.scatter(all_labels, all_preds, alpha=0.5)
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title("Predicted Values vs Actual Values")
    plt.plot([min(all_labels), max(all_labels)], [min(all_labels), max(all_labels)], 'r--')  # Diagonal line
    plt.grid(True)
    plt.show()


def main():

    config_path = " "


    config = load_config_evaluate(config_path)


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model_type = config['model']['type']
    model_path = config['model']['path']
    model_architecture = config['model'].get('architecture', 'resnet18')  # Default to resnet18

    test_data_dir = config['test']['data_dir']
    batch_size = config['test']['batch_size']
    num_workers = config['test']['num_workers']

    resize = tuple(config['transform']['resize'])
    normalize_mean = config['transform']['normalize']['mean']
    normalize_std = config['transform']['normalize']['std']

    transform = transforms.Compose([
        transforms.Resize(resize),
        transforms.ToTensor(),
        transforms.Normalize(mean=normalize_mean, std=normalize_std),
    ])

    if model_type == 'checkpoint':
        model = load_model_from_checkpoint(model_path, device)
    elif model_type == 'state_dict':
        model = load_model_from_state_dict(model_architecture, model_path, device)
    else:
        raise ValueError("Invalid model_type in config. Choose either 'checkpoint' or 'state_dict'.")

    raw_flag = config.get('dataset', {}).get('raw', False)
    print(f"Dataset raw flag: {raw_flag}")

    test_loader = get_test_loader(test_data_dir, batch_size, num_workers, transform, raw_flag)

    print(f"Loaded {len(test_loader.dataset)} images from {test_data_dir}.")

    all_labels, all_preds = evaluate_model(model, test_loader, device)

    plot_predictions(all_labels, all_preds)


if __name__ == '__main__':
    main()