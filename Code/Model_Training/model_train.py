
import os
import yaml
import math
import timm
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.models as models
import torch
import torch.nn as nn
import pytorch_lightning as pl
from ultralytics import YOLO
from pytorch_lightning.loggers import CSVLogger
from torchvision.models import resnet18, ResNet18_Weights, vgg11, VGG11_Weights, vgg16, VGG16_Weights, densenet121, DenseNet121_Weights, \
    efficientnet_b0, EfficientNet_B0_Weights, regnet_y_400mf, RegNet_Y_400MF_Weights, regnet_x_400mf, \
    RegNet_X_400MF_Weights, regnet_y_800mf, RegNet_Y_800MF_Weights, efficientnet_v2_s, EfficientNet_V2_S_Weights, \
    resnet34, mobilenet_v3_small, MobileNet_V3_Small_Weights
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, TQDMProgressBar
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from yolo_modified import ModifiedYOLO

#  Load configuration file
def load_config(config_path='config/config_train.yaml'):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


#  Custom Dataset class
class CIFAR10RegressionDataset(Dataset):
    def __init__(self, root_dir, transform=None, raw=False):

        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.raw = raw  # Add raw flag

        for class_name in os.listdir(root_dir):
            class_path = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_path):
                print(f"Warning: {class_path} is not a directory, skipping.")
                continue
            try:
                label = float(class_name)
            except ValueError:
                print(f"Warning: Cannot convert folder name '{class_name}' to float label, skipping.")
                continue
            for img_name in os.listdir(class_path):
                img_path = os.path.join(class_path, img_name)
                if not os.path.isfile(img_path):
                    print(f"Warning: {img_path} is not a file, skipping.")
                    continue
                self.image_paths.append(img_path)
                self.labels.append(label)

        print(f"Loaded {len(self.image_paths)} images from {root_dir}.")

        if len(self.image_paths) == 0:
            print(f"Warning: No images found in {root_dir}. Please check data directories and folder names.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")

            image = Image.new('RGB', (224, 224))
            label = 0.0
        else:
            label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

            # Apply normalization based on raw flag
            if self.raw:
                try:
                    # Ensure label is >0 for log calculation
                    if label <= 0:
                        raise ValueError(f"Label value must be >0 for logarithm, but got {label}")
                    #label = math.log(label / 100)
                    label = (label - 100) / 50
                except Exception as e:
                    print(f"Error normalizing label {label} for image {img_path}: {e}")
                    # Set to 0.0 in case of error
                    label = 0.0

        # Convert label to float tensor
        label = torch.tensor(label, dtype=torch.float32)

        return image, label


#  Define PyTorch Lightning model
class BaseRegressionModel(pl.LightningModule):
    def __init__(self, model_architecture, learning_rate=1e-3, scheduler_config=None, model_config=None):
        super(BaseRegressionModel, self).__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.scheduler_config = scheduler_config
        self.model_config = model_config

        if model_architecture == "resnet18":
            self.model = resnet18(weights=ResNet18_Weights.DEFAULT)
            num_features = self.model.fc.in_features
            self.model.fc = nn.Sequential(
                nn.Linear(num_features, 1),

            )
        elif model_architecture == "vgg16":
            self.model = vgg16(weights=VGG16_Weights.DEFAULT)
            num_features = self.model.classifier[-1].in_features
            self.model.classifier[-1] = nn.Sequential(
                nn.Linear(num_features, 1),

            )
        elif model_architecture == "vgg11":
            self.model = vgg11(weights=VGG11_Weights.DEFAULT)
            num_features = self.model.classifier[-1].in_features
            self.model.classifier[-1] = nn.Sequential(
                nn.Linear(num_features, 1),

            )
        elif model_architecture == "densenet121":
            self.model = densenet121(weights=DenseNet121_Weights.DEFAULT)
            num_features = self.model.classifier.in_features
            self.model.classifier = nn.Sequential(
                nn.Linear(num_features, 1),

            )
        elif model_architecture == "regnet_y_400mf":
            self.model = regnet_y_400mf(weights=RegNet_Y_400MF_Weights.DEFAULT)
            num_features = self.model.fc.in_features
            self.model.fc = nn.Sequential(
                nn.Linear(num_features, 1),

            )
        elif model_architecture == "regnet_x_400mf":
            self.model = regnet_x_400mf(weights=RegNet_X_400MF_Weights.DEFAULT)
            num_features = self.model.fc.in_features
            self.model.fc = nn.Sequential(
                nn.Linear(num_features, 1),

            )
        elif model_architecture == "efficientnet_b0":
            self.model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
            num_features = self.model.classifier[1].in_features
            self.model.classifier[1] = nn.Sequential(
                nn.Linear(num_features, 1),

            )
        elif model_architecture == "efficientnet_v2_s":
            self.model = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.DEFAULT)
            num_features = self.model.classifier[1].in_features
            self.model.classifier[1] = nn.Sequential(
                nn.Linear(num_features, 1),

            )
        elif model_architecture == "mobilenet_v3_small":
            self.model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
            num_features = self.model.classifier[-1].in_features
            self.model.classifier[-1] = nn.Linear(num_features, 1)
        elif model_architecture == "yolov11s-cls":

            original_model = YOLO('model/yolo_model/yolo11s-cls.pt')
            original_model.eval()

            self.model = ModifiedYOLO(original_model)
            print("YOLO model successfully adapted for regression.")
        elif model_architecture == "yolov11m-cls":
            original_model = YOLO('model/yolo_model/yolo11m-cls.pt')
            original_model.eval()
            self.model = ModifiedYOLO(original_model)
            print("yolo11m-cls successfully adapted for regression.")
        elif model_architecture == "visformer_tiny":

            self.model = timm.create_model('visformer_tiny', pretrained=True)
            num_features = self.model.head.in_features
            self.model.head = nn.Sequential(
                nn.Linear(num_features, 1)
            )
        elif model_architecture == "visformer_small":
            self.model = timm.create_model('visformer_small', pretrained=True)
            num_features = self.model.head.in_features
            self.model.head = nn.Sequential(
                nn.Linear(num_features, 1)
            )
        else:
            raise ValueError(f"Unsupported model architecture: {model_architecture}")

        # Loss function
        self.criterion = nn.MSELoss()

    def forward(self, x):
        return self.model(x).squeeze()  # Output shape [batch_size]

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        current_lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log('train_lr', current_lr, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_mse', loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log('val_mae', nn.L1Loss()(y_hat, y), on_step=False, on_epoch=True, prog_bar=False)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('test_mse', loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log('test_mae', nn.L1Loss()(y_hat, y), on_step=False, on_epoch=True, prog_bar=False)
        r2 = r2_score(y.cpu(), y_hat.cpu())
        self.log('test_r2', r2, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

        # Add scheduler if configured
        if self.scheduler_config:
            scheduler_type = self.scheduler_config.get('type', None)
            if scheduler_type == "ReduceLROnPlateau":
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    factor=self.scheduler_config.get('factor', 0.1),
                    patience=self.scheduler_config.get('patience_s', 3),
                    min_lr=self.scheduler_config.get('min_lr', 1e-6),
                    mode=self.scheduler_config.get('mode', 'min')
                )
                return {
                    'optimizer': optimizer,
                    'lr_scheduler': {
                        'scheduler': scheduler,
                        'monitor': 'val_loss',
                        'frequency': 1
                    }
                }
            elif scheduler_type == "StepLR":
                scheduler = torch.optim.lr_scheduler.StepLR(
                    optimizer,
                    step_size=self.scheduler_config.get('step_size', 10),
                    gamma=self.scheduler_config.get('gamma', 0.1)
                )
                return {
                    'optimizer': optimizer,
                    'lr_scheduler': scheduler
                }
            elif scheduler_type == "ExponentialLR":
                scheduler = torch.optim.lr_scheduler.ExponentialLR(
                    optimizer,
                    gamma=self.scheduler_config.get('gamma', 0.95)
                )
                return {
                    'optimizer': optimizer,
                    'lr_scheduler': scheduler
                }
            elif scheduler_type == "CosineAnnealingLR":
                T_max = self.scheduler_config.get('T_max', 10)
                eta_min = self.scheduler_config.get('eta_min', 1e-6)
                last_epoch = self.scheduler_config.get('last_epoch', -1)
                print(f"Scheduler Config: T_max={T_max}, eta_min={eta_min}, last_epoch={last_epoch}")
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=T_max,
                    eta_min=eta_min,
                    last_epoch=last_epoch
                )
                return {
                    'optimizer': optimizer,
                    'lr_scheduler': {
                        'scheduler': scheduler,
                        'interval': 'epoch',
                        'frequency': 1,
                        'strict': True
                    }
                }
            else:
                raise ValueError(f"Unsupported scheduler type: {scheduler_type}")
        else:
            return optimizer


#   Define DataModule
class CIFAR10DataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size=32, num_workers=4, raw=False):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.raw = raw  # Add raw flag

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet mean and std
                                 std=[0.229, 0.224, 0.225]),
        ])

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            train_dir = os.path.join(self.data_dir, 'train')
            val_dir = os.path.join(self.data_dir, 'val')

            self.train_dataset = CIFAR10RegressionDataset(train_dir, transform=self.transform, raw=self.raw)
            self.val_dataset = CIFAR10RegressionDataset(val_dir, transform=self.transform, raw=self.raw)
            print(f"Training samples: {len(self.train_dataset)}")
            print(f"Validation samples: {len(self.val_dataset)}")

        if stage == 'test' or stage is None:
            test_dir = os.path.join(self.data_dir, 'test')
            self.test_dataset = CIFAR10RegressionDataset(test_dir, transform=self.transform, raw=self.raw)
            print(f"Test samples: {len(self.test_dataset)}")

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers,
                          persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers,
                          persistent_workers=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers,
                          persistent_workers=True)


def main():

    config = load_config('config/config_train.yaml')

    data_dir = os.path.commonpath([config['data']['train_dir'], config['data']['val_dir'], config['data']['test_dir']])
    print(f"Data directory: {data_dir}")
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory {data_dir} does not exist.")
    else:
        print(f"Data directory {data_dir} exists.")

    raw_flag = config.get('dataset', {}).get('raw', False)
    print(f"Dataset raw flag: {raw_flag}")

    data_module = CIFAR10DataModule(
        data_dir=data_dir,
        batch_size=config['training']['batch_size'],
        num_workers=config['training']['num_workers'],
        raw=raw_flag
    )

    model = BaseRegressionModel(
        model_architecture=config['model']['architecture'],
        learning_rate=config['model']['learning_rate'],
        scheduler_config=config.get('scheduler', None),
        model_config=config.get('model', None)
    )

    logger = CSVLogger(
        save_dir=config['logging']['logger_dir'],
        name=config['logging']['logger_name']
    )

    version = logger.version

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath=os.path.join(logger.save_dir, logger.name, f"version_{version}", 'checkpoints'),
        filename='best-checkpoint',
        save_top_k=1,
        mode='min'
    )

    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=config.get('early_stopping', {}).get('patience_e', 15),
        verbose=True,
        mode='min'
    )

    progress_bar = TQDMProgressBar(refresh_rate=config.get('training', {}).get('log_every_n_steps', 20))

    trainer = pl.Trainer(
        max_epochs=config['training']['max_epochs'],
        logger=logger,
        callbacks=[checkpoint_callback, early_stop_callback, progress_bar],
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1 if torch.cuda.is_available() else None,
        log_every_n_steps=config.get('training', {}).get('log_every_n_steps', 2),
        # limit_train_batches=0.1,
        # limit_val_batches=0.1
    )

    trainer.fit(model, datamodule=data_module)

    trainer.test(model, datamodule=data_module, ckpt_path='best')


    version_dir = os.path.join(logger.save_dir, logger.name, f"version_{version}")
    os.makedirs(version_dir, exist_ok=True)
    best_model_path = os.path.join(version_dir, "best_model.pt")
    torch.save(model.state_dict(), best_model_path)
    print(f"Model saved to {best_model_path}")


if __name__ == '__main__':
    main()