# classification_main.py
import os
import yaml
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from torchmetrics import Accuracy
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, TQDMProgressBar, LearningRateMonitor
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
import matplotlib.pyplot as plt
import numpy as np
from torchmetrics import ConfusionMatrix
from sklearn.metrics import confusion_matrix
import seaborn as sns


def load_config(config_path='config/config_classify.yaml'):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config



class ClassificationDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size=32, num_workers=4):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_classes = None

        # Augmentation configuration
        self.train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        self.val_transform = transforms.Compose([

            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def setup(self, stage=None):
        if stage in ('fit', None):
            self.train_ds = ImageFolder(
                os.path.join(self.data_dir, 'train'),
                transform=self.train_transform
            )
            self.val_ds = ImageFolder(
                os.path.join(self.data_dir, 'val'),
                transform=self.val_transform
            )
            self.num_classes = len(self.train_ds.classes)

        if stage == 'test' or stage is None:
            self.test_ds = ImageFolder(
                os.path.join(self.data_dir, 'test'),
                transform=self.val_transform
            )

    def train_dataloader(self):
        return DataLoader(self.train_ds,
                          batch_size=self.batch_size,
                          shuffle=True,
                          num_workers=self.num_workers,
                          persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          persistent_workers=True)

    def test_dataloader(self):
        return DataLoader(self.test_ds,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          persistent_workers=True)



class MobileNetV3Classifier(pl.LightningModule):
    def __init__(self, num_classes, learning_rate=1e-3, pretrained=True, model_config=None):
        super().__init__()
        self.save_hyperparameters()

        self.model = mobilenet_v3_small(weights=None)
        self.model_config = model_config

        self.test_confmat = ConfusionMatrix(task='multiclass', num_classes=num_classes)
        self.all_test_preds = []
        self.all_test_labels = []
        self.class_names = None

        pretrained_model_path = self.model_config.get('pretrained_path')

        if os.path.exists(pretrained_model_path):
            try:

                pretrained_state = torch.load(pretrained_model_path, map_location="cpu")

                adjusted_state = {}
                for k, v in pretrained_state.items():
                    if k.startswith("model."):
                        adjusted_k = k[6:]
                        adjusted_state[adjusted_k] = v
                    else:
                        adjusted_state[k] = v

                self.model.load_state_dict(adjusted_state, strict=False)
                print(f"Successfully loaded pretrained weights: {pretrained_model_path}")

                num_features = self.model.classifier[-1].in_features
                self.model.classifier[-1] = nn.Linear(num_features, 4)

                freeze_backbone = self.model_config.get('freeze_backbone', True)

                if freeze_backbone:

                    for param in self.model.features.parameters():
                        param.requires_grad = False

                    for param in self.model.classifier.parameters():
                        param.requires_grad = True

                    exclude_layers = self.model_config.get('freeze_exclude', [])
                    for name, param in self.model.named_parameters():
                        if any(name.startswith(ex_layer) for ex_layer in exclude_layers):
                            param.requires_grad = True
                            # print(f"Unfrozen layer: {name}")

                    trainable_layers = sorted(list({
                        name.split('.weight')[0].split('.bias')[0]
                        for name, param in self.model.named_parameters()
                        if param.requires_grad
                    }))

                    classifier_layers = [l for l in trainable_layers if 'classifier' in l]
                    features_layers = [l for l in trainable_layers if 'features' in l]

                    output = [
                        "Current trainable layers status:",
                        f"├─ Features layers unfrozen: {len(features_layers)}",
                        f"├─ Classifier status: {'Trainable' if any(classifier_layers) else 'Frozen'}",
                        "└─ Detailed trainable layers:"
                    ]

                    for layer in features_layers + classifier_layers:
                        module_type = ""
                        if 'features' in layer:
                            module_type = "Features"
                        elif 'classifier' in layer:
                            module_type = "Classifier"
                        output.append(f"   - [{module_type}] {layer}")

                    print('\n'.join(output))

            except Exception as e:
                print(f"Error loading pretrained weights: {e}")
                raise

        self.criterion = nn.CrossEntropyLoss()
        self.train_acc = Accuracy(task='multiclass', num_classes=num_classes)
        self.val_acc = Accuracy(task='multiclass', num_classes=num_classes)
        self.test_acc = Accuracy(task='multiclass', num_classes=num_classes)

    def forward(self, x):
        return self.model(x)

    def _shared_step(self, batch):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        return loss, preds, y

    def training_step(self, batch, batch_idx):
        loss, preds, y = self._shared_step(batch)
        self.train_acc(preds, y)
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_acc', self.train_acc, on_step=False, on_epoch=True, prog_bar=True)

        current_lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log('train_lr', current_lr, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        loss, preds, y = self._shared_step(batch)
        self.val_acc(preds, y)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_acc', self.val_acc, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        loss, preds, y = self._shared_step(batch)

        self.all_test_preds.extend(preds.cpu().numpy())
        self.all_test_labels.extend(y.cpu().numpy())

        self.test_acc(preds, y)
        self.log('test_loss', loss, prog_bar=True)
        self.log('test_acc', self.test_acc, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            eps=1e-6

        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=15,
            eta_min=1e-6,
            last_epoch = -1
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                'frequency': 1,
                'strict': True
            }
        }

    def on_test_epoch_end(self):

        if len(self.all_test_preds) > 0 and self.trainer.is_global_zero:
            self._plot_confusion_matrix()

        self.all_test_preds.clear()
        self.all_test_labels.clear()

    def _plot_confusion_matrix(self):

        if self.class_names is None:
            datamodule = self.trainer.datamodule
            self.class_names = datamodule.train_ds.classes

        cm = confusion_matrix(self.all_test_labels, self.all_test_preds)

        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d',
                    xticklabels=self.class_names,
                    yticklabels=self.class_names,
                    cmap='Blues')

        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)

        version_dir = self._get_version_dir()
        save_path = os.path.join(version_dir, "confusion_matrix.png")
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()

        print(f"Confusion matrix saved to: {save_path}")

    def _get_version_dir(self):

        logger = self.trainer.logger
        return os.path.join(
            logger.save_dir,
            logger.name,
            f"version_{logger.version}"
        )



def main():
    config = load_config()

    dm = ClassificationDataModule(
        data_dir=config['data']['root_dir'],
        batch_size=config['training']['batch_size'],
        num_workers=config['training']['num_workers'],
    )
    dm.setup(stage='fit')

    model = MobileNetV3Classifier(
        num_classes=dm.num_classes,
        learning_rate=config['model']['learning_rate'],
        pretrained=config['model']['pretrained'],
        model_config=config.get('model', None)
    )

    logger = CSVLogger(
        save_dir=config['logging']['logger_dir'],
        name=config['logging']['logger_name']
    )
    version = logger.version

    checkpoint_callback = ModelCheckpoint(
        monitor='val_acc',
        dirpath=os.path.join(logger.save_dir, logger.name, f"version_{version}", 'checkpoints'),
        filename='best-{epoch}-{val_acc:.2f}',
        save_top_k=1,
        mode='max'
    )

    callbacks = [
        checkpoint_callback,
        EarlyStopping(monitor='val_acc', patience=10, mode='max'),
        TQDMProgressBar(refresh_rate=20),
        LearningRateMonitor(logging_interval='epoch')
    ]

    trainer = pl.Trainer(
        max_epochs=config['training']['max_epochs'],
        logger=logger,
        callbacks=callbacks,
        accelerator='auto',
        devices=1,
        log_every_n_steps=20
    )

    trainer.fit(model, dm)

    trainer.test(datamodule=dm, ckpt_path='best')

    version_dir = os.path.join(logger.save_dir, logger.name, f"version_{version}")
    os.makedirs(version_dir, exist_ok=True)

    best_model_path = os.path.join(version_dir, "best_model.pt")
    torch.save(model.state_dict(), best_model_path)


    full_model_path = os.path.join(version_dir, "full_model.pth")
    torch.save(model, full_model_path)


if __name__ == '__main__':
    main()