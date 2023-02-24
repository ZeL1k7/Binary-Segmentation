import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torch
from utils import SegmentDataset, visualize_masks


class SegmentModule(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self._config = config
        self.model = MODELS[config["model_type"]](**config["model_params"])
        self.criterion = CRITERIONS[config["criterion_type"]]

    def train_dataset(self):
        params = self._config["dataset_train_params"]
        return SegmentDataset(split="train", **params)

    def val_dataset(self):
        params = self._config["dataset_val_params"]
        return SegmentDataset(split="val", **params)

    def test_dataset(self):
        params = self._config["dataset_test_params"]
        return SegmentDataset(split="test", **params)

    def train_dataloader(self):
        dataset = self.train_dataset()
        params = self._config["dataloader_params"]
        return torch.utils.data.DataLoader(
            dataset, shuffle=True, num_workers=2, **params
        )

    def val_dataloader(self):
        dataset = self.val_dataset()
        params = self._config["dataloader_params"]
        return torch.utils.data.DataLoader(
            dataset, shuffle=False, num_workers=2, **params
        )

    def test_dataloader(self):
        dataset = self.test_dataset()
        params = self._config["dataloader_params"]
        return torch.utils.data.DataLoader(
            dataset, shuffle=False, num_workers=2, **params
        )

    def training_step(self, batch, batch_idx):
        image, mask = batch
        predicted_heatmaps = self.model(image)
        loss = self.criterion(predicted_heatmaps, mask)
        predicted_masks = (predicted_heatmaps.sigmoid() >= 0.5).long().squeeze(1)
        tp, fp, fn, tn = smp.metrics.get_stats(predicted_masks, mask, mode="binary")
        iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro-imagewise")
        accuracy = smp.metrics.accuracy(tp, fp, fn, tn, reduction="micro")
        self.log("loss", loss)
        self.log("iou", iou)
        self.log("accuracy", accuracy)
        return {"loss": loss, "iou": iou, "accuracy": accuracy}

    def test_step(self, batch, batch_idx):
        image, mask = batch
        predicted_heatmaps = self.model(image)
        loss = self.criterion(predicted_heatmaps, mask)
        predicted_masks = (predicted_heatmaps.sigmoid() >= 0.5).long().squeeze(1)

        for sample in range(image.size(0)):
            visualize_masks(
                predicted_masks[sample].squeeze(0).detach().cpu(),
                mask[sample].squeeze(0).cpu(),
                image[sample].detach().cpu(),
            )

        tp, fp, fn, tn = smp.metrics.get_stats(predicted_masks, mask, mode="binary")
        iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro-imagewise")
        accuracy = smp.metrics.accuracy(tp, fp, fn, tn, reduction="micro")
        self.log("test_loss", loss)
        self.log("test_iou", iou)
        self.log("test_accuracy", accuracy)
        return {"test_loss": loss, "test_iou": iou, "test_accuracy": accuracy}

    def validation_step(self, batch, batch_idx):
        image, mask = batch
        predicted_heatmaps = self.model(image)
        loss = self.criterion(predicted_heatmaps, mask)
        predicted_masks = (predicted_heatmaps.sigmoid() >= 0.5).long().squeeze(1)
        tp, fp, fn, tn = smp.metrics.get_stats(predicted_masks, mask, mode="binary")
        iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro-imagewise")
        accuracy = smp.metrics.accuracy(tp, fp, fn, tn, reduction="micro")
        self.log("val_loss", loss)
        self.log("val_iou", iou)
        self.log("val_accuracy", accuracy)
        return {"val_loss": loss, "val_iou": iou, "val_accuracy": accuracy}

    def configure_optimizers(self):
        params = self._config["optimizer_params"]
        optimizer = OPTIMIZERS[config["optimizer_type"]](
            self.model.parameters(), **params
        )
        return optimizer
