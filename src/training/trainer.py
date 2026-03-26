import json
import time
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


def _safe_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
	if len(np.unique(y_true)) < 2:
		return 0.0
	return float(roc_auc_score(y_true, y_prob))


class Trainer:
	def __init__(
		self,
		model,
		train_loader,
		val_loader,
		loss_fn,
		optimizer,
		scheduler,
		config,
		device,
		output_dir: str,
		fold_name: str,
	):
		self.model = model
		self.train_loader = train_loader
		self.val_loader = val_loader
		self.loss_fn = loss_fn
		self.optimizer = optimizer
		self.scheduler = scheduler
		self.config = config
		self.device = device
		self.fold_name = str(fold_name)
		self.output_dir = Path(output_dir)
		self.output_dir.mkdir(parents=True, exist_ok=True)

		self.logs_dir = self.output_dir / "logs" / self.fold_name
		self.logs_dir.mkdir(parents=True, exist_ok=True)
		self.writer = SummaryWriter(log_dir=str(self.logs_dir))

		self.best_val_f1 = -1.0
		self.best_epoch = -1
		self.history = []

		self.ckpt_dir = self.output_dir / "checkpoints"
		self.ckpt_dir.mkdir(parents=True, exist_ok=True)
		self.best_ckpt_path = self.ckpt_dir / f"{self.fold_name}_best.pt"

	def _epoch_pass(self, loader, train: bool) -> Dict[str, float]:
		if train:
			self.model.train()
		else:
			self.model.eval()

		losses = []
		y_true_all = []
		y_prob_all = []

		bar = tqdm(loader, desc="train" if train else "val", leave=False)
		for batch in bar:
			x = batch["spectrogram"].to(self.device)
			y = batch["label"].float().to(self.device)

			if train:
				self.optimizer.zero_grad(set_to_none=True)

			with torch.set_grad_enabled(train):
				out = self.model(x)
				loss_dict = self.loss_fn(out, y)
				loss = loss_dict["total_loss"]
				if train:
					loss.backward()
					torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
					self.optimizer.step()

			losses.append(float(loss.detach().cpu()))
			probs = torch.sigmoid(out["logits"]).detach().cpu().numpy()
			y_prob_all.extend(probs.tolist())
			y_true_all.extend(y.detach().cpu().numpy().tolist())

		y_true = np.asarray(y_true_all, dtype=np.int32)
		y_prob = np.asarray(y_prob_all, dtype=np.float32)
		y_pred = (y_prob >= 0.5).astype(np.int32)

		metrics = {
			"loss": float(np.mean(losses)) if losses else 0.0,
			"f1": float(f1_score(y_true, y_pred, zero_division=0)) if len(y_true) else 0.0,
			"precision": float(precision_score(y_true, y_pred, zero_division=0)) if len(y_true) else 0.0,
			"recall": float(recall_score(y_true, y_pred, zero_division=0)) if len(y_true) else 0.0,
			"auc_roc": _safe_auc(y_true, y_prob) if len(y_true) else 0.0,
		}
		return metrics

	def train_epoch(self) -> Dict[str, float]:
		return self._epoch_pass(self.train_loader, train=True)

	def validate(self) -> Dict[str, float]:
		with torch.no_grad():
			return self._epoch_pass(self.val_loader, train=False)

	def save_checkpoint(self, path: str, epoch: int, metrics: Dict[str, float]) -> None:
		payload = {
			"model_state_dict": self.model.state_dict(),
			"optimizer_state_dict": self.optimizer.state_dict(),
			"epoch": int(epoch),
			"metrics": metrics,
		}
		torch.save(payload, path)

	def load_checkpoint(self, path: str) -> Dict[str, float]:
		payload = torch.load(path, map_location=self.device)
		self.model.load_state_dict(payload["model_state_dict"])
		self.optimizer.load_state_dict(payload["optimizer_state_dict"])
		return payload.get("metrics", {})

	def train(self, n_epochs: int) -> Dict[str, object]:
		patience = int(getattr(self.config.training, "early_stopping_patience", 10))
		stale = 0
		t0 = time.time()

		for epoch in range(1, int(n_epochs) + 1):
			train_metrics = self.train_epoch()
			val_metrics = self.validate()

			if self.scheduler is not None:
				if hasattr(self.scheduler, "step"):
					if self.scheduler.__class__.__name__.lower().startswith("reducelronplateau"):
						self.scheduler.step(val_metrics["loss"])
					else:
						self.scheduler.step()

			lr = float(self.optimizer.param_groups[0]["lr"])
			self.writer.add_scalar("train/loss", train_metrics["loss"], epoch)
			self.writer.add_scalar("train/f1", train_metrics["f1"], epoch)
			self.writer.add_scalar("val/loss", val_metrics["loss"], epoch)
			self.writer.add_scalar("val/f1", val_metrics["f1"], epoch)
			self.writer.add_scalar("val/auc_roc", val_metrics["auc_roc"], epoch)
			self.writer.add_scalar("train/lr", lr, epoch)

			rec = {"epoch": epoch, "lr": lr, "train": train_metrics, "val": val_metrics}
			self.history.append(rec)

			if val_metrics["f1"] > self.best_val_f1:
				self.best_val_f1 = val_metrics["f1"]
				self.best_epoch = epoch
				stale = 0
				self.save_checkpoint(str(self.best_ckpt_path), epoch, val_metrics)
			else:
				stale += 1

			if stale >= patience:
				break

		best_metrics = {}
		if self.best_ckpt_path.exists():
			best_metrics = self.load_checkpoint(str(self.best_ckpt_path))

		total_time = float(time.time() - t0)
		result = {
			"best_epoch": self.best_epoch,
			"best_val_f1": self.best_val_f1,
			"best_metrics": best_metrics,
			"history": self.history,
			"total_time_s": total_time,
		}

		with (self.output_dir / f"{self.fold_name}_history.json").open("w", encoding="utf-8") as f:
			json.dump(result, f, indent=2)

		self.writer.flush()
		self.writer.close()
		return result
