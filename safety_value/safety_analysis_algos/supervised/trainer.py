"""Supervised safety analysis trainer implementation."""

import os
import time
from typing import Optional, Callable

import numpy as np
import torch
from torch.utils.data import DataLoader
from ..dataset import hj_collate

from ..model import build_mlp
from ..loss import SupervisedRegressionLoss


class SupervisedTrainer:
    """Trainer for supervised safety analysis.
    
    Directly trains the network to predict the sample safety value using
    mean squared error (regression).
    """
    
    def __init__(
            self,
            input_dim: int,
            train_dataset,
            test_dataset=None,
            device: str = "cpu",
            hidden_dims=(256, 256),
            lr: float = 1e-3,
            batch_size: int = 256,
            eval_steps: int = 500,
            eval_batches: int = 0,
            plot_dir: Optional[str] = None,
            train_data_dir: Optional[str] = None,
            collate_fn: Optional[Callable] = hj_collate,
    ):
        self.device = torch.device(device)
        self.model = build_mlp(input_dim=input_dim, hidden_dims=hidden_dims).to(self.device)
        # print model size
        print("MLP structure:\n", self.model)
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.collate_fn = collate_fn or hj_collate
        self.dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=self.collate_fn)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = SupervisedRegressionLoss()

        # training settings
        self.eval_interval = int(eval_steps)
        self.eval_batches = max(0, int(eval_batches))
        self.loss_history = []
        self.loss_epoch_history = []
        self.global_step = 0
        
        # plotting/metrics
        self.plot_dir = plot_dir
        self.train_data_dir = train_data_dir
        self.run_suffix = ""

        # Per-step training metrics
        self.train_steps = []
        self.train_value_mse = []
        self.train_inv_acc = []

        # Test-set evaluation metrics (recorded every eval_steps and at step 0)
        self.eval_steps = []
        self.eval_value_mse = []
        self.eval_inv_acc = []

    def save_checkpoint(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "model_state": self.model.state_dict(),
            "opt_state": self.opt.state_dict(),
        }, path)

    def load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.opt.load_state_dict(ckpt["opt_state"])

    def save_plots(self):
        """Save training plots for MSE and invariant accuracy vs steps."""
        if self.plot_dir is None:
            return
        
        import matplotlib.pyplot as plt
        suffix = self.run_suffix
        os.makedirs(self.plot_dir, exist_ok=True)

        # MSE plot (train steps vs eval steps)
        if len(self.train_steps) > 0:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(self.train_steps, self.train_value_mse, label="Train MSE")
            if len(self.eval_value_mse) > 0:
                ax.plot(self.eval_steps, self.eval_value_mse, label="Eval MSE")
            ax.set_xlabel("Global step", fontsize=12)
            ax.set_ylabel("MSE vs sample_safety_value", fontsize=12)
            ax.set_title("Supervised MSE vs steps", fontsize=14)
            ax.grid(True, linestyle="--", alpha=0.5)
            ax.legend()
            plt.tight_layout()
            mse_plot_path = os.path.join(self.plot_dir, f"mse_vs_steps{suffix}.png")
            plt.savefig(mse_plot_path, dpi=150)
            plt.close()
            print(f"Saved MSE plot to {mse_plot_path}")

        # Invariant accuracy plot (train vs test)
        if len(self.train_steps) > 0:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(self.train_steps, self.train_inv_acc, label="Train Inv Acc")
            if len(self.eval_inv_acc) > 0:
                ax.plot(self.eval_steps, self.eval_inv_acc, label="Eval Inv Acc")
            ax.set_xlabel("Global step", fontsize=12)
            ax.set_ylabel("Invariant sign accuracy", fontsize=12)
            ax.set_title("Supervised invariant accuracy vs steps", fontsize=14)
            ax.grid(True, linestyle="--", alpha=0.5)
            ax.legend()
            plt.tight_layout()
            inv_plot_path = os.path.join(self.plot_dir, f"inv_acc_vs_steps{suffix}.png")
            plt.savefig(inv_plot_path, dpi=150)
            plt.close()
            print(f"Saved invariant accuracy plot to {inv_plot_path}")

        # Plot 3: Training loss vs epochs (averaged per epoch)
        if len(self.loss_history) > 0:
            fig, ax = plt.subplots(figsize=(10, 6))
            epochs = self.loss_epoch_history if len(self.loss_epoch_history) == len(self.loss_history) else list(range(1, len(self.loss_history) + 1))
            ax.plot(epochs, self.loss_history, 'g-', label='Train Loss', linewidth=2)
            ax.set_xlabel('Epoch', fontsize=12)
            ax.set_ylabel('Average Loss', fontsize=12)
            ax.set_title('Training Loss vs Epochs', fontsize=14)
            ax.legend(fontsize=10)
            ax.grid(True, linestyle='--', alpha=0.5)
            plt.tight_layout()
            loss_plot_path = os.path.join(self.plot_dir, f'loss_vs_epochs{suffix}.png')
            plt.savefig(loss_plot_path, dpi=150)
            plt.close()
            print(f"Saved loss plot to {loss_plot_path}")

    def save_training_data(self):
        """Save train/test metrics to CSV files."""
        if self.train_data_dir is None:
            return
        
        import pandas as pd
        os.makedirs(self.train_data_dir, exist_ok=True)
        suffix = self.run_suffix

        df_train = pd.DataFrame({
            "step": self.train_steps,
            "train_value_mse": self.train_value_mse,
            "train_inv_acc": self.train_inv_acc,
        })
        train_csv = os.path.join(self.train_data_dir, f"training_metrics{suffix}.csv")
        df_train.to_csv(train_csv, index=False)
        print(f"Saved train metrics to {train_csv}")

        df_test = pd.DataFrame({
            "step": self.eval_steps,
            "eval_value_mse": self.eval_value_mse,
            "eval_inv_acc": self.eval_inv_acc,
        })
        test_csv = os.path.join(self.train_data_dir, f"evaluation_metrics{suffix}.csv")
        df_test.to_csv(test_csv, index=False)
        print(f"Saved test metrics to {test_csv}")

        # Save loss history separately for clarity
        if len(self.loss_history) > 0:
            loss_epochs = self.loss_epoch_history if len(self.loss_epoch_history) == len(self.loss_history) else list(range(1, len(self.loss_history) + 1))
            loss_df = pd.DataFrame({
                'epoch': loss_epochs,
                'avg_loss': self.loss_history,
            })
            loss_csv_path = os.path.join(self.train_data_dir, f'training_loss{suffix}.csv')
            loss_df.to_csv(loss_csv_path, index=False)
            print(f"Saved training loss data to {loss_csv_path}")

    def train(
        self,
        total_steps: int,
        ckpt_dir: Optional[str] = None,
        log_every: int = 100,
        save_last_only: bool = True,
        run_suffix: str = "",
    ):
        self.run_suffix = run_suffix or ""
        start_time = time.time()

        def _eta_str() -> str:
            if self.global_step <= 0:
                return "ETA: --"
            elapsed = time.time() - start_time
            if elapsed <= 0:
                return "ETA: --"
            remaining = max(total_steps - self.global_step, 0)
            if remaining == 0:
                return "ETA: 00:00:00"
            rate = self.global_step / elapsed
            if rate <= 0:
                return "ETA: --"
            eta_sec = remaining / rate
            hrs = int(eta_sec // 3600)
            mins = int((eta_sec % 3600) // 60)
            secs = int(eta_sec % 60)
            return f"ETA: {hrs:02d}:{mins:02d}:{secs:02d}"
        # evaluate before training starts (test set only)
        print("Evaluating before training...")
        if self.test_dataset is not None:
            test_loader = DataLoader(self.test_dataset, batch_size=self.dl.batch_size, collate_fn=self.collate_fn)
            pre_stats_test = self.evaluate_invariant_sign(test_loader)
            test_mse = pre_stats_test.get("value_mse", float('nan'))
            test_inv_acc = pre_stats_test.get("inv_acc", float('nan'))
            self.eval_steps.append(0)
            self.eval_value_mse.append(test_mse)
            self.eval_inv_acc.append(test_inv_acc)
            print(f"Pre-train eval metrics | mse: {test_mse:.4f} inv_acc: {test_inv_acc:.4f}")

        self.model.train()
        epoch = 0
        while self.global_step < total_steps:
            epoch += 1
            epoch_start_step = self.global_step
            epoch_losses = []
            t0 = time.time()
            print(f"\nEpoch {epoch} (steps {self.global_step}/{total_steps}) started...")
            
            for it, batch in enumerate(self.dl):
                x, x_next, l, sample_safety_value, invariant, _xf, _lf = batch
                x = x.to(self.device)
                sample_safety_value = sample_safety_value.to(self.device)
                invariant = invariant.to(self.device)

                self.opt.zero_grad()
                v_x = self.model(x)
                # Supervised regression: fit v_x to sample_safety_value
                loss = self.criterion.compute_loss(v_x, sample_safety_value)
                loss.backward()
                self.opt.step()
                
                loss_val = loss.item()
                epoch_losses.append(loss_val)

                # Per-step training metrics (current batch)
                with torch.no_grad():
                    v_flat = v_x.view(-1)
                    ssv_flat = sample_safety_value.view(-1)
                    inv_flat = invariant.view(-1)
                    mse_batch = ((v_flat - ssv_flat) ** 2).mean().item()
                    inv_acc_batch = ((v_flat <= 0) == (inv_flat > 0)).float().mean().item()
                    self.train_steps.append(self.global_step)
                    self.train_value_mse.append(mse_batch)
                    self.train_inv_acc.append(inv_acc_batch)

                self.global_step += 1
                
                # Evaluate on test every eval_steps
                if self.global_step % self.eval_interval == 0:
                    self.model.eval()
                    self._eval_and_record(ckpt_dir, save_last_only)
                    self.model.train()

                if self.global_step >= total_steps:
                    break

            avg_loss = float(np.mean(epoch_losses)) if len(epoch_losses) > 0 else float('nan')
            self.loss_history.append(avg_loss)
            self.loss_epoch_history.append(epoch)
            print(
                f"Epoch {epoch} finished in {time.time()-t0:.1f}s | avg_loss={avg_loss:.6f} "
                f"| global_step={self.global_step}/{total_steps} | {_eta_str()}"
            )

            if ckpt_dir is not None and not save_last_only:
                os.makedirs(ckpt_dir, exist_ok=True)
                epoch_ckpt = os.path.join(ckpt_dir, f"supervised_{epoch}{self.run_suffix}.pt")
                self.save_checkpoint(epoch_ckpt)
            
            if self.global_step >= total_steps:
                break
            if epoch_start_step == self.global_step:
                print("No progress made in this epoch; stopping early.")
                break
        
        # Save plots after training completes
        print("\nSaving training plots...")
        self.save_plots()
        
        # Save training data
        print("Saving training metrics data...")
        self.save_training_data()

        # Save final checkpoint
        if ckpt_dir is not None:
            os.makedirs(ckpt_dir, exist_ok=True)
            final_ckpt = os.path.join(ckpt_dir, f"last{self.run_suffix}.pt")
            self.save_checkpoint(final_ckpt)
            print(f"Saved final checkpoint to {final_ckpt}")

    def _eval_and_record(self, ckpt_dir: Optional[str], save_last_only: bool):
        """Evaluate on test set and record metrics keyed by global step."""
        test_mse = float('nan')
        test_inv_acc = float('nan')
        if self.test_dataset is not None:
            test_loader = DataLoader(self.test_dataset, batch_size=self.dl.batch_size, collate_fn=self.collate_fn)
            test_stats = self.evaluate_invariant_sign(test_loader)
            test_mse = test_stats.get("value_mse", float('nan'))
            test_inv_acc = test_stats.get("inv_acc", float('nan'))

            self.eval_steps.append(self.global_step)
            self.eval_value_mse.append(test_mse)
            self.eval_inv_acc.append(test_inv_acc)
        print(
                f"[Step {self.global_step}] eval_value_mse={test_mse:.4f} "
                f"eval_inv_acc={test_inv_acc:.4f}"
        )

        if ckpt_dir is not None and not save_last_only:
            os.makedirs(ckpt_dir, exist_ok=True)
            eval_ckpt = os.path.join(ckpt_dir, f"step_{self.global_step}{self.run_suffix}.pt")
            self.save_checkpoint(eval_ckpt)
            print(f"[Step {self.global_step}] Saved checkpoint to {eval_ckpt}")

    def evaluate_invariant_sign(self, data_loader: DataLoader) -> dict:
        """Evaluate invariant sign accuracy and value MSE vs sample_safety_value."""
        self.model.eval()
        mse_list = []
        inv_acc_list = []
        preds_list = []
        labels_list = []
        processed = 0
        max_samples = self.eval_batches
        with torch.no_grad():
            for batch in data_loader:
                x, _x_next, _l, sample_safety_value, invariant, _xf, _lf = batch
                x = x.to(self.device)
                sample_safety_value = sample_safety_value.to(self.device).view(-1)
                invariant = invariant.to(self.device).view(-1)

                v = self.model(x).view(-1)
                preds = (v <= 0).float()
                labels = (invariant > 0).float()

                preds_list.append(preds.cpu().numpy())
                labels_list.append(labels.cpu().numpy())
                mse_list.append(((v - sample_safety_value) ** 2).mean().item())
                inv_acc_list.append(((preds == labels).float().mean()).item())

                processed += len(invariant)
                if max_samples > 0 and processed >= max_samples:
                    break

        preds = np.concatenate(preds_list, axis=0) if len(preds_list) > 0 else np.array([])
        labels = np.concatenate(labels_list, axis=0) if len(labels_list) > 0 else np.array([])
        value_mse = float(np.mean(mse_list)) if len(mse_list) > 0 else float('nan')
        inv_acc = float(np.mean(inv_acc_list)) if len(inv_acc_list) > 0 else float('nan')
        return {"preds": preds, "labels": labels, "value_mse": value_mse, "inv_acc": inv_acc}
