"""Discounted policy evaluation trainer implementation."""

import os
import time
from typing import Optional, Callable
import copy

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from ..model import build_mlp
from ..loss import DiscountedBellmanLoss
from ..dataset import hj_collate

def init_value_head_negative(net: torch.nn.Module, bias: float = -0.5):
    # assuming last layer is Linear
    last = None
    for m in net.modules():
        if isinstance(m, torch.nn.Linear):
            last = m
    assert last is not None
    torch.nn.init.zeros_(last.weight)
    torch.nn.init.constant_(last.bias, bias)

class DiscountedPolicyEvaluationTrainer:
    """Trainer for discounted policy evaluation approach to safety analysis.
    
    Uses temporal difference learning with discount factor (lambda) to estimate
    safety values, similar to policy evaluation in reinforcement learning.
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
            initial_lambda: float = 0.0,
            lambda_increase_ratio: float = 0.1,
            lambda_max: float = 0.99,
            lambda_update_steps: int = 500,
            loss_converge_tol: float = 0.01,
            eval_steps: int = 500,
            eval_batches: int = 0,
            plot_dir: Optional[str] = None,
            train_data_dir: Optional[str] = None,
            use_target_network: bool = True,
            target_update_steps: int = 10,
            target_tau: float = 0.05,
            enable_lambda_annealing: bool = True,
            collate_fn: Optional[Callable] = hj_collate,
    ):
        self.device = torch.device(device)
        self.model = build_mlp(input_dim=input_dim, hidden_dims=hidden_dims).to(self.device)
        init_value_head_negative(self.model, bias=-2.0)
        # print model size
        print("MLP structure:\n", self.model)
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.collate_fn = collate_fn or hj_collate
        self.dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=self.collate_fn)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = DiscountedBellmanLoss(lam=initial_lambda)

        # target network (similar to Q-learning)
        self.use_target_network = use_target_network
        self.target_update_steps = target_update_steps
        self.target_tau = target_tau
        if self.use_target_network:
            self.target_model = build_mlp(input_dim=input_dim, hidden_dims=hidden_dims).to(self.device)
            self.target_model.load_state_dict(self.model.state_dict())
            self.target_model.eval()
            print(f"Using target network with tau={target_tau}, update_steps={target_update_steps}")
        else:
            self.target_model = None

        # lambda scheduling
        self.enable_lambda_annealing = enable_lambda_annealing
        self.lambda_increase_ratio = lambda_increase_ratio
        self.lambda_max = lambda_max
        self.lambda_update_steps = lambda_update_steps
        self.eval_interval = int(eval_steps)
        self.eval_batches = max(0, int(eval_batches))
        self.loss_history = []
        self._recent_losses = []
        self._prev_window_avg_loss = None
        # Convergence threshold for lambda annealing. Interpreted as *relative* change in
        # the average loss between consecutive windows (e.g., 0.01 == 1% change).
        self.loss_converge_tol = float(loss_converge_tol)
        self.global_step = 0
        
        # plotting
        self.plot_dir = plot_dir
        self.train_data_dir = train_data_dir
        # Per-step training metrics
        self.train_steps = []
        self.train_value_mse = []
        self.train_inv_acc = []

        # Evaluation metrics (test set, recorded every eval_interval and at step 0)
        self.eval_steps = []
        self.eval_inv_acc = []
        self.eval_value_mse = []
        self.lambda_history = []

        self.run_suffix = ""

    def _eval_loader(self, dataset, fraction: float, batch_size: int) -> Optional[DataLoader]:
        if dataset is None:
            return None
        total = len(dataset)
        if total == 0:
            return None
        if fraction >= 1.0:
            indices = list(range(total))
        else:
            n = max(1, int(total * fraction))
            g = torch.Generator()
            g.manual_seed(0)
            indices = torch.randperm(total, generator=g)[:n].tolist()
        subset = Subset(dataset, indices)
        return DataLoader(subset, batch_size=batch_size, shuffle=False, collate_fn=self.collate_fn)

    def save_checkpoint(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        ckpt = {
            "model_state": self.model.state_dict(),
            "opt_state": self.opt.state_dict(),
            "criterion_lam": self.criterion.lam,
        }
        if self.use_target_network:
            ckpt["target_model_state"] = self.target_model.state_dict()
        torch.save(ckpt, path)

    def load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.opt.load_state_dict(ckpt["opt_state"])
        self.criterion.set_lambda(ckpt.get("criterion_lam", self.criterion.lam))
        if self.use_target_network and "target_model_state" in ckpt:
            self.target_model.load_state_dict(ckpt["target_model_state"])

    def save_plots(self):
        """Save training plots for accuracy/MSE and lambda vs steps."""
        if self.plot_dir is None:
            return
        
        import matplotlib.pyplot as plt
        suffix = self.run_suffix
        os.makedirs(self.plot_dir, exist_ok=True)

        if len(self.eval_steps) > 0 and len(self.eval_inv_acc) > 0:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(self.eval_steps, self.eval_inv_acc, label="Test Inv Acc", linewidth=2)
            if len(self.train_steps) > 0 and len(self.train_inv_acc) > 0:
                ax.plot(self.train_steps, self.train_inv_acc, label="Train Inv Acc", linewidth=1.5)
            ax.set_xlabel("Step", fontsize=12)
            ax.set_ylabel("Invariant sign accuracy", fontsize=12)
            ax.set_title("Invariant accuracy vs steps", fontsize=14)
            ax.grid(True, linestyle="--", alpha=0.5)
            ax.legend()
            plt.tight_layout()
            acc_plot_path = os.path.join(self.plot_dir, f"accuracy_vs_steps{suffix}.png")
            plt.savefig(acc_plot_path, dpi=150)
            plt.close()
            print(f"Saved accuracy plot to {acc_plot_path}")

        if len(self.eval_steps) > 0 and len(self.eval_value_mse) > 0:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(self.eval_steps, self.eval_value_mse, label="Eval value MSE", linewidth=2)
            if len(self.train_steps) > 0 and len(self.train_value_mse) > 0:
                ax.plot(self.train_steps, self.train_value_mse, label="Train value MSE", linewidth=1.5)
            ax.set_xlabel("Step", fontsize=12)
            ax.set_ylabel("MSE(v, sample_safety_value)", fontsize=12)
            ax.set_title("Value MSE vs steps", fontsize=14)
            ax.grid(True, linestyle="--", alpha=0.5)
            ax.legend()
            plt.tight_layout()
            eval_mse_plot_path = os.path.join(self.plot_dir, f"value_mse_vs_steps{suffix}.png")
            plt.savefig(eval_mse_plot_path, dpi=150)
            plt.close()
            print(f"Saved eval value MSE plot to {eval_mse_plot_path}")

        if len(self.train_steps) > 0:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(self.train_steps, self.train_value_mse, label="Train value MSE", linewidth=2)
            ax.set_xlabel("Step", fontsize=12)
            ax.set_ylabel("MSE(v, sample_safety_value)", fontsize=12)
            ax.set_title("Value MSE vs steps", fontsize=14)
            ax.grid(True, linestyle="--", alpha=0.5)
            ax.legend()
            plt.tight_layout()
            mse_plot_path = os.path.join(self.plot_dir, f"train_value_mse{suffix}.png")
            plt.savefig(mse_plot_path, dpi=150)
            plt.close()
            print(f"Saved value MSE plot to {mse_plot_path}")

        if len(self.eval_steps) > 0 and len(self.lambda_history) > 0:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(self.eval_steps, self.lambda_history, 'g-', linewidth=2)
            ax.set_xlabel('Step', fontsize=12)
            ax.set_ylabel('Lambda (λ)', fontsize=12)
            ax.set_title('Lambda Schedule vs Steps', fontsize=14)
            ax.grid(True, linestyle='--', alpha=0.5)
            plt.tight_layout()
            lambda_plot_path = os.path.join(self.plot_dir, f'lambda_vs_steps{suffix}.png')
            plt.savefig(lambda_plot_path, dpi=150)
            plt.close()
            print(f"Saved lambda plot to {lambda_plot_path}")

    def save_training_data(self):
        """Save training/evaluation metrics to CSV files."""
        if self.train_data_dir is None:
            return
        
        import pandas as pd
        suffix = self.run_suffix
        os.makedirs(self.train_data_dir, exist_ok=True)

        if len(self.train_steps) > 0:
            train_df = pd.DataFrame({
                'step': self.train_steps,
                'train_value_mse': self.train_value_mse,
                'train_inv_acc': self.train_inv_acc,
            })
            train_csv = os.path.join(self.train_data_dir, f'training_metrics{suffix}.csv')
            train_df.to_csv(train_csv, index=False)
            print(f"Saved training data to {train_csv}")

        if len(self.eval_steps) > 0:
            eval_df = pd.DataFrame({
                'step': self.eval_steps,
                'eval_value_mse': self.eval_value_mse,
                'eval_inv_acc': self.eval_inv_acc,
                'lambda': self.lambda_history,
            })
            eval_csv = os.path.join(self.train_data_dir, f'evaluation_metrics{suffix}.csv')
            eval_df.to_csv(eval_csv, index=False)
            print(f"Saved evaluation data to {eval_csv}")

    def _update_target_network(self):
        """Update target network using soft update (Polyak averaging)."""
        if not self.use_target_network:
            return
        
        # Soft update: θ_target = τ * θ + (1 - τ) * θ_target
        for target_param, param in zip(self.target_model.parameters(), self.model.parameters()):
            target_param.data.copy_(
                self.target_tau * param.data + (1.0 - self.target_tau) * target_param.data
            )

    def _maybe_update_lambda(self):
        """Update lambda based on training steps."""
        if not self.enable_lambda_annealing:
            return
        
        if self.global_step % self.lambda_update_steps == 0 and self.global_step > 0:
            # Calculate average loss over recent steps
            if len(self._recent_losses) > 0:
                avg_recent_loss = float(np.mean(self._recent_losses))
                
                # Check if loss has converged by comparing with previous window
                loss_converged = False
                if self._prev_window_avg_loss is not None:
                    loss_change = abs(avg_recent_loss - self._prev_window_avg_loss)
                    denom = max(abs(self._prev_window_avg_loss), 1e-12)
                    rel_change = loss_change / denom
                    loss_converged = rel_change < self.loss_converge_tol
                    print(f"Step {self.global_step}: Avg loss: {avg_recent_loss:.6f}, Prev: {self._prev_window_avg_loss:.6f}, RelChange: {rel_change:.6f}, Converged: {loss_converged}")
                else:
                    print(f"Step {self.global_step}: Avg loss: {avg_recent_loss:.6f} (first window)")
                
                # Only update lambda if loss has converged
                if loss_converged:
                    new_lam = min(self.lambda_max, self.criterion.lam + self.lambda_increase_ratio * (self.lambda_max - self.criterion.lam))
                    if new_lam > self.criterion.lam:
                        print(f"Step {self.global_step}: Loss converged! Increasing lambda from {self.criterion.lam:.4f} -> {new_lam:.4f}")
                        self.criterion.set_lambda(new_lam)
                    elif new_lam >= self.lambda_max:
                        print(f"Step {self.global_step}: Lambda already at maximum ({self.lambda_max:.4f})")
                
                # Update previous window average
                self._prev_window_avg_loss = avg_recent_loss
                self._recent_losses = []

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
        test_acc = float('nan')
        test_value_mse = float('nan')
        if self.test_dataset is not None:
            test_loader = DataLoader(self.test_dataset, batch_size=self.dl.batch_size, collate_fn=self.collate_fn)
            pre_stats_test = self.evaluate_invariant_sign(test_loader)
            test_acc = float(pre_stats_test.get("inv_acc", float('nan')))
            test_value_mse = float(pre_stats_test.get("value_mse", float('nan')))
            print(f"Pre-train invariant sign accuracy (test): {test_acc:.4f}")

        # Record pre-training accuracy (step 0)
        self.eval_steps.append(0)
        self.eval_inv_acc.append(test_acc)
        self.eval_value_mse.append(test_value_mse)
        self.lambda_history.append(self.criterion.lam)

        self.model.train()
        epoch = 0
        while self.global_step < total_steps:
            epoch += 1
            epoch_losses = []
            t0 = time.time()
            print(f"\nEpoch {epoch} (steps {self.global_step}/{total_steps}) started...")

            for it, batch in enumerate(self.dl):
                x, x_next, l, sample_safety_value, invariant, _x_future, _l_future = batch
                x = x.to(self.device)
                x_next = x_next.to(self.device)
                l = l.to(self.device)
                sample_safety_value = sample_safety_value.to(self.device)
                invariant = invariant.to(self.device)

                self.opt.zero_grad()
                v_x = self.model(x)
                
                # Use target network for v_x_next if enabled
                if self.use_target_network:
                    with torch.no_grad():
                        v_x_next = self.target_model(x_next)
                else:
                    v_x_next = self.model(x_next)
                
                loss = self.criterion.compute_loss(v_x, v_x_next, l, invariant)
                loss.backward()
                self.opt.step()
                
                loss_val = loss.item()
                epoch_losses.append(loss_val)
                self._recent_losses.append(loss_val)
                self.global_step += 1

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

                if self.global_step % log_every == 0:
                    print(
                        f"[Step {self.global_step} | Epoch {epoch} Iter {it+1}] "
                        f"loss={loss_val:.4f} mse_v_ssv={mse_batch:.6f} "
                        f"train_inv_acc={inv_acc_batch:.4f} lam={self.criterion.lam:.4f} | {_eta_str()}"
                    )

                # Update target network if enabled
                if self.use_target_network and self.global_step % self.target_update_steps == 0:
                    self._update_target_network()
                    # if self.global_step % (self.target_update_steps * 10) == 0:  # Log every 10 updates
                    #     print(f"[Step {self.global_step}] Updated target network")

                # Update lambda based on steps
                self._maybe_update_lambda()

                # Evaluate on test every eval_interval
                if self.global_step % self.eval_interval == 0:
                    self.model.eval()
                    test_acc = float('nan')
                    test_value_mse = float('nan')
                    if self.test_dataset is not None:
                        test_loader = DataLoader(self.test_dataset, batch_size=self.dl.batch_size, collate_fn=self.collate_fn)
                        eval_stats_test = self.evaluate_invariant_sign(test_loader)
                        test_acc = float(eval_stats_test.get("inv_acc", float('nan')))
                        test_value_mse = float(eval_stats_test.get("value_mse", float('nan')))
                        print(f"[Step {self.global_step}] Test accuracy: {test_acc:.4f} | {_eta_str()}")

                    # Record for plotting
                    self.eval_steps.append(self.global_step)
                    self.eval_inv_acc.append(test_acc)
                    self.eval_value_mse.append(test_value_mse)
                    self.lambda_history.append(self.criterion.lam)

                    # Save checkpoint at every evaluation
                    if ckpt_dir is not None and not save_last_only:
                        os.makedirs(ckpt_dir, exist_ok=True)
                        eval_ckpt = os.path.join(ckpt_dir, f"step_{self.global_step}{self.run_suffix}.pt")
                        self.save_checkpoint(eval_ckpt)

                    self.model.train()

                if self.global_step >= total_steps:
                    break
            
            avg_loss = float(np.mean(epoch_losses)) if len(epoch_losses) > 0 else float('nan')
            self.loss_history.append(avg_loss)
            print(f"Epoch {epoch} finished in {time.time()-t0:.1f}s | avg_loss={avg_loss:.6f} | lam={self.criterion.lam:.4f} | global_step={self.global_step}/{total_steps} | {_eta_str()}")

            if self.global_step >= total_steps:
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

    def evaluate_invariant_sign(self, data_loader: DataLoader) -> dict:
        """Evaluate invariant sign accuracy and value MSE against sample_safety_value."""
        self.model.eval()
        preds_list = []
        labels_list = []
        inv_acc_list = []
        mse_list = []
        processed = 0
        max_samples = self.eval_batches
        with torch.no_grad():
            for batch in data_loader:
                x, x_next, l, sample_safety_value, invariant, _x_future, _l_future = batch
                x = x.to(self.device)
                sample_safety_value = sample_safety_value.to(self.device)
                invariant = invariant.to(self.device)

                v = self.model(x)
                v_flat = v.view(-1)
                ssv_flat = sample_safety_value.view(-1)
                inv_flat = invariant.view(-1)

                preds = (v_flat <= 0).float()
                labels = (inv_flat > 0).float()

                preds_list.append(preds.cpu().numpy())
                labels_list.append(labels.cpu().numpy())
                inv_acc_list.append(((preds == labels).float().mean()).cpu().item())
                mse_list.append(((v_flat - ssv_flat) ** 2).mean().item())

                processed += len(labels)
                if max_samples > 0 and processed >= max_samples:
                    break

        preds = np.concatenate(preds_list, axis=0) if len(preds_list) > 0 else np.array([])
        labels = np.concatenate(labels_list, axis=0) if len(labels_list) > 0 else np.array([])
        inv_acc = float(np.mean(inv_acc_list)) if len(inv_acc_list) > 0 else float('nan')
        value_mse = float(np.mean(mse_list)) if len(mse_list) > 0 else float('nan')

        return {"preds": preds, "labels": labels, "inv_acc": inv_acc, "value_mse": value_mse}
