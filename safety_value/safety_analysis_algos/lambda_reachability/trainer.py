"""Lambda reachability trainer implementation with stochastic absorption.

Implements the batch training procedure with n-step targets and random absorption
(survival gating). Each training step performs:
  1) Sample a batch of trajectories (obs[t:], l[t:]).
  2) For each sample, draw a geometric horizon n_i ~ Geom(1-lambda), truncate by
     available future length (and max_horizon).
  3) Compute max_l_i = max(l_t, ..., l_{t+n_i-1}) over the n-step prefix.
  4) Sample survival indicator S_i ~ Bernoulli(delta^n_i):
       - If S_i = 1 (survive): bootstrap = min(V1^-(x_{t+n_i}), V2^-(x_{t+n_i}))
         (or l_{t+n_i} if terminal).
       - If S_i = 0 (absorbed): bootstrap = v_term (fixed baseline, e.g., -1e6).
     Compute target y_i = max(max_l_i, bootstrap_i).
  5) Train both critics toward y_i with squared loss, plus auxiliaries:
        - Lower-bound hinge: V(x_t) >= l_t
        - Monotonicity hinge: V(x_t) >= V(x_{t+1})
        - BCE on sign of V(x_t) to predict 1[y_i > 0] via sigma(alpha V)
  6) Polyak update target networks.

The survival rate delta controls stochastic discounting via absorption:
  - delta close to 1: almost always survive, bootstrap from learned V.
  - delta close to 0: high absorption probability, bootstrap from v_term.
Weights on all loss terms are exposed for ablations (set any to 0).
"""

import os
import time
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from ..dataset import hj_collate
from ..model import build_mlp

def init_value_head_negative(net: torch.nn.Module, bias: float = -0.5):
    # assuming last layer is Linear
    last = None
    for m in net.modules():
        if isinstance(m, torch.nn.Linear):
            last = m
    assert last is not None
    torch.nn.init.zeros_(last.weight)
    torch.nn.init.constant_(last.bias, bias)

class LambdaReachabilityTrainer:
    """Trainer implementing λ-reachability with two critics, target networks, and stochastic absorption."""

    def __init__(
        self,
        input_dim: int,
        train_dataset,
        test_dataset=None,
        device: str = "cpu",
        hidden_dims: Sequence[int] = (256, 256),
        lr: float = 1e-3,
        batch_size: int = 256,
        lambda_param: float = 0.99,
        max_horizon: int = 250,
        delta: float = 0.99,
        v_term: float = -1e6,
        alpha_bce: float = 5.0,
        weight_main: float = 1.0,
        weight_hinge_lb: float = 0.5,
        weight_hinge_mono: float = 0.5,
        weight_bce: float = 0.5,
        target_tau: float = 0.05,
        target_update_period: int = 10,
        eval_steps: int = 500,
        eval_batches: int = 0,
        plot_dir: Optional[str] = None,
        train_data_dir: Optional[str] = None,
        collate_fn=None,
    ):
        self.device = torch.device(device)
        self.lambda_param = float(lambda_param)
        self.max_horizon = int(max_horizon)
        
        # Stochastic absorption parameters
        if not (0.0 < delta <= 1.0):
            raise ValueError(f"delta must be in (0, 1], got {delta}")
        self.delta = float(delta)
        self.v_term = float(v_term)
        
        self.alpha_bce = float(alpha_bce)
        self.weight_main = float(weight_main)
        self.weight_hinge_lb = float(weight_hinge_lb)
        self.weight_hinge_mono = float(weight_hinge_mono)
        self.weight_bce = float(weight_bce)
        self.target_tau = float(target_tau)
        self.target_update_period = int(target_update_period)
        self.eval_interval = int(eval_steps)
        self.eval_batches = max(0, int(eval_batches))

        # Critics
        self.critic1 = build_mlp(input_dim=input_dim, hidden_dims=hidden_dims).to(self.device)
        self.critic2 = build_mlp(input_dim=input_dim, hidden_dims=hidden_dims).to(self.device)
        init_value_head_negative(self.critic1, bias=-2.0)
        init_value_head_negative(self.critic2, bias=-2.0)
        print("Critic 1:\n", self.critic1)
        print("Critic 2:\n", self.critic2)

        # Target critics
        self.target1 = build_mlp(input_dim=input_dim, hidden_dims=hidden_dims).to(self.device)
        self.target2 = build_mlp(input_dim=input_dim, hidden_dims=hidden_dims).to(self.device)
        self._hard_update_targets()

        # Data
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=hj_collate)

        # Optimizer over both critics
        self.opt = torch.optim.Adam(
            list(self.critic1.parameters()) + list(self.critic2.parameters()), lr=lr
        )

        # Buffers
        self.loss_history = []
        self.global_step = 0

        self.train_steps = []  # shared index for all per-step training metrics
        self.train_value_mse = []
        self.train_inv_acc = []

        self.eval_steps = []
        self.eval_value_mse = []
        self.eval_inv_acc = []
        
        self.plot_dir = plot_dir
        self.train_data_dir = train_data_dir
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
        return DataLoader(subset, batch_size=batch_size, shuffle=False, collate_fn=hj_collate)

    def _hard_update_targets(self):
        self.target1.load_state_dict(self.critic1.state_dict())
        self.target2.load_state_dict(self.critic2.state_dict())
        self.target1.eval()
        self.target2.eval()

    @torch.no_grad()
    def _soft_update_targets(self):
        tau = self.target_tau
        for tgt, src in zip(self.target1.parameters(), self.critic1.parameters()):
            tgt.data.mul_(1.0 - tau).add_(tau * src.data)
        for tgt, src in zip(self.target2.parameters(), self.critic2.parameters()):
            tgt.data.mul_(1.0 - tau).add_(tau * src.data)

    # def _sample_horizon(self, max_len: int) -> int:
    #     """Sample geometric horizon n_i truncated by available length and max_horizon."""
    #     p = 1.0 - self.lambda_param
    #     g = torch.distributions.Geometric(probs=torch.tensor(p)).sample().item()
    #     n = int(g)  # geometric support starts at 1
    #     n = max(1, n)
    #     n = min(n, max_len, self.max_horizon)
    #     return n

    def _sample_horizon(self, max_len: int) -> int:
        """
        Sample n ~ Geometric(1-lambda), truncated to [1, L] and renormalized.

        Returns:
            n in {1, ..., L}
        """
        L = max_len
        if L <= 1:
            return 1

        lam = self.lambda_param

        if lam >= 1.0:
            raise ValueError("lambda_param must be less than 1.0 for geometric distribution.")

        # k = 1..L
        ks = torch.arange(1, L + 1, device=self.device, dtype=torch.float32)

        # Unnormalized probabilities: (1-lam) * lam^(k-1)
        probs = (1.0 - lam) * torch.pow(lam, ks - 1)

        # Renormalize: divide by (1 - lam^L)
        probs = probs / (1.0 - lam ** L)

        # Sample
        n = torch.distributions.Categorical(probs=probs).sample().item()

        # Categorical returns index in [0, L-1], shift to [1, L]
        return int(n + 1)

    def _compute_targets(
        self, batch
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute anchors and targets using per-state future trajectories with stochastic absorption.

        For each sample:
          1) Sample horizon n from truncated geometric distribution.
          2) Compute max_l = max(l_t, ..., l_{t+n-1}) over the n-step prefix.
          3) Compute bootstrap candidate v = min(target1(xn), target2(xn)).
          4) Sample survival indicator S ~ Bernoulli(delta^n):
               - If S = 1: use v (or ln if terminal) as bootstrap.
               - If S = 0: use v_term as bootstrap (absorption).
          5) Set target y = max(max_l, bootstrap).

        Batch is produced by hj_collate and contains:
          x, x_next, l, sample_safety_value, invariant, x_future(list), l_future(list)
        """
        x, x_next, l, sample_safety_value, invariant, x_future, l_future = batch

        x0_list: List[torch.Tensor] = []
        x1_list: List[torch.Tensor] = []
        xn_list: List[torch.Tensor] = []
        l0_list: List[torch.Tensor] = []
        ln_list: List[torch.Tensor] = []
        max_l_list: List[torch.Tensor] = []  # max over horizon prefix
        is_terminal_list: List[float] = []
        mono_mask: List[float] = []  # 1 if successor exists
        ssv_list: List[torch.Tensor] = []
        inv_list: List[torch.Tensor] = []
        survival_prob_list: List[float] = []  # delta^n for each sample

        batch_size = x.shape[0]
        for i in range(batch_size):
            fut_obs = x_future[i]
            fut_l = l_future[i]

            # Ensure there is at least one successor
            if fut_obs is None or len(fut_obs) == 0:
                continue

            fut_obs = fut_obs.to(self.device)
            fut_l = fut_l.to(self.device)

            # Limit horizon by available length and max_horizon
            max_len = min(fut_obs.shape[0], self.max_horizon)
            if max_len <= 0:
                continue

            n = self._sample_horizon(max_len)
            bootstrap_idx = n - 1  # because fut_obs starts at t+1
            is_terminal = (bootstrap_idx == fut_obs.shape[0] - 1)

            l_anchor = l[i].to(self.device).reshape(-1)[0]
            l_prefix = fut_l[:bootstrap_idx].view(-1)
            if l_prefix.numel() == 0:
                max_l = l_anchor
            else:
                max_l = torch.max(torch.cat([l_anchor.view(1), l_prefix], dim=0))

            x0_list.append(x[i].to(self.device))
            x1_list.append(x_next[i].to(self.device))
            l0_list.append(l_anchor)
            max_l_list.append(max_l)
            xn_list.append(fut_obs[bootstrap_idx])
            ln_list.append(fut_l[bootstrap_idx])
            is_terminal_list.append(float(is_terminal))
            mono_mask.append(1.0)
            ssv_list.append(sample_safety_value[i].to(self.device).reshape(-1)[0])
            inv_list.append(invariant[i].to(self.device).reshape(-1)[0])
            survival_prob_list.append(self.delta ** n)

        if len(x0_list) == 0:
            return None, None, None, None, None, None, None

        x0            = torch.stack(x0_list, dim=0)
        x1            = torch.stack(x1_list, dim=0)
        xn            = torch.stack(xn_list, dim=0)
        l0            = torch.stack(l0_list, dim=0).view(-1)
        ln            = torch.stack(ln_list, dim=0).view(-1)
        max_l         = torch.stack(max_l_list, dim=0).view(-1)
        terminal_mask = torch.tensor(is_terminal_list, device=self.device, dtype=torch.float32).view(-1)
        mono_mask_t   = torch.tensor(mono_mask, device=self.device, dtype=torch.float32)
        ssv0          = torch.stack(ssv_list, dim=0).view(-1)
        inv0          = torch.stack(inv_list, dim=0).view(-1)
        survival_probs = torch.tensor(survival_prob_list, device=self.device, dtype=torch.float32).view(-1)

        # Target network eval at bootstrapped states
        with torch.no_grad():
            v1 = self.target1(xn).view(-1)
            v2 = self.target2(xn).view(-1)
            v = torch.minimum(v1, v2)

        # For terminal states, use l_bootstrap as the bootstrap value
        v_bootstrap = torch.where(terminal_mask > 0.5, ln, v)

        # Stochastic absorption: sample survival indicator S ~ Bernoulli(delta^n)
        # survive_mask = 1 if survive (use v_bootstrap), 0 if absorbed (use v_term)
        survive_mask = (torch.rand(len(survival_probs), device=self.device) < survival_probs).float()

        # Final bootstrap: survive uses learned value, absorbed uses fixed v_term
        bootstrap = torch.where(survive_mask > 0.5, v_bootstrap, torch.full_like(v_bootstrap, self.v_term))

        y = torch.maximum(max_l, bootstrap)

        return x0, x1, l0, y, mono_mask_t, ssv0, inv0

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
        bce_loss_fn = nn.BCEWithLogitsLoss()

        # Evaluate before training for consistency with other trainers (test set only)
        print("Evaluating before training...")
        test_acc = float("nan")
        test_mse = float("nan")
        if self.test_dataset is not None:
            test_loader = DataLoader(self.test_dataset, batch_size=self.dl.batch_size, collate_fn=hj_collate)
            pre_stats_test = self.evaluate_invariant_sign(test_loader)
            test_acc = pre_stats_test.get("inv_acc", float("nan"))
            test_mse = pre_stats_test.get("value_mse", float("nan"))

        self.eval_steps.append(0)
        self.eval_value_mse.append(test_mse)
        self.eval_inv_acc.append(test_acc)
        print(f"Pre-train invariant sign accuracy | test: {test_acc:.4f} mse={test_mse:.6f}")

        epoch = 0
        while self.global_step < total_steps:
            epoch += 1
            epoch_start_step = self.global_step
            epoch_losses = []
            t0 = time.time()
            print(f"\nEpoch {epoch} (steps {self.global_step}/{total_steps}) started...")

            for it, batch in enumerate(self.dl):
                res = self._compute_targets(batch)
                if res[0] is None:
                    continue
                x0, x1, l0, y, mono_mask, ssv0, inv0 = res

                self.opt.zero_grad()
                v1 = self.critic1(x0).view(-1)
                v2 = self.critic2(x0).view(-1)
                v_min = torch.minimum(v1, v2)
                v_mean = 0.5 * (v1 + v2)

                # Main regression loss (shared target for both critics)
                main_loss = ((v1 - y) ** 2 + (v2 - y) ** 2) * 0.5
                main_loss = main_loss.mean()

                # Lower-bound hinge: enforce V >= l0
                hinge_lb = (F.relu(l0 - v1) + F.relu(l0 - v2)).mean()

                # Monotonicity hinge: V(x_t) >= V(x_{t+1})
                v1_next = self.critic1(x1).view(-1)
                v2_next = self.critic2(x1).view(-1)
                mono_term = (F.relu(v1_next - v1) + F.relu(v2_next - v2))
                # mask to avoid counting invalid successors (should be 1 for valid)
                hinge_mono = (mono_term * mono_mask).sum() / torch.clamp(mono_mask.sum(), min=1.0)

                # BCE on sign: target 1 if y > 0
                y_pos = (y > 0).float()
                logits = torch.cat([self.alpha_bce * v1, self.alpha_bce * v2], dim=0)
                labels = torch.cat([y_pos, y_pos], dim=0)
                bce_loss = bce_loss_fn(logits, labels)

                mse_v_ssv = F.mse_loss(v_min, ssv0)
                inv_pred = (v_mean.view(-1) <= 0).float()
                inv_acc = float((inv_pred == inv0.view(-1)).float().mean())

                total_loss = (
                    self.weight_main * main_loss
                    + self.weight_hinge_lb * hinge_lb
                    + self.weight_hinge_mono * hinge_mono
                    + self.weight_bce * bce_loss
                )

                total_loss.backward()
                self.opt.step()
                self.global_step += 1

                # Track regression error vs. sample safety value by step
                self.train_steps.append(self.global_step)
                self.train_value_mse.append(float(mse_v_ssv))
                self.train_inv_acc.append(inv_acc)

                # Target update
                if self.global_step % self.target_update_period == 0:
                    self._soft_update_targets()

                # Logging
                loss_val = float(total_loss.item())
                epoch_losses.append(loss_val)
                if self.global_step % log_every == 0:
                    print(
                        f"[Step {self.global_step} | Epoch {epoch} Iter {it+1}] loss={loss_val:.4f} "
                        f"main={float(main_loss):.4f} lb={float(hinge_lb):.4f} mono={float(hinge_mono):.4f} "
                        f"bce={float(bce_loss):.4f} mse_v_ssv={float(mse_v_ssv):.6f} train_inv_acc={inv_acc:.4f} | {_eta_str()}"
                    )

                if self.global_step % self.eval_interval == 0:
                    self._eval_and_record()

                if self.global_step >= total_steps:
                    break

            avg_loss = float(np.mean(epoch_losses)) if len(epoch_losses) > 0 else float('nan')
            self.loss_history.append(avg_loss)
            print(
                f"Epoch {epoch} finished in {time.time()-t0:.1f}s | avg_loss={avg_loss:.6f} "
                f"| global_step={self.global_step}/{total_steps} | {_eta_str()}"
            )

            if ckpt_dir is not None and not save_last_only:
                os.makedirs(ckpt_dir, exist_ok=True)
                epoch_ckpt = os.path.join(ckpt_dir, f"lambda_reachability_{epoch}{self.run_suffix}.pt")
                self.save_checkpoint(epoch_ckpt)
                print(f"Saved epoch checkpoint to {epoch_ckpt}")

            if self.global_step >= total_steps:
                break
            if epoch_start_step == self.global_step:
                print("No progress made in this epoch; stopping early.")
                break

        # Save final checkpoint and plots/data
        if ckpt_dir is not None:
            os.makedirs(ckpt_dir, exist_ok=True)
            final_ckpt = os.path.join(ckpt_dir, f"last{self.run_suffix}.pt")
            self.save_checkpoint(final_ckpt)
            print(f"Saved final checkpoint to {final_ckpt}")
        self.save_plots()
        self.save_training_data()

    def _eval_and_record(self):
        # Evaluate invariant sign accuracy on test set using first state and invariant flag
        test_acc = float('nan')
        test_mse = float('nan')
        if self.test_dataset is not None:
            test_loader = DataLoader(self.test_dataset, batch_size=self.dl.batch_size, collate_fn=hj_collate)
            test_stats = self.evaluate_invariant_sign(test_loader)
            test_acc = test_stats.get("inv_acc", float('nan'))
            test_mse = test_stats.get("value_mse", float('nan'))
        self.eval_steps.append(self.global_step)
        self.eval_inv_acc.append(test_acc)
        self.eval_value_mse.append(test_mse)
        print(f"[Step {self.global_step}] test_acc={test_acc:.4f} eval_value_mse={test_mse:.6f}")

    def save_checkpoint(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "critic1": self.critic1.state_dict(),
            "critic2": self.critic2.state_dict(),
            "opt": self.opt.state_dict(),
        }, path)

    def load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.critic1.load_state_dict(ckpt["critic1"])
        self.critic2.load_state_dict(ckpt["critic2"])
        self.opt.load_state_dict(ckpt["opt"])
        self._hard_update_targets()

    def save_plots(self):
        if self.plot_dir is None:
            return
        import matplotlib.pyplot as plt
        suffix = self.run_suffix
        os.makedirs(self.plot_dir, exist_ok=True)

        if len(self.eval_steps) > 0 and len(self.eval_inv_acc) > 0:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(self.eval_steps, self.eval_inv_acc, label="Test Acc")
            if len(self.train_steps) > 0 and len(self.train_inv_acc) > 0:
                ax.plot(self.train_steps, self.train_inv_acc, label="Train Inv Acc")
            ax.set_xlabel("Step")
            ax.set_ylabel("Invariant sign accuracy")
            ax.set_title("λ-Reachability invariant accuracy vs steps")
            ax.grid(True, linestyle="--", alpha=0.5)
            ax.legend()
            path = os.path.join(self.plot_dir, f"accuracy_vs_steps{suffix}.png")
            plt.tight_layout()
            plt.savefig(path, dpi=150)
            plt.close()
            print(f"Saved accuracy plot to {path}")

        if len(self.train_steps) > 0:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(self.train_steps, self.train_value_mse, label="Train value MSE")
            ax.set_xlabel("Step")
            ax.set_ylabel("MSE(v, sample_safety_value)")
            ax.set_title("Value MSE vs steps")
            ax.grid(True, linestyle="--", alpha=0.5)
            ax.legend()
            path = os.path.join(self.plot_dir, f"train_value_mse{suffix}.png")
            plt.tight_layout()
            plt.savefig(path, dpi=150)
            plt.close()
            print(f"Saved value MSE plot to {path}")

    def save_training_data(self):
        if self.train_data_dir is None:
            return
        import pandas as pd
        os.makedirs(self.train_data_dir, exist_ok=True)
        suffix = self.run_suffix
        if len(self.eval_steps) > 0:
            df_eval = pd.DataFrame({
                "step": self.eval_steps,
                "eval_value_mse": self.eval_value_mse,
                "eval_inv_acc": self.eval_inv_acc,
            })
            path_eval = os.path.join(self.train_data_dir, f"evaluation_metrics{suffix}.csv")
            df_eval.to_csv(path_eval, index=False)
            print(f"Saved evaluation data to {path_eval}")

        if len(self.train_steps) > 0:
            df_train = pd.DataFrame({
                "step": self.train_steps,
                "train_value_mse": self.train_value_mse,
                "train_inv_acc": self.train_inv_acc,
            })
            path_train = os.path.join(self.train_data_dir, f"training_metrics{suffix}.csv")
            df_train.to_csv(path_train, index=False)
            print(f"Saved training metrics to {path_train}")

    def evaluate_invariant_sign(self, data_loader: DataLoader) -> dict:
        """Evaluate invariant sign accuracy and value MSE vs sample_safety_value."""
        self.critic1.eval()
        self.critic2.eval()
        preds = []
        labels = []
        inv_acc_list = []
        mse_list = []
        processed = 0
        max_samples = self.eval_batches
        with torch.no_grad():
            for batch in data_loader:
                x, _x_next, _l, sample_safety_value, invariant, _xf, _lf = batch
                x = x.to(self.device)
                sample_safety_value = sample_safety_value.to(self.device).view(-1)
                invariant = invariant.to(self.device).view(-1)

                v_mean = 0.5 * (self.critic1(x) + self.critic2(x)).view(-1)
                pred = (v_mean <= 0).float()
                label = (invariant > 0).float()

                preds.append(pred.cpu().numpy())
                labels.append(label.cpu().numpy())
                inv_acc_list.append(((pred == label).float().mean()).item())
                mse_list.append(((v_mean - sample_safety_value) ** 2).mean().item())

                processed += len(invariant)
                if max_samples > 0 and processed >= max_samples:
                    break
        preds_np = np.concatenate(preds, axis=0) if len(preds) > 0 else np.array([])
        labels_np = np.concatenate(labels, axis=0) if len(labels) > 0 else np.array([])
        inv_acc = float(np.mean(inv_acc_list)) if len(inv_acc_list) > 0 else float('nan')
        value_mse = float(np.mean(mse_list)) if len(mse_list) > 0 else float('nan')
        return {"inv_acc": inv_acc, "value_mse": value_mse, "preds": preds_np, "labels": labels_np}
