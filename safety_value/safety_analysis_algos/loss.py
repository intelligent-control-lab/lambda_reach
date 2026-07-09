import torch


class DiscountedBellmanLoss:

    def __init__(self, lam: float = 0.8):
        self.lam = float(lam)

    def set_lambda(self, lam: float):
        self.lam = float(lam)

    def compute_loss(self, v_x: torch.Tensor, v_x_next: torch.Tensor, l: torch.Tensor, invariant_flag: torch.Tensor) -> torch.Tensor:
        """Compute scalar loss given current predictions v_x, next-step predictions v_x_next and cost l.

        For now leave as not implemented. Return a scalar tensor when implemented.
        """
        # Implemented loss according to user spec:
        # value_target = (1 - lambda) * l(x) + lambda * max(l(x), v_x_detached)
        # loss = mean squared error between v_x and value_target
        if not (v_x.shape == l.shape):
            # try to squeeze or expand l
            l = l.view_as(v_x)

        v_x_next_det = v_x_next.detach()
        value_target = (1.0 - self.lam) * l + self.lam * torch.max(l, v_x_next_det)
        loss = torch.mean((v_x - value_target) ** 2)
        return loss


class WeakSupervisionLoss:
    """Loss function for weak supervision approach.
    
    If invariant_flag == 1, the value should be <= 0 (safe/invariant)
    If invariant_flag == 0, the value should be > 0 (unsafe/not invariant)
    
    We use a hinge loss approach to encourage:
    - v_x <= 0 when invariant_flag == 1
    - v_x > 0 when invariant_flag == 0
    """
    
    def __init__(self, margin: float = 0.1):
        """
        Args:
            margin: Safety margin for the hinge loss
        """
        self.margin = float(margin)
    
    def set_margin(self, margin: float):
        self.margin = float(margin)
    
    def compute_loss(self, v_x: torch.Tensor, invariant_flag: torch.Tensor) -> torch.Tensor:
        """Compute loss for direct labeling.
        
        Args:
            v_x: Predicted values, shape (batch, 1)
            invariant_flag: Binary labels, 1 for invariant (should be negative), 0 for not invariant (should be positive)
        
        Returns:
            Scalar loss tensor
        """
        # Ensure shapes match
        if v_x.shape != invariant_flag.shape:
            invariant_flag = invariant_flag.view_as(v_x)
        
        # For invariant samples (flag == 1): loss = max(0, v_x + margin)
        # We want v_x <= -margin, so penalize when v_x > -margin
        invariant_loss = torch.where(
            invariant_flag > 0.5,
            torch.clamp(v_x + self.margin, min=0.0),
            torch.zeros_like(v_x)
        )
        
        # For non-invariant samples (flag == 0): loss = max(0, margin - v_x)
        # We want v_x >= margin, so penalize when v_x < margin
        non_invariant_loss = torch.where(
            invariant_flag <= 0.5,
            torch.clamp(self.margin - v_x, min=0.0),
            torch.zeros_like(v_x)
        )
        
        # Combine losses
        total_loss = invariant_loss + non_invariant_loss
        return torch.mean(total_loss)


class SupervisedRegressionLoss:
    """Loss function for supervised regression approach.
    
    Directly fits the predicted value to the sample safety value using
    mean squared error (MSE).
    """
    
    def __init__(self):
        """Initialize supervised regression loss."""
        pass
    
    def compute_loss(self, v_x: torch.Tensor, sample_safety_value: torch.Tensor) -> torch.Tensor:
        """Compute MSE loss between predictions and sample safety values.
        
        Args:
            v_x: Predicted values, shape (batch, 1)
            sample_safety_value: Ground truth safety values, shape (batch, 1)
        
        Returns:
            Scalar loss tensor
        """
        # Ensure shapes match
        if v_x.shape != sample_safety_value.shape:
            sample_safety_value = sample_safety_value.view_as(v_x)
        
        # Simple MSE loss
        loss = torch.mean((v_x - sample_safety_value) ** 2)
        return loss
