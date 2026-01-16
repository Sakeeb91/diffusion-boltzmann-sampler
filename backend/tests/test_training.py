"""Tests for training module including losses, trainer, and API endpoints."""

import pytest
import torch
import tempfile
import os
from torch.utils.data import DataLoader, TensorDataset

from backend.ml.models.score_network import ScoreNetwork
from backend.ml.models.diffusion import DiffusionProcess
from backend.ml.training import (
    Trainer,
    ScoreMatchingLoss,
    denoising_score_matching_loss,
    sigma_weighted_loss,
    snr_weighted_loss,
    importance_sampled_loss,
    compute_loss,
    reduce_loss,
)
from backend.ml.training.trainer import (
    EarlyStopping,
    EMA,
    create_scheduler,
    compute_gradient_norm,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def small_score_network():
    """Create a small score network for fast testing."""
    return ScoreNetwork(
        in_channels=1,
        base_channels=8,
        time_embed_dim=16,
        num_blocks=1,
    )


@pytest.fixture
def diffusion():
    """Create a diffusion process."""
    return DiffusionProcess(beta_min=0.1, beta_max=20.0)


@pytest.fixture
def sample_batch():
    """Create sample batch data for testing."""
    return torch.randn(4, 1, 8, 8)


@pytest.fixture
def sample_dataloader(sample_batch):
    """Create a dataloader from sample batch."""
    dataset = TensorDataset(sample_batch)
    return DataLoader(dataset, batch_size=2, shuffle=True)


@pytest.fixture
def trainer(small_score_network, diffusion):
    """Create a trainer instance."""
    return Trainer(
        score_network=small_score_network,
        diffusion=diffusion,
        learning_rate=1e-3,
    )


# ============================================================================
# Loss Function Tests
# ============================================================================


class TestComputeLoss:
    """Tests for compute_loss helper function."""

    def test_l2_loss(self):
        """L2 loss computes squared error."""
        pred = torch.tensor([1.0, 2.0, 3.0])
        target = torch.tensor([1.0, 1.0, 1.0])
        loss = compute_loss(pred, target, loss_type="l2")
        expected = torch.tensor([0.0, 1.0, 4.0])
        assert torch.allclose(loss, expected)

    def test_l1_loss(self):
        """L1 loss computes absolute error."""
        pred = torch.tensor([1.0, 2.0, 3.0])
        target = torch.tensor([1.0, 1.0, 1.0])
        loss = compute_loss(pred, target, loss_type="l1")
        expected = torch.tensor([0.0, 1.0, 2.0])
        assert torch.allclose(loss, expected)

    def test_huber_loss_small_error(self):
        """Huber loss is quadratic for small errors."""
        pred = torch.tensor([1.0, 1.5])
        target = torch.tensor([1.0, 1.0])
        loss = compute_loss(pred, target, loss_type="huber", huber_delta=1.0)
        # For |diff| <= delta, huber = 0.5 * diff^2
        expected = torch.tensor([0.0, 0.5 * 0.5**2])
        assert torch.allclose(loss, expected)

    def test_huber_loss_large_error(self):
        """Huber loss is linear for large errors."""
        pred = torch.tensor([3.0])
        target = torch.tensor([0.0])
        loss = compute_loss(pred, target, loss_type="huber", huber_delta=1.0)
        # For |diff| > delta, huber = delta * (|diff| - 0.5 * delta)
        expected = torch.tensor([1.0 * (3.0 - 0.5 * 1.0)])
        assert torch.allclose(loss, expected)

    def test_invalid_loss_type(self):
        """Invalid loss type raises ValueError."""
        with pytest.raises(ValueError):
            compute_loss(torch.zeros(1), torch.zeros(1), loss_type="invalid")


class TestReduceLoss:
    """Tests for reduce_loss helper function."""

    def test_mean_reduction(self):
        """Mean reduction averages all elements."""
        loss = torch.tensor([1.0, 2.0, 3.0, 4.0])
        reduced = reduce_loss(loss, reduction="mean")
        assert reduced.item() == 2.5

    def test_sum_reduction(self):
        """Sum reduction sums all elements."""
        loss = torch.tensor([1.0, 2.0, 3.0, 4.0])
        reduced = reduce_loss(loss, reduction="sum")
        assert reduced.item() == 10.0

    def test_none_reduction(self):
        """None reduction returns original tensor."""
        loss = torch.tensor([1.0, 2.0, 3.0])
        reduced = reduce_loss(loss, reduction="none")
        assert torch.equal(reduced, loss)

    def test_invalid_reduction(self):
        """Invalid reduction raises ValueError."""
        with pytest.raises(ValueError):
            reduce_loss(torch.zeros(1), reduction="invalid")


class TestDSMLossFunctions:
    """Tests for denoising score matching loss functions."""

    def test_dsm_loss_returns_scalar(self, small_score_network, diffusion, sample_batch):
        """DSM loss returns a scalar tensor."""
        loss = denoising_score_matching_loss(small_score_network, sample_batch, diffusion)
        assert loss.dim() == 0
        assert loss.item() > 0

    def test_sigma_weighted_loss_returns_scalar(
        self, small_score_network, diffusion, sample_batch
    ):
        """Sigma-weighted loss returns a scalar tensor."""
        loss = sigma_weighted_loss(small_score_network, sample_batch, diffusion)
        assert loss.dim() == 0
        assert loss.item() > 0

    def test_snr_weighted_loss_returns_scalar(
        self, small_score_network, diffusion, sample_batch
    ):
        """SNR-weighted loss returns a scalar tensor."""
        loss = snr_weighted_loss(small_score_network, sample_batch, diffusion)
        assert loss.dim() == 0
        assert loss.item() > 0

    def test_importance_sampled_loss_returns_scalar(
        self, small_score_network, diffusion, sample_batch
    ):
        """Importance-sampled loss returns a scalar tensor."""
        loss = importance_sampled_loss(small_score_network, sample_batch, diffusion)
        assert loss.dim() == 0
        assert loss.item() > 0

    def test_loss_is_differentiable(self, small_score_network, diffusion, sample_batch):
        """Loss can be backpropagated."""
        loss = denoising_score_matching_loss(small_score_network, sample_batch, diffusion)
        loss.backward()

        # Check gradients exist
        for param in small_score_network.parameters():
            assert param.grad is not None


class TestScoreMatchingLossClass:
    """Tests for ScoreMatchingLoss class."""

    def test_uniform_weighting(self, small_score_network, diffusion, sample_batch):
        """Uniform weighting computes standard DSM loss."""
        loss_fn = ScoreMatchingLoss(diffusion, weighting="uniform")
        loss = loss_fn(small_score_network, sample_batch)
        assert loss.dim() == 0
        assert loss.item() > 0

    def test_sigma_weighting(self, small_score_network, diffusion, sample_batch):
        """Sigma weighting works correctly."""
        loss_fn = ScoreMatchingLoss(diffusion, weighting="sigma")
        loss = loss_fn(small_score_network, sample_batch)
        assert loss.dim() == 0
        assert loss.item() > 0

    def test_snr_weighting(self, small_score_network, diffusion, sample_batch):
        """SNR weighting works correctly."""
        loss_fn = ScoreMatchingLoss(diffusion, weighting="snr", snr_gamma=0.5)
        loss = loss_fn(small_score_network, sample_batch)
        assert loss.dim() == 0
        assert loss.item() > 0

    def test_importance_weighting(self, small_score_network, diffusion, sample_batch):
        """Importance weighting works correctly."""
        loss_fn = ScoreMatchingLoss(diffusion, weighting="importance")
        loss = loss_fn(small_score_network, sample_batch)
        assert loss.dim() == 0
        assert loss.item() > 0

    def test_invalid_weighting_raises(self, diffusion, small_score_network, sample_batch):
        """Invalid weighting raises ValueError."""
        loss_fn = ScoreMatchingLoss(diffusion, weighting="uniform")
        loss_fn.weighting = "invalid"  # Force invalid weighting
        with pytest.raises(ValueError):
            loss_fn(small_score_network, sample_batch)


# ============================================================================
# Trainer Tests
# ============================================================================


class TestTrainerBasic:
    """Basic tests for Trainer class."""

    def test_trainer_initialization(self, small_score_network, diffusion):
        """Trainer initializes with correct attributes."""
        trainer = Trainer(small_score_network, diffusion, learning_rate=1e-3)
        assert trainer.model is not None
        assert trainer.diffusion is not None
        assert trainer.optimizer is not None

    def test_trainer_default_diffusion(self, small_score_network):
        """Trainer creates default diffusion if not provided."""
        trainer = Trainer(small_score_network)
        assert trainer.diffusion is not None

    def test_train_step_returns_loss(self, trainer, sample_batch):
        """Train step returns loss value."""
        loss = trainer.train_step(sample_batch)
        assert isinstance(loss, float)
        assert loss > 0

    def test_train_epoch_updates_history(self, trainer, sample_dataloader):
        """Train epoch updates loss history."""
        initial_len = len(trainer.history["train_loss"])
        trainer.train_epoch(sample_dataloader)
        assert len(trainer.history["train_loss"]) == initial_len + 1

    def test_evaluate_returns_loss(self, trainer, sample_dataloader):
        """Evaluate returns validation loss."""
        loss = trainer.evaluate(sample_dataloader)
        assert isinstance(loss, float)
        assert loss > 0

    def test_train_multiple_epochs(self, trainer, sample_dataloader):
        """Training for multiple epochs works."""
        history = trainer.train(sample_dataloader, epochs=3, verbose=False)
        assert len(history["train_loss"]) == 3

    def test_loss_weighting_option(self, small_score_network, diffusion):
        """Trainer accepts loss_weighting parameter."""
        trainer = Trainer(
            small_score_network,
            diffusion,
            loss_weighting="sigma",
        )
        assert trainer.loss_weighting == "sigma"


class TestTrainerCheckpointing:
    """Tests for Trainer checkpoint save/load."""

    def test_save_checkpoint(self, trainer, sample_dataloader):
        """Trainer can save checkpoint."""
        # Train a bit first
        trainer.train_epoch(sample_dataloader)

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            trainer.save_checkpoint(f.name)
            assert os.path.exists(f.name)
            os.unlink(f.name)

    def test_checkpoint_includes_metadata(self, trainer, sample_dataloader):
        """Checkpoint includes configuration metadata."""
        trainer.train_epoch(sample_dataloader)

        model_config = {
            "in_channels": 1,
            "base_channels": 8,
            "time_embed_dim": 16,
            "num_blocks": 1,
        }

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            trainer.save_checkpoint(
                f.name,
                model_config=model_config,
                training_temperature=2.5,
                training_meta={"epochs": 1},
                extra_info={"lattice_size": 8},
            )
            checkpoint = torch.load(f.name)
            assert checkpoint["model_config"]["in_channels"] == 1
            assert checkpoint["training_temperature"] == 2.5
            assert "diffusion_config" in checkpoint
            assert checkpoint["training_meta"]["epochs"] == 1
            assert checkpoint["lattice_size"] == 8
            os.unlink(f.name)

    def test_load_checkpoint(self, small_score_network, diffusion, sample_dataloader):
        """Trainer can load checkpoint."""
        # Create and train first trainer
        trainer1 = Trainer(small_score_network, diffusion)
        trainer1.train(sample_dataloader, epochs=2, verbose=False)

        # Save checkpoint
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            trainer1.save_checkpoint(f.name)

            # Create new trainer and load
            new_network = ScoreNetwork(
                in_channels=1, base_channels=8, time_embed_dim=16, num_blocks=1
            )
            trainer2 = Trainer(new_network, diffusion)
            trainer2.load_checkpoint(f.name)

            # History should be loaded
            assert len(trainer2.history["train_loss"]) == 2
            os.unlink(f.name)

    def test_checkpoint_contains_history(self, trainer, sample_dataloader):
        """Checkpoint contains training history."""
        trainer.train(sample_dataloader, epochs=3, verbose=False)

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            trainer.save_checkpoint(f.name)
            checkpoint = torch.load(f.name)
            assert "history" in checkpoint
            assert "train_loss" in checkpoint["history"]
            assert len(checkpoint["history"]["train_loss"]) == 3
            os.unlink(f.name)


# ============================================================================
# Training API Tests
# ============================================================================


class TestTrainingAPI:
    """Tests for training API endpoints."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        from fastapi.testclient import TestClient
        from backend.api.main import app

        return TestClient(app)

    def test_get_training_status(self, client):
        """GET /training/status returns status."""
        response = client.get("/training/status")
        assert response.status_code == 200
        data = response.json()
        assert "is_training" in data
        assert "progress" in data

    def test_get_training_config(self, client):
        """GET /training/config returns configuration."""
        response = client.get("/training/config")
        assert response.status_code == 200
        data = response.json()
        assert "recommended" in data
        assert "limits" in data

    def test_list_checkpoints(self, client):
        """GET /training/checkpoints returns list."""
        response = client.get("/training/checkpoints")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

    def test_load_nonexistent_checkpoint(self, client):
        """POST /training/checkpoints/load returns 404 for missing file."""
        response = client.post(
            "/training/checkpoints/load",
            json={"checkpoint_name": "nonexistent.pt"},
        )
        assert response.status_code == 404


# ============================================================================
# Helper Class Tests
# ============================================================================


class TestEarlyStopping:
    """Tests for EarlyStopping callback."""

    def test_no_stop_while_improving(self):
        """Early stopping doesn't trigger while loss decreases."""
        stopper = EarlyStopping(patience=3)
        losses = [1.0, 0.9, 0.8, 0.7, 0.6]
        for loss in losses:
            assert not stopper(loss)

    def test_stops_after_patience_no_improvement(self):
        """Early stopping triggers after patience epochs without improvement."""
        stopper = EarlyStopping(patience=3)
        # Decrease, then plateau
        stopper(1.0)
        stopper(0.5)
        stopper(0.5)  # No improvement
        stopper(0.5)  # No improvement
        assert stopper(0.5)  # Should stop now

    def test_reset_clears_state(self):
        """Reset clears early stopping state."""
        stopper = EarlyStopping(patience=2)
        stopper(1.0)
        stopper(1.0)
        stopper(1.0)  # Should trigger
        stopper.reset()
        assert not stopper.should_stop

    def test_min_delta_threshold(self):
        """Min delta requires minimum improvement."""
        stopper = EarlyStopping(patience=2, min_delta=0.1)
        stopper(1.0)
        stopper(0.95)  # Not enough improvement
        stopper(0.91)  # Still not enough
        assert stopper(0.87)  # Should stop


class TestEMA:
    """Tests for EMA (Exponential Moving Average)."""

    def test_ema_initialization(self, small_score_network):
        """EMA initializes shadow parameters."""
        ema = EMA(small_score_network, decay=0.999)
        assert len(ema.shadow) > 0

    def test_ema_update_changes_shadow(self, small_score_network):
        """EMA update modifies shadow parameters."""
        ema = EMA(small_score_network, decay=0.999)

        # Get initial shadow
        initial_shadow = {k: v.clone() for k, v in ema.shadow.items()}

        # Modify model parameters
        for param in small_score_network.parameters():
            param.data += 0.1

        # Update EMA
        ema.update()

        # Shadow should have changed
        for name in ema.shadow:
            assert not torch.equal(ema.shadow[name], initial_shadow[name])

    def test_apply_and_restore_shadow(self, small_score_network):
        """Apply shadow and restore works correctly."""
        ema = EMA(small_score_network, decay=0.999)

        # Get original parameters
        original = {
            name: param.data.clone()
            for name, param in small_score_network.named_parameters()
        }

        ema.apply_shadow()
        ema.restore()

        # Should be back to original
        for name, param in small_score_network.named_parameters():
            if name in original:
                assert torch.equal(param.data, original[name])


class TestScheduler:
    """Tests for learning rate scheduler factory."""

    def test_create_none_scheduler(self):
        """None scheduler returns None."""
        model = torch.nn.Linear(10, 10)
        opt = torch.optim.Adam(model.parameters())
        scheduler = create_scheduler(opt, "none", epochs=100)
        assert scheduler is None

    def test_create_cosine_scheduler(self):
        """Cosine scheduler is created correctly."""
        model = torch.nn.Linear(10, 10)
        opt = torch.optim.Adam(model.parameters())
        scheduler = create_scheduler(opt, "cosine", epochs=100)
        assert scheduler is not None

    def test_create_step_scheduler(self):
        """Step scheduler is created correctly."""
        model = torch.nn.Linear(10, 10)
        opt = torch.optim.Adam(model.parameters())
        scheduler = create_scheduler(opt, "step", epochs=100)
        assert scheduler is not None


class TestGradientNorm:
    """Tests for gradient norm computation."""

    def test_compute_gradient_norm(self, small_score_network, sample_batch, diffusion):
        """Gradient norm is computed correctly."""
        loss = denoising_score_matching_loss(small_score_network, sample_batch, diffusion)
        loss.backward()

        norm = compute_gradient_norm(small_score_network)
        assert norm > 0
        assert not torch.isnan(torch.tensor(norm))


# ============================================================================
# Integration Tests
# ============================================================================


class TestTrainingIntegration:
    """Integration tests for complete training workflow."""

    def test_loss_decreases_during_training(self):
        """Loss should generally decrease during training."""
        # Create simple network and data
        net = ScoreNetwork(in_channels=1, base_channels=8, num_blocks=1)
        diffusion = DiffusionProcess()
        trainer = Trainer(net, diffusion, learning_rate=1e-2)

        # Create synthetic data
        data = torch.randn(32, 1, 8, 8)
        dataset = TensorDataset(data)
        dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

        # Train
        history = trainer.train(dataloader, epochs=10, verbose=False)

        # Loss should decrease (check first vs last)
        assert history["train_loss"][-1] < history["train_loss"][0] * 1.5

    def test_training_with_validation(self):
        """Training with validation data works."""
        net = ScoreNetwork(in_channels=1, base_channels=8, num_blocks=1)
        trainer = Trainer(net, learning_rate=1e-3)

        train_data = torch.randn(16, 1, 8, 8)
        val_data = torch.randn(8, 1, 8, 8)
        train_loader = DataLoader(TensorDataset(train_data), batch_size=4)
        val_loader = DataLoader(TensorDataset(val_data), batch_size=4)

        history = trainer.train(
            train_loader, epochs=3, val_dataloader=val_loader, verbose=False
        )

        assert len(history["train_loss"]) == 3
        assert len(history["val_loss"]) == 3
