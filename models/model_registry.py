"""
models/model_registry.py - Model Versioning and Registry

MLflow-style model registry for:
- Versioning trained models
- A/B testing different model versions
- Tracking model performance over time
- Rolling back to previous versions
- Model lifecycle management (staging, production, archived)

Usage:
    registry = ModelRegistry()
    version = registry.register_model(model, "signal_generator", metrics)
    production_model = registry.get_production_model("signal_generator")
"""

import json
import pickle
import shutil
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ModelStage(Enum):
    """Model lifecycle stages."""
    DEVELOPMENT = "development"  # Being developed/tested
    STAGING = "staging"          # Ready for A/B testing
    PRODUCTION = "production"    # Active in production
    ARCHIVED = "archived"        # Retired/archived


@dataclass
class ModelMetrics:
    """Model performance metrics."""
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    max_drawdown: float = 0.0
    directional_accuracy: float = 0.0
    total_return: float = 0.0
    num_trades: int = 0

    # Training metrics
    train_loss: float = 0.0
    val_loss: float = 0.0
    train_accuracy: float = 0.0
    val_accuracy: float = 0.0

    def to_dict(self) -> dict:
        return {
            'sharpe_ratio': self.sharpe_ratio,
            'sortino_ratio': self.sortino_ratio,
            'win_rate': self.win_rate,
            'profit_factor': self.profit_factor,
            'max_drawdown': self.max_drawdown,
            'directional_accuracy': self.directional_accuracy,
            'total_return': self.total_return,
            'num_trades': self.num_trades,
            'train_loss': self.train_loss,
            'val_loss': self.val_loss,
            'train_accuracy': self.train_accuracy,
            'val_accuracy': self.val_accuracy
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'ModelMetrics':
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class ModelVersion:
    """A specific version of a model."""
    model_name: str
    version: int
    stage: ModelStage
    created_at: datetime
    updated_at: datetime
    metrics: ModelMetrics
    parameters: Dict[str, Any]
    tags: Dict[str, str]
    description: str
    model_path: Path
    checksum: str
    training_data_info: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            'model_name': self.model_name,
            'version': self.version,
            'stage': self.stage.value,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'metrics': self.metrics.to_dict(),
            'parameters': self.parameters,
            'tags': self.tags,
            'description': self.description,
            'model_path': str(self.model_path),
            'checksum': self.checksum,
            'training_data_info': self.training_data_info
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'ModelVersion':
        return cls(
            model_name=data['model_name'],
            version=data['version'],
            stage=ModelStage(data['stage']),
            created_at=datetime.fromisoformat(data['created_at']),
            updated_at=datetime.fromisoformat(data['updated_at']),
            metrics=ModelMetrics.from_dict(data['metrics']),
            parameters=data['parameters'],
            tags=data['tags'],
            description=data['description'],
            model_path=Path(data['model_path']),
            checksum=data['checksum'],
            training_data_info=data.get('training_data_info', {})
        )


class ModelRegistry:
    """
    Central registry for model versioning and lifecycle management.

    Features:
    - Version tracking with automatic incrementing
    - Stage management (dev, staging, production, archived)
    - Metrics tracking and comparison
    - Model artifact storage
    - Rollback support
    - A/B testing support
    """

    def __init__(self, registry_dir: Path = Path('models/registry')):
        """
        Initialize model registry.

        Args:
            registry_dir: Directory to store model artifacts and metadata
        """
        self.registry_dir = registry_dir
        self.registry_dir.mkdir(parents=True, exist_ok=True)

        # Model metadata file
        self.metadata_file = self.registry_dir / 'registry.json'

        # Load existing registry
        self.models: Dict[str, Dict[int, ModelVersion]] = {}
        self._load_registry()

        logger.info(f"📦 Model Registry initialized at {registry_dir}")
        logger.info(f"   Registered models: {len(self.models)}")

    def _load_registry(self):
        """Load registry metadata from disk."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file) as f:
                    data = json.load(f)

                for model_name, versions in data.items():
                    self.models[model_name] = {}
                    for version_str, version_data in versions.items():
                        version = ModelVersion.from_dict(version_data)
                        self.models[model_name][version.version] = version

                logger.info(f"Loaded {len(self.models)} models from registry")

            except Exception as e:
                logger.error(f"Failed to load registry: {e}")
                self.models = {}

    def _save_registry(self):
        """Save registry metadata to disk."""
        try:
            data = {}
            for model_name, versions in self.models.items():
                data[model_name] = {}
                for version_num, version in versions.items():
                    data[model_name][str(version_num)] = version.to_dict()

            with open(self.metadata_file, 'w') as f:
                json.dump(data, f, indent=2)

        except Exception as e:
            logger.error(f"Failed to save registry: {e}")

    def _compute_checksum(self, model_path: Path) -> str:
        """Compute checksum of model file."""
        hasher = hashlib.sha256()
        with open(model_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                hasher.update(chunk)
        return hasher.hexdigest()[:16]

    def _get_next_version(self, model_name: str) -> int:
        """Get next version number for a model."""
        if model_name not in self.models:
            return 1
        return max(self.models[model_name].keys()) + 1

    def register_model(
        self,
        model: Any,
        model_name: str,
        metrics: ModelMetrics,
        parameters: Dict[str, Any] = None,
        tags: Dict[str, str] = None,
        description: str = "",
        training_data_info: Dict[str, Any] = None,
        stage: ModelStage = ModelStage.DEVELOPMENT
    ) -> ModelVersion:
        """
        Register a new model version.

        Args:
            model: The trained model object (must be picklable)
            model_name: Name of the model (e.g., "signal_generator")
            metrics: Model performance metrics
            parameters: Hyperparameters used
            tags: Optional tags (e.g., {"experiment": "baseline"})
            description: Human-readable description
            training_data_info: Info about training data
            stage: Initial stage

        Returns:
            ModelVersion object
        """
        version_num = self._get_next_version(model_name)

        # Create model directory
        model_dir = self.registry_dir / model_name / f"v{version_num}"
        model_dir.mkdir(parents=True, exist_ok=True)

        # Save model artifact
        model_path = model_dir / 'model.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)

        # Compute checksum
        checksum = self._compute_checksum(model_path)

        # Create version
        now = datetime.now()
        version = ModelVersion(
            model_name=model_name,
            version=version_num,
            stage=stage,
            created_at=now,
            updated_at=now,
            metrics=metrics,
            parameters=parameters or {},
            tags=tags or {},
            description=description,
            model_path=model_path,
            checksum=checksum,
            training_data_info=training_data_info or {}
        )

        # Store in registry
        if model_name not in self.models:
            self.models[model_name] = {}
        self.models[model_name][version_num] = version

        # Save metadata
        self._save_registry()

        logger.info(f"✅ Registered {model_name} v{version_num} ({stage.value})")
        logger.info(f"   Metrics: Sharpe={metrics.sharpe_ratio:.2f}, WinRate={metrics.win_rate:.1%}")

        return version

    def load_model(self, model_name: str, version: int = None) -> Tuple[Any, ModelVersion]:
        """
        Load a model from the registry.

        Args:
            model_name: Name of the model
            version: Specific version (latest if None)

        Returns:
            Tuple of (model, version_info)
        """
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found in registry")

        if version is None:
            version = max(self.models[model_name].keys())

        if version not in self.models[model_name]:
            raise ValueError(f"Version {version} not found for model '{model_name}'")

        version_info = self.models[model_name][version]

        # Load model artifact
        with open(version_info.model_path, 'rb') as f:
            model = pickle.load(f)

        # Verify checksum
        current_checksum = self._compute_checksum(version_info.model_path)
        if current_checksum != version_info.checksum:
            logger.warning(f"⚠️ Checksum mismatch for {model_name} v{version}!")

        logger.info(f"Loaded {model_name} v{version} ({version_info.stage.value})")

        return model, version_info

    def get_production_model(self, model_name: str) -> Tuple[Any, ModelVersion]:
        """
        Get the production version of a model.

        Args:
            model_name: Name of the model

        Returns:
            Tuple of (model, version_info)
        """
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found")

        # Find production version
        for version_num, version in sorted(self.models[model_name].items(), reverse=True):
            if version.stage == ModelStage.PRODUCTION:
                return self.load_model(model_name, version_num)

        raise ValueError(f"No production version found for '{model_name}'")

    def promote_model(
        self,
        model_name: str,
        version: int,
        to_stage: ModelStage,
        demote_current: bool = True
    ):
        """
        Promote a model to a new stage.

        Args:
            model_name: Model name
            version: Version to promote
            to_stage: Target stage
            demote_current: If True, demote current model in that stage
        """
        if model_name not in self.models or version not in self.models[model_name]:
            raise ValueError(f"Model {model_name} v{version} not found")

        # Demote current model in target stage
        if demote_current and to_stage in [ModelStage.PRODUCTION, ModelStage.STAGING]:
            for v, model_version in self.models[model_name].items():
                if model_version.stage == to_stage and v != version:
                    if to_stage == ModelStage.PRODUCTION:
                        model_version.stage = ModelStage.ARCHIVED
                    else:
                        model_version.stage = ModelStage.DEVELOPMENT
                    model_version.updated_at = datetime.now()
                    logger.info(f"Demoted {model_name} v{v} to {model_version.stage.value}")

        # Promote target version
        self.models[model_name][version].stage = to_stage
        self.models[model_name][version].updated_at = datetime.now()

        self._save_registry()
        logger.info(f"✅ Promoted {model_name} v{version} to {to_stage.value}")

    def compare_versions(
        self,
        model_name: str,
        version_a: int,
        version_b: int
    ) -> Dict[str, Any]:
        """
        Compare two model versions.

        Args:
            model_name: Model name
            version_a: First version
            version_b: Second version

        Returns:
            Comparison dict with metrics differences
        """
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found")

        va = self.models[model_name].get(version_a)
        vb = self.models[model_name].get(version_b)

        if not va or not vb:
            raise ValueError("One or both versions not found")

        metrics_a = va.metrics.to_dict()
        metrics_b = vb.metrics.to_dict()

        comparison = {
            'version_a': version_a,
            'version_b': version_b,
            'metrics_diff': {},
            'winner': None
        }

        # Compare each metric
        for metric in metrics_a:
            diff = metrics_b[metric] - metrics_a[metric]
            comparison['metrics_diff'][metric] = {
                'version_a': metrics_a[metric],
                'version_b': metrics_b[metric],
                'diff': diff,
                'pct_change': diff / metrics_a[metric] * 100 if metrics_a[metric] != 0 else 0
            }

        # Determine winner based on Sharpe ratio
        if metrics_b['sharpe_ratio'] > metrics_a['sharpe_ratio']:
            comparison['winner'] = version_b
        else:
            comparison['winner'] = version_a

        return comparison

    def get_model_history(self, model_name: str) -> List[ModelVersion]:
        """Get all versions of a model."""
        if model_name not in self.models:
            return []
        return sorted(self.models[model_name].values(), key=lambda v: v.version)

    def list_models(self) -> Dict[str, Dict]:
        """List all registered models with summary info."""
        result = {}
        for model_name, versions in self.models.items():
            latest = max(versions.values(), key=lambda v: v.version)
            production = next(
                (v for v in versions.values() if v.stage == ModelStage.PRODUCTION),
                None
            )

            result[model_name] = {
                'num_versions': len(versions),
                'latest_version': latest.version,
                'production_version': production.version if production else None,
                'latest_metrics': latest.metrics.to_dict(),
                'stages': {stage.value: sum(1 for v in versions.values() if v.stage == stage)
                          for stage in ModelStage}
            }

        return result

    def delete_version(self, model_name: str, version: int, force: bool = False):
        """
        Delete a model version.

        Args:
            model_name: Model name
            version: Version to delete
            force: Force delete even if production
        """
        if model_name not in self.models or version not in self.models[model_name]:
            raise ValueError(f"Model {model_name} v{version} not found")

        model_version = self.models[model_name][version]

        if model_version.stage == ModelStage.PRODUCTION and not force:
            raise ValueError("Cannot delete production model without force=True")

        # Delete model artifact
        if model_version.model_path.exists():
            shutil.rmtree(model_version.model_path.parent)

        # Remove from registry
        del self.models[model_name][version]

        # Remove model entry if no versions left
        if not self.models[model_name]:
            del self.models[model_name]

        self._save_registry()
        logger.info(f"🗑️ Deleted {model_name} v{version}")

    def update_metrics(
        self,
        model_name: str,
        version: int,
        metrics: ModelMetrics
    ):
        """Update metrics for a model version (e.g., after live trading)."""
        if model_name not in self.models or version not in self.models[model_name]:
            raise ValueError(f"Model {model_name} v{version} not found")

        self.models[model_name][version].metrics = metrics
        self.models[model_name][version].updated_at = datetime.now()
        self._save_registry()

        logger.info(f"Updated metrics for {model_name} v{version}")

    def get_status(self) -> Dict:
        """Get registry status for dashboard."""
        total_versions = sum(len(v) for v in self.models.values())
        production_count = sum(
            1 for versions in self.models.values()
            for v in versions.values()
            if v.stage == ModelStage.PRODUCTION
        )

        return {
            'num_models': len(self.models),
            'total_versions': total_versions,
            'production_models': production_count,
            'models': list(self.models.keys())
        }
