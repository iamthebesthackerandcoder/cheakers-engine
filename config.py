"""
Central configuration for paths and tunables.
Enhanced with Pydantic models for type-safe configuration management.
"""
from __future__ import annotations

import os
import logging
import sys
from typing import Dict, Any, Optional, List, Tuple, Union
from pydantic import BaseModel, Field, validator, field_validator


# Type aliases for better readability (avoiding circular imports)
Board = List[int]
Move = List[int]
GameResult = Tuple[int, Optional[Move]]
Player = int  # 1 for black, -1 for red
ConfigDict = Dict[str, Any]
SettingsDict = Dict[str, Union[str, int, float, bool]]


class UISettings(BaseModel):
    """UI display and interaction settings."""

    use_color: bool = Field(default=True, description="Enable colored terminal output")
    use_unicode: bool = Field(default=True, description="Use Unicode characters for pieces")
    show_indices: bool = Field(default=True, description="Show board square indices")
    compact: bool = Field(default=False, description="Use compact board layout")
    animations: bool = Field(default=True, description="Enable animations and delays")
    show_borders: bool = Field(default=True, description="Show board borders")
    highlight_moves: bool = Field(default=True, description="Highlight legal move destinations")

    @field_validator('use_color', 'use_unicode', 'show_indices', 'compact', 'animations', 'show_borders', 'highlight_moves', mode='before')
    @classmethod
    def validate_bool_fields(cls, v):
        return bool(v)


class EngineSettings(BaseModel):
    """Game engine configuration settings."""

    default_depth: int = Field(default=6, ge=1, le=10, description="Default search depth")
    batch_size: int = Field(default=64, ge=1, le=1024, description="Neural evaluation batch size (increased for better GPU utilization)")
    tt_max_size: int = Field(default=200000, ge=1000, description="Transposition table max size (doubled for better caching)")
    max_cache_size: int = Field(default=20000, ge=1000, description="Evaluation cache max size (doubled for better performance)")
    aspiration_margin: int = Field(default=100, ge=1, description="Aspiration search margin")
    null_move_reduction: int = Field(default=3, ge=1, description="Null move pruning depth reduction")
    # New performance optimization settings
    use_iterative_deepening: bool = Field(default=True, description="Use iterative deepening for better move ordering")
    parallel_search: bool = Field(default=False, description="Enable parallel search (experimental)")
    cache_compression: bool = Field(default=True, description="Enable cache compression for memory efficiency")

    @field_validator('default_depth', 'batch_size', 'tt_max_size', 'max_cache_size', 'aspiration_margin', 'null_move_reduction', mode='before')
    @classmethod
    def validate_int_fields(cls, v):
        return int(v)


class TrainingSettings(BaseModel):
    """Neural network training configuration."""

    learning_rate: float = Field(default=1e-3, gt=0, le=1.0, description="Training learning rate (increased for faster convergence)")
    epochs: int = Field(default=2, ge=1, le=100, description="Number of training epochs (increased for better learning)")
    batch_size: int = Field(default=2048, ge=32, le=8192, description="Training batch size (doubled for better GPU utilization)")
    l2_regularization: float = Field(default=1e-6, ge=0, description="L2 regularization strength")
    train_val_split: float = Field(default=0.8, gt=0, lt=1, description="Train/validation split ratio")
    shuffle_training: bool = Field(default=True, description="Shuffle training data")
    # New training optimizations
    use_mixed_precision: bool = Field(default=False, description="Use mixed precision training for faster computation")
    gradient_clipping: float = Field(default=1.0, ge=0, description="Gradient clipping threshold")
    early_stopping_patience: int = Field(default=10, ge=0, description="Early stopping patience")

    @field_validator('learning_rate', 'l2_regularization', 'train_val_split', mode='before')
    @classmethod
    def validate_float_fields(cls, v):
        return float(v)


class GameRulesSettings(BaseModel):
    """Game rules and variant settings."""

    captures_mandatory: bool = Field(default=False, description="Require captures when available")
    allow_undo: bool = Field(default=True, description="Allow undoing moves")
    auto_promotion: bool = Field(default=True, description="Auto-promote pieces to kings")

    # Use Pydantic v2 style validators for stricter runtime validation
    @field_validator('captures_mandatory', 'allow_undo', 'auto_promotion', mode='before')
    @classmethod
    def validate_bool_fields(cls, v):
        return bool(v)


class LoggingSettings(BaseModel):
    """Logging and monitoring configuration."""

    log_level: str = Field(default="INFO", description="Logging level (DEBUG, INFO, WARNING, ERROR)")
    log_to_file: bool = Field(default=False, description="Write logs to file")
    log_file_path: str = Field(default="checkers.log", description="Log file path")
    enable_profiling: bool = Field(default=False, description="Enable performance profiling")
    profile_output_dir: str = Field(default="profiles", description="Profile output directory")

    @field_validator('log_level', mode='before')
    @classmethod
    def validate_log_level(cls, v):
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR']
        v_upper = v.upper() if isinstance(v, str) else str(v).upper()
        if v_upper not in valid_levels:
            raise ValueError(f"log_level must be one of {valid_levels}")
        return v_upper


class CheckersConfig(BaseModel):
    """Main configuration model for the Checkers AI project."""

    ui: UISettings = Field(default_factory=UISettings)
    engine: EngineSettings = Field(default_factory=EngineSettings)
    training: TrainingSettings = Field(default_factory=TrainingSettings)
    rules: GameRulesSettings = Field(default_factory=GameRulesSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)

    # Metadata
    version: str = Field(default="1.0.0", description="Configuration version")
    config_file: Optional[str] = Field(default=None, description="Path to config file")

    def __init__(self, **data):
        super().__init__(**data)
        # Auto-detect terminal capabilities
        try:
            if not sys.stdout.isatty():
                self.ui.use_color = False
                self.ui.animations = False
        except (AttributeError, OSError):
            # Fallback for environments without isatty support
            self.ui.use_color = False
            self.ui.animations = False

    @classmethod
    def from_env(cls) -> 'CheckersConfig':
        """Create configuration from environment variables."""
        return cls(
            ui=UISettings(
                use_color=os.getenv('CHECKERS_COLOR', 'true').lower() == 'true',
                use_unicode=os.getenv('CHECKERS_UNICODE', 'true').lower() == 'true',
                compact=os.getenv('CHECKERS_COMPACT', 'false').lower() == 'true',
                animations=os.getenv('CHECKERS_ANIMATIONS', 'true').lower() == 'true',
            ),
            engine=EngineSettings(
                default_depth=int(os.getenv('CHECKERS_DEPTH', '6')),
                batch_size=int(os.getenv('CHECKERS_BATCH_SIZE', '32')),
            ),
            rules=GameRulesSettings(
                captures_mandatory=os.getenv('CHECKERS_MANDATORY', 'false').lower() == 'true',
            ),
            logging=LoggingSettings(
                log_level=os.getenv('CHECKERS_LOG_LEVEL', 'INFO'),
                log_to_file=os.getenv('CHECKERS_LOG_FILE', 'false').lower() == 'true',
            )
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'ui': self.ui.dict(),
            'engine': self.engine.dict(),
            'training': self.training.dict(),
            'rules': self.rules.dict(),
            'logging': self.logging.dict(),
            'version': self.version,
            'config_file': self.config_file,
        }

    def save_to_file(self, filepath: str) -> None:
        """Save configuration to JSON file."""
        import json
        config_dict = self.to_dict()
        config_dict['config_file'] = filepath

        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=2)

    @classmethod
    def load_from_file(cls, filepath: str) -> 'CheckersConfig':
        """Load configuration from JSON file."""
        import json

        with open(filepath, 'r') as f:
            data = json.load(f)

        return cls(
            ui=UISettings(**data.get('ui', {})),
            engine=EngineSettings(**data.get('engine', {})),
            training=TrainingSettings(**data.get('training', {})),
            rules=GameRulesSettings(**data.get('rules', {})),
            logging=LoggingSettings(**data.get('logging', {})),
            version=data.get('version', '1.0.0'),
            config_file=filepath,
        )

    def update_from_dict(self, updates: Dict[str, Any]) -> None:
        """Update configuration from dictionary."""
        for section, settings in updates.items():
            if hasattr(self, section) and isinstance(settings, dict):
                section_model = getattr(self, section)
                for key, value in settings.items():
                    if hasattr(section_model, key):
                        setattr(section_model, key, value)


# Global configuration instance
_config: Optional[CheckersConfig] = None


def get_config() -> CheckersConfig:
    """Get or create the global configuration instance."""
    global _config
    if _config is None:
        _config = CheckersConfig.from_env()
    return _config


def load_config_from_file(filepath: str) -> CheckersConfig:
    """Load configuration from file and update global instance."""
    global _config
    _config = CheckersConfig.load_from_file(filepath)
    return _config


def reset_config() -> None:
    """Reset the global configuration to defaults."""
    global _config
    _config = None


# Convenience functions for common configuration access
def get_ui_settings() -> UISettings:
    """Get UI configuration settings."""
    return get_config().ui


def get_engine_settings() -> EngineSettings:
    """Get engine configuration settings."""
    return get_config().engine


def get_training_settings() -> TrainingSettings:
    """Get training configuration settings."""
    return get_config().training


def get_game_rules() -> GameRulesSettings:
    """Get game rules configuration settings."""
    return get_config().rules


def get_logging_settings() -> LoggingSettings:
    """Get logging configuration settings."""
    return get_config().logging


# Directories
DATA_DIR = os.environ.get("CHECKERS_DATA_DIR", os.path.join("data"))
MODELS_DIR = os.environ.get("CHECKERS_MODELS_DIR", os.path.join("models"))

# Default artifact paths
DEFAULT_MODEL_PATH = os.path.join(DATA_DIR, "neural_model.pkl")
DEFAULT_DATA_PATH = os.path.join(DATA_DIR, "training_data.pkl")  # will be saved as .npz
DEFAULT_CHECKPOINT_PATH = os.path.join(DATA_DIR, "training_checkpoint.pkl")

# Training exploration schedule
EPSILON_START = float(os.environ.get("CHECKERS_EPS_START", 0.10))
EPSILON_END = float(os.environ.get("CHECKERS_EPS_END", 0.01))
EPSILON_DECAY_GAMES = int(os.environ.get("CHECKERS_EPS_DECAY", 100))

# Search defaults
DEFAULT_SEARCH_DEPTHS = (4, 5, 6)

# Logging level (DEBUG, INFO, WARNING, ERROR)
LOG_LEVEL = os.environ.get("CHECKERS_LOG_LEVEL", "INFO").upper()


def ensure_dirs() -> None:
    """Ensure required directories exist."""
    for d in (DATA_DIR, MODELS_DIR):
        if d and not os.path.exists(d):
            os.makedirs(d, exist_ok=True)


def setup_logging() -> None:
    """Configure root logging once, controlled by env var CHECKERS_LOG_LEVEL."""
    if getattr(setup_logging, "_configured", False):
        return
    level: int = getattr(logging, LOG_LEVEL, logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    setup_logging._configured = True  # type: ignore[attr-defined]


# Backward compatibility: expose configuration as dictionaries
def get_ui_config() -> Dict[str, bool]:
    """Get UI configuration as dictionary (backward compatibility)."""
    ui_settings = get_ui_settings()
    return {
        "use_color": ui_settings.use_color,
        "use_unicode": ui_settings.use_unicode,
        "show_indices": ui_settings.show_indices,
        "compact": ui_settings.compact,
        "animations": ui_settings.animations,
        "show_borders": ui_settings.show_borders,
        "highlight_moves": ui_settings.highlight_moves,
    }


def get_engine_config() -> Dict[str, int]:
    """Get engine configuration as dictionary (backward compatibility)."""
    engine_settings = get_engine_settings()
    return {
        "default_depth": engine_settings.default_depth,
        "batch_size": engine_settings.batch_size,
        "tt_max_size": engine_settings.tt_max_size,
        "max_cache_size": engine_settings.max_cache_size,
        "aspiration_margin": engine_settings.aspiration_margin,
        "null_move_reduction": engine_settings.null_move_reduction,
    }


def get_rules_config() -> Dict[str, bool]:
    """Get rules configuration as dictionary (backward compatibility)."""
    rules_settings = get_game_rules()
    return {
        "captures_mandatory": rules_settings.captures_mandatory,
        "allow_undo": rules_settings.allow_undo,
        "auto_promotion": rules_settings.auto_promotion,
    }
