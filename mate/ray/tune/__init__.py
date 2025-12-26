"""
Ray Tune compatibility wrapper
Supports Ray 1.13.x and Ray 2.x
"""

from ray.tune.error import TuneError
from ray.tune.tune import run_experiments, run
from ray.tune.syncer import SyncConfig
from ray.tune.experiment import Experiment
from ray.tune.analysis import Analysis, ExperimentAnalysis
from ray.tune.stopper import Stopper
from ray.tune.registry import register_env, register_trainable
from ray.tune.trainable import Trainable
from ray.tune.durable_trainable import DurableTrainable, durable
from ray.tune.callback import Callback
from ray.tune.suggest import grid_search
from ray.tune.progress_reporter import (
    ProgressReporter,
    CLIReporter,
    JupyterNotebookReporter,
)
from ray.tune.sample import (
    function,
    sample_from,
    uniform,
    quniform,
    choice,
    randint,
    lograndint,
    qrandint,
    qlograndint,
    randn,
    qrandn,
    loguniform,
    qloguniform,
)
from ray.tune.suggest import create_searcher
from ray.tune.schedulers import create_scheduler
from ray.tune.utils.placement_groups import PlacementGroupFactory
from ray.tune.utils.trainable import with_parameters

# ============================
# Session compatibility
# ============================

try:
    # Ray >= 2.x
    from ray.tune.session import (
        report,
        get_trial_dir,
        get_trial_name,
        get_trial_id,
        get_trial_resources,
        save_checkpoint,
        is_session_enabled,
    )

    def checkpoint_dir(step):
        return save_checkpoint(step)

except ImportError:
    # Ray 1.13.x fallback
    from ray import tune

    def report(**kwargs):
        tune.report(**kwargs)

    def checkpoint_dir(step):
        return tune.checkpoint_dir(step)

    def save_checkpoint(_):
        return

    def is_session_enabled():
        return False

    def get_trial_dir():
        return None

    def get_trial_name():
        return None

    def get_trial_id():
        return None

    def get_trial_resources():
        return None


__all__ = [
    "Trainable",
    "DurableTrainable",
    "durable",
    "Callback",
    "TuneError",
    "grid_search",
    "register_env",
    "register_trainable",
    "run",
    "run_experiments",
    "with_parameters",
    "Stopper",
    "Experiment",
    "function",
    "sample_from",
    "uniform",
    "quniform",
    "choice",
    "randint",
    "lograndint",
    "qrandint",
    "qlograndint",
    "randn",
    "qrandn",
    "loguniform",
    "qloguniform",
    "Analysis",
    "ExperimentAnalysis",
    "CLIReporter",
    "JupyterNotebookReporter",
    "ProgressReporter",
    "report",
    "get_trial_dir",
    "get_trial_name",
    "get_trial_id",
    "get_trial_resources",
    "checkpoint_dir",
    "save_checkpoint",
    "is_session_enabled",
    "SyncConfig",
    "create_searcher",
    "create_scheduler",
    "PlacementGroupFactory",
]
