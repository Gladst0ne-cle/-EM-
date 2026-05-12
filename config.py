from dataclasses import dataclass


@dataclass
class ExperimentConfig:
    dataset_name: str = "iris"
    n_components: int = 3
    max_iter: int = 300
    tol: float = 1e-6
    random_state: int = 42


@dataclass
class BatchConfig:
    seed_start: int = 0
    n_runs: int = 10
