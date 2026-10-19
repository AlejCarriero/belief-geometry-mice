"""Milestone 0: check that this machine can run the project.

None of these tests check any science. They exist so that a fresh clone fails
early and clearly if a dependency is missing or broken.
"""

import importlib
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from src.seeding import set_global_seed

REQUIRED_PACKAGES = ["numpy", "torch", "matplotlib", "sklearn", "pytest"]


def test_python_version_is_at_least_3_10() -> None:
    assert sys.version_info >= (3, 10)


# One test per package, so a missing package is reported by name
# instead of as a single import error for the whole file.
@pytest.mark.parametrize("package_name", REQUIRED_PACKAGES)
def test_required_package_imports(package_name: str) -> None:
    importlib.import_module(package_name)


def test_torch_can_run_forward_and_backward_on_cpu() -> None:
    layer = torch.nn.Linear(in_features=4, out_features=2)
    inputs = torch.randn(3, 4)

    loss = layer(inputs).pow(2).mean()
    loss.backward()

    assert layer.weight.grad is not None
    assert torch.isfinite(layer.weight.grad).all()


def test_matplotlib_can_save_a_figure_without_a_display(tmp_path: Path) -> None:
    import matplotlib

    # "Agg" draws to files only, so this also works on a machine with no screen.
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots()
    axes.plot([0, 1], [0, 1])
    output_path = tmp_path / "smoke_test.png"
    figure.savefig(output_path)
    plt.close(figure)

    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_set_global_seed_makes_numpy_and_torch_reproducible() -> None:
    first_numpy_generator = set_global_seed(123)
    first_torch_draw = torch.randn(5)
    first_numpy_draw = first_numpy_generator.random(5)

    second_numpy_generator = set_global_seed(123)
    second_torch_draw = torch.randn(5)
    second_numpy_draw = second_numpy_generator.random(5)

    assert torch.equal(first_torch_draw, second_torch_draw)
    np.testing.assert_array_equal(first_numpy_draw, second_numpy_draw)


def test_different_seeds_give_different_draws() -> None:
    # Guards against a seeding helper that ignores its argument.
    first_numpy_generator = set_global_seed(1)
    first_torch_draw = torch.randn(5)
    first_numpy_draw = first_numpy_generator.random(5)

    second_numpy_generator = set_global_seed(2)
    second_torch_draw = torch.randn(5)
    second_numpy_draw = second_numpy_generator.random(5)

    assert not torch.equal(first_torch_draw, second_torch_draw)
    assert not np.array_equal(first_numpy_draw, second_numpy_draw)
