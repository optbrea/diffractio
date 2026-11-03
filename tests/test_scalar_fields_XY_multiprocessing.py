"""Multiprocessing tests for scalar XY fields."""

import datetime
import os
import time
from pathlib import Path

import numpy as np
import pytest

from diffractio import no_date, num_max_processors, plt, um
from diffractio.scalar_sources_XY import Scalar_source_XY
from diffractio.utils_multiprocessing import execute_multiprocessing

if no_date is True:
    date = '0'
else:
    now = datetime.datetime.now()
    date = now.strftime("%Y-%m-%d_%H")

path_base = "tests_results"
path_class = "scalar_fields_XY_multiprocessing"
newpath = "{}_{}/{}/".format(path_base, date, path_class)
os.makedirs(newpath, exist_ok=True)


def _propagate_xy_field(parameters):
    num_points = parameters['num_points']
    x = np.linspace(-100 * um, 100 * um, num_points)
    y = np.linspace(-100 * um, 100 * um, num_points)
    field = Scalar_source_XY(x, y, parameters['wavelength'])
    field.gauss_beam(
        r0=parameters['r0'],
        w0=parameters['w0'],
        z0=0,
        A=1,
    )

    for z in parameters['distances']:
        field = field.RS(z=z, new_field=True, verbose=False)

    return field.u


def test_xy_propagation_time_by_processor_count():
    if num_max_processors < 2:
        pytest.skip("This benchmark needs at least two available CPUs.")

    parameters = [
        {
            'num_points': 512,
            'wavelength': (0.5 + index * 0.01) * um,
            'r0': ((index % 4 - 1.5) * 5 * um, (index // 4 - 1.5) * 5 * um),
            'w0': (30 * um, 35 * um),
            'distances': (10 * um, 20 * um, 30 * um),
        }
        for index in range(num_max_processors)
    ]

    start = time.perf_counter()
    sequential_results = [_propagate_xy_field(item) for item in parameters]
    sequential_time = time.perf_counter() - start
    processor_counts = list(range(1, num_max_processors + 1))
    execution_times = [sequential_time]

    for num_processors in processor_counts[1:]:
        parallel_results, parallel_time = execute_multiprocessing(
            _propagate_xy_field,
            parameters,
            num_processors,
            verbose=False,
        )

        assert len(parallel_results) == len(sequential_results)
        for sequential, parallel in zip(sequential_results, parallel_results):
            np.testing.assert_allclose(
                parallel, sequential, rtol=1e-10, atol=1e-12
            )

        execution_times.append(parallel_time)
        print(f"{num_processors} processes: {parallel_time:.3f}s")

    output_path = Path(newpath) / "test_xy_propagation_time_by_processor_count.png"

    figure, axis = plt.subplots()
    axis.plot(processor_counts, execution_times, marker='o')
    axis.set(
        title='XY RS propagation time vs. number of processes',
        xlabel='Number of processes',
        ylabel='Execution time (s)',
    )
    axis.set_xticks(processor_counts)
    axis.grid(True)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)

    print(f"1 process: {sequential_time:.3f}s")
    print(f"Scaling graph saved to {output_path}")
