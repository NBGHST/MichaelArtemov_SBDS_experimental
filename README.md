# Spatial Birth Death Simulator

This repository contains a high-performance C++ implementation of a spatial birth-death point process simulator with explicit spatial interactions. It is built as a Python extension using Cython, offering C++ execution speeds with Python's accessibility.

Simulator Features:

- Create 1D, 2D, or 3D grids
- Highly optimized interaction kernels (pre-computed neighbour lists, `distSq` evaluation)
- Custom birth/death intensities and radial interaction kernels
- Establish initial populations
- Trigger stochastic events (random or user-defined)
- Obtain cell-level granular data (coordinates, death rates, etc.)
- Spatial pattern and population dynamics analysis

---

## 1. Directory Structure

The project structure is organized as follows:

```
Spatial_Birth_Death_Simulator/
├── examples/                           # Example Jupyter notebooks
├── include/
│   └── SpatialBirthDeath.h             # C++ header with class definitions
├── src/
│   └── SpatialBirthDeath.cpp           # C++ simulator implementation
├── simulation/
│   ├── __init__.py                     # Python package initialization
│   └── SpatialBirthDeathWrapper.pyx    # Cython wrapper for the C++ code
├── tests/                              # Automated testing & benchmarking suite
├── .gitignore   
├── README.md                           # This file
├── clean.py                            # Script to clean generated build files
├── compare_benchmarks.sh               # Run cross-branch performance tests
├── optimization_summary.md             # Summary of C++ optimizations
├── requirements.txt                    # Python dependencies
└── setup.py                            # Build script
```

---

## 2. Building the Extension

From the root directory (`Spatial_Birth_Death_Simulator/`), execute the following commands (or run `full_setup.sh` and run notebooks under `venv_SBDS`):

```bash
pip install -r requirements.txt --break-system-packages
python setup.py build_ext --inplace
```

This compiles and generates the shared library in the `simulation` directory.

If you modify any `.cpp` or `.pyx` files, re-run `python setup.py build_ext --inplace` to rebuild.

---

## 3. Automated Testing and Benchmarks

This project includes a comprehensive suite of automated tests to ensure deterministic correctness and evaluate performance against baseline branches. 

To run the PyTest correctness suite:
```bash
python3 -m pytest tests/test_correctness.py -v
```

To compare performance across optimization branches (e.g., `extreme_opt` vs `cpp_opt`):
```bash
./compare_benchmarks.sh
```
*For detailed information on the test suite, refer to `tests/README.md`.*

---

## 4. Basic Usage from Python

After building, you can import the extension in Python:

```python
import sys
sys.path.append("your_path_to_project/Spatial_Birth_Death_Simulator")
import simulation  # or: from simulation import PyGrid1, PyGrid2, PyGrid3
```

Three Python classes are available for modeling, one for each dimension:

- **`PyGrid1`**: 1D simulator
- **`PyGrid2`**: 2D simulator
- **`PyGrid3`**: 3D simulator

### 4.1 Creating a Grid

**Primary Parameters**:

1. **M** (int): Number of species.
2. **areaLen** (list of float): Domain dimensions.
3. **cellCount** (list of int): Number of cells per dimension. This defines spatial partitioning for optimized neighbour searches. It is recommended to choose values such that the average interaction radius roughly matches 1-3 cell lengths.
4. **isPeriodic** (bool): `True` - periodic boundary conditions, `False` - absorbing boundary conditions.
5. **birthRates** (list of float of length `M`): Base birth intensities for each species.
6. **deathRates** (list of float of length `M`): Base natural death intensities for each species.
7. **ddMatrix** (list of float of length `M*M`): Flattened row-major matrix of pairwise competition. Element `ddMatrix[i*M + j]` defines the strength of species `i`'s effect on the mortality of species `j`.
8. **birthX**, **birthY** (list of lists of floats): Data for determining the inverse radial cumulative distribution function (ICDF) for offspring placement. For each species `s`:
   - **birthX[s]** contains sorted probabilities (quantiles) in range `[0,1)`
   - **birthY[s]** contains corresponding placement radii
9. **deathX**, **deathY** (3-level lists): Radial death kernel for each species pair `(s1, s2)`.
   - **deathX_[s1][s2]** contains sorted distances between individuals (0 to max interaction radius)
   - **deathY_[s1][s2]** contains corresponding density influence values
   
   When individuals of species `s1` and `s2` interact at distance $r$, their influence is calculated via linear interpolation between the `(deathX, deathY)` points. 
10. **cutoffs** (list of float of length `M*M`): Cutoff interaction distances for each species pair.
11. **seed** (int): RNG seed.
12. **rtimeLimit** (float): Maximum real-time limit in seconds for the simulation.

### 4.2 Placing Initial Populations

Use the `placePopulation(...)` method, which accepts a list of coordinate lists:

- For 1D: `[x]`
- For 2D: `[x, y]`
- For 3D: `[x, y, z]`

Alternatively, place specific individuals via `spawn_at(species_idx, position)`.

### 4.3 Triggering Events

The simulator provides the following methods for event execution:

- **`make_event()`**: Execute one stochastically chosen birth or death event.
- **`spawn_random()`**: Force one random birth event.
- **`kill_random()`**: Force one random death event.
- **`run_events(n)`**: Sequentially execute `n` `make_event()` calls.
- **`run_for(t)`**: Continue executing events until simulation time reaches or exceeds `t`.

### 4.4 Cell Data

For each grid cell, you can extract:

- **`get_cell_coords(cell_index, species_idx)`**: Returns coordinate list for this species in this cell.
- **`get_cell_death_rates(cell_index, species_idx)`**: Returns list of individual death rates.
- **`get_cell_population(cell_index)`**: Returns list of populations for each species.
- **`get_cell_birth_rate(cell_index)`** & **`get_cell_death_rate(cell_index)`**: Returns aggregated rates for this cell.

### 4.5 Global Particle Retrieval

The **`get_all_particle_coords()`** method returns all positions in a single call, grouped by species.

Similarly, **`get_all_particle_death_rates()`** returns individual death rates mapped 1-to-1 to the coordinates.

---

## 5. Defining Birth and Death Kernels

The simulator uses **radially symmetric** kernels for birth and death processes, defined as piecewise-linear functions `(X, Y)` that are interpolated at runtime.

### 5.1 Birth Kernels (Inverse Radial CDF)

For each species `s`, provide `(birthX[s], birthY[s])` representing the **Inverse Radial CDF** from `[0..1]` to `[0..∞)`:

1. **`birthX[s]`** - Sorted quantiles `[0..1)`.  
2. **`birthY[s]`** - Corresponding radii (`ICDF(u)`).

When a birth event occurs:
1. Generate uniform random `u ∈ [0,1]`.
2. Determine radius $r$ via linear interpolation.
3. Apply a random sign/direction based on dimensionality.

### 5.2 Death Kernels

For each pair `(s1, s2)`, provide `(deathX[s1][s2], deathY[s1][s2])` and a **cutoff radius**. The spatial interaction is evaluated linearly between these points.

**Crucial**: Death kernels must be **normalized** to integrate to 1 over the full spatial domain:

$$\int_{-\infty}^{\infty} K(x) dx = 1$$
$$\iint_{-\infty}^{\infty} K(x,y) dx dy = 1$$
$$\iiint_{-\infty}^{\infty} K(x,y,z) dx dy dz = 1$$

Normalization guarantees that the total influence of a single particle on the infinite space equals exactly one.
