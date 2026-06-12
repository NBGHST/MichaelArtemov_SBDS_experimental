/**
 * @file SpatialBirthDeath.h
 * @brief Header for a spatial birth-death point process simulator.
 *
 * This simulator models spatially explicit population dynamics with the ability
 * to spawn and kill individuals in a grid of cells.
 *
 * ## Features:
 * - Create grids in 1, 2, or 3 dimensions.
 * - Specify birth/death rates and radial kernels.
 * - Place initial populations.
 * - Periodic or non-periodic boundaries.
 * - Run stochastic events (random or user-specified spawns/kills).
 * - Inspect cell-level data (coordinates, death rates, etc.).
 * - Analyze spatial patterns and population dynamics.
 */

#include <vector>
#include <array>
#include <cmath>
#include <random>
#include <chrono>

/**
 * @brief Performs linear interpolation on tabulated (x, y) data.
 *
 * @param xdat The x-values of the tabulated data.
 * @param ydat The y-values of the tabulated data.
 * @param x The x-value at which to interpolate.
 * @return The interpolated y-value.
 */
inline double linearInterpolate(const std::vector<double> &xdat, const std::vector<double> &ydat, double x) {
    if (x >= xdat.back()) {
        return ydat.back();
    }
    if (x <= xdat.front()) {
        return ydat.front();
    }
    auto i = std::lower_bound(xdat.begin(), xdat.end(), x);
    const size_t k = i - xdat.begin();
    const size_t l = (k > 0) ? k - 1 : 0;
    const double x1 = xdat[l];
    const double x2 = xdat[k];
    const double y1 = ydat[l];
    const double y2 = ydat[k];
    return y1 + ((y2 - y1) * (x - x1) / (x2 - x1));
}

/**
 * @brief Fast O(1) linear interpolation for uniformly-spaced x data.
 *
 * Assumes x values are evenly spaced from x0 with step dx.
 * Falls back to clamping at boundaries.
 *
 * @param ydat The y-values of the tabulated data.
 * @param x0 The first x value.
 * @param dx The step between consecutive x values.
 * @param inv_dx Precomputed 1.0/dx.
 * @param n Number of data points.
 * @param x The x-value at which to interpolate.
 * @return The interpolated y-value.
 */
inline double linearInterpolateUniform(const double* __restrict__ ydat,
                                        double x0, double dx, double inv_dx,
                                        int n, double x) {
    if (x <= x0) return ydat[0];
    double idx_f = (x - x0) * inv_dx;
    int idx = static_cast<int>(idx_f);
    if (idx >= n - 1) return ydat[n - 1];
    double frac = idx_f - idx;
    return ydat[idx] + frac * (ydat[idx + 1] - ydat[idx]);
}

/**
 * @brief Computes the squared Euclidean distance between two points in DIM dimensions,
 *        with optional periodic wrapping.
 *
 * @param point_a The first point.
 * @param point_b The second point.
 * @param length The domain size along each dimension.
 * @param periodic Whether to apply periodic wrapping.
 * @return The squared Euclidean distance between `point_a` and `point_b`.
 */
template <int DIM>
inline double distancePeriodicSq(const std::array<double, DIM> &point_a, const std::array<double, DIM> &point_b,
                                  const std::array<double, DIM> &length, bool periodic) {
    double sumSq = 0.0;
    for (int i = 0; i < DIM; ++i) {
        double diff = point_a[i] - point_b[i];
        if (periodic) {
            if (diff > 0.5 * length[i]) {
                diff -= length[i];
            } else if (diff < -0.5 * length[i]) {
                diff += length[i];
            }
        }
        sumSq += diff * diff;
    }
    return sumSq;
}

/**
 * @brief Computes the Euclidean distance between two points in DIM dimensions,
 *        with optional periodic wrapping.
 *
 * For each dimension `d`, if `periodic` is true and the difference is larger
 * than half the domain length, the difference is wrapped accordingly.
 *
 * @param point_a The first point.
 * @param point_b The second point.
 * @param length The domain size along each dimension.
 * @param periodic Whether to apply periodic wrapping.
 * @return The Euclidean distance between `point_a` and `point_b`.
 */
template <int DIM>
inline double distancePeriodic(const std::array<double, DIM> &point_a, const std::array<double, DIM> &point_b,
                        const std::array<double, DIM> &length, bool periodic) {
    return std::sqrt(distancePeriodicSq<DIM>(point_a, point_b, length, periodic));
}

/**
 * @brief Iterates over all neighbor cell indices within a specified range around a center cell.
 *
 * Visits all cells within a hypercubic region centered at the given cell index.
 * For each neighbor cell, the provided callback function is invoked with the neighbor's index.
 *
 * @tparam DIM The dimension of the domain (1, 2, or 3).
 * @tparam FUNC A callable type that accepts a cell index array, e.g., `[](const std::array<int,
 * DIM> &nIdx) { ... }`.
 * @param centerIdx The center cell index around which to search.
 * @param range The maximum offset in each dimension to search (from `centerIdx - range` to
 * `centerIdx + range`).
 * @param callback The function to invoke for each neighbor cell index.
 */
template <int DIM, typename FUNC>
inline void forNeighbors(const std::array<int, DIM> &centerIdx, const std::array<int, DIM> &range, FUNC &&callback) {
    std::array<int, DIM> neighborIdx;
    if constexpr (DIM == 1) {
        const int minX = centerIdx[0] - range[0];
        const int maxX = centerIdx[0] + range[0];
        for (int x = minX; x <= maxX; ++x) {
            neighborIdx[0] = x;
            callback(neighborIdx);
        }
    } else if constexpr (DIM == 2) {
        const int minX = centerIdx[0] - range[0];
        const int maxX = centerIdx[0] + range[0];
        const int minY = centerIdx[1] - range[1];
        const int maxY = centerIdx[1] + range[1];
        for (int x = minX; x <= maxX; ++x) {
            neighborIdx[0] = x;
            for (int y = minY; y <= maxY; ++y) {
                neighborIdx[1] = y;
                callback(neighborIdx);
            }
        }
    } else if constexpr (DIM == 3) {
        const int minX = centerIdx[0] - range[0];
        const int maxX = centerIdx[0] + range[0];
        const int minY = centerIdx[1] - range[1];
        const int maxY = centerIdx[1] + range[1];
        const int minZ = centerIdx[2] - range[2];
        const int maxZ = centerIdx[2] + range[2];
        for (int x = minX; x <= maxX; ++x) {
            neighborIdx[0] = x;
            for (int y = minY; y <= maxY; ++y) {
                neighborIdx[1] = y;
                for (int z = minZ; z <= maxZ; ++z) {
                    neighborIdx[2] = z;
                    callback(neighborIdx);
                }
            }
        }
    }
}

/**
 * @brief Fenwick Tree (Binary Indexed Tree) for O(log N) prefix sums and sampling.
 */
class FenwickTree {
    std::vector<double> tree;
public:
    void init(int n) { tree.assign(n + 1, 0.0); }
    
    void update(int i, double delta) {
        for (++i; i < static_cast<int>(tree.size()); i += i & -i) {
            tree[i] += delta;
        }
    }
    
    double query(int i) const {
        double sum = 0.0;
        for (++i; i > 0; i -= i & -i) {
            sum += tree[i];
        }
        return sum;
    }
    int find(double target) const {
        if (tree.size() <= 1) return 0;
        int idx = 0;
        int n = static_cast<int>(tree.size()) - 1;
        
        int bitMask = 1;
        while (bitMask <= n) bitMask <<= 1;
        bitMask >>= 1;

        double current_sum = 0.0;
        for (int step = bitMask; step > 0; step >>= 1) {
            int next_idx = idx + step;
            if (next_idx <= n && current_sum + tree[next_idx] < target) {
                current_sum += tree[next_idx];
                idx = next_idx;
            }
        }
        return (idx >= n) ? (n - 1) : idx;
    }
};



/**
 * @brief The main simulation Grid. Partitions the domain into cells.
 *
 * The `Grid` class simulates spatially explicit population dynamics using a grid-based
 * approach. It supports multiple species, periodic boundaries, and stochastic birth/death
 * events. The simulation operates by repeatedly calling `make_event()`, which selects
 * either a birth or death event based on the global rates.
 *
 * @tparam DIM The dimensionality of the simulation (1D, 2D, or 3D).
 *
 *
 * ## Key Methods:
 * - `spawn_at(...)` / `kill_at(...)`: Low-level particle addition/removal.
 * - `placePopulation(...)`: Initializes populations by calling `spawn_at`.
 * - `spawn_random(...)` / `kill_random(...)`: Executes random birth/death events.
 * - `make_event()`: Performs a single birth or death event.
 * - `run_events(...)`: Runs a fixed number of events.
 * - `run_for(...)`: Runs the simulation for a specified duration.
 * - `get_all_particle_coords()`: Retrieves all particle coordinates.
 * - `get_all_particle_death_rates()`: Retrieves all particle death rates.
 */
template <int DIM>
class Grid {
public:
    int M_;                                ///< Number of species
    std::array<double, DIM> area_length_;  ///< Physical length of the domain in each dimension
    std::array<int, DIM> cell_count_;      ///< Number of cells along each dimension
    bool periodic_;                        ///< If true, boundaries are periodic

    std::vector<double> b_;                ///< Birth rates (b[s])
    std::vector<double> d_;                ///< Death rates (d[s])
    std::vector<std::vector<double>> dd_;  ///< Pairwise competition magnitudes: dd[s1][s2]

    std::vector<std::vector<double>> birth_x_;  ///< Birth kernel x-values for each species
    std::vector<std::vector<double>> birth_y_;  ///< Birth kernel y-values for each species

    std::vector<std::vector<std::vector<double>>> death_x_;  ///< Death kernel x-values for species pairs
    std::vector<std::vector<std::vector<double>>> death_y_;  ///< Death kernel y-values for species pairs

    std::vector<std::vector<double>> cutoff_;              ///< Maximum interaction distance: cutoff[s1][s2]
    std::vector<std::vector<double>> cutoff_sq_;           ///< Squared cutoff for fast comparison
    std::vector<std::vector<std::array<int, DIM>>> cull_;  ///< Neighbor cell search range: cull[s1][s2][dim]

    // Precomputed uniform-grid interpolation data for death kernels
    struct UniformInterpData {
        double x0;      ///< First x value
        double dx;      ///< Step between consecutive x values
        double inv_dx;  ///< Precomputed 1.0/dx
        int n;          ///< Number of data points
        bool is_uniform;///< Whether this kernel has uniform spacing
    };
    std::vector<std::vector<UniformInterpData>> death_interp_;  ///< [s1][s2]
    std::vector<UniformInterpData> birth_interp_;               ///< [s]

    struct UniformInterpDataSq {
        double r_sq_0;
        double d_r_sq;
        double inv_d_r_sq;
        int n;
        std::vector<double> y_data;
    };
    std::vector<std::vector<UniformInterpDataSq>> death_interp_sq_; ///< [s1][s2]
    std::vector<std::vector<std::vector<std::vector<int>>>> neighbor_list_; ///< [s1][s2][cIdxFlat] -> list of nIdxFlat

    std::vector<double> cell_death_delta_buffer_;
    std::vector<int> active_cells_;

    // Helper for 1D flattened access
    inline int getSpeciesCellIdx(int s, int cIdxFlat) const {
        return s * total_num_cells_ + cIdxFlat;
    }
    inline int getCoordIdx(int s, int d, int cIdxFlat) const {
        return (s * DIM + d) * total_num_cells_ + cIdxFlat;
    }

    // Global SoA: State for each cell
    std::vector<int> cell_population_;               ///< size: M_ * total_num_cells_
    std::vector<double> cell_birth_rate_by_species_; ///< size: M_ * total_num_cells_
    std::vector<double> cell_death_rate_by_species_; ///< size: M_ * total_num_cells_
    std::vector<double> cell_birth_rate_;            ///< size: total_num_cells_
    std::vector<double> cell_death_rate_;            ///< size: total_num_cells_

    // Global SoA: Particle data per cell
    std::vector<std::vector<double>> cell_coords_;               ///< size: M_ * DIM * total_num_cells_
    std::vector<std::vector<double>> cell_particle_death_rates_; ///< size: M_ * total_num_cells_

    // Working buffers to avoid allocation during events
    std::vector<double> dist_buffer_;
    std::vector<int> species_pop_;
    int total_num_cells_;           ///< Total number of cells (product of cell_count[dim])

    double total_birth_rate_ = {0.0};  ///< Global sum of birth rates for event selection
    double total_death_rate_ = {0.0};  ///< Global sum of death rates for event selection
    int total_population_ = {0};       ///< Total population count across all species

    std::mt19937 rng_;       ///< Random number generator
    std::uniform_real_distribution<double> dist_uniform_{0.0, 1.0};
    double time_ = {0.0};    ///< Simulation time
    int event_count_ = {0};  ///< Total number of events processed

    std::chrono::system_clock::time_point init_time_;  ///< Real-time simulation start point
    double realtime_limit_;                            ///< Real-time limit for simulation (in seconds)
    bool realtime_limit_reached_ = {false};            ///< Flag indicating if the real-time limit was reached

    // Precomputed inverse cell sizes for fast coordinate-to-cell mapping
    std::array<double, DIM> cell_size_inv_;  ///< cell_count_[d] / area_length_[d]

    // Fenwick trees for O(log N) global event sampling
    FenwickTree birth_tree_;
    FenwickTree death_tree_;

    /**
     * @brief Main constructor. Initializes the grid and simulation parameters.
     */
    Grid(int M, const std::array<double, DIM> &areaLen, const std::array<int, DIM> &cellCount, bool isPeriodic,
         const std::vector<double> &birthRates, const std::vector<double> &deathRates,
         const std::vector<double> &ddMatrix, const std::vector<std::vector<double>> &birthX,
         const std::vector<std::vector<double>> &birthY, const std::vector<std::vector<std::vector<double>>> &deathX,
         const std::vector<std::vector<std::vector<double>>> &deathY, const std::vector<double> &cutoffs, int seed,
         double rtimeLimit);

    /**
     * @brief Converts a multi-dimensional cell index to a flat index
     * @param idx Multi-dimensional cell index
     * @return Flattened one-dimensional index
     */
    int flattenIdx(const std::array<int, DIM> &idx) const;

    /**
     * @brief Converts a flat index to a multi-dimensional cell index
     * @param cellIndex Flattened one-dimensional index
     * @return Multi-dimensional cell index
     */
    std::array<int, DIM> unflattenIdx(int cellIndex) const;

    /**
     * @brief Wraps or clamps an index in a specific dimension
     * @param i The index to wrap/clamp
     * @param dim The dimension (0 to DIM-1)
     * @return The wrapped/clamped index
     */
    int wrapIndex(int i, int dim) const;

    /**
     * @brief Checks if a multi-dimensional index is within the domain
     * @param idx Multi-dimensional index to check
     * @return true if index is within domain, false otherwise
     */
    bool inDomain(const std::array<int, DIM> &idx) const;

    /**
     * @brief Gets a reference to the cell at the given raw index (handles wrapping)
     * @param raw Raw multi-dimensional index (may be outside domain if periodic)
     * @return Reference to the cell
     */
    int cellIdx(const std::array<double, DIM> &pos) const;

    /**
     * @brief Evaluates the birth kernel for a species at a given quantile
     * @param s Species index
     * @param x Quantile value in [0,1]
     * @return The radius corresponding to the quantile
     */
    double evalBirthKernel(int s, double x) const;

    /**
     * @brief Evaluates the death kernel for a species pair at a given distance
     * @param s1 First species index
     * @param s2 Second species index
     * @param dist Distance between individuals
     * @return The kernel value at the given distance
     */
    double evalDeathKernel(int s1, int s2, double dist) const;
    double evalDeathKernelSq(int s1, int s2, double distSq) const;

    /**
     * @brief Create a random unit vector in DIM dimensions.
     *        (In 1D, returns either +1 or -1).
     *
     * @param rng Random number generator
     * @return A random unit vector
     */
    std::array<double, DIM> randomUnitVector(std::mt19937 &rng);

    /**
     * @brief Place a new particle of species s at position inPos (wrapping or discarding
     *        if outside domain and periodic==true or false). Update local and global rates.
     *
     * @param s Species index
     * @param inPos The desired real-space position
     */
    void spawn_at(int s, const std::array<double, DIM> &inPos);

    /**
     * @brief Remove exactly one particle of species s in cell cIdx at the specified index.
     *        Updates local and global rates.
     *
     * @param s Species index
     * @param cIdx The cell index array
     * @param victimIdx The index of the particle to remove within the cell's species array
     */
    void kill_at(int s, const std::array<int, DIM> &cIdx, int victimIdx);

    /**
     * @brief Removes the interactions contributed by a single particle
     *        (sVictim, victimIdx) in cell cIdx.
     *
     * For each neighbor cell (within cull[sVictim][s2]), we subtract i->j
     * and j->i interactions from occupant j in species s2.
     *
     * @param cIdx The cell index array containing the victim particle
     * @param sVictim Species index of the victim particle
     * @param victimIdx Index of the victim particle within the cell's species array
     */
    void removeInteractionsOfParticle(const std::array<int, DIM> &cIdx, int sVictim, int victimIdx);

    /**
     * @brief Loop over a list of positions for each species and call spawn_at.
     *        Useful to initialize a population or add partial subpopulations.
     *
     * @param initCoords initCoords[s] is a vector of positions for species s.
     */
    void placePopulation(const std::vector<std::vector<std::array<double, DIM>>> &initCoords);

    /**
     * @brief Perform a random spawn event:
     *   1) pick cell by cellBirthRate
     *   2) pick species by cellBirthRateBySpecies
     *   3) pick a random parent occupant
     *   4) sample a radius from the species' birth kernel, pick random direction
     *   5) call spawn_at(...)
     */
    void spawn_random();

    /**
     * @brief Perform a random kill event:
     *   1) pick cell by cellDeathRate
     *   2) pick species by cellDeathRateBySpecies
     *   3) pick a victim occupant by that species' per-particle deathRates
     *   4) call kill_at(...)
     */
    void kill_random();

    /**
     * @brief Perform one birth or death event, chosen by ratio of total_birth_rate
     *        to total_death_rate, then sample the waiting time exponentially.
     *
     * Does nothing if total_birth_rate + total_death_rate < 1e-12.
     */
    void make_event();

    /**
     * @brief Run a fixed number of events (birth or death).
     *
     * Terminates early if the real-time limit is reached.
     *
     * @param events Number of events to perform.
     */
    void run_events(int events);

    /**
     * @brief Run the simulation until the specified amount of simulated time has elapsed.
     *
     * Terminates if real-time limit is reached or if total rates vanish.
     *
     * @param duration How much additional simulation time to run.
     */
    void run_for(double duration);

    /**
     * @brief Returns aggregated coordinates for all particles for each species.
     *
     * For each species s (0 <= s < M), returns a vector of particle coordinates.
     * The return type is a vector (per species) of std::array<double, DIM>.
     *
     * @return A vector of vectors containing coordinates for each particle of each species
     */
    std::vector<std::vector<std::array<double, DIM>>> get_all_particle_coords() const;

    /**
     * @brief Gets the coordinates of particles of a given species in a given cell.
     */
    std::vector<std::array<double, DIM>> get_cell_coords(int cell_idx, int species_idx) const;
    std::vector<double> get_cell_death_rates(int cell_idx, int species_idx) const;

    /**
     * @brief Returns death rates for all particles for each species.
     *
     * For each species s (0 <= s < M), returns a vector of particle death rates.
     * The return type is a vector (per species) of doubles.
     * The order matches the order of coordinates returned by get_all_particle_coords().
     *
     * @return A vector of vectors containing death rates for each particle of each species
     */
    std::vector<std::vector<double>> get_all_particle_death_rates() const;
};

extern template class Grid<1>;
extern template class Grid<2>;
extern template class Grid<3>;
