/**
 * @file SpatialBirthDeath.cpp
 * @brief Implementation of a spatial birth-death point process simulator.
 *
 * This file contains the implementation of the Grid class template and related functions
 * for simulating spatial birth-death processes in 1, 2, or 3 dimensions.
 *
 * Optimizations applied:
 * - Squared-distance cutoff comparison (avoids sqrt in the common reject case)
 * - Pre-computed cell_size_inv_ for fast coordinate-to-cell mapping
 * - Scratch buffer for cell rate vectors (eliminates per-event heap allocations)
 * - Uniform-grid O(1) interpolation for death kernels
 * - Inlined hot functions via header
 * - AVX2 vectorized batch distance computation for DIM=2
 * - sample_discrete with pre-known totals
 */

#include <vector>
#include <cstddef>
#include <algorithm>
#include <array>
#include <cmath>
#include <chrono>
#include <random>
#include <numeric>
#include "../include/SpatialBirthDeath.h"

#if defined(__AVX2__)
#include <immintrin.h>
#endif

template<bool known_total>
inline int sample_discrete(const std::vector<double>& rates, std::mt19937& rng, double total = 0) {
    if constexpr (!known_total) {
        total = 0.0;
        for (double r : rates) {
            total += r;
        }
    }
    double u = std::uniform_real_distribution<double>(0.0, total)(rng);
    double acc = 0.0;
    for (int i = 0; i < (int)rates.size(); ++i) {
        acc += rates[i];
        if (u < acc) {
            return i;
        }
    }
    return rates.size() - 1;
}

/**
 * @brief Check if x-data is uniformly spaced (within tolerance).
 */
static bool isUniformSpacing(const std::vector<double>& xdat, double &x0, double &dx) {
    if (xdat.size() < 2) return false;
    x0 = xdat[0];
    dx = xdat[1] - xdat[0];
    if (dx <= 0.0) return false;
    for (size_t i = 2; i < xdat.size(); ++i) {
        double expected = x0 + i * dx;
        if (std::abs(xdat[i] - expected) > 1e-10 * dx) {
            return false;
        }
    }
    return true;
}

#if defined(__AVX2__)
/**
 * @brief AVX2 batch squared-distance computation for DIM=2, non-periodic.
 *
 * Computes squared distances from point_a to 4 points stored contiguously
 * as [x0,y0, x1,y1, x2,y2, x3,y3].
 * Returns 4 squared distances in a __m256d.
 */
static inline __m256d batchDistSq2D_nonperiodic(
    __m256d ax_broadcast, __m256d ay_broadcast,
    const double* __restrict__ coords_ptr)
{
    // Load [x0, y0, x1, y1]
    __m256d c01 = _mm256_loadu_pd(coords_ptr);
    // Load [x2, y2, x3, y3]
    __m256d c23 = _mm256_loadu_pd(coords_ptr + 4);

    // ax_broadcast = [ax, ax, ax, ax], ay_broadcast = [ay, ay, ay, ay]
    // Interleave: need [ax, ay, ax, ay]
    __m256d a01 = _mm256_unpacklo_pd(ax_broadcast, ay_broadcast); // [ax, ay, ax, ay]
    __m256d a23 = a01;

    // diff = a - c
    __m256d d01 = _mm256_sub_pd(a01, c01);
    __m256d d23 = _mm256_sub_pd(a23, c23);

    // d^2
    __m256d dsq01 = _mm256_mul_pd(d01, d01);
    __m256d dsq23 = _mm256_mul_pd(d23, d23);

    // Horizontal add pairs: [dx0^2+dy0^2, dx1^2+dy1^2, dx2^2+dy2^2, dx3^2+dy3^2]
    __m256d result = _mm256_hadd_pd(dsq01, dsq23);
    // After hadd: [dsq0, dsq2, dsq1, dsq3] - need to permute
    result = _mm256_permute4x64_pd(result, 0b11011000); // [dsq0, dsq1, dsq2, dsq3]
    return result;
}

/**
 * @brief AVX2 batch squared-distance computation for DIM=2, periodic.
 */
static inline __m256d batchDistSq2D_periodic(
    __m256d ax_broadcast, __m256d ay_broadcast,
    const double* __restrict__ coords_ptr,
    __m256d Lx_broadcast, __m256d Ly_broadcast,
    __m256d halfLx_broadcast, __m256d halfLy_broadcast)
{
    __m256d c01 = _mm256_loadu_pd(coords_ptr);
    __m256d c23 = _mm256_loadu_pd(coords_ptr + 4);

    __m256d a01 = _mm256_unpacklo_pd(ax_broadcast, ay_broadcast);
    __m256d a23 = a01;

    __m256d d01 = _mm256_sub_pd(a01, c01);
    __m256d d23 = _mm256_sub_pd(a23, c23);

    // Periodic wrapping for d01: interleaved [dx0, dy0, dx1, dy1]
    __m256d L01 = _mm256_unpacklo_pd(Lx_broadcast, Ly_broadcast);
    __m256d halfL01 = _mm256_unpacklo_pd(halfLx_broadcast, halfLy_broadcast);

    // If diff > halfL: diff -= L
    __m256d mask_pos01 = _mm256_cmp_pd(d01, halfL01, _CMP_GT_OQ);
    d01 = _mm256_sub_pd(d01, _mm256_and_pd(mask_pos01, L01));
    // If diff < -halfL: diff += L
    __m256d neg_halfL01 = _mm256_sub_pd(_mm256_setzero_pd(), halfL01);
    __m256d mask_neg01 = _mm256_cmp_pd(d01, neg_halfL01, _CMP_LT_OQ);
    d01 = _mm256_add_pd(d01, _mm256_and_pd(mask_neg01, L01));

    // Same for d23
    __m256d mask_pos23 = _mm256_cmp_pd(d23, halfL01, _CMP_GT_OQ);
    d23 = _mm256_sub_pd(d23, _mm256_and_pd(mask_pos23, L01));
    __m256d mask_neg23 = _mm256_cmp_pd(d23, neg_halfL01, _CMP_LT_OQ);
    d23 = _mm256_add_pd(d23, _mm256_and_pd(mask_neg23, L01));

    __m256d dsq01 = _mm256_mul_pd(d01, d01);
    __m256d dsq23 = _mm256_mul_pd(d23, d23);

    __m256d result = _mm256_hadd_pd(dsq01, dsq23);
    result = _mm256_permute4x64_pd(result, 0b11011000);
    return result;
}
#endif  // __AVX2__

template <int DIM>
Grid<DIM>::Grid(int M, const std::array<double, DIM> &areaLen, const std::array<int, DIM> &cellCount, bool isPeriodic,
                const std::vector<double> &birthRates, const std::vector<double> &deathRates,
                const std::vector<double> &ddMatrix, const std::vector<std::vector<double>> &birthX,
                const std::vector<std::vector<double>> &birthY,
                const std::vector<std::vector<std::vector<double>>> &deathX,
                const std::vector<std::vector<std::vector<double>>> &deathY, const std::vector<double> &cutoffs,
                int seed, double rtimeLimit)
    : M_(M),
      area_length_(areaLen),
      cell_count_(cellCount),
      periodic_(isPeriodic),
      rng_(seed),
      realtime_limit_(rtimeLimit) {

    init_time_ = std::chrono::system_clock::now();

    species_pop_.resize(M_);
    b_ = birthRates;
    d_ = deathRates;

    // Precompute inverse cell sizes for fast coordinate-to-cell mapping
    for (int d = 0; d < DIM; ++d) {
        cell_size_inv_[d] = cell_count_[d] / area_length_[d];
    }

    dd_.resize(M_);
    for (int s1 = 0; s1 < M_; ++s1) {
        dd_[s1].resize(M_);
        for (int s2 = 0; s2 < M_; ++s2) {
            dd_[s1][s2] = ddMatrix[(s1 * M_) + s2];
        }
    }

    birth_x_ = birthX;
    birth_y_ = birthY;

    death_x_.resize(M_);
    death_y_.resize(M_);
    for (int s1 = 0; s1 < M_; ++s1) {
        death_x_[s1].resize(M_);
        death_y_[s1].resize(M_);
        for (int s2 = 0; s2 < M_; ++s2) {
            death_x_[s1][s2] = deathX[s1][s2];
            death_y_[s1][s2] = deathY[s1][s2];
        }
    }

    // Precompute uniform interpolation data for death kernels
    death_interp_.resize(M_);
    for (int s1 = 0; s1 < M_; ++s1) {
        death_interp_[s1].resize(M_);
        for (int s2 = 0; s2 < M_; ++s2) {
            UniformInterpData &uid = death_interp_[s1][s2];
            uid.n = static_cast<int>(death_x_[s1][s2].size());
            uid.is_uniform = isUniformSpacing(death_x_[s1][s2], uid.x0, uid.dx);
            if (uid.is_uniform) {
                uid.inv_dx = 1.0 / uid.dx;
            } else {
                uid.x0 = 0.0;
                uid.dx = 0.0;
                uid.inv_dx = 0.0;
            }
        }
    }

    cutoff_.resize(M_);
    cutoff_sq_.resize(M_);
    for (int s1 = 0; s1 < M_; ++s1) {
        cutoff_[s1].resize(M_);
        cutoff_sq_[s1].resize(M_);
        for (int s2 = 0; s2 < M_; ++s2) {
            double c = cutoffs[(s1 * M_) + s2];
            cutoff_[s1][s2] = c;
            cutoff_sq_[s1][s2] = c * c;
        }
    }

    cull_.resize(M_);
    for (int s1 = 0; s1 < M_; ++s1) {
        cull_[s1].resize(M_);
        for (int s2 = 0; s2 < M_; ++s2) {
            for (int dim = 0; dim < DIM; ++dim) {
                const double cellSize = area_length_[dim] / cell_count_[dim];
                const int needed = static_cast<int>(std::ceil(cutoff_[s1][s2] / cellSize));
                cull_[s1][s2][dim] = std::max(needed, 3);
            }
        }
    }

    total_num_cells_ = std::accumulate(cell_count_.begin(), cell_count_.end(), 1, std::multiplies<int>());
    cells_.resize(total_num_cells_);
    for (auto &c : cells_) {
        c.initSpecies(M_);
    }

    // Pre-allocate scratch buffer
    scratch_cell_rates_.resize(total_num_cells_);
}

template <int DIM>
int Grid<DIM>::flattenIdx(const std::array<int, DIM> &idx) const {
    int f = 0;
    int mul = 1;
    for (int dim = 0; dim < DIM; ++dim) {
        f += idx[dim] * mul;
        mul *= cell_count_[dim];
    }
    return f;
}

template <int DIM>
std::array<int, DIM> Grid<DIM>::unflattenIdx(int cellIndex) const {
    std::array<int, DIM> cIdx;
    for (int dim = 0; dim < DIM; ++dim) {
        cIdx[dim] = cellIndex % cell_count_[dim];
        cellIndex /= cell_count_[dim];
    }
    return cIdx;
}

template <int DIM>
int Grid<DIM>::wrapIndex(int i, int dim) const {
    if (!periodic_) {
        return i;
    }
    const int n = cell_count_[dim];
    if (i < 0) {
        i += n;
    } else if (i >= n) {
        i -= n;
    }
    return i;
}

template <int DIM>
bool Grid<DIM>::inDomain(const std::array<int, DIM> &idx) const {
    for (int dim = 0; dim < DIM; ++dim) {
        if (idx[dim] < 0 || idx[dim] >= cell_count_[dim]) {
            return false;
        }
    }
    return true;
}

template <int DIM>
Cell<DIM> &Grid<DIM>::cellAt(const std::array<int, DIM> &raw) {
    std::array<int, DIM> w;
    for (int dim = 0; dim < DIM; ++dim) {
        w[dim] = wrapIndex(raw[dim], dim);
    }
    return cells_[flattenIdx(w)];
}

template <int DIM>
double Grid<DIM>::evalBirthKernel(int s, double x) const {
    return linearInterpolate(birth_x_[s], birth_y_[s], x);
}

template <int DIM>
double Grid<DIM>::evalDeathKernel(int s1, int s2, double dist) const {
    const auto &uid = death_interp_[s1][s2];
    if (uid.is_uniform) {
        return linearInterpolateUniform(death_y_[s1][s2].data(),
                                         uid.x0, uid.dx, uid.inv_dx,
                                         uid.n, dist);
    }
    return linearInterpolate(death_x_[s1][s2], death_y_[s1][s2], dist);
}

template <int DIM>
std::array<double, DIM> Grid<DIM>::randomUnitVector(std::mt19937 &rng) {
    std::array<double, DIM> dir;
    if constexpr (DIM == 1) {
        std::uniform_real_distribution<double> u(0.0, 1.0);
        dir[0] = (u(rng) < 0.5) ? -1.0 : 1.0;
    } else {
        std::normal_distribution<double> gauss(0.0, 1.0);
        double sumSq = 0.0;
        for (int d = 0; d < DIM; ++d) {
            const double val = gauss(rng);
            dir[d] = val;
            sumSq += val * val;
        }
        const double inv = 1.0 / std::sqrt(sumSq + 1e-14);
        for (int d = 0; d < DIM; ++d) {
            dir[d] *= inv;
        }
    }
    return dir;
}

template <int DIM>
void Grid<DIM>::spawn_at(int s, const std::array<double, DIM> &inPos) {
    std::array<double, DIM> pos = inPos;
    for (int d = 0; d < DIM; ++d) {
        if (pos[d] < 0.0 || pos[d] > area_length_[d]) {
            if (!periodic_) {
                return;
            }
            const double L = area_length_[d];
            while (pos[d] < 0.0) {
                pos[d] += L;
            }
            while (pos[d] >= L) {
                pos[d] -= L;
            }
        }
    }
    std::array<int, DIM> cIdx;
    for (int d = 0; d < DIM; ++d) {
        int c = static_cast<int>(std::floor(pos[d] * cell_size_inv_[d]));
        if (c >= cell_count_[d]) {
            c = cell_count_[d] - 1;
        }
        cIdx[d] = c;
    }
    Cell<DIM> &cell = cellAt(cIdx);
    cell.coords[s].push_back(pos);
    cell.deathRates[s].push_back(d_[s]);
    ++cell.population[s];
    ++total_population_;
    ++species_pop_[s];
    cell.cellBirthRateBySpecies[s] += b_[s];
    cell.cellBirthRate += b_[s];
    total_birth_rate_ += b_[s];
    cell.cellDeathRateBySpecies[s] += d_[s];
    cell.cellDeathRate += d_[s];
    total_death_rate_ += d_[s];
    auto &posNew = cell.coords[s].back();
    int newIdx = static_cast<int>(cell.coords[s].size()) - 1;

    for (int s2 = 0; s2 < M_; ++s2) {
        const double cutoff_s_s2 = cutoff_[s][s2];
        const double cutoff_s2_s = cutoff_[s2][s];
        const double dd_s_s2 = dd_[s][s2];
        const double dd_s2_s = dd_[s2][s];
        auto cullRange = cull_[s][s2];

        forNeighbors<DIM>(cIdx, cullRange, [&](const std::array<int, DIM> &nIdx) {
            if (!periodic_ && !inDomain(nIdx)) {
                return;
            }
            Cell<DIM> &neighCell = cellAt(nIdx);
            const int nParticles = static_cast<int>(neighCell.coords[s2].size());

#if defined(__AVX2__)
            if constexpr (DIM == 2) {
                // AVX2 fast path: process 4 particles at a time
                const double* coords_data = reinterpret_cast<const double*>(neighCell.coords[s2].data());
                int j = 0;

                if (nParticles >= 4) {
                    __m256d ax_bc = _mm256_set1_pd(posNew[0]);
                    __m256d ay_bc = _mm256_set1_pd(posNew[1]);

                    __m256d Lx_bc, Ly_bc, halfLx_bc, halfLy_bc;
                    if (periodic_) {
                        Lx_bc = _mm256_set1_pd(area_length_[0]);
                        Ly_bc = _mm256_set1_pd(area_length_[1]);
                        halfLx_bc = _mm256_set1_pd(0.5 * area_length_[0]);
                        halfLy_bc = _mm256_set1_pd(0.5 * area_length_[1]);
                    }

                    for (; j + 3 < nParticles; j += 4) {
                        // Skip self check: if any of j..j+3 is self, fall through to scalar
                        if (&neighCell == &cell && s2 == s &&
                            newIdx >= j && newIdx < j + 4) {
                            // Process this batch scalar (rare case)
                            for (int jj = j; jj < j + 4; ++jj) {
                                if (&neighCell == &cell && s2 == s && jj == newIdx) continue;
                                auto &pos2 = neighCell.coords[s2][jj];
                                const double dist = distancePeriodic<DIM>(posNew, pos2, area_length_, periodic_);
                                if (dist <= cutoff_s_s2) {
                                    const double inter_ij = dd_s_s2 * evalDeathKernel(s, s2, dist);
                                    neighCell.deathRates[s2][jj] += inter_ij;
                                    neighCell.cellDeathRateBySpecies[s2] += inter_ij;
                                    neighCell.cellDeathRate += inter_ij;
                                    total_death_rate_ += inter_ij;
                                }
                                if (dist <= cutoff_s2_s) {
                                    const double inter_ji = dd_s2_s * evalDeathKernel(s2, s, dist);
                                    cell.deathRates[s][newIdx] += inter_ji;
                                    cell.cellDeathRateBySpecies[s] += inter_ji;
                                    cell.cellDeathRate += inter_ji;
                                    total_death_rate_ += inter_ji;
                                }
                            }
                            continue;
                        }

                        __m256d distSq;
                        if (periodic_) {
                            distSq = batchDistSq2D_periodic(ax_bc, ay_bc, coords_data + j * 2,
                                                             Lx_bc, Ly_bc, halfLx_bc, halfLy_bc);
                        } else {
                            distSq = batchDistSq2D_nonperiodic(ax_bc, ay_bc, coords_data + j * 2);
                        }

                        // Extract and process
                        alignas(32) double distSq_arr[4];
                        _mm256_store_pd(distSq_arr, distSq);

                        for (int k = 0; k < 4; ++k) {
                            const int jj = j + k;
                            const double dist = std::sqrt(distSq_arr[k]);
                            if (dist <= cutoff_s_s2) {
                                const double inter_ij = dd_s_s2 * evalDeathKernel(s, s2, dist);
                                neighCell.deathRates[s2][jj] += inter_ij;
                                neighCell.cellDeathRateBySpecies[s2] += inter_ij;
                                neighCell.cellDeathRate += inter_ij;
                                total_death_rate_ += inter_ij;
                            }
                            if (dist <= cutoff_s2_s) {
                                const double inter_ji = dd_s2_s * evalDeathKernel(s2, s, dist);
                                cell.deathRates[s][newIdx] += inter_ji;
                                cell.cellDeathRateBySpecies[s] += inter_ji;
                                cell.cellDeathRate += inter_ji;
                                total_death_rate_ += inter_ji;
                            }
                        }
                    }
                }

                // Scalar tail
                for (; j < nParticles; ++j) {
                    if (&neighCell == &cell && s2 == s && j == newIdx) {
                        continue;
                    }
                    auto &pos2 = neighCell.coords[s2][j];
                    const double dist = distancePeriodic<DIM>(posNew, pos2, area_length_, periodic_);
                    if (dist <= cutoff_s_s2) {
                        const double inter_ij = dd_s_s2 * evalDeathKernel(s, s2, dist);
                        neighCell.deathRates[s2][j] += inter_ij;
                        neighCell.cellDeathRateBySpecies[s2] += inter_ij;
                        neighCell.cellDeathRate += inter_ij;
                        total_death_rate_ += inter_ij;
                    }
                    if (dist <= cutoff_s2_s) {
                        const double inter_ji = dd_s2_s * evalDeathKernel(s2, s, dist);
                        cell.deathRates[s][newIdx] += inter_ji;
                        cell.cellDeathRateBySpecies[s] += inter_ji;
                        cell.cellDeathRate += inter_ji;
                        total_death_rate_ += inter_ji;
                    }
                }
            } else
#endif  // __AVX2__
            {
                // Scalar path for DIM != 2 or no AVX2
                for (int j = 0; j < nParticles; ++j) {
                    if (&neighCell == &cell && s2 == s && j == newIdx) {
                        continue;
                    }
                    auto &pos2 = neighCell.coords[s2][j];
                    const double dist = distancePeriodic<DIM>(posNew, pos2, area_length_, periodic_);
                    if (dist <= cutoff_s_s2) {
                        const double inter_ij = dd_s_s2 * evalDeathKernel(s, s2, dist);
                        neighCell.deathRates[s2][j] += inter_ij;
                        neighCell.cellDeathRateBySpecies[s2] += inter_ij;
                        neighCell.cellDeathRate += inter_ij;
                        total_death_rate_ += inter_ij;
                    }
                    if (dist <= cutoff_s2_s) {
                        const double inter_ji = dd_s2_s * evalDeathKernel(s2, s, dist);
                        cell.deathRates[s][newIdx] += inter_ji;
                        cell.cellDeathRateBySpecies[s] += inter_ji;
                        cell.cellDeathRate += inter_ji;
                        total_death_rate_ += inter_ji;
                    }
                }
            }
        });
    }
}

template <int DIM>
void Grid<DIM>::kill_at(int s, const std::array<int, DIM> &cIdx, int victimIdx) {
    Cell<DIM> &cell = cellAt(cIdx);
    const double victimRate = cell.deathRates[s][victimIdx];
    --cell.population[s];
    --total_population_;
    --species_pop_[s];
    cell.cellDeathRateBySpecies[s] -= victimRate;
    cell.cellDeathRate -= victimRate;
    total_death_rate_ -= victimRate;
    cell.cellBirthRateBySpecies[s] -= b_[s];
    cell.cellBirthRate -= b_[s];
    total_birth_rate_ -= b_[s];
    removeInteractionsOfParticle(cIdx, s, victimIdx);
    const int lastIdx = static_cast<int>(cell.coords[s].size()) - 1;
    if (victimIdx != lastIdx) {
        cell.coords[s][victimIdx] = cell.coords[s][lastIdx];
        cell.deathRates[s][victimIdx] = cell.deathRates[s][lastIdx];
    }
    cell.coords[s].pop_back();
    cell.deathRates[s].pop_back();
}

template <int DIM>
void Grid<DIM>::removeInteractionsOfParticle(const std::array<int, DIM> &cIdx, int sVictim, int victimIdx) {
    Cell<DIM> &victimCell = cellAt(cIdx);
    auto &posVictim = victimCell.coords[sVictim][victimIdx];
    for (int s2 = 0; s2 < M_; ++s2) {
        const double cutoff_sv_s2 = cutoff_[sVictim][s2];
        const double dd_sv_s2 = dd_[sVictim][s2];
        auto range = cull_[sVictim][s2];

        forNeighbors<DIM>(cIdx, range, [&](const std::array<int, DIM> &nIdx) {
            if (!periodic_ && !inDomain(nIdx)) {
                return;
            }
            Cell<DIM> &neighCell = cellAt(nIdx);
            const int nParticles = static_cast<int>(neighCell.coords[s2].size());

#if defined(__AVX2__)
            if constexpr (DIM == 2) {
                const double* coords_data = reinterpret_cast<const double*>(neighCell.coords[s2].data());
                int j = 0;

                if (nParticles >= 4) {
                    __m256d ax_bc = _mm256_set1_pd(posVictim[0]);
                    __m256d ay_bc = _mm256_set1_pd(posVictim[1]);

                    __m256d Lx_bc, Ly_bc, halfLx_bc, halfLy_bc;
                    if (periodic_) {
                        Lx_bc = _mm256_set1_pd(area_length_[0]);
                        Ly_bc = _mm256_set1_pd(area_length_[1]);
                        halfLx_bc = _mm256_set1_pd(0.5 * area_length_[0]);
                        halfLy_bc = _mm256_set1_pd(0.5 * area_length_[1]);
                    }

                    for (; j + 3 < nParticles; j += 4) {
                        if (&neighCell == &victimCell && s2 == sVictim &&
                            victimIdx >= j && victimIdx < j + 4) {
                            for (int jj = j; jj < j + 4; ++jj) {
                                if (&neighCell == &victimCell && s2 == sVictim && jj == victimIdx) continue;
                                auto &pos2 = neighCell.coords[s2][jj];
                                const double dist = distancePeriodic<DIM>(posVictim, pos2, area_length_, periodic_);
                                if (dist <= cutoff_sv_s2) {
                                    const double inter_ij = dd_sv_s2 * evalDeathKernel(sVictim, s2, dist);
                                    neighCell.deathRates[s2][jj] -= inter_ij;
                                    neighCell.cellDeathRateBySpecies[s2] -= inter_ij;
                                    neighCell.cellDeathRate -= inter_ij;
                                    total_death_rate_ -= inter_ij;
                                }
                            }
                            continue;
                        }

                        __m256d distSq;
                        if (periodic_) {
                            distSq = batchDistSq2D_periodic(ax_bc, ay_bc, coords_data + j * 2,
                                                             Lx_bc, Ly_bc, halfLx_bc, halfLy_bc);
                        } else {
                            distSq = batchDistSq2D_nonperiodic(ax_bc, ay_bc, coords_data + j * 2);
                        }

                        // Compare distSq <= cutoffSq for early batch reject
                        __m256d cutSq_bc_val = _mm256_set1_pd(cutoff_sv_s2 * cutoff_sv_s2);
                        __m256d cmp_mask = _mm256_cmp_pd(distSq, cutSq_bc_val, _CMP_LE_OQ);
                        int mask = _mm256_movemask_pd(cmp_mask);

                        if (mask == 0) continue;  // No interactions in this batch

                        alignas(32) double distSq_arr[4];
                        _mm256_store_pd(distSq_arr, distSq);

                        for (int k = 0; k < 4; ++k) {
                            if (mask & (1 << k)) {
                                const int jj = j + k;
                                const double dist = std::sqrt(distSq_arr[k]);
                                if (dist <= cutoff_sv_s2) {
                                    const double inter_ij = dd_sv_s2 * evalDeathKernel(sVictim, s2, dist);
                                    neighCell.deathRates[s2][jj] -= inter_ij;
                                    neighCell.cellDeathRateBySpecies[s2] -= inter_ij;
                                    neighCell.cellDeathRate -= inter_ij;
                                    total_death_rate_ -= inter_ij;
                                }
                            }
                        }
                    }
                }

                // Scalar tail
                for (; j < nParticles; ++j) {
                    if (&neighCell == &victimCell && s2 == sVictim && j == victimIdx) {
                        continue;
                    }
                    auto &pos2 = neighCell.coords[s2][j];
                    const double dist = distancePeriodic<DIM>(posVictim, pos2, area_length_, periodic_);
                    if (dist <= cutoff_sv_s2) {
                        const double inter_ij = dd_sv_s2 * evalDeathKernel(sVictim, s2, dist);
                        neighCell.deathRates[s2][j] -= inter_ij;
                        neighCell.cellDeathRateBySpecies[s2] -= inter_ij;
                        neighCell.cellDeathRate -= inter_ij;
                        total_death_rate_ -= inter_ij;
                    }
                }
            } else
#endif  // __AVX2__
            {
                for (int j = 0; j < nParticles; ++j) {
                    if (&neighCell == &victimCell && s2 == sVictim && j == victimIdx) {
                        continue;
                    }
                    auto &pos2 = neighCell.coords[s2][j];
                    const double dist = distancePeriodic<DIM>(posVictim, pos2, area_length_, periodic_);
                    if (dist <= cutoff_sv_s2) {
                        const double inter_ij = dd_sv_s2 * evalDeathKernel(sVictim, s2, dist);
                        neighCell.deathRates[s2][j] -= inter_ij;
                        neighCell.cellDeathRateBySpecies[s2] -= inter_ij;
                        neighCell.cellDeathRate -= inter_ij;
                        total_death_rate_ -= inter_ij;
                    }
                }
            }
        });
    }
}

template <int DIM>
void Grid<DIM>::placePopulation(const std::vector<std::vector<std::array<double, DIM>>> &initCoords) {
    for (int s = 0; s < M_; ++s) {
        for (auto &pos : initCoords[s]) {
            spawn_at(s, pos);
        }
    }
}

template <int DIM>
void Grid<DIM>::spawn_random() {
    if (total_birth_rate_ < 1e-12) {
        return;
    }
    // Reuse scratch buffer instead of allocating each time
    for (int i = 0; i < total_num_cells_; ++i) {
        scratch_cell_rates_[i] = cells_[i].cellBirthRate;
    }
    const int parentCellIndex = sample_discrete<true>(scratch_cell_rates_, rng_, total_birth_rate_);
    Cell<DIM> &parentCell = cells_[parentCellIndex];
    const int s = sample_discrete<true>(parentCell.cellBirthRateBySpecies, rng_, parentCell.cellBirthRate);
    const int parentIdx = std::uniform_int_distribution<int>(0, parentCell.population[s] - 1)(rng_);
    auto &parentPos = parentCell.coords[s][parentIdx];
    const double u = std::uniform_real_distribution<double>(0.0, 1.0)(rng_);
    const double radius = evalBirthKernel(s, u);
    auto dir = randomUnitVector(rng_);
    for (int d = 0; d < DIM; ++d) {
        dir[d] *= radius;
    }
    std::array<double, DIM> childPos;
    for (int d = 0; d < DIM; ++d) {
        childPos[d] = parentPos[d] + dir[d];
    }
    spawn_at(s, childPos);
}

template <int DIM>
void Grid<DIM>::kill_random() {
    if (total_death_rate_ < 1e-12) {
        return;
    }
    // Reuse scratch buffer instead of allocating each time
    for (int i = 0; i < total_num_cells_; ++i) {
        scratch_cell_rates_[i] = cells_[i].cellDeathRate;
    }
    const int cellIndex = sample_discrete<true>(scratch_cell_rates_, rng_, total_death_rate_);
    Cell<DIM> &cell = cells_[cellIndex];
    const int s = sample_discrete<true>(cell.cellDeathRateBySpecies, rng_, cell.cellDeathRate);
    if (cell.population[s] == 0) {
        return;
    }
    const int victimIdx = sample_discrete<true>(cell.deathRates[s], rng_, cell.cellDeathRateBySpecies[s]);
    const std::array<int, DIM> cIdx = unflattenIdx(cellIndex);
    kill_at(s, cIdx, victimIdx);
}

template <int DIM>
void Grid<DIM>::make_event() {
    const double sumRate = total_birth_rate_ + total_death_rate_;
    if (sumRate < 1e-12) {
        return;
    }
    ++event_count_;
    std::exponential_distribution<double> expDist(sumRate);
    const double dt = expDist(rng_);
    time_ += dt;
    const double r = std::uniform_real_distribution<double>(0.0, sumRate)(rng_);
    const bool isBirth = (r < total_birth_rate_);
    if (isBirth) {
        spawn_random();
    } else {
        kill_random();
    }
}

template <int DIM>
void Grid<DIM>::run_events(int events) {
    for (int i = 0; i < events; ++i) {
        if (std::chrono::system_clock::now() > init_time_ + std::chrono::duration<double>(realtime_limit_)) {
            realtime_limit_reached_ = true;
            return;
        }
        make_event();
    }
}

template <int DIM>
void Grid<DIM>::run_for(double duration) {
    const double endTime = time_ + duration;
    while (time_ < endTime) {
        if (std::chrono::system_clock::now() > init_time_ + std::chrono::duration<double>(realtime_limit_)) {
            realtime_limit_reached_ = true;
            return;
        }
        make_event();
        if (total_birth_rate_ + total_death_rate_ < 1e-12) {
            return;
        }
    }
}

template <int DIM>
std::vector<std::vector<std::array<double, DIM>>> Grid<DIM>::get_all_particle_coords() const {
    std::vector<std::vector<std::array<double, DIM>>> result(M_);
    // Pre-allocate based on species populations
    for (int s = 0; s < M_; ++s) {
        result[s].reserve(species_pop_[s]);
    }
    for (const auto &cell : cells_) {
        for (int s = 0; s < M_; ++s) {
            result[s].insert(result[s].end(), cell.coords[s].begin(), cell.coords[s].end());
        }
    }
    return result;
}

template <int DIM>
std::vector<std::vector<double>> Grid<DIM>::get_all_particle_death_rates() const {
    std::vector<std::vector<double>> result(M_);
    // Pre-allocate based on species populations
    for (int s = 0; s < M_; ++s) {
        result[s].reserve(species_pop_[s]);
    }
    for (const auto &cell : cells_) {
        for (int s = 0; s < M_; ++s) {
            result[s].insert(result[s].end(), cell.deathRates[s].begin(), cell.deathRates[s].end());
        }
    }
    return result;
}

template class Grid<1>;
template class Grid<2>;
template class Grid<3>;

template struct Cell<1>;
template struct Cell<2>;
template struct Cell<3>;
