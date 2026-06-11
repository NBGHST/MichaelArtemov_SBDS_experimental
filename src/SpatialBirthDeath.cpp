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
    
    cell_population_.resize(M_ * total_num_cells_, 0);
    cell_birth_rate_by_species_.resize(M_ * total_num_cells_, 0.0);
    cell_death_rate_by_species_.resize(M_ * total_num_cells_, 0.0);
    cell_birth_rate_.resize(total_num_cells_, 0.0);
    cell_death_rate_.resize(total_num_cells_, 0.0);

    cell_coords_.resize(M_ * DIM * total_num_cells_);
    cell_particle_death_rates_.resize(M_ * total_num_cells_);

    // Initialize Fenwick trees
    birth_tree_.init(total_num_cells_);
    death_tree_.init(total_num_cells_);
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
int Grid<DIM>::cellIdx(const std::array<double, DIM> &pos) const {
    std::array<int, DIM> cIdx;
    for (int d = 0; d < DIM; ++d) {
        int c = static_cast<int>(std::floor(pos[d] * cell_size_inv_[d]));
        if (c >= cell_count_[d]) {
            c = cell_count_[d] - 1;
        }
        cIdx[d] = wrapIndex(c, d);
    }
    return flattenIdx(cIdx);
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
    int cIdxFlat = flattenIdx(cIdx);
    int sCIdxFlat = getSpeciesCellIdx(s, cIdxFlat);

    for (int d = 0; d < DIM; ++d) {
        cell_coords_[getCoordIdx(s, d, cIdxFlat)].push_back(pos[d]);
    }
    cell_particle_death_rates_[sCIdxFlat].push_back(d_[s]);
    ++cell_population_[sCIdxFlat];
    ++total_population_;
    ++species_pop_[s];
    cell_birth_rate_by_species_[sCIdxFlat] += b_[s];
    cell_birth_rate_[cIdxFlat] += b_[s];
    total_birth_rate_ += b_[s];
    cell_death_rate_by_species_[sCIdxFlat] += d_[s];
    cell_death_rate_[cIdxFlat] += d_[s];
    total_death_rate_ += d_[s];
    
    std::array<double, DIM> posNew;
    for (int d = 0; d < DIM; ++d) {
        posNew[d] = cell_coords_[getCoordIdx(s, d, cIdxFlat)].back();
    }
    int newIdx = static_cast<int>(cell_coords_[getCoordIdx(s, 0, cIdxFlat)].size()) - 1;

    birth_tree_.update(cIdxFlat, b_[s]);
    death_tree_.update(cIdxFlat, d_[s]);

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
            std::array<int, DIM> wrappedNIdx = nIdx;
            if (periodic_) {
                for (int d = 0; d < DIM; ++d) {
                    wrappedNIdx[d] = (wrappedNIdx[d] % cell_count_[d] + cell_count_[d]) % cell_count_[d];
                }
            }
            int nIdxFlat = flattenIdx(wrappedNIdx);
            int s2NIdxFlat = getSpeciesCellIdx(s2, nIdxFlat);
            const auto& coords_s2_0 = cell_coords_[getCoordIdx(s2, 0, nIdxFlat)];
            const int nParticles = static_cast<int>(coords_s2_0.size());
            if (nParticles == 0) return;

            dist_buffer_.resize(nParticles);

            if constexpr (DIM == 1) {
                const double* __restrict__ p0 = coords_s2_0.data();
                const double pn0 = posNew[0];
                const double len0 = area_length_[0];
                const bool per = periodic_;
if (per) {
#pragma GCC ivdep
                    for (int j = 0; j < nParticles; ++j) {
                        double diff = std::abs(pn0 - p0[j]);
                        dist_buffer_[j] = std::min(diff, len0 - diff);
                    }
                } else {
#pragma GCC ivdep
                    for (int j = 0; j < nParticles; ++j) {
                        dist_buffer_[j] = std::abs(pn0 - p0[j]);
                    }
                }
                if (nIdxFlat == cIdxFlat && s2 == s) dist_buffer_[newIdx] = 1e9;
            } else if constexpr (DIM == 2) {
                const auto& coords_s2_1 = cell_coords_[getCoordIdx(s2, 1, nIdxFlat)];
                const double* __restrict__ p0 = coords_s2_0.data();
                const double* __restrict__ p1 = coords_s2_1.data();
                const double pn0 = posNew[0];
                const double pn1 = posNew[1];
                const double len0 = area_length_[0];
                const double len1 = area_length_[1];
                const bool per = periodic_;
if (per) {
#pragma GCC ivdep
                    for (int j = 0; j < nParticles; ++j) {
                        double diff0 = std::abs(pn0 - p0[j]);
                        diff0 = std::min(diff0, len0 - diff0);
                        double diff1 = std::abs(pn1 - p1[j]);
                        diff1 = std::min(diff1, len1 - diff1);
                        dist_buffer_[j] = diff0*diff0 + diff1*diff1;
                    }
                } else {
#pragma GCC ivdep
                    for (int j = 0; j < nParticles; ++j) {
                        double diff0 = pn0 - p0[j];
                        double diff1 = pn1 - p1[j];
                        dist_buffer_[j] = diff0*diff0 + diff1*diff1;
                    }
                }
                if (nIdxFlat == cIdxFlat && s2 == s) dist_buffer_[newIdx] = 1e9;
            } else {
                const auto& coords_s2_1 = cell_coords_[getCoordIdx(s2, 1, nIdxFlat)];
                const auto& coords_s2_2 = cell_coords_[getCoordIdx(s2, 2, nIdxFlat)];
                const double* __restrict__ p0 = coords_s2_0.data();
                const double* __restrict__ p1 = coords_s2_1.data();
                const double* __restrict__ p2 = coords_s2_2.data();
                const double pn0 = posNew[0];
                const double pn1 = posNew[1];
                const double pn2 = posNew[2];
                const double len0 = area_length_[0];
                const double len1 = area_length_[1];
                const double len2 = area_length_[2];
                const bool per = periodic_;
if (per) {
#pragma GCC ivdep
                    for (int j = 0; j < nParticles; ++j) {
                        double diff0 = std::abs(pn0 - p0[j]);
                        diff0 = std::min(diff0, len0 - diff0);
                        double diff1 = std::abs(pn1 - p1[j]);
                        diff1 = std::min(diff1, len1 - diff1);
                        double diff2 = std::abs(pn2 - p2[j]);
                        diff2 = std::min(diff2, len2 - diff2);
                        dist_buffer_[j] = diff0*diff0 + diff1*diff1 + diff2*diff2;
                    }
                } else {
#pragma GCC ivdep
                    for (int j = 0; j < nParticles; ++j) {
                        double diff0 = pn0 - p0[j];
                        double diff1 = pn1 - p1[j];
                        double diff2 = pn2 - p2[j];
                        dist_buffer_[j] = diff0*diff0 + diff1*diff1 + diff2*diff2;
                    }
                }
                if (nIdxFlat == cIdxFlat && s2 == s) dist_buffer_[newIdx] = 1e9;
            }

            double delta_neigh = 0.0;
            double delta_cell = 0.0;

            const double cutoffSq_s_s2 = cutoff_s_s2 * cutoff_s_s2 + 1e-12;
            const double cutoffSq_s2_s = cutoff_s2_s * cutoff_s2_s + 1e-12;
            for (int j = 0; j < nParticles; ++j) {
                double dist = dist_buffer_[j];
                double distSq = dist;
                if constexpr (DIM > 1) {
                    distSq = dist; // dist_buffer_ holds squared distance for DIM > 1
                } else {
                    distSq = dist * dist; // For 1D, dist_buffer_ holds absolute distance
                }
                
                if (distSq <= cutoffSq_s_s2 || distSq <= cutoffSq_s2_s) {
                    double actual_dist = dist;
                    if constexpr (DIM > 1) {
                        actual_dist = std::sqrt(distSq);
                    }
                    if (actual_dist <= cutoff_s_s2) {
                        const double inter_ij = dd_s_s2 * evalDeathKernel(s, s2, actual_dist);
                        cell_particle_death_rates_[s2NIdxFlat][j] += inter_ij;
                        delta_neigh += inter_ij;
                    }
                    if (actual_dist <= cutoff_s2_s) {
                        const double inter_ji = dd_s2_s * evalDeathKernel(s2, s, actual_dist);
                        delta_cell += inter_ji;
                    }
                }
            }

            if (delta_neigh > 0.0) {
                cell_death_rate_by_species_[s2NIdxFlat] += delta_neigh;
                cell_death_rate_[nIdxFlat] += delta_neigh;
                total_death_rate_ += delta_neigh;
                death_tree_.update(nIdxFlat, delta_neigh);
            }
            if (delta_cell > 0.0) {
                cell_particle_death_rates_[sCIdxFlat][newIdx] += delta_cell;
                cell_death_rate_by_species_[sCIdxFlat] += delta_cell;
                cell_death_rate_[cIdxFlat] += delta_cell;
                total_death_rate_ += delta_cell;
                death_tree_.update(cIdxFlat, delta_cell);
            }
        });
    }
}

template <int DIM>
void Grid<DIM>::kill_at(int s, const std::array<int, DIM> &cIdx, int victimIdx) {
    int cIdxFlat = flattenIdx(cIdx);
    int sCIdxFlat = getSpeciesCellIdx(s, cIdxFlat);
    const double victimRate = cell_particle_death_rates_[sCIdxFlat][victimIdx];
    --cell_population_[sCIdxFlat];
    --total_population_;
    --species_pop_[s];
    cell_death_rate_by_species_[sCIdxFlat] -= victimRate;
    cell_death_rate_[cIdxFlat] -= victimRate;
    total_death_rate_ -= victimRate;
    cell_birth_rate_by_species_[sCIdxFlat] -= b_[s];
    cell_birth_rate_[cIdxFlat] -= b_[s];
    total_birth_rate_ -= b_[s];

    birth_tree_.update(cIdxFlat, -b_[s]);
    death_tree_.update(cIdxFlat, -victimRate);

    std::array<double, DIM> posVictim;
    for (int d = 0; d < DIM; ++d) {
        posVictim[d] = cell_coords_[getCoordIdx(s, d, cIdxFlat)][victimIdx];
    }
    removeInteractionsOfParticle(cIdx, s, victimIdx);
    const int lastIdx = static_cast<int>(cell_coords_[getCoordIdx(s, 0, cIdxFlat)].size()) - 1;
    if (victimIdx != lastIdx) {
        for (int d = 0; d < DIM; ++d) {
            cell_coords_[getCoordIdx(s, d, cIdxFlat)][victimIdx] = cell_coords_[getCoordIdx(s, d, cIdxFlat)][lastIdx];
        }
        cell_particle_death_rates_[sCIdxFlat][victimIdx] = cell_particle_death_rates_[sCIdxFlat][lastIdx];
    }
    for (int d = 0; d < DIM; ++d) {
        cell_coords_[getCoordIdx(s, d, cIdxFlat)].pop_back();
    }
    cell_particle_death_rates_[sCIdxFlat].pop_back();
}

template <int DIM>
void Grid<DIM>::removeInteractionsOfParticle(const std::array<int, DIM> &cIdx, int sVictim, int victimIdx) {
    int cIdxFlat = flattenIdx(cIdx);
    std::array<double, DIM> posVictim;
    for (int d = 0; d < DIM; ++d) {
        posVictim[d] = cell_coords_[getCoordIdx(sVictim, d, cIdxFlat)][victimIdx];
    }
    for (int s2 = 0; s2 < M_; ++s2) {
        const double cutoff_sv_s2 = cutoff_[sVictim][s2];
        const double dd_sv_s2 = dd_[sVictim][s2];
        auto range = cull_[sVictim][s2];

        forNeighbors<DIM>(cIdx, range, [&](const std::array<int, DIM> &nIdx) {
            if (!periodic_ && !inDomain(nIdx)) {
                return;
            }
            std::array<int, DIM> wrappedNIdx = nIdx;
            if (periodic_) {
                for (int d = 0; d < DIM; ++d) {
                    wrappedNIdx[d] = (wrappedNIdx[d] % cell_count_[d] + cell_count_[d]) % cell_count_[d];
                }
            }
            int nIdxFlat = flattenIdx(wrappedNIdx);
            int s2NIdxFlat = getSpeciesCellIdx(s2, nIdxFlat);
            const auto& coords_s2_0 = cell_coords_[getCoordIdx(s2, 0, nIdxFlat)];
            const int nParticles = static_cast<int>(coords_s2_0.size());
            if (nParticles == 0) return;

            dist_buffer_.resize(nParticles);

            if constexpr (DIM == 1) {
                const double* __restrict__ p0 = coords_s2_0.data();
                const double pv0 = posVictim[0];
                const double len0 = area_length_[0];
                const bool per = periodic_;
if (per) {
#pragma GCC ivdep
                    for (int j = 0; j < nParticles; ++j) {
                        double diff = std::abs(pv0 - p0[j]);
                        dist_buffer_[j] = std::min(diff, len0 - diff);
                    }
                } else {
#pragma GCC ivdep
                    for (int j = 0; j < nParticles; ++j) {
                        dist_buffer_[j] = std::abs(pv0 - p0[j]);
                    }
                }
                if (nIdxFlat == cIdxFlat && s2 == sVictim) dist_buffer_[victimIdx] = 1e9;
            } else if constexpr (DIM == 2) {
                const auto& coords_s2_1 = cell_coords_[getCoordIdx(s2, 1, nIdxFlat)];
                const double* __restrict__ p0 = coords_s2_0.data();
                const double* __restrict__ p1 = coords_s2_1.data();
                const double pv0 = posVictim[0];
                const double pv1 = posVictim[1];
                const double len0 = area_length_[0];
                const double len1 = area_length_[1];
                const bool per = periodic_;
if (per) {
#pragma GCC ivdep
                    for (int j = 0; j < nParticles; ++j) {
                        double diff0 = std::abs(pv0 - p0[j]);
                        diff0 = std::min(diff0, len0 - diff0);
                        double diff1 = std::abs(pv1 - p1[j]);
                        diff1 = std::min(diff1, len1 - diff1);
                        dist_buffer_[j] = diff0*diff0 + diff1*diff1;
                    }
                } else {
#pragma GCC ivdep
                    for (int j = 0; j < nParticles; ++j) {
                        double diff0 = pv0 - p0[j];
                        double diff1 = pv1 - p1[j];
                        dist_buffer_[j] = diff0*diff0 + diff1*diff1;
                    }
                }
                if (nIdxFlat == cIdxFlat && s2 == sVictim) dist_buffer_[victimIdx] = 1e9;
            } else {
                const auto& coords_s2_1 = cell_coords_[getCoordIdx(s2, 1, nIdxFlat)];
                const auto& coords_s2_2 = cell_coords_[getCoordIdx(s2, 2, nIdxFlat)];
                const double* __restrict__ p0 = coords_s2_0.data();
                const double* __restrict__ p1 = coords_s2_1.data();
                const double* __restrict__ p2 = coords_s2_2.data();
                const double pv0 = posVictim[0];
                const double pv1 = posVictim[1];
                const double pv2 = posVictim[2];
                const double len0 = area_length_[0];
                const double len1 = area_length_[1];
                const double len2 = area_length_[2];
                const bool per = periodic_;
if (per) {
#pragma GCC ivdep
                    for (int j = 0; j < nParticles; ++j) {
                        double diff0 = std::abs(pv0 - p0[j]);
                        diff0 = std::min(diff0, len0 - diff0);
                        double diff1 = std::abs(pv1 - p1[j]);
                        diff1 = std::min(diff1, len1 - diff1);
                        double diff2 = std::abs(pv2 - p2[j]);
                        diff2 = std::min(diff2, len2 - diff2);
                        dist_buffer_[j] = diff0*diff0 + diff1*diff1 + diff2*diff2;
                    }
                } else {
#pragma GCC ivdep
                    for (int j = 0; j < nParticles; ++j) {
                        double diff0 = pv0 - p0[j];
                        double diff1 = pv1 - p1[j];
                        double diff2 = pv2 - p2[j];
                        dist_buffer_[j] = diff0*diff0 + diff1*diff1 + diff2*diff2;
                    }
                }
                if (nIdxFlat == cIdxFlat && s2 == sVictim) dist_buffer_[victimIdx] = 1e9;
            }

            double delta_neigh = 0.0;

            const double cutoffSq_sv_s2 = cutoff_sv_s2 * cutoff_sv_s2 + 1e-12;
            for (int j = 0; j < nParticles; ++j) {
                double dist = dist_buffer_[j];
                double distSq = dist;
                if constexpr (DIM > 1) {
                    distSq = dist;
                } else {
                    distSq = dist * dist;
                }
                if (distSq <= cutoffSq_sv_s2) {
                    double actual_dist = dist;
                    if constexpr (DIM > 1) {
                        actual_dist = std::sqrt(distSq);
                    }
                    if (actual_dist <= cutoff_sv_s2) {
                        const double inter_ij = dd_sv_s2 * evalDeathKernel(sVictim, s2, actual_dist);
                        cell_particle_death_rates_[s2NIdxFlat][j] -= inter_ij;
                        delta_neigh -= inter_ij;
                    }
                }
            }

            if (delta_neigh < 0.0) {
                cell_death_rate_by_species_[s2NIdxFlat] += delta_neigh;
                cell_death_rate_[nIdxFlat] += delta_neigh;
                total_death_rate_ += delta_neigh;
                death_tree_.update(nIdxFlat, delta_neigh);
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
    const int parentCellIndex = birth_tree_.sample(rng_, total_birth_rate_);
    double acc_species = 0.0;
    const double target_species = std::uniform_real_distribution<double>(0.0, cell_birth_rate_[parentCellIndex])(rng_);
    int s = 0;
    for (int i = 0; i < M_; ++i) {
        acc_species += cell_birth_rate_by_species_[getSpeciesCellIdx(i, parentCellIndex)];
        if (target_species < acc_species) {
            s = i;
            break;
        }
    }
    if (cell_population_[getSpeciesCellIdx(s, parentCellIndex)] == 0) {
        return;
    }
    const int parentIdx = std::uniform_int_distribution<int>(0, cell_population_[getSpeciesCellIdx(s, parentCellIndex)] - 1)(rng_);
    std::array<double, DIM> parentPos;
    for (int d = 0; d < DIM; ++d) {
        parentPos[d] = cell_coords_[getCoordIdx(s, d, parentCellIndex)][parentIdx];
    }
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
    const int cellIndex = death_tree_.sample(rng_, total_death_rate_);
    double acc_species = 0.0;
    const double target_species = std::uniform_real_distribution<double>(0.0, cell_death_rate_[cellIndex])(rng_);
    int s = 0;
    for (int i = 0; i < M_; ++i) {
        acc_species += cell_death_rate_by_species_[getSpeciesCellIdx(i, cellIndex)];
        if (target_species < acc_species) {
            s = i;
            break;
        }
    }
    if (cell_population_[getSpeciesCellIdx(s, cellIndex)] == 0) {
        return;
    }
    const int victimIdx = sample_discrete<true>(cell_particle_death_rates_[getSpeciesCellIdx(s, cellIndex)], rng_, cell_death_rate_by_species_[getSpeciesCellIdx(s, cellIndex)]);
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
std::vector<std::array<double, DIM>> Grid<DIM>::get_cell_coords(int cell_idx, int species_idx) const {
    const int nParticles = cell_population_[getSpeciesCellIdx(species_idx, cell_idx)];
    std::vector<std::array<double, DIM>> res(nParticles);
    for (int j = 0; j < nParticles; ++j) {
        for (int d = 0; d < DIM; ++d) {
            res[j][d] = cell_coords_[getCoordIdx(species_idx, d, cell_idx)][j];
        }
    }
    return res;
}

template <int DIM>
std::vector<double> Grid<DIM>::get_cell_death_rates(int cell_idx, int species_idx) const {
    return cell_particle_death_rates_[getSpeciesCellIdx(species_idx, cell_idx)];
}

template <int DIM>
std::vector<std::vector<std::array<double, DIM>>> Grid<DIM>::get_all_particle_coords() const {
    std::vector<std::vector<std::array<double, DIM>>> result(M_);
    for (int s = 0; s < M_; ++s) {
        result[s].reserve(species_pop_[s]);
    }
    for (int cell_idx = 0; cell_idx < total_num_cells_; ++cell_idx) {
        for (int s = 0; s < M_; ++s) {
            const int nParticles = cell_population_[getSpeciesCellIdx(s, cell_idx)];
            for (int j = 0; j < nParticles; ++j) {
                std::array<double, DIM> pos;
                for (int d = 0; d < DIM; ++d) {
                    pos[d] = cell_coords_[getCoordIdx(s, d, cell_idx)][j];
                }
                result[s].push_back(pos);
            }
        }
    }
    return result;
}

template <int DIM>
std::vector<std::vector<double>> Grid<DIM>::get_all_particle_death_rates() const {
    std::vector<std::vector<double>> result(M_);
    for (int s = 0; s < M_; ++s) {
        result[s].reserve(species_pop_[s]);
    }
    for (int cell_idx = 0; cell_idx < total_num_cells_; ++cell_idx) {
        for (int s = 0; s < M_; ++s) {
            int scIdx = getSpeciesCellIdx(s, cell_idx);
            result[s].insert(result[s].end(), cell_particle_death_rates_[scIdx].begin(), cell_particle_death_rates_[scIdx].end());
        }
    }
    return result;
}

template class Grid<1>;
template class Grid<2>;
template class Grid<3>;
