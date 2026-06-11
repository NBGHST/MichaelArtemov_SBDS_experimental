/**
 * @file SpatialBirthDeath.cpp
 * @brief Implementation of a spatial birth-death point process simulator.
 *
 * This file contains the implementation of the Grid class template and related functions
 * for simulating spatial birth-death processes in 1, 2, or 3 dimensions.
 */

#include <vector>
#include <cstddef>
#include <algorithm>
#include <array>
#include <cmath>
#include <chrono>
#include <random>
#include "../include/SpatialBirthDeath.h"

double linearInterpolate(const std::vector<double> &xdat, const std::vector<double> &ydat, double x) {
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

template <int DIM>
double distancePeriodic(const std::array<double, DIM> &point_a, const std::array<double, DIM> &point_b,
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
    return std::sqrt(sumSq);
}

template <int DIM, typename FUNC>
void forNeighbors(const std::array<int, DIM> &centerIdx, const std::array<int, DIM> &range, FUNC &&callback) {
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
template<bool known_total>
int sample_discrete(const std::vector<double>& rates, std::mt19937& rng, double total = 0) {
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

    cutoff_.resize(M_);
    for (int s1 = 0; s1 < M_; ++s1) {
        cutoff_[s1].resize(M_);
        for (int s2 = 0; s2 < M_; ++s2) {
            cutoff_[s1][s2] = cutoffs[(s1 * M_) + s2];
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
        int c = static_cast<int>(std::floor(pos[d] * cell_count_[d] / area_length_[d]));
        if (c == cell_count_[d]) {
            --c;
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
        auto cullRange = cull_[s][s2];
        forNeighbors<DIM>(cIdx, cullRange, [&](const std::array<int, DIM> &nIdx) {
            if (!periodic_ && !inDomain(nIdx)) {
                return;
            }
            Cell<DIM> &neighCell = cellAt(nIdx);
            for (int j = 0; j < static_cast<int>(neighCell.coords[s2].size()); ++j) {
                if (&neighCell == &cell && s2 == s && j == newIdx) {
                    continue;
                }
                auto &pos2 = neighCell.coords[s2][j];
                const double dist = distancePeriodic<DIM>(posNew, pos2, area_length_, periodic_);
                if (dist <= cutoff_[s][s2]) {
                    const double inter_ij = dd_[s][s2] * evalDeathKernel(s, s2, dist);
                    neighCell.deathRates[s2][j] += inter_ij;
                    neighCell.cellDeathRateBySpecies[s2] += inter_ij;
                    neighCell.cellDeathRate += inter_ij;
                    total_death_rate_ += inter_ij;
                }
                if (dist <= cutoff_[s2][s]) {
                    const double inter_ji = dd_[s2][s] * evalDeathKernel(s2, s, dist);
                    cell.deathRates[s][newIdx] += inter_ji;
                    cell.cellDeathRateBySpecies[s] += inter_ji;
                    cell.cellDeathRate += inter_ji;
                    total_death_rate_ += inter_ji;
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
        auto range = cull_[sVictim][s2];
        forNeighbors<DIM>(cIdx, range, [&](const std::array<int, DIM> &nIdx) {
            if (!periodic_ && !inDomain(nIdx)) {
                return;
            }
            Cell<DIM> &neighCell = cellAt(nIdx);
            for (int j = 0; j < static_cast<int>(neighCell.coords[s2].size()); ++j) {
                if (&neighCell == &victimCell && s2 == sVictim && j == victimIdx) {
                    continue;
                }
                auto &pos2 = neighCell.coords[s2][j];
                const double dist = distancePeriodic<DIM>(posVictim, pos2, area_length_, periodic_);
                if (dist <= cutoff_[sVictim][s2]) {
                    const double inter_ij = dd_[sVictim][s2] * evalDeathKernel(sVictim, s2, dist);
                    neighCell.deathRates[s2][j] -= inter_ij;
                    neighCell.cellDeathRateBySpecies[s2] -= inter_ij;
                    neighCell.cellDeathRate -= inter_ij;
                    total_death_rate_ -= inter_ij;
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
    std::vector<double> cellRateVec(total_num_cells_);
    for (int i = 0; i < total_num_cells_; ++i) {
        cellRateVec[i] = cells_[i].cellBirthRate;
    }
    //std::discrete_distribution<int> cellDist(cellRateVec.begin(), cellRateVec.end());
    const int parentCellIndex = sample_discrete<true>(cellRateVec, rng_, total_birth_rate_);
    Cell<DIM> &parentCell = cells_[parentCellIndex];
    //std::discrete_distribution<int> spDist(parentCell.cellBirthRateBySpecies.begin(),
    //                                       parentCell.cellBirthRateBySpecies.end());
    //const int s = spDist(rng_);
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
    std::vector<double> cellRateVec(total_num_cells_);
    for (int i = 0; i < total_num_cells_; ++i) {
        cellRateVec[i] = cells_[i].cellDeathRate;
    }
    //std::discrete_distribution<int> cellDist(cellRateVec.begin(), cellRateVec.end());
    const int cellIndex = sample_discrete<true>(cellRateVec, rng_, total_death_rate_);
    Cell<DIM> &cell = cells_[cellIndex];
    //std::discrete_distribution<int> spDist(cell.cellDeathRateBySpecies.begin(), cell.cellDeathRateBySpecies.end());
    const int s = sample_discrete<true>(cell.cellDeathRateBySpecies, rng_, cell.cellDeathRate);
    if (cell.population[s] == 0) {
        return;
    }
    //std::discrete_distribution<int> victimDist(cell.deathRates[s].begin(), cell.deathRates[s].end());
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
