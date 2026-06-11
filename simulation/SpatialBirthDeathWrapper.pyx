#cython: language_level=3
#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True
"""
SpatialBirthDeathWrapper.pyx - Cython wrapper for the C++ spatial birth-death simulator.

This module provides Python classes (PyGrid1, PyGrid2, PyGrid3) that wrap the C++ Grid<DIM>
template class, allowing Python users to create and manipulate spatial birth-death
simulations in 1, 2, or 3 dimensions.
"""

from libcpp.vector cimport vector
from libcpp cimport bool
from libc.stdlib cimport malloc, free
from libc.stddef cimport size_t

# 1) Declare std::array<T,N> for needed combinations
cdef extern from "<array>" namespace "std" nogil:

    # -- double, dimension=1 --
    cdef cppclass arrayDouble1 "std::array<double, 1>":
        arrayDouble1() except +
        double& operator[](size_t) except +

    # -- double, dimension=2 --
    cdef cppclass arrayDouble2 "std::array<double, 2>":
        arrayDouble2() except +
        double& operator[](size_t) except +

    # -- double, dimension=3 --
    cdef cppclass arrayDouble3 "std::array<double, 3>":
        arrayDouble3() except +
        double& operator[](size_t) except +

    # -- int, dimension=1 --
    cdef cppclass arrayInt1 "std::array<int, 1>":
        arrayInt1() except +
        int& operator[](size_t) except +

    # -- int, dimension=2 --
    cdef cppclass arrayInt2 "std::array<int, 2>":
        arrayInt2() except +
        int& operator[](size_t) except +

    # -- int, dimension=3 --
    cdef cppclass arrayInt3 "std::array<int, 3>":
        arrayInt3() except +
        int& operator[](size_t) except +


# 2) Provide helper functions to convert Python lists to C++ std::array

# Python list with 1 float to std::array<double, 1>
cdef arrayDouble1 pyToStdArrayDouble1(list arr) except *:
    cdef arrayDouble1 result = arrayDouble1()
    result[0] = 0.0

    if len(arr) >= 1:
        result[0] = <double>arr[0]

    return result

# Python list with 2 float to to std::array<double, 2>
cdef arrayDouble2 pyToStdArrayDouble2(list arr) except *:
    cdef arrayDouble2 result = arrayDouble2()
    result[0] = 0.0
    result[1] = 0.0

    if len(arr) >= 2:
        result[0] = <double>arr[0]
        result[1] = <double>arr[1]

    return result

# Python list with 3 float to to std::array<double, 3>
cdef arrayDouble3 pyToStdArrayDouble3(list arr) except *:
    cdef arrayDouble3 result = arrayDouble3()
    result[0] = 0.0
    result[1] = 0.0
    result[2] = 0.0

    if len(arr) >= 3:
        result[0] = <double>arr[0]
        result[1] = <double>arr[1]
        result[2] = <double>arr[2]

    return result

# Python list with 1 integer to std::array<int, 1>
cdef arrayInt1 pyToStdArrayInt1(list arr) except *:
    cdef arrayInt1 result = arrayInt1()
    result[0] = 0

    if len(arr) >= 1:
        result[0] = <int>arr[0]

    return result

# Python list with 2 integers to std::array<int, 2>
cdef arrayInt2 pyToStdArrayInt2(list arr) except *:
    cdef arrayInt2 result = arrayInt2()
    result[0] = 0
    result[1] = 0

    if len(arr) >= 2:
        result[0] = <int>arr[0]
        result[1] = <int>arr[1]

    return result

# Python list with 3 integers to std::array<int, 3>
cdef arrayInt3 pyToStdArrayInt3(list arr) except *:
    cdef arrayInt3 result = arrayInt3()
    result[0] = 0
    result[1] = 0
    result[2] = 0

    if len(arr) >= 3:
        result[0] = <int>arr[0]
        result[1] = <int>arr[1]
        result[2] = <int>arr[2]

    return result

# 3) Convert Python lists to C++ std::vector<double> and nested vectors

# Python list of floats to std::vector<double>
cdef vector[double] pyListToVectorDouble(object pyList) except *:
    cdef int n = len(pyList)
    cdef vector[double] vec
    vec.resize(n)
    cdef int i
    cdef double val
    for i in range(n):
        val = pyList[i]
        vec[i] = val
    return vec

# Python list of lists of floats to std::vector<std::vector<double>>
cdef vector[vector[double]] pyListOfListToVectorVectorDouble(object pyList) except *:
    cdef int outer_size = len(pyList)
    cdef vector[vector[double]] result
    result.resize(outer_size)

    cdef int i, inner_size
    cdef object inner_list
    cdef int j
    cdef double val
    for i in range(outer_size):
        inner_list = pyList[i]
        inner_size = len(inner_list)
        result[i].resize(inner_size)
        for j in range(inner_size):
            val = inner_list[j]
            result[i][j] = val
    return result

# Python list of lists of lists of floats to std::vector<std::vector<std::vector<double>>>
cdef vector[vector[vector[double]]] pyListOfListOfListToVector3Double(object pyList) except *:
    cdef int s1_count = len(pyList)
    cdef vector[vector[vector[double]]] out3
    out3.resize(s1_count)

    cdef object second_level
    cdef object third_level
    cdef int s1, s2_count, s2
    for s1 in range(s1_count):
        second_level = pyList[s1]
        s2_count = len(second_level)
        out3[s1].resize(s2_count)
        for s2 in range(s2_count):
            third_level = second_level[s2]
            out3[s1][s2] = pyListToVectorDouble(third_level)
    return out3

# 4) These functions convert Python coordinate lists to C++ vector of vectors of arrays for use with the placePopulation method

cdef vector[vector[arrayDouble1]] pyToCoordsD1(object pyCoords) except *:
    """
    Convert Python coordinate lists to C++ format for 1D simulations.
    
    Parameters:
        pyCoords: List of lists where pyCoords[s] is a list of positions for species s.
                  Each position is a list [x].
    
    Returns:
        A vector of vectors of arrayDouble1 for use with Grid<1>::placePopulation
    """
    cdef int nSpecies = len(pyCoords)
    cdef vector[vector[arrayDouble1]] result
    result.resize(nSpecies)

    cdef int s, nPos, i
    cdef object posList
    cdef list singlePos
    for s in range(nSpecies):
        posList = pyCoords[s]
        nPos = len(posList)
        result[s].resize(nPos)
        for i in range(nPos):
            singlePos = posList[i]
            result[s][i] = pyToStdArrayDouble1(singlePos)
    return result

cdef vector[vector[arrayDouble2]] pyToCoordsD2(object pyCoords) except *:
    """
    Convert Python coordinate lists to C++ format for 2D simulations.
    
    Parameters:
        pyCoords: List of lists where pyCoords[s] is a list of positions for species s.
                  Each position is a list [x, y].
    
    Returns:
        A vector of vectors of arrayDouble2 for use with Grid<2>::placePopulation
    """
    cdef int nSpecies = len(pyCoords)
    cdef vector[vector[arrayDouble2]] result
    result.resize(nSpecies)

    cdef int s, nPos, i
    cdef object posList
    cdef list xy
    for s in range(nSpecies):
        posList = pyCoords[s]
        nPos = len(posList)
        result[s].resize(nPos)
        for i in range(nPos):
            xy = posList[i]
            result[s][i] = pyToStdArrayDouble2(xy)
    return result

cdef vector[vector[arrayDouble3]] pyToCoordsD3(object pyCoords) except *:
    """
    Convert Python coordinate lists to C++ format for 3D simulations.
    
    Parameters:
        pyCoords: List of lists where pyCoords[s] is a list of positions for species s.
                  Each position is a list [x, y, z].
    
    Returns:
        A vector of vectors of arrayDouble3 for use with Grid<3>::placePopulation
    """
    cdef int nSpecies = len(pyCoords)
    cdef vector[vector[arrayDouble3]] result
    result.resize(nSpecies)

    cdef int s, nPos, i
    cdef object posList
    cdef list xyz
    for s in range(nSpecies):
        posList = pyCoords[s]
        nPos = len(posList)
        result[s].resize(nPos)
        for i in range(nPos):
            xyz = posList[i]
            result[s][i] = pyToStdArrayDouble3(xyz)
    return result

# 6) These declarations allow Cython to access the C++ Grid template classes

cdef extern from "SpatialBirthDeath.h":

    cdef cppclass Grid1 "Grid<1>":
        Grid1(int M,
              arrayDouble1 areaLen,
              arrayInt1 cellCount,
              bool isPeriodic,
              vector[double] &birthRates,
              vector[double] &deathRates,
              vector[double] &ddMatrix,
              vector[vector[double]] &birthX,
              vector[vector[double]] &birthY,
              vector[vector[vector[double]]] &deathX,
              vector[vector[vector[double]]] &deathY,
              vector[double] &cutoffs,
              int seed,
              double rtimeLimit
        ) except +

        void placePopulation(const vector[vector[arrayDouble1]] &initCoords) except +
        void spawn_at(int s, const arrayDouble1 &inPos) except +
        void kill_at(int s, const arrayInt1 &cIdx, int victimIdx) except +

        void spawn_random() except +
        void kill_random() except +
        void make_event() except +
        void run_events(int events) except +
        void run_for(double time) except +
        vector[vector[arrayDouble1]] get_all_particle_coords() except +
        vector[arrayDouble1] get_cell_coords(int cell_idx, int species_idx) except +
        vector[double] get_cell_death_rates(int cell_idx, int species_idx) except +
        vector[vector[double]] get_all_particle_death_rates() except +

        double total_birth_rate_
        double total_death_rate_
        int    total_num_cells_
        int    total_population_
        double time_
        int    event_count_
        vector[int] cell_population_
        vector[double] cell_birth_rate_by_species_
        vector[double] cell_death_rate_by_species_
        vector[double] cell_birth_rate_
        vector[double] cell_death_rate_
        vector[int] species_pop_

    cdef cppclass Grid2 "Grid<2>":
        Grid2(int M,
              arrayDouble2 areaLen,
              arrayInt2 cellCount,
              bool isPeriodic,
              vector[double] &birthRates,
              vector[double] &deathRates,
              vector[double] &ddMatrix,
              vector[vector[double]] &birthX,
              vector[vector[double]] &birthY,
              vector[vector[vector[double]]] &deathX,
              vector[vector[vector[double]]] &deathY,
              vector[double] &cutoffs,
              int seed,
              double rtimeLimit
        ) except +

        void placePopulation(const vector[ vector[ arrayDouble2 ] ] &initCoords) except +
        void spawn_at(int s, const arrayDouble2 &inPos) except +
        void kill_at(int s, const arrayInt2 &cIdx, int victimIdx) except +

        void spawn_random() except +
        void kill_random() except +
        void make_event() except +
        void run_events(int events) except +
        void run_for(double time) except +
        vector[vector[arrayDouble2]] get_all_particle_coords() except +
        vector[arrayDouble2] get_cell_coords(int cell_idx, int species_idx) except +
        vector[double] get_cell_death_rates(int cell_idx, int species_idx) except +
        vector[vector[double]] get_all_particle_death_rates() except +

        double total_birth_rate_
        double total_death_rate_
        int    total_num_cells_
        int    total_population_
        double time_
        int    event_count_
        vector[int] cell_population_
        vector[double] cell_birth_rate_by_species_
        vector[double] cell_death_rate_by_species_
        vector[double] cell_birth_rate_
        vector[double] cell_death_rate_
        vector[int] species_pop_

    cdef cppclass Grid3 "Grid<3>":
        Grid3(int M,
              arrayDouble3 areaLen,
              arrayInt3 cellCount,
              bool isPeriodic,
              vector[double] &birthRates,
              vector[double] &deathRates,
              vector[double] &ddMatrix,
              vector[vector[double]] &birthX,
              vector[vector[double]] &birthY,
              vector[vector[vector[double]]] &deathX,
              vector[vector[vector[double]]] &deathY,
              vector[double] &cutoffs,
              int seed,
              double rtimeLimit
        ) except +

        void placePopulation(const vector[ vector[ arrayDouble3 ] ] &initCoords) except +
        void spawn_at(int s, const arrayDouble3 &inPos) except +
        void kill_at(int s, const arrayInt3 &cIdx, int victimIdx) except +

        void spawn_random() except +
        void kill_random() except +
        void make_event() except +
        void run_events(int events) except +
        void run_for(double time) except +
        vector[vector[arrayDouble3]] get_all_particle_coords() except +
        vector[arrayDouble3] get_cell_coords(int cell_idx, int species_idx) except +
        vector[double] get_cell_death_rates(int cell_idx, int species_idx) except +
        vector[vector[double]] get_all_particle_death_rates() except +

        double total_birth_rate_
        double total_death_rate_
        int    total_num_cells_
        int    total_population_
        double time_
        int    event_count_
        vector[int] cell_population_
        vector[double] cell_birth_rate_by_species_
        vector[double] cell_death_rate_by_species_
        vector[double] cell_birth_rate_
        vector[double] cell_death_rate_
        vector[int] species_pop_

# 7) Base class for PyGrid classes to avoid code duplication
cdef class PyGridBase:
    # These properties and methods are common to all dimensions
    @property
    def species_pop(self):
        return self._get_species_pop()

    @property
    def total_birth_rate(self):
        return self._get_total_birth_rate()
    
    @property
    def total_death_rate(self):
        return self._get_total_death_rate()
    
    @property
    def total_population(self):
        return self._get_total_population()
    
    @property
    def time(self):
        return self._get_time()
    
    @property
    def event_count(self):
        return self._get_event_count()
    
    def spawn_random(self):
        self._spawn_random()
    
    def kill_random(self):
        self._kill_random()
    
    def make_event(self):
        self._make_event()
    
    def run_events(self, n):
        self._run_events(n)
    
    def run_for(self, duration):
        self._run_for(duration)
    
    def get_num_cells(self):
        return self._get_num_cells()
    
    def get_cell_birth_rate(self, cell_index):
        return self._get_cell_birth_rate(cell_index)
    
    def get_cell_death_rate(self, cell_index):
        return self._get_cell_death_rate(cell_index)
    
    def get_cell_population(self, cell_index):
        return self._get_cell_population(cell_index)
    
    # Abstract methods to be implemented by dimension-specific subclasses
    cdef double _get_total_birth_rate(self):
        raise NotImplementedError()
    
    cdef double _get_total_death_rate(self):
        raise NotImplementedError()
    
    cdef int _get_total_population(self):
        raise NotImplementedError()
    
    cdef double _get_time(self):
        raise NotImplementedError()
    
    cdef int _get_event_count(self):
        raise NotImplementedError()
    
    cdef int _get_num_cells(self):
        raise NotImplementedError()
    
    cdef void _spawn_random(self):
        raise NotImplementedError()
    
    cdef void _kill_random(self):
        raise NotImplementedError()
    
    cdef void _make_event(self):
        raise NotImplementedError()
    
    cdef void _run_events(self, int n):
        raise NotImplementedError()
    
    cdef void _run_for(self, double duration):
        raise NotImplementedError()
    
    cdef double _get_cell_birth_rate(self, int cell_index):
        raise NotImplementedError()
    
    cdef double _get_cell_death_rate(self, int cell_index):
        raise NotImplementedError()
    
    cdef list _get_cell_population(self, int cell_index):
        raise NotImplementedError()
    
    cdef list _get_species_pop(self):
        raise NotImplementedError()

# 8) Updated PyGrid classes inheriting from PyGridBase

# PyGrid1 (wrapper for Grid<1>)
cdef class PyGrid1(PyGridBase):
    cdef Grid1* cpp_grid  # Owned pointer

    def __cinit__(self,
                  M,                # Number of species
                  areaLen,          # Domain size, e.g. [25.0]
                  cellCount,        # Number of cells, e.g. [25]
                  isPeriodic,       # Whether to use periodic boundaries
                  birthRates,       # Baseline birth rates for each species
                  deathRates,       # Baseline death rates for each species
                  ddMatrix,         # Flattened MxM pairwise interaction magnitudes
                  birthX,           # Birth kernel x-values (quantiles)
                  birthY,           # Birth kernel y-values (radii)
                  deathX,           # Death kernel x-values (distances)
                  deathY,           # Death kernel y-values (kernel values)
                  cutoffs,          # Flattened MxM cutoff distances
                  seed,             # Random number generator seed
                  rtimeLimit):      # Real-time limit in seconds

        cdef arrayDouble1 c_areaLen = pyToStdArrayDouble1(areaLen)
        cdef arrayInt1    c_cellCount = pyToStdArrayInt1(cellCount)
        cdef vector[double] c_birthRates = pyListToVectorDouble(birthRates)
        cdef vector[double] c_deathRates = pyListToVectorDouble(deathRates)
        cdef vector[double] c_ddMatrix   = pyListToVectorDouble(ddMatrix)
        cdef vector[vector[double]] c_birthX = pyListOfListToVectorVectorDouble(birthX)
        cdef vector[vector[double]] c_birthY = pyListOfListToVectorVectorDouble(birthY)
        cdef vector[vector[vector[double]]] c_deathX = pyListOfListOfListToVector3Double(deathX)
        cdef vector[vector[vector[double]]] c_deathY = pyListOfListOfListToVector3Double(deathY)
        cdef vector[double] c_cutoffs = pyListToVectorDouble(cutoffs)

        self.cpp_grid = new Grid1(
            M,
            c_areaLen,
            c_cellCount,
            <bool>isPeriodic,
            c_birthRates,
            c_deathRates,
            c_ddMatrix,
            c_birthX,
            c_birthY,
            c_deathX,
            c_deathY,
            c_cutoffs,
            seed,
            <double>rtimeLimit
        )

    def __dealloc__(self):
        if self.cpp_grid != NULL:
            del self.cpp_grid
            self.cpp_grid = NULL

    def placePopulation(self, initCoords):
        cdef vector[ vector[arrayDouble1] ] c_init = pyToCoordsD1(initCoords)
        self.cpp_grid.placePopulation(c_init)

    def spawn_at(self, s, pos):
        cdef arrayDouble1 cpos = pyToStdArrayDouble1(pos)
        self.cpp_grid.spawn_at(s, cpos)

    def kill_at(self, s, cell_idx, victimIdx):
        cdef arrayInt1 cc = pyToStdArrayInt1(cell_idx)
        self.cpp_grid.kill_at(s, cc, victimIdx)

    # Implement abstract methods from PyGridBase
    cdef double _get_total_birth_rate(self):
        return self.cpp_grid.total_birth_rate_
    
    cdef double _get_total_death_rate(self):
        return self.cpp_grid.total_death_rate_
    
    cdef int _get_total_population(self):
        return self.cpp_grid.total_population_
    
    cdef list _get_species_pop(self):
        cdef vector[int] pop = self.cpp_grid.species_pop_
        cdef int m = pop.size()
        cdef list out = [0] * m
        for s in range(m):
            out[s] = pop[s]
        return out
    
    cdef double _get_time(self):
        return self.cpp_grid.time_
    
    cdef int _get_event_count(self):
        return self.cpp_grid.event_count_
    
    cdef int _get_num_cells(self):
        return self.cpp_grid.total_num_cells_
    
    cdef void _spawn_random(self):
        self.cpp_grid.spawn_random()
    
    cdef void _kill_random(self):
        self.cpp_grid.kill_random()
    
    cdef void _make_event(self):
        self.cpp_grid.make_event()
    
    cdef void _run_events(self, int n):
        self.cpp_grid.run_events(n)
    
    cdef void _run_for(self, double duration):
        self.cpp_grid.run_for(duration)
    
    cdef double _get_cell_birth_rate(self, int cell_index):
        return self.cpp_grid.cell_birth_rate_[cell_index]
    
    cdef double _get_cell_death_rate(self, int cell_index):
        return self.cpp_grid.cell_death_rate_[cell_index]
    
    cdef list _get_cell_population(self, int cell_index):
        cdef int m = self.cpp_grid.species_pop_.size()
        cdef list out = [0]*m
        for s in range(m):
            out[s] = self.cpp_grid.cell_population_[s * self.M + cell_index]
        return out

    def get_cell_coords(self, cell_index, species_idx):
        cdef vector[arrayDouble1] coords_vec = self.cpp_grid.get_cell_coords(cell_index, species_idx)
        cdef int n = coords_vec.size()
        cdef list out = []
        for i in range(n):
            out.append(coords_vec[i][0])
        return out

    def get_cell_death_rates(self, cell_index, species_idx):
        cdef vector[double] drates = self.cpp_grid.get_cell_death_rates(cell_index, species_idx)
        cdef int n = drates.size()
        cdef list out = []
        for i in range(n):
            out.append(drates[i])
        return out

    def get_all_particle_coords(self):
        cdef vector[vector[arrayDouble1]] c_all = self.cpp_grid.get_all_particle_coords()
        cdef size_t ns = c_all.size()
        py_out = []
        cdef size_t i, j
        for i in range(ns):
            species_coords = []
            for j in range(c_all[i].size()):
                species_coords.append(c_all[i][j][0])
            py_out.append(species_coords)
        return py_out
        
    def get_all_particle_death_rates(self):
        cdef vector[vector[double]] c_all = self.cpp_grid.get_all_particle_death_rates()
        cdef size_t ns = c_all.size()
        py_out = []
        cdef size_t i, j
        for i in range(ns):
            species_rates = []
            for j in range(c_all[i].size()):
                species_rates.append(c_all[i][j])
            py_out.append(species_rates)
        return py_out

# PyGrid2 (wrapper for Grid<2>)
cdef class PyGrid2(PyGridBase):
    cdef Grid2* cpp_grid

    def __cinit__(self,
                  M,                # Number of species
                  areaLen,          # Domain size, e.g. [width, height]
                  cellCount,        # Number of cells, e.g. [nx, ny]
                  isPeriodic,       # Whether to use periodic boundaries
                  birthRates,       # Baseline birth rates for each species
                  deathRates,       # Baseline death rates for each species
                  ddMatrix,         # Flattened MxM pairwise interaction magnitudes
                  birthX,           # Birth kernel x-values (quantiles)
                  birthY,           # Birth kernel y-values (radii)
                  deathX,           # Death kernel x-values (distances)
                  deathY,           # Death kernel y-values (kernel values)
                  cutoffs,          # Flattened MxM cutoff distances
                  seed,             # Random number generator seed
                  rtimeLimit):      # Real-time limit in seconds
        
        cdef arrayDouble2 c_areaLen = pyToStdArrayDouble2(areaLen)
        cdef arrayInt2    c_cellCount = pyToStdArrayInt2(cellCount)
        cdef vector[double] c_birthRates = pyListToVectorDouble(birthRates)
        cdef vector[double] c_deathRates = pyListToVectorDouble(deathRates)
        cdef vector[double] c_ddMatrix   = pyListToVectorDouble(ddMatrix)
        cdef vector[ vector[double] ] c_birthX = pyListOfListToVectorVectorDouble(birthX)
        cdef vector[ vector[double] ] c_birthY = pyListOfListToVectorVectorDouble(birthY)
        cdef vector[ vector[ vector[double] ] ] c_deathX = pyListOfListOfListToVector3Double(deathX)
        cdef vector[ vector[ vector[double] ] ] c_deathY = pyListOfListOfListToVector3Double(deathY)
        cdef vector[double] c_cutoffs = pyListToVectorDouble(cutoffs)

        self.cpp_grid = new Grid2(
            M,
            c_areaLen,
            c_cellCount,
            <bool>isPeriodic,
            c_birthRates,
            c_deathRates,
            c_ddMatrix,
            c_birthX,
            c_birthY,
            c_deathX,
            c_deathY,
            c_cutoffs,
            seed,
            <double>rtimeLimit
        )

    def __dealloc__(self):
        if self.cpp_grid != NULL:
            del self.cpp_grid
            self.cpp_grid = NULL

    def placePopulation(self, initCoords):
        cdef vector[ vector[arrayDouble2] ] c_init = pyToCoordsD2(initCoords)
        self.cpp_grid.placePopulation(c_init)

    def spawn_at(self, s, pos):
        cdef arrayDouble2 cpos = pyToStdArrayDouble2(pos)
        self.cpp_grid.spawn_at(s, cpos)

    def kill_at(self, s, cell_idx, victimIdx):
        cdef arrayInt2 cc = pyToStdArrayInt2(cell_idx)
        self.cpp_grid.kill_at(s, cc, victimIdx)

    # Implement abstract methods from PyGridBase
    cdef double _get_total_birth_rate(self):
        return self.cpp_grid.total_birth_rate_
    
    cdef double _get_total_death_rate(self):
        return self.cpp_grid.total_death_rate_
    
    cdef int _get_total_population(self):
        return self.cpp_grid.total_population_
    
    cdef list _get_species_pop(self):
        cdef vector[int] pop = self.cpp_grid.species_pop_
        cdef int m = pop.size()
        cdef list out = [0] * m
        for s in range(m):
            out[s] = pop[s]
        return out
    
    cdef double _get_time(self):
        return self.cpp_grid.time_
    
    cdef int _get_event_count(self):
        return self.cpp_grid.event_count_
    
    cdef int _get_num_cells(self):
        return self.cpp_grid.total_num_cells_
    
    cdef void _spawn_random(self):
        self.cpp_grid.spawn_random()
    
    cdef void _kill_random(self):
        self.cpp_grid.kill_random()
    
    cdef void _make_event(self):
        self.cpp_grid.make_event()
    
    cdef void _run_events(self, int n):
        self.cpp_grid.run_events(n)
    
    cdef void _run_for(self, double duration):
        self.cpp_grid.run_for(duration)
    
    cdef double _get_cell_birth_rate(self, int cell_index):
        return self.cpp_grid.cell_birth_rate_[cell_index]
    
    cdef double _get_cell_death_rate(self, int cell_index):
        return self.cpp_grid.cell_death_rate_[cell_index]
    
    cdef list _get_cell_population(self, int cell_index):
        cdef int m = self.cpp_grid.species_pop_.size()
        cdef list out = [0]*m
        for s in range(m):
            out[s] = self.cpp_grid.cell_population_[s * self.M + cell_index]
        return out

    def get_cell_coords(self, cell_index, species_idx):
        cdef vector[arrayDouble2] coords_vec = self.cpp_grid.get_cell_coords(cell_index, species_idx)
        cdef int n = coords_vec.size()
        cdef list out = []
        for i in range(n):
            out.append([coords_vec[i][0], coords_vec[i][1]])
        return out

    def get_cell_death_rates(self, cell_index, species_idx):
        cdef vector[double] drates = self.cpp_grid.get_cell_death_rates(cell_index, species_idx)
        cdef int n = drates.size()
        cdef list out = []
        for i in range(n):
            out.append(drates[i])
        return out

    def get_all_particle_coords(self):
        cdef vector[vector[arrayDouble2]] c_all = self.cpp_grid.get_all_particle_coords()
        cdef size_t ns = c_all.size()
        py_out = []
        cdef size_t i, j
        for i in range(ns):
            species_coords = []
            for j in range(c_all[i].size()):
                species_coords.append([c_all[i][j][0], c_all[i][j][1]])
            py_out.append(species_coords)
        return py_out
        
    def get_all_particle_death_rates(self):
        cdef vector[vector[double]] c_all = self.cpp_grid.get_all_particle_death_rates()
        cdef size_t ns = c_all.size()
        py_out = []
        cdef size_t i, j
        for i in range(ns):
            species_rates = []
            for j in range(c_all[i].size()):
                species_rates.append(c_all[i][j])
            py_out.append(species_rates)
        return py_out

# PyGrid3 (wrapper for Grid<3>)
cdef class PyGrid3(PyGridBase):
    cdef Grid3* cpp_grid

    def __cinit__(self,
                  M,                # Number of species
                  areaLen,          # Domain size, e.g. [x_size, y_size, z_size]
                  cellCount,        # Number of cells, e.g. [nx, ny, nz]
                  isPeriodic,       # Whether to use periodic boundaries
                  birthRates,       # Baseline birth rates for each species
                  deathRates,       # Baseline death rates for each species
                  ddMatrix,         # Flattened MxM pairwise interaction magnitudes
                  birthX,           # Birth kernel x-values (quantiles)
                  birthY,           # Birth kernel y-values (radii)
                  deathX,           # Death kernel x-values (distances)
                  deathY,           # Death kernel y-values (kernel values)
                  cutoffs,          # Flattened MxM cutoff distances
                  seed,             # Random number generator seed
                  rtimeLimit):      # Real-time limit in seconds
        
        cdef arrayDouble3 c_areaLen = pyToStdArrayDouble3(areaLen)
        cdef arrayInt3    c_cellCount = pyToStdArrayInt3(cellCount)
        cdef vector[double] c_birthRates = pyListToVectorDouble(birthRates)
        cdef vector[double] c_deathRates = pyListToVectorDouble(deathRates)
        cdef vector[double] c_ddMatrix   = pyListToVectorDouble(ddMatrix)
        cdef vector[ vector[double] ] c_birthX = pyListOfListToVectorVectorDouble(birthX)
        cdef vector[ vector[double] ] c_birthY = pyListOfListToVectorVectorDouble(birthY)
        cdef vector[ vector[ vector[double] ] ] c_deathX = pyListOfListOfListToVector3Double(deathX)
        cdef vector[ vector[ vector[double] ] ] c_deathY = pyListOfListOfListToVector3Double(deathY)
        cdef vector[double] c_cutoffs = pyListToVectorDouble(cutoffs)

        self.cpp_grid = new Grid3(
            M,
            c_areaLen,
            c_cellCount,
            <bool>isPeriodic,
            c_birthRates,
            c_deathRates,
            c_ddMatrix,
            c_birthX,
            c_birthY,
            c_deathX,
            c_deathY,
            c_cutoffs,
            seed,
            <double>rtimeLimit
        )

    def __dealloc__(self):
        if self.cpp_grid != NULL:
            del self.cpp_grid
            self.cpp_grid = NULL

    def placePopulation(self, initCoords):
        cdef vector[ vector[arrayDouble3] ] c_init = pyToCoordsD3(initCoords)
        self.cpp_grid.placePopulation(c_init)

    def spawn_at(self, s, pos):
        cdef arrayDouble3 cpos = pyToStdArrayDouble3(pos)
        self.cpp_grid.spawn_at(s, cpos)

    def kill_at(self, s, cell_idx, victimIdx):
        cdef arrayInt3 cc = pyToStdArrayInt3(cell_idx)
        self.cpp_grid.kill_at(s, cc, victimIdx)

    # Implement abstract methods from PyGridBase
    cdef double _get_total_birth_rate(self):
        return self.cpp_grid.total_birth_rate_
    
    cdef double _get_total_death_rate(self):
        return self.cpp_grid.total_death_rate_
    
    cdef int _get_total_population(self):
        return self.cpp_grid.total_population_
    
    cdef list _get_species_pop(self):
        cdef vector[int] pop = self.cpp_grid.species_pop_
        cdef int m = pop.size()
        cdef list out = [0] * m
        for s in range(m):
            out[s] = pop[s]
        return out
    
    cdef double _get_time(self):
        return self.cpp_grid.time_
    
    cdef int _get_event_count(self):
        return self.cpp_grid.event_count_
    
    cdef int _get_num_cells(self):
        return self.cpp_grid.total_num_cells_
    
    cdef void _spawn_random(self):
        self.cpp_grid.spawn_random()
    
    cdef void _kill_random(self):
        self.cpp_grid.kill_random()
    
    cdef void _make_event(self):
        self.cpp_grid.make_event()
    
    cdef void _run_events(self, int n):
        self.cpp_grid.run_events(n)
    
    cdef void _run_for(self, double duration):
        self.cpp_grid.run_for(duration)
    
    cdef double _get_cell_birth_rate(self, int cell_index):
        return self.cpp_grid.cell_birth_rate_[cell_index]
    
    cdef double _get_cell_death_rate(self, int cell_index):
        return self.cpp_grid.cell_death_rate_[cell_index]
    
    cdef list _get_cell_population(self, int cell_index):
        cdef int m = self.cpp_grid.species_pop_.size()
        cdef list out = [0]*m
        for s in range(m):
            out[s] = self.cpp_grid.cell_population_[s * self.M + cell_index]
        return out

    def get_cell_coords(self, cell_index, species_idx):
        cdef vector[arrayDouble3] coords_vec = self.cpp_grid.get_cell_coords(cell_index, species_idx)
        cdef int n = coords_vec.size()
        cdef list out = []
        for i in range(n):
            out.append([coords_vec[i][0],
                        coords_vec[i][1],
                        coords_vec[i][2]])
        return out

    def get_cell_death_rates(self, cell_index, species_idx):
        cdef vector[double] drates = self.cpp_grid.get_cell_death_rates(cell_index, species_idx)
        cdef int n = drates.size()
        cdef list out = []
        for i in range(n):
            out.append(drates[i])
        return out

    def get_all_particle_coords(self):
        cdef vector[vector[arrayDouble3]] c_all = self.cpp_grid.get_all_particle_coords()
        cdef size_t ns = c_all.size()
        py_out = []
        cdef size_t i, j
        for i in range(ns):
            species_coords = []
            for j in range(c_all[i].size()):
                species_coords.append([c_all[i][j][0],
                                       c_all[i][j][1],
                                       c_all[i][j][2]])
            py_out.append(species_coords)
        return py_out
        
    def get_all_particle_death_rates(self):
        cdef vector[vector[double]] c_all = self.cpp_grid.get_all_particle_death_rates()
        cdef size_t ns = c_all.size()
        py_out = []
        cdef size_t i, j
        for i in range(ns):
            species_rates = []
            for j in range(c_all[i].size()):
                species_rates.append(c_all[i][j])
            py_out.append(species_rates)
        return py_out
