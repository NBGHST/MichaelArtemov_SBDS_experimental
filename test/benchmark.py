import numpy as np
import pandas as pd
import sys
import os
import time
from scipy.stats import rayleigh
sys.path.append(".") # ваш путь к папке с проектом
sys.path.append("..")
import simulation

L = 10.0       # Размер области
M = 2          # Количество видов
my_seed = 42   # seed для генератора случайных чисел

birth_rates = [0.33, 0.33]        # Вероятность рождаемости
natural_death_rates = [0.1, 0.1]  # Вероятность естественной смерти
competition_matrix = [            # Матрица конкуренции
    0.02, 0.012,                  # Влияние вида 1 на вид 1 и на вид 2
    0.012, 0.02                   # Влияние вида 2 на вид 1 и на вид 2
]                               

sigma_m = [1, 1]                  # Радиус распространения потомства

sigma_w = np.array([              # Радиус влияния конкуренции
    [0.4, 0.4],
    [0.4, 0.4],
])

def exponential_2d_radial(r, lambda_param):
    return (lambda_param**2) * r * np.exp(-lambda_param * r)

q_values = np.arange(0, 1.0, 0.001)
birth_inverse_values = []
for i in range(M):
    inverse_vals = np.sqrt(-2*np.log(1-q_values))/sigma_m[i]
    birth_inverse_values.append(inverse_vals.tolist())

death_r_values = []
death_density_values = []

for i in range(M):
    r_values_row = []
    density_row = []
    
    for j in range(M):
        r_max = min(5/sigma_w[i,j], L/2)
        r_vals = np.linspace(0, r_max, 500)
        
        density = exponential_2d_radial(r_vals, 1/sigma_w[i, j])
        
        r_values_row.append(r_vals.tolist())
        density_row.append(density.tolist())
    
    death_r_values.append(r_values_row)
    death_density_values.append(density_row)

cutoffs = []
for i in range(M):
    for j in range(M):
        cutoffs.append(min(10 * sigma_w[i,j], L/2))

np.random.seed(my_seed)

coordinates = []
for _ in range(M):
    group = [[np.random.uniform(0, L), np.random.uniform(0, L)] for _ in range(400)]
    coordinates.append(group)

sumtime = 0
sumtime -= time.time()
g2 = simulation.PyGrid2(
    M = M,
    areaLen = [L, L],
    cellCount = [50, 50],
    isPeriodic = False,
    birthRates = birth_rates,
    deathRates = natural_death_rates,
    ddMatrix = competition_matrix,
    birthX = [q_values.tolist()] * M,
    birthY = birth_inverse_values,
    deathX = death_r_values,
    deathY = death_density_values,
    cutoffs = cutoffs,
    seed = my_seed,
    rtimeLimit = 7200.0
)

g2.placePopulation(coordinates)

for t in range(1000):
    g2.run_for(1)

sumtime += time.time()










#ТЕСТ2

L = 2.0                           # Размер области

birth_rates = [0.4, 0.4]          # Вероятность рождаемости
natural_death_rates = [0.2, 0.2]  # Вероятность естественной смерти

# Вид 1 выживает, вид 2 вымирает
competition_matrix = [            # Матрица конкуренции
    0.001, 0.001,                 # Влияние вида 1 на вид 1 и на вид 2
    0.0008, 0.001                 # Влияние вида 2 на вид 1 и на вид 2
]                               
sigma_m = [0.04, 0.04]            # Радиус распространения потомства

sigma_w = np.array([              # Радиус влияния конкуренции
    [0.04, 0.04],
    [0.04, 0.04],
])

def normal_2d_radial(r, sigma):
    return (1 / (2 * np.pi * sigma**2)) * np.exp(-r**2 / (2 * sigma**2))


q_values = np.arange(0, 1.0, 0.001)

birth_inverse_values = []
for i in range(M):
    inverse_vals = rayleigh.ppf(q_values, scale=sigma_m[i])
    birth_inverse_values.append(inverse_vals.tolist())

death_r_values = []
death_density_values = []

for i in range(M):
    r_values_row = []
    density_row = []
    
    for j in range(M):
        r_max = min(10 * sigma_w[i,j], L/2)
        r_vals = np.linspace(0, r_max, 500)
        
        density = normal_2d_radial(r_vals, sigma_w[i, j])
        
        r_values_row.append(r_vals.tolist())
        density_row.append(density.tolist())
    
    death_r_values.append(r_values_row)
    death_density_values.append(density_row)


cutoffs = []
for i in range(M):
    for j in range(M):
        cutoffs.append(min(10 * sigma_w[i,j], L/2))

np.random.seed(my_seed)
coordinates = []
for _ in range(M):
    group = [[np.random.uniform(0, L), np.random.uniform(0, L)] for _ in range(500)]
    coordinates.append(group)

sumtime -= time.time()

g2 = simulation.PyGrid2(
    M = M,
    areaLen = [L, L],
    cellCount = [50, 50],
    isPeriodic = False,
    birthRates = birth_rates,
    deathRates = natural_death_rates,
    ddMatrix = competition_matrix,
    birthX = [q_values.tolist()] * M,
    birthY = birth_inverse_values,
    deathX = death_r_values,
    deathY = death_density_values,
    cutoffs = cutoffs,
    seed = my_seed,
    rtimeLimit = 7200.0
)

g2.placePopulation(coordinates)
for t in range(1500):
    g2.run_for(1)

sumtime += time.time()









#ТЕСТ3

L = 2.0                           # Размер области

birth_rates = [0.4, 0.4]          # Вероятность рождаемости
natural_death_rates = [0.2, 0.2]  # Вероятность естественной смерти
competition_matrix = [            # Матрица конкуренции
    0.001, 0.001,                 # Влияние вида 1 на вид 1 и на вид 2
    0.001, 0.001                  # Влияние вида 2 на вид 1 и на вид 2
]                               

sigma_m = [0.04, 0.04]            # Радиус распространения потомства

#  Для сосущестования двух видов
sigma_w = np.array([              # Радиус влияния конкуренции
    [0.06, 0.03],
    [0.03, 0.06],
])


def normal_2d_radial(r, sigma):
    return (1 / (2 * np.pi * sigma**2)) * np.exp(-r**2 / (2 * sigma**2))


q_values = np.arange(0, 1.0, 0.001)

birth_inverse_values = []
for i in range(M):
    inverse_vals = rayleigh.ppf(q_values, scale=sigma_m[i])
    birth_inverse_values.append(inverse_vals.tolist())

death_r_values = []
death_density_values = []

for i in range(M):
    r_values_row = []
    density_row = []
    
    for j in range(M):
        r_max = min(10 * sigma_w[i,j], L/2)
        r_vals = np.linspace(0, r_max, 500)
        
        density = normal_2d_radial(r_vals, sigma_w[i, j])
        
        r_values_row.append(r_vals.tolist())
        density_row.append(density.tolist())
    
    death_r_values.append(r_values_row)
    death_density_values.append(density_row)

cutoffs = []
for i in range(M):
    for j in range(M):
        cutoffs.append(min(10 * sigma_w[i,j], L/2))

np.random.seed(my_seed)
coordinates = []
for _ in range(M):
    group = [[np.random.uniform(0, L), np.random.uniform(0, L)] for _ in range(500)]
    coordinates.append(group)

sumtime -= time.time()
g2 = simulation.PyGrid2(
    M = M,
    areaLen = [L, L],
    cellCount = [50, 50],
    isPeriodic = False,
    birthRates = birth_rates,
    deathRates = natural_death_rates,
    ddMatrix = competition_matrix,
    birthX = [q_values.tolist()] * M,
    birthY = birth_inverse_values,
    deathX = death_r_values,
    deathY = death_density_values,
    cutoffs = cutoffs,
    seed = my_seed,
    rtimeLimit = 7200.0
)

g2.placePopulation(coordinates)
for t in range(2000):
    g2.run_for(1)

sumtime += time.time()












#ТЕСТ4

L = 2.0                           # Размер области

birth_rates = [0.4, 0.4]          # Вероятность рождаемости
natural_death_rates = [0.2, 0.2]  # Вероятность естественной смерти
competition_matrix = [            # Матрица конкуренции
    0.001, 0.001,                 # Влияние вида 1 на вид 1 и на вид 2
    0.001, 0.001                  # Влияние вида 2 на вид 1 и на вид 2
]                               

sigma_m = [0.04, 0.04]            # Радиус распространения потомства
sigma_w = np.array([              # Радиус влияния конкуренции
    [0.04, 0.04],
    [0.04, 0.04],
])

def normal_2d_radial(r, sigma):
    return (1 / (2 * np.pi * sigma**2)) * np.exp(-r**2 / (2 * sigma**2))


q_values = np.arange(0, 1.0, 0.001)

birth_inverse_values = []
for i in range(M):
    inverse_vals = rayleigh.ppf(q_values, scale=sigma_m[i])
    birth_inverse_values.append(inverse_vals.tolist())

death_r_values = []
death_density_values = []

for i in range(M):
    r_values_row = []
    density_row = []
    
    for j in range(M):
        r_max = min(10 * sigma_w[i,j], L/2)
        r_vals = np.linspace(0, r_max, 500)
        
        density = normal_2d_radial(r_vals, sigma_w[i, j])
        
        r_values_row.append(r_vals.tolist())
        density_row.append(density.tolist())
    
    death_r_values.append(r_values_row)
    death_density_values.append(density_row)


cutoffs = []
for i in range(M):
    for j in range(M):
        cutoffs.append(min(10 * sigma_w[i,j], L/2))


np.random.seed(my_seed)
coordinates = []
for _ in range(M):
    group = [[np.random.uniform(0, L), np.random.uniform(0, L)] for _ in range(500)]
    coordinates.append(group)

sumtime -= time.time()

g2 = simulation.PyGrid2(
    M = M,
    areaLen = [L, L],
    cellCount = [50, 50],
    isPeriodic = False,
    birthRates = birth_rates,
    deathRates = natural_death_rates,
    ddMatrix = competition_matrix,
    birthX = [q_values.tolist()] * M,
    birthY = birth_inverse_values,
    deathX = death_r_values,
    deathY = death_density_values,
    cutoffs = cutoffs,
    seed = my_seed,
    rtimeLimit = 7200.0
)

g2.placePopulation(coordinates)
for t in range(3000):
    g2.run_for(1)

sumtime += time.time()

print(f"exec time {sumtime} seconds")