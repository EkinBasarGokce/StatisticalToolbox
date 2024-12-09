#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# necessary imports
import numpy as np
import scipy.stats as stats 
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from scipy.stats import f, t
from ANOVALR import *

alpha = 0.05

# The data is previously prepared by me

# ANOVA 1 Inputs
data_1 = [
    [18, 19, 17, 16, 22, 21, 20, 19, 23, 18],
    [22, 23, 21, 27, 23.5, 25, 24, 26, 22, 23.8],
    [29, 28, 30, 31, 27, 29, 28, 30, 31, 27],
    [33, 32, 34, 33, 32, 34, 33, 32, 34, 33]
]

C_1 = np.array([
    [2, -2, 0, 0],  
    [0, 2, -2, 0], 
    [0, 0, 2, -2],  
    [2, 0, -2, 0],  
    [2, 0, 0, -2]   
])

# ANOVA 2 Inputs
X = np.array([
[[15, 17, 16, 18, 19, 20], [25, 27, 26, 28, 29, 30], [35, 37, 36, 38, 39, 40], [45, 47, 46, 48, 49, 50]],
[[16, 18, 17, 19, 20, 21], [26, 28, 27, 29, 30, 31], [36, 38, 37, 39, 40, 41], [46, 48, 47, 49, 50, 51]],
[[17, 19, 18, 20, 21, 22], [27, 29, 28, 30, 31, 32], [37, 39, 38, 40, 41, 42], [47, 49, 48, 50, 51, 52]],
[[18, 20, 19, 21, 22, 23], [28, 30, 29, 31, 32, 33], [38, 40, 39, 41, 42, 43], [48, 50, 49, 51, 52, 53]],
[[19, 21, 20, 22, 23, 24], [29, 31, 30, 32, 33, 34], [39, 41, 40, 42, 43, 44], [49, 51, 50, 52, 53, 54]]
])


# LR Inputs
indices = np.array([1, 2]) # Test if beta_1 = beta_2 = 0

X_lr = np.array([
    [1, 44.0, 42.0], [1, 65.0, 95.0], [1, 39.0, 37.0], [1, 41.0, 32.0], 
    [1, 47.0, 54.0], [1, 40.0, 19.0], [1, 45.0, 40.0], [1, 24.0, 10.0], 
    [1, 39.0, 35.0], [1, 25.0, 11.0], [1, 35.0, 23.0], [1, 60.0, 81.0],
    [1, 49.0, 44.0], [1, 68.0, 98.0], [1, 42.0, 40.0], [1, 44.0, 35.0], 
    [1, 50.0, 57.0], [1, 43.0, 22.0], [1, 48.0, 43.0], [1, 27.0, 13.0], 
    [1, 42.0, 38.0], [1, 28.0, 14.0], [1, 38.0, 26.0], [1, 63.0, 84.0],
    [1, 52.0, 47.0], [1, 71.0, 101.0], [1, 45.0, 43.0], [1, 47.0, 38.0], 
    [1, 53.0, 60.0], [1, 46.0, 25.0], [1, 50.0, 46.0], [1, 30.0, 16.0], 
    [1, 45.0, 41.0], [1, 31.0, 17.0], [1, 41.0, 29.0], [1, 66.0, 87.0],
    [1, 55.0, 50.0], [1, 74.0, 104.0], [1, 48.0, 46.0], [1, 50.0, 41.0], 
    [1, 56.0, 63.0], [1, 49.0, 28.0], [1, 54.0, 49.0], [1, 33.0, 19.0], 
    [1, 48.0, 44.0], [1, 34.0, 20.0], [1, 44.0, 32.0], [1, 69.0, 90.0]
])
y_lr = np.array([
    38.5, 51.0, 36.0, 37.5, 44.5, 29.5, 38.5, 21.5, 35.0, 32.0, 40.0, 48.5,
    41.5, 54.0, 39.0, 40.5, 47.5, 32.5, 41.5, 24.5, 37.5, 34.5, 42.5, 51.5,
    44.5, 57.0, 42.0, 43.5, 50.5, 35.5, 44.5, 27.5, 40.5, 37.0, 45.0, 54.5,
    47.5, 60.0, 45.0, 46.5, 53.5, 38.5, 47.5, 30.5, 43.5, 39.5, 47.5, 57.5
])
D_lr = np.array([
    [1, 32.0, 32.0], [1, 42.0, 42.0], [1, 52.0, 52.0], 
    [1, 62.0, 62.0], [1, 72.0, 72.0], [1, 82.0, 82.0]
])

C_lr = np.array([
    [0, 1, -1], 
    [0, 0, 1]   
])

c0 = np.zeros(C_lr.shape[0])


print("\nANOVA Functions Tests\n")
# Test for ANOVA1_is_contrast
coefficients_1 = [1, -1, 0, 0]
coefficients_2 = [0, 1, -1, 0]
coefficients_3 = [1, 1, -1, -1]
print("ANOVA1_is_contrast results:")
print(f"Is {coefficients_1} a contrast? {ANOVA1_is_contrast(coefficients_1)}")
print(f"Is {coefficients_2} a contrast? {ANOVA1_is_contrast(coefficients_2)}")
print(f"Is {coefficients_3} a contrast? {ANOVA1_is_contrast(coefficients_3)}")
print()

# Test for ANOVA1_is_orthogonal
print("ANOVA1_is_orthogonal results:")
print(f"Are {coefficients_1} and {coefficients_2} orthogonal contrasts? {ANOVA1_is_orthogonal(coefficients_1, coefficients_2)}")
print(f"Are {coefficients_1} and {coefficients_3} orthogonal contrasts? {ANOVA1_is_orthogonal(coefficients_1, coefficients_3)}")
print()

# Test for Bonferroni_correction
num_tests = 5
print("Bonferroni_correction results:")
print(f"Bonferroni correction for alpha={alpha} and {num_tests} tests: {Bonferroni_correction(alpha, num_tests)}")
print()

# Test for Sidak_correction
print("Sidak_correction results:")
print(f"Sidak correction for alpha={alpha} and {num_tests} tests: {Sidak_correction(alpha, num_tests)}")
print()

#print("Bigger Functions")
## Tests for bigger ANOVA1 functions
# Test ANOVA 1 Functions
print("Testing ANOVA 1 Functions:")
#print("\nANOVA1_partition_TSS example:")
SS_t, SS_b, SS_w = ANOVA1_partition_TSS(data_1)
print(f"SS_t: {SS_t}, SS_b: {SS_b}, SS_w: {SS_w}")

print("\nANOVA1_test_equality example:")
ANOVA1_test_equality(data_1, alpha)

# Test function for ANOVA1_CI_linear_combs
def test_ANOVA1_CI_methods(data, alpha, C):
    methods = ["Scheffe", "Tukey", "Bonferroni", "Sidak", "best"]
    for method in methods:
        results = ANOVA1_CI_linear_combs(data, alpha, C, method=method)
        print(f"{method} method results:")
        for ci in results:
            print(ci)
        print()

# Test test_ANOVA1_CI_methods
test_ANOVA1_CI_methods(data_1, alpha, C_1)

# Testing the ANOVA1_test_linear_combs  
print("\nTest Linear Combinations example:")
linear_combination_results = ANOVA1_test_linear_combs(data_1, alpha, C_1, np.zeros((5, 1)), "best")
for hypothesis, outcome in linear_combination_results.items():
    print(f"{hypothesis}: Reject: {outcome[0]}, p-value: {outcome[1]}")

# Test ANOVA 2 Functions
print("\nTesting ANOVA 2 Functions:")

print("\nANOVA2_partition_TSS example:")
ANOVA2_partition_results = ANOVA2_partition_TSS(X)
for key, value in ANOVA2_partition_results.items():
    print(f"{key}: {value}")

print("\nANOVA2_MLE example:")
ANOVA2_mle_results = ANOVA2_MLE(X)
for key, value in ANOVA2_mle_results.items():
    print(f"{key}: {value}")

print("\nANOVA2_test_equality example for 'A':")
ANOVA2_test_equality(X, alpha, "A")

print("\nANOVA2_test_equality example for 'B':")
ANOVA2_test_equality(X, alpha, "B")

print("\nANOVA2_test_equality example for 'AB':")
ANOVA2_test_equality(X, alpha, "AB")

print("\n Linear Regression Functions Tests")

# Test least squares solution
beta_hat, sigma2_hat, sigma2_hat_biased = Mult_LR_Least_squares(X_lr, y_lr)
print("\nLeast Squares Solution:")
print("Beta_hat:", beta_hat)
print("Unbiased Sigma^2_hat:", sigma2_hat)
print("Sigma^2_hat MLE:", sigma2_hat_biased)

# Test partition TSS
TSS, RegSS, RSS = Mult_LR_partition_TSS(X_lr, y_lr)
print("\nPartition of TSS:")
print("Total sum of squares (TSS):", TSS)
print("Regression Sum of Squares (RegSS):", RegSS)
print("Residual Sum of Squares (RSS):", RSS)

# Test simultaneous confidence intervals
simul_CI = Mult_normLR_simul_CI(X_lr, y_lr, alpha)
print("\nSimultaneous Confidence Intervals for Beta:")
for i, interval in enumerate(simul_CI):
    print(f"Beta_{i}: {interval}")

# Test confidence region for beta 
center, cov_matrix, radius_squared = Mult_norm_LR_CR(X_lr, y_lr, C_lr, alpha)
print("\nConfidence Region for Cβ:")
print("Center:", center)
print("Covariance Matrix:\n", cov_matrix)
print("Radius Squared:", radius_squared)

# Testing lines
c0 = np.array([0, 0])  # Example point to check
is_in_CR = Mult_normLR_is_in_CR(X_lr, y_lr, C_lr, c0, alpha)
# Print the result based on the boolean value returned
if is_in_CR:
    print("Yes, c0 is in the 100(1 − α)% confidence region for Cβ")
else:
    print("No, c0 is not in the 100(1 − α)% confidence region for Cβ")

# Test general hypothesis
general_test = Mult_normLR_test_general(X_lr, y_lr, C_lr, c0, alpha)
print("\nGeneral Hypothesis Test Result:", general_test)

# Print the result based on the boolean value returned
if general_test:
    print("Do not reject the null hypothesis: Yes, c0 is in the 100(1 − α)% confidence region for Cβ")
else:
    print("Reject the null hypothesis: No, c0 is not in the 100(1 − α)% confidence region for Cβ")

# Test multiple comparisons
comp_test = Mult_normLR_test_comp(X_lr, y_lr, indices, alpha)
print("\nMultiple Comparisons Test Result:", comp_test)

# Print the result based on the boolean value returned
if comp_test:
    print("Do not reject the null hypothesis: Yes, c0 is in the 100(1 − α)% confidence region for Cβ")
else:
    print("Reject the null hypothesis: No, c0 is not in the 100(1 − α)% confidence region for Cβ")

# Test linear regression
linear_reg_test = Mult_normLR_test_linear_reg(X_lr, y_lr, alpha)
print("\nLinear Regression Existence Test Result:", linear_reg_test)

# Print the result based on the boolean value returned
if linear_reg_test:
    print("Do not reject the null hypothesis: Yes, c0 is in the 100(1 − α)% confidence region for Cβ")
else:
    print("Reject the null hypothesis: No, c0 is not in the 100(1 − α)% confidence region for Cβ")

    
# Test Mult_norm_LR_pred_CI

# Compute intervals using Bonferroni method
predictions_bonferroni, intervals_bonferroni = Mult_norm_LR_pred_CI(X_lr, y_lr, D_lr, alpha, "Bonferroni")
print("\nBonferroni Method:")
print("Predictions:", predictions_bonferroni)
print("Confidence Intervals:", intervals_bonferroni)

# Compute intervals using Scheffe method
predictions_scheffe, intervals_scheffe = Mult_norm_LR_pred_CI(X_lr, y_lr, D_lr, alpha, "Scheffe")
print("\nScheffe Method:")
print("Predictions:", predictions_scheffe)
print("Confidence Intervals:", intervals_scheffe)

predictions_best, intervals_best = Mult_norm_LR_pred_CI(X_lr, y_lr, D_lr, alpha, "best")
print("\nBest Results:")
print("Predictions:", predictions_best)
print("Confidence Intervals:", intervals_best)


