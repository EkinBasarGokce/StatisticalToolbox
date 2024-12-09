# ANOVALR Tool Box and main.py README for ENS505 Term Project

by Ekin Başar Gökçe

## Overview

This project contains two primary Python files:

1. **ANOVALR.py**: A toolbox for performing various ANOVA (Analysis of Variance) and Linear Regression analyses.
2. **main.py**: A script that calls the functions from ANOVALR.py and tests them with artificially created data.

## ANOVALR.py Functions

##### More detailed descriptions of the functions can be found in the ANOVALR.py file, inside the functions.

### ANOVA Functions 
 

- `ANOVA1_partition_TSS(dataset)`: Calculate total, within-group, and between-group sum of squares.
- `ANOVA1_test_equality(dataset, alpha)`: Perform ANOVA test and print ANOVA table and decision based on F-statistic.
- `ANOVA1_is_contrast(*coefficients)`: Check if the given coefficients form a contrast.
- `ANOVA1_is_orthogonal(coefficients1, coefficients2)`: Check if two sets of coefficients are orthogonal contrasts.
- `ANOVA1_CI_linear_combs(data, alpha, C, method)`: Calculate simultaneous confidence intervals for specified linear combinations of group means in one-way ANOVA.
- `ANOVA1_test_linear_combs(X, alpha, C, d, method)`: Tests hypotheses on linear combinations of group means with FWER control.
- `ANOVA2_partition_TSS(data)`: Partition the sum of squares in a two-way ANOVA layout.
- `ANOVA2_MLE(data)`: Calculate the MLE for the parameters of a two-way ANOVA.
- `ANOVA2_test_equality(data, alpha, test)`: Perform one of the basic three tests in the two-way ANOVA layout.

### Linear Regression Functions

- `Mult_LR_Least_squares(X, y)`: Find the least squares solution for a multiple linear regression model.
- `Mult_LR_partition_TSS(X, y)`: Partition the Total Sum of Squares (TSS) into the Regression Sum of Squares (RegSS) and the Residual Sum of Squares (RSS) for a multiple linear regression model.
- `Mult_normLR_simul_CI(X, y, alpha)`: Calculate simultaneous confidence intervals for the coefficients of a multiple linear regression model.
- `Mult_norm_LR_CR(X, y, C, alpha)`: Calculates the 100(1 - alpha)% confidence region for Cβ according to the normal multiple linear regression model.
- `Mult_normLR_is_in_CR(X, y, C, c0, alpha)`: Checks if a point is within the 100(1 - alpha)% confidence region for Cβ.
- `Mult_normLR_test_general(X, y, C, c0, alpha)`: Perform a general linear hypothesis test for multiple linear regression.
- `Mult_normLR_test_comp(X, y, indices, alpha)`: Tests a composite hypothesis for a subset of regression coefficients.
- `Mult_normLR_test_linear_reg(X, y, alpha)`: Tests the null hypothesis that all regression coefficients are zero.
- `Mult_norm_LR_pred_CI(X, y, D, alpha, method)`: Compute simultaneous confidence bounds for multiple linear regression predictions.

## main.py

The `main.py` script is designed to demonstrate the usage of the functions in `ANOVALR.py`. It includes:

1. **Data Creation**: Generation of artificial datasets for testing purposes.
2. **Function Calls**: Example calls to the functions defined in `ANOVALR.py`, showcasing how to perform ANOVA and Linear Regression analyses.
3. **Output**: Display of results, including ANOVA tables, confidence intervals, and test decisions.

## Usage

2. **Run the main script**:
    ```bash
    python main.py
    ```

## Dependencies
The main.py and ANOVALR.py files must be in the same folder.

```bash
pip install numpy 
pip install scipy 
pip install statsmodels
```