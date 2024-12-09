#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# necessary imports
import numpy as np
import scipy.stats as stats 
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from scipy.stats import f, t

def ANOVA1_partition_TSS(dataset):
    """Calculate total, within-group, and between-group sum of squares."""
    total = sum(sum(group) for group in dataset)  # Sum of all elements in the dataset
    count = sum(len(group) for group in dataset)  # Total number of elements in the dataset
    total_mean = total / count  # Mean of all elements in the dataset

    # Calculate total sum of squares
    SS_t = sum((value - total_mean) ** 2 for group in dataset for value in group)

    # Calculate within-group and between-group sum of squares
    SS_w = sum(sum((value - np.mean(group)) ** 2 for value in group) for group in dataset)
    SS_b = sum(len(group) * (np.mean(group) - total_mean) ** 2 for group in dataset)
    
    return SS_t, SS_b, SS_w

def ANOVA1_test_equality(dataset, alpha):
    """Perform ANOVA test and print ANOVA table and decision based on F-statistic."""
    SS_t, SS_b, SS_w = ANOVA1_partition_TSS(dataset)
    
    num_groups = len(dataset)  # Number of groups
    num_total = sum(len(group) for group in dataset)  # Total number of elements in the dataset
    
    df_b = num_groups - 1  # Degrees of freedom between groups
    df_w = num_total - num_groups  # Degrees of freedom within groups

    MS_b = SS_b / df_b  # Mean square between groups
    MS_w = SS_w / df_w  # Mean square within groups

    f_stat = MS_b / MS_w  # F-statistic
    
    critical_val = stats.f.ppf(1 - alpha, df_b, df_w)  # Critical value from F-distribution
    p_val = 1 - stats.f.cdf(f_stat, df_b, df_w)  # p-value from F-distribution
    
    print("\nANOVA Table:")
    print(f"{'Source':<12}{'df':<10}{'SS':<10}{'MS':<10}{'F':<10}")
    print(f"{'Between':<12}{df_b:<10}{SS_b:<10.2f}{MS_b:<10.2f}{f_stat:<10.2f}")
    print(f"{'Within':<12}{df_w:<10}{SS_w:<10.2f}{MS_w:<10.2f}{'-':<10}")
    print(f"{'Total':<12}{df_b + df_w:<10}{SS_t:<10.2f}{'-':<10}{'-':<10}")
    
    print(f"\nCritical value for alpha({alpha}):\t{critical_val:.2f}")
    print(f"p-value:\t{p_val:.2f}")
    
    decision = "Reject the null hypothesis!" if p_val < alpha else "Do not reject the null hypothesis!"
    print("Decision:", decision)

    return

def ANOVA1_is_contrast(*coefficients):
    """Check if the given coefficients form a contrast."""
    if len(coefficients) == 1 and isinstance(coefficients[0], (list, tuple, np.ndarray)):
        coefficients = coefficients[0]
    return np.sum(coefficients) == 0 and np.sum(coefficients != 0) >= 2

def ANOVA1_is_orthogonal(coefficients1, coefficients2):
    """Check if two sets of coefficients are orthogonal contrasts."""
    if not (ANOVA1_is_contrast(coefficients1) and ANOVA1_is_contrast(coefficients2)):
        return "Warning! At least one of the coefficients is not a contrast!"
    if len(coefficients1) != len(coefficients2):
        return "Warning! Vectors must be of the same length!"
    return sum(x * y for x, y in zip(coefficients1, coefficients2)) == 0

def Bonferroni_correction(alpha, m):
    """Apply Bonferroni correction to the significance level."""
    if m == 0:
        return "Number of tests cannot be zero!"
    adjusted_alpha = alpha / m  # Adjusted alpha level for Bonferroni correction
    return float("%.4f" % adjusted_alpha)

def Sidak_correction(alpha, m):
    """Apply Sidak correction to the significance level."""
    if m == 0:
        return "Number of tests cannot be zero!"
    adjusted_alpha = 1 - (1 - alpha) ** (1 / m)  # Adjusted alpha level for Sidak correction
    return float("%.4f" % adjusted_alpha)

def ANOVA1_CI_linear_combs(data, alpha, C, method):
    """
    Calculate simultaneous confidence intervals for specified linear combinations of group means in one-way ANOVA.
    
    Parameters:
    - X (list of np.array): Data set where each array represents a group.
    - alpha (float): Significance level.
    - C (np.array): m x I matrix defining m linear combinations of I group means.
    - method (str): Statistical method to use ("Scheffe", "Tukey", "Bonferroni", "Sidak", "best").
    
    Returns:
    - dict: Dictionary of confidence intervals if valid, None with a warning if not.
    """
    data = np.array(data)
    
    # Number of groups
    I = data.shape[0]
    
    # Total number of observations
    n = data.size
    
    # Number of observations in each group
    n_i = np.array([len(group) for group in data])
    
    # Means of each group
    X_bar = np.mean(data, axis=1)
    
    # Sum of squares within (SSw)
    SSw = np.sum((data - X_bar[:, None])**2)
    
    # Compute the critical value M_alpha, I-1, n-I
    f_critical = stats.f.ppf(1 - alpha, I - 1, n - I)
    M_alpha = np.sqrt((I - 1) * f_critical)
    
    # Compute the contrast estimates using matrix multiplication
    contrast_estimates = np.dot(C, X_bar)
    
    means = np.mean(data, axis=1) # overall mean
    df_error = n - I
    df_total = n - 1
    MSE = SSw / (n - I)

    results = []
    
    if method == "best":
        all_contrasts = True
        for contrast in C:
            if not ANOVA1_is_contrast(contrast):
                all_contrasts = False
                break

        if all_contrasts:
            for i, contrast in enumerate(C):
                orthogonal = True
                for j in range(i + 1, len(C)):
                    if not ANOVA1_is_orthogonal(contrast, C[j]):
                        orthogonal = False
                        break
                
                contrast_estimate = contrast_estimates[i]
                
                if orthogonal:
                    # Compute Sidak correction CI
                    t_crit_sidak = t.ppf(1 - Sidak_correction(alpha, len(C)), df_error)
                    error_margin_sidak = t_crit_sidak * np.sqrt(MSE * np.sum(contrast**2 / np.array(n_i)))
                    lower_bound_sidak = contrast_estimate - error_margin_sidak
                    upper_bound_sidak = contrast_estimate + error_margin_sidak
                    ci_sidak = (lower_bound_sidak, upper_bound_sidak)
                    
                    # Compute Theorem 2.8 CI
                    contrast_estimate = contrast_estimates[i]
                    standard_error = np.sqrt(SSw / (n - I) * np.sum(contrast**2 / n_i))
                    margin_of_error = M_alpha * standard_error
                    lower_bound = contrast_estimate - margin_of_error
                    upper_bound = contrast_estimate + margin_of_error
                    ci_28 = (lower_bound, upper_bound)
              
                    # Choose better CI
                    if (ci_sidak[1] - ci_sidak[0]) < (ci_28[1] - ci_28[0]):
                        results.append(ci_sidak)
                    else:
                        results.append(ci_28)
                else:
                    # Compute Bonferroni correction CI
                    t_crit_bonferroni = t.ppf(1 - Bonferroni_correction(alpha, len(C)), df_error)
                    error_margin_bonferroni = t_crit_bonferroni * np.sqrt(MSE * np.sum(contrast**2 / np.array(n_i)))
                    lower_bound_bonferroni = contrast_estimate - error_margin_bonferroni
                    upper_bound_bonferroni = contrast_estimate + error_margin_bonferroni
                    ci_bonferroni = (lower_bound_bonferroni, upper_bound_bonferroni)
                    
                    # Compute Theorem 2.8 CI
                    M_bonferroni = np.sqrt((I-1) * f.ppf(1 - Bonferroni_correction(alpha, len(C)), I-1, sum(n_i) - I))
                    SE_bonferroni = np.sqrt((SSw / (sum(n_i) - I)) * np.sum(contrast**2 / np.array(n_i)))
                    CI_28_bonferroni = M_bonferroni * SE_bonferroni
                    ci_28 = (contrast_estimate - CI_28_bonferroni, contrast_estimate + CI_28_bonferroni)
                    
                    # Choose better CI
                    if (ci_bonferroni[1] - ci_bonferroni[0]) < (ci_28[1] - ci_28[0]):
                        results.append(ci_bonferroni)
                    else:
                        results.append(ci_28)
        else:
            for i, contrast in enumerate(C):
                contrast_estimate = contrast_estimates[i]
                # Compute Bonferroni correction CI
                t_crit_bonferroni = t.ppf(1 - Bonferroni_correction(alpha, len(C)), df_error)
                error_margin_bonferroni = t_crit_bonferroni * np.sqrt(MSE * np.sum(contrast**2 / np.array(n_i)))
                lower_bound_bonferroni = contrast_estimate - error_margin_bonferroni
                upper_bound_bonferroni = contrast_estimate + error_margin_bonferroni
                ci_bonferroni = (lower_bound_bonferroni, upper_bound_bonferroni)
                
                # Compute Theorem 2.7 CI
                M_bonferroni = np.sqrt(I * f.ppf(1 - Bonferroni_correction(alpha, len(C)), I, sum(n_i) - I))
                SE_bonferroni = np.sqrt((SSw / (sum(n_i) - I)) * np.sum(contrast**2 / np.array(n_i)))
                CI_27_bonferroni = M_bonferroni * SE_bonferroni
                ci_27 = (contrast_estimate - CI_27_bonferroni, contrast_estimate + CI_27_bonferroni)
                
                # Choose better CI
                if (ci_bonferroni[1] - ci_bonferroni[0]) < (ci_27[1] - ci_27[0]):
                    results.append(ci_bonferroni)
                else:
                    results.append(ci_27)
            

    elif method == "Scheffe":
        for idx, contrast in enumerate(C):
            contrast_estimate = contrast_estimates[idx]
            standard_error = np.sqrt(SSw / (n - I) * np.sum(contrast**2 / n_i))
            margin_of_error = M_alpha * standard_error
            lower_bound = contrast_estimate - margin_of_error
            upper_bound = contrast_estimate + margin_of_error
            ci = (lower_bound, upper_bound)
            results.append(ci)

    elif method == "Tukey":
        for idx, contrast in enumerate(C):
            if np.all(np.isin(contrast, [-1, 0, 1])) and np.sum(contrast) == 0:
                contrast_estimate = contrast_estimates[idx]
                q_crit = t.ppf(1 - alpha / 2, df_error)
                error_margin = q_crit * np.sqrt(MSE * 2 / len(means))
                lower_bound = contrast_estimate - error_margin
                upper_bound = contrast_estimate + error_margin
                ci = (lower_bound, upper_bound)
                results.append(ci)
            else:
                print("Warning: Tukey's method only valid for pairwise comparisons.")

    elif method == "Bonferroni":
         for idx, contrast in enumerate(C):
            contrast_estimate = contrast_estimates[idx]
            t_crit = t.ppf(1 - Bonferroni_correction(alpha, len(C)), df_error)
            error_margin = t_crit * np.sqrt(MSE * np.sum(contrast**2 / np.array(n_i)))
            lower_bound = contrast_estimate - error_margin
            upper_bound = contrast_estimate + error_margin
            ci = (lower_bound, upper_bound)
            results.append(ci)
            
    elif method == "Sidak":
        for i, contrast in enumerate(C):
            contrast_estimate = contrast_estimates[i]
            orthogonal = True
            for j in range(i + 1, len(C)):
                if not ANOVA1_is_orthogonal(contrast, C[j]):
                    orthogonal = False
                    break
            if orthogonal:          
                t_crit = t.ppf(1 - Sidak_correction(alpha, len(C)), df_error)
                error_margin = t_crit * np.sqrt(MSE * np.sum(contrast**2 / np.array(n_i)))
                lower_bound = contrast_estimate - error_margin
                upper_bound = contrast_estimate + error_margin
                ci = (lower_bound, upper_bound)
                results.append(ci)
            else:
                print("Warning: Sidak's method only valid for orthogonal comparisons.")

    else:
        return "Error: Method not recognized."

    return results


def ANOVA1_test_linear_combs(X, alpha, C, d, method):
    """
    Tests hypotheses on linear combinations of group means with FWER control.
    
    Parameters:
    - X (list of arrays): Data set where each array contains observations for a group.
    - alpha (float): Familywise error rate.
    - C (np.array): m x I matrix defining linear combinations of I group means.
    - d (np.array): m x 1 vector of hypothesized values for each combination.
    - method (str): Method to control FWER ("Scheffe", "Tukey", "Bonferroni", "Sidak", "best").
    
    Returns:
    - dict: Dictionary with keys as hypothesis and values as (reject, p-value).
    """
    group_means = np.array([np.mean(group) for group in X])  # Calculate group means
    test_statistics = np.dot(C, group_means) - d.flatten()  # Calculate the test statistics
    
    n_groups = len(X)  # Number of groups
    total_samples = sum(len(group) for group in X)  # Total number of samples
    df = total_samples - n_groups  # Degrees of freedom
    MSE = sum(np.var(group, ddof=1) * (len(group) - 1) for group in X) / df  # Mean squared error
    epsilon = 1e-6  # Small value to avoid singular matrix
    group_cov = np.cov([np.mean(group) for group in X], rowvar=False) + epsilon * np.eye(n_groups)  # Covariance matrix
    
    def calculate_p_values(method):
        outcomes = {}
        if method == "Scheffe":
            F_crit = stats.f.ppf(1 - alpha, len(C), df)  # Critical value for F-distribution
            scheffe_multiplier = np.sqrt((len(C) * F_crit) * MSE)  # Scheffe multiplier
            for i, value in enumerate(test_statistics):
                se = np.sqrt(np.dot(np.dot(C[i], np.linalg.inv(group_cov)), C[i].T))  # Standard error
                t_stat = value / se  # Test statistic
                p_value = 1 - stats.f.cdf(t_stat**2, 1, df)  # p-value from F-distribution
                outcomes[f"Hypothesis {i+1}"] = (abs(t_stat) > scheffe_multiplier, p_value)
        elif method == "Tukey":
            flat_data = [item for sublist in X for item in sublist]  # Flatten the data
            groups = np.repeat(range(n_groups), [len(group) for group in X])  # Group labels
            tukey_results = pairwise_tukeyhsd(flat_data, groups, alpha=alpha)  # Tukey's HSD test
            for i, value in enumerate(test_statistics):
                se = np.sqrt(np.dot(np.dot(C[i], np.linalg.inv(group_cov)), C[i].T))  # Standard error
                t_stat = value / se  # Test statistic
                reject, p_value = False, 1.0
                for result in tukey_results.summary().data[1:]:
                    if np.isclose(result[2], t_stat, atol=1e-2):  # Check if t_stat is close to result
                        reject = result[6] < alpha  # Check if p-value < alpha
                        p_value = result[5]
                        break
                outcomes[f"Hypothesis {i+1}"] = (reject, p_value)
        elif method == "Bonferroni":
            t_crit = stats.t.ppf(1 - alpha / (2 * len(C)), df)  # Critical value for t-distribution
            for i, value in enumerate(test_statistics):
                se = np.sqrt(np.dot(np.dot(C[i], np.linalg.inv(group_cov)), C[i].T))  # Standard error
                t_stat = value / se  # Test statistic
                p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df))  # p-value from t-distribution
                outcomes[f"Hypothesis {i+1}"] = (abs(t_stat) > t_crit, p_value)
        elif method == "Sidak":
            sidak_crit = 1 - (1 - alpha) ** (1 / len(C))  # Sidak correction
            t_crit = stats.t.ppf(1 - sidak_crit / 2, df)  # Critical value for t-distribution
            for i, value in enumerate(test_statistics):
                se = np.sqrt(np.dot(np.dot(C[i], np.linalg.inv(group_cov)), C[i].T))  # Standard error
                t_stat = value / se  # Test statistic
                p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df))  # p-value from t-distribution
                outcomes[f"Hypothesis {i+1}"] = (abs(t_stat) > t_crit, p_value)
    
        return outcomes
    
    if method == "best":
        best_outcomes = {}
        for i in range(len(test_statistics)):
            best_p_value = float('inf')
            best_decision = False
            for method in ["Scheffe", "Tukey", "Bonferroni", "Sidak"]:
                outcomes = calculate_p_values(method)  # Calculate p-values for each method
                hypothesis = f"Hypothesis {i+1}"
                decision, p_value = outcomes[hypothesis]
                if p_value < best_p_value:
                    best_p_value = p_value
                    best_decision = decision
            best_outcomes[f"Hypothesis {i+1}"] = (best_decision, best_p_value)
        return best_outcomes

    return calculate_p_values(method)
   

def ANOVA2_partition_TSS(data):
    """
    Partition the sum of squares in a two-way ANOVA layout.
    Parameters:
        data (numpy.array): A 3D numpy array with dimensions (I, J, K)
    Returns:
        dict: Dictionary containing SStotal, SSA, SSB, SSAB, and SSE.
    """
    # Calculating the grand mean
    grand_mean = np.mean(data)

    # Dimensions
    I, J, K = data.shape

    # Calculating SStotal
    SStotal = np.sum((data - grand_mean)**2)

    # Calculating SSA
    means_A = np.mean(data, axis=(1, 2))
    SSA = J * K * np.sum((means_A - grand_mean)**2)

    # Calculating SSB
    means_B = np.mean(data, axis=(0, 2))
    SSB = I * K * np.sum((means_B - grand_mean)**2)

    # Calculating SSAB
    means_AB = np.mean(data, axis=2)
    SSAB = K * np.sum((means_AB - means_A[:, np.newaxis] - means_B + grand_mean)**2)

    # Calculating SSE
    SSE = SStotal - SSA - SSB - SSAB

    return {
        'SStotal': SStotal,
        'SSA': SSA,
        'SSB': SSB,
        'SSAB': SSAB,
        'SSE': SSE
    }


def ANOVA2_MLE(data):
    """
    Calculate the MLE for the parameters of a two-way ANOVA.
    Parameters:
        data (numpy.array): A 3D numpy array with dimensions (I, J, K)
    Returns:
        dict: Dictionary containing estimates for mu, ai, bj, and delta_ij.
    """
    I, J, K = data.shape
    
    # Calculate the overall mean (mu)
    mu = np.mean(data)
    
    # Calculate the effect of factor A (ai)
    means_A = np.mean(data, axis=(1, 2))
    ai = means_A - mu
    
    # Calculate the effect of factor B (bj)
    means_B = np.mean(data, axis=(0, 2))
    bj = means_B - mu
    
    # Calculate the interaction effect (delta_ij)
    means_AB = np.mean(data, axis=2)
    delta_ij = means_AB - (ai[:, np.newaxis] + bj + mu)
    
    # Adjust ai and bj to ensure their sums are zero
    ai -= np.mean(ai)
    bj -= np.mean(bj)
    
    # Adjust delta_ij to ensure their sums are zero
    for i in range(I):
        delta_ij[i, :] -= np.mean(delta_ij[i, :])
    for j in range(J):
        delta_ij[:, j] -= np.mean(delta_ij[:, j])
    
    return {
        'mu': mu,
        'ai': ai,
        'bj': bj,
        'delta_ij': delta_ij
    }

def ANOVA2_test_equality(data, alpha, test):
    """
    Perform one of the basic three tests in the two-way ANOVA layout.
    Parameters:
        data (numpy.array): A 3D numpy array with dimensions (I, J, K)
        alpha (float): Significance level
        test (str): The test to perform, should be one of "A", "B", or "AB"
    Returns:
        None: Prints the ANOVA table row for the specified test
    """
    # Calculate MLEs using ANOVA2_MLE
    mle_results = ANOVA2_MLE(data)
    
    # Calculate sum of squares using ANOVA2_partition_TSS
    ss_results = ANOVA2_partition_TSS(data)
    
    I, J, K = data.shape
    
    # Extracting the necessary components
    SSA = ss_results['SSA']
    SSB = ss_results['SSB']
    SSAB = ss_results['SSAB']
    SSE = ss_results['SSE']
    
    df_A = I - 1
    df_B = J - 1
    df_AB = (I - 1) * (J - 1)
    df_E = I * J * (K - 1)
    
    MSA = SSA / df_A
    MSB = SSB / df_B
    MSAB = SSAB / df_AB
    MSE = SSE / df_E
    
    if test == "A":
        F = MSA / MSE
        p_value = 1 - stats.f.cdf(F, df_A, df_E)
        print(f"Source: A\nDegrees of freedom: {df_A}\nSS: {SSA:.2f}\nMS: {MSA:.2f}\nF: {F:.2f}\np-value: {p_value:.4f}")
    elif test == "B":
        F = MSB / MSE
        p_value = 1 - stats.f.cdf(F, df_B, df_E)
        print(f"Source: B\nDegrees of freedom: {df_B}\nSS: {SSB:.2f}\nMS: {MSB:.2f}\nF: {F:.2f}\np-value: {p_value:.4f}")
    elif test == "AB":
        F = MSAB / MSE
        p_value = 1 - stats.f.cdf(F, df_AB, df_E)
        print(f"Source: AB\nDegrees of freedom: {df_AB}\nSS: {SSAB:.2f}\nMS: {MSAB:.2f}\nF: {F:.2f}\np-value: {p_value:.4f}")
    else:
        print("Invalid test type. Please choose 'A', 'B', or 'AB'.")
        
        
        
def Mult_LR_Least_squares(X, y):
    """
    Find the least squares solution for a multiple linear regression model.
    
    Parameters:
        X (numpy.ndarray): An n x (k+1) matrix of predictors.
        y (numpy.ndarray): An n x 1 vector of responses.
        
    Returns:
        beta_hat (numpy.ndarray): Estimated coefficients for the predictors.
        sigma2_hat (float): Unbiased estimate of the variance of the residuals.
        sigma2_hat_biased (float): Biased estimate of the variance of the residuals.
    """
    
    # Convert X to a numpy array
    X = np.array(X)
    
    # Convert y to a numpy array
    y = np.array(y)
    
    # Compute the maximum likelihood estimate of beta (beta_hat)
    beta_hat = np.linalg.inv(X.T @ X) @ X.T @ y
    
    # Compute the residuals
    residuals = y - X @ beta_hat
    
    # Compute the unbiased estimate of sigma^2 (sigma2_hat)
    sigma2_hat = np.sum(residuals**2) / (len(y) - X.shape[1])
    
    # Compute the biased estimate of sigma^2
    sigma2_hat_biased = np.sum(residuals**2) / len(y)
    
    # Return the estimates of beta, unbiased sigma^2, and biased sigma^2
    return beta_hat, sigma2_hat, sigma2_hat_biased

def Mult_LR_partition_TSS(X, y):
    """
    Partition the Total Sum of Squares (TSS) into the Regression Sum of Squares (RegSS) 
    and the Residual Sum of Squares (RSS) for a multiple linear regression model.
    
    Parameters:
        X (numpy.ndarray): An n x (k+1) matrix of predictors.
        y (numpy.ndarray): An n x 1 vector of responses.
        
    Returns:
        TSS (float): Total Sum of Squares.
        RegSS (float): Regression Sum of Squares.
        RSS (float): Residual Sum of Squares.
    """
    # Convert X and y to numpy arrays (ensure they are numpy arrays)
    X = np.array(X)
    y = np.array(y)
    
    # Calculate the mean of the response variable y
    y_mean = np.mean(y)
    
    # Calculate the Total Sum of Squares (TSS)
    TSS = np.sum((y - y_mean)**2)
    
    # Fit the multiple linear regression model to get the estimated coefficients
    beta_hat, _, _ = Mult_LR_Least_squares(X, y)
    
    # Compute the predicted values of y (y_hat)
    y_hat = X @ beta_hat
    
    # Calculate the Residual Sum of Squares (RSS)
    RSS = np.sum((y - y_hat)**2)
    
    # Calculate the Regression Sum of Squares (RegSS)
    RegSS = TSS - RSS
    
    # Return the TSS, RegSS, and RSS
    return TSS, RegSS, RSS

def Mult_normLR_simul_CI(X, y, alpha):
    """
    Calculate simultaneous confidence intervals for the coefficients of a multiple linear regression model.
    
    Parameters:
        X (numpy.ndarray): An n x (k+1) matrix of predictors.
        y (numpy.ndarray): An n x 1 vector of responses.
        alpha (float): Significance level for the confidence intervals.
        
    Returns:
        intervals (list of tuples): List of confidence intervals for each coefficient.
    """
    # Convert X and y to numpy arrays (ensure they are numpy arrays)
    X = np.array(X)
    y = np.array(y)
    
    # Fit the multiple linear regression model to get the estimated coefficients and error variance
    beta_hat, sigma2_hat, _ = Mult_LR_Least_squares(X, y)
    
    # Get the number of observations (n) and the number of predictors (k)
    n, k = X.shape
    
    # Calculate the covariance matrix of the estimated coefficients
    cov_beta_hat = sigma2_hat * np.linalg.inv(X.T @ X)
    
    # Determine the critical value from the t-distribution, adjusted for multiple comparisons
    crit_value = t.ppf(1 - alpha / (2 * k), n - k)
    
    # Initialize a list to store the confidence intervals
    intervals = []
    
    # Calculate the confidence intervals for each coefficient
    for i in range(k):
        # Compute the margin of error
        margin_of_error = crit_value * np.sqrt(cov_beta_hat[i, i])
        # Append the confidence interval for the i-th coefficient to the list
        intervals.append((beta_hat[i] - margin_of_error, beta_hat[i] + margin_of_error))
    
    # Return the list of confidence intervals
    return intervals

def Mult_norm_LR_CR(X, y, C, alpha): 
    """
    Calculates the 100(1 - alpha)% confidence region for Cβ according to the normal multiple linear regression model.
    
    Parameters:
    X (np.ndarray): An n x (k+1) matrix of predictors.
    y (np.ndarray): An n x 1 vector of responses.
    C (np.ndarray): An r x (k+1) matrix with rank r.
    alpha (float): Significance level.
    
    Returns:
    tuple: Center, covariance matrix, and radius squared of the confidence region.
    """
    # Ensure inputs are numpy arrays
    X = np.asarray(X)
    y = np.asarray(y).reshape(-1, 1)
    C = np.asarray(C)
    
    # Get the dimensions of the matrices
    n, k_plus_1 = X.shape
    r, _ = C.shape
    
    # Calculate the least squares estimate of β
    XtX_inv = np.linalg.inv(X.T @ X)
    beta_hat = XtX_inv @ X.T @ y
    
    # Estimate the variance of the error term
    residuals = y - X @ beta_hat
    sigma_squared_hat = (residuals.T @ residuals) / (n - k_plus_1)
    
    # Calculate the covariance matrix of β
    cov_beta_hat = sigma_squared_hat * XtX_inv
    
    # Calculate the covariance matrix of Cβ
    cov_Cbeta_hat = C @ cov_beta_hat @ C.T
    
    # Calculate the F-statistic critical value
    F_critical = stats.f.ppf(1 - alpha, r, n - k_plus_1)
    
    # Calculate the center of the confidence region
    center = C @ beta_hat

    # Calculate the specifications of the ellipsoid using the given formula
    radius_squared = r * F_critical * sigma_squared_hat
    
    return center.flatten(), cov_Cbeta_hat, radius_squared

def Mult_normLR_is_in_CR(X, y, C, c0, alpha):
    """
    Checks if a point is within the 100(1 - alpha)% confidence region for Cβ.
    
    Parameters:
    X (np.ndarray): An n x (k+1) matrix of predictors.
    y (np.ndarray): An n x 1 vector of responses.
    C (np.ndarray): An r x (k+1) matrix with rank r.
    c0 (np.ndarray): The point to check.
    alpha (float): Significance level.
    
    Returns:
    bool: True if the point is within the confidence region, False otherwise.
    """
    center, cov_matrix, radius_squared = Mult_norm_LR_CR(X, y, C, alpha)
    
    # Calculate the distance based on the given formula
    diff = c0 - center
    test_statistic = diff.T @ np.linalg.inv(cov_matrix) @ diff
    
    return test_statistic <= radius_squared

def Mult_normLR_test_general(X, y, C, c0, alpha):
    """
    Perform a general linear hypothesis test for multiple linear regression.
    
    Parameters:
        X (numpy.ndarray): An n x (k+1) matrix of predictors.
        y (numpy.ndarray): An n x 1 vector of responses.
        C (numpy.ndarray): A matrix representing the linear constraints.
        c0 (numpy.ndarray): A vector representing the hypothesized values under the null hypothesis.
        alpha (float): The significance level for the test.
        
    Returns:
        bool: True if the test statistic is less than or equal to the radius squared, indicating the point is within the confidence region.
    """
    
    # Calculate the center, covariance matrix, and radius squared for the confidence region
    center, cov_matrix, radius_squared = Mult_norm_LR_CR(X, y, C, alpha)
    
    # Calculate the difference between the point c0 and the center of the confidence region
    diff = c0 - center
    
    # Calculate the test statistic using the Mahalanobis distance formula
    test_statistic = diff.T @ np.linalg.inv(cov_matrix) @ diff
    
    # debugging
    #print(f"Test Statistic: {test_statistic}")
    #print(f"Radius Squared: {radius_squared}")
    
    # Return True if the test statistic is less than or equal to the radius squared, indicating the point is within the confidence region
    return test_statistic <= radius_squared


def Mult_normLR_test_comp(X, y, indices, alpha):
    """
    Tests a composite hypothesis for a subset of regression coefficients.

    Parameters:
    X (np.ndarray): An n x (k+1) matrix of predictors.
    y (np.ndarray): An n x 1 vector of responses.
    indices (list of int): Indices of the coefficients to test.
    alpha (float): Significance level.

    Returns:
    bool: True if the test statistic is within the confidence region, False otherwise.
    """
    # Create a matrix C selecting only the rows corresponding to the specified indices
    C = np.eye(X.shape[1])[indices]
    
    # Define the null hypothesis value for the selected coefficients
    c0 = np.zeros(C.shape[0])
    
    # Perform the general hypothesis test
    return Mult_normLR_test_general(X, y, C, c0, alpha)

def Mult_normLR_test_linear_reg(X, y, alpha):
    """
    Tests the null hypothesis that all regression coefficients are zero.

    Parameters:
    X (np.ndarray): An n x (k+1) matrix of predictors.
    y (np.ndarray): An n x 1 vector of responses.
    alpha (float): Significance level.

    Returns:
    bool: True if the test statistic is within the confidence region, False otherwise.
    """
    # Create a matrix C that selects all coefficients (identity matrix)
    C = np.eye(X.shape[1])
    
    # Define the null hypothesis value for all coefficients
    c0 = np.zeros(C.shape[0])
    
    # Perform the general hypothesis test
    return Mult_normLR_test_general(X, y, C, c0, alpha)

def Mult_norm_LR_pred_CI(X, y, D, alpha, method):
    """
    Compute simultaneous confidence bounds for multiple linear regression predictions.
    
    Parameters:
        X (numpy.ndarray): An n x (k+1) matrix of predictors.
        y (numpy.ndarray): An n x 1 vector of responses.
        D (numpy.ndarray): An m x (k+1) matrix of new predictors.
        alpha (float): Significance level.
        method (str): Method for confidence intervals ("Bonferroni", "Scheffe", "best").
        
    Returns:
        predictions (numpy.ndarray): Predictions for the rows of D.
        intervals (numpy.ndarray): Confidence intervals for the predictions.
    """
    # Get the dimensions of X
    n, k_plus_1 = X.shape

    # Fit the multiple linear regression model
    beta_hat, sigma2, _ = Mult_LR_Least_squares(X, y)
    
    # Calculate the standard deviation of the errors
    sigma = np.sqrt(sigma2)
    
    # Compute predictions for the new data points
    predictions = D @ beta_hat

    # Determine the critical value based on the specified method
    if method == "Bonferroni":
        # Bonferroni method: adjust alpha for multiple comparisons
        critical_value = stats.t.ppf(1 - alpha / (2 * D.shape[0]), n - k_plus_1)
    elif method == "Scheffe":
        # Scheffe method: use the F-distribution to adjust for multiple comparisons
        critical_value = np.sqrt((k_plus_1) * stats.f.ppf(1 - alpha, k_plus_1, n - k_plus_1))
    elif method == "best":
        # Best method: choose the smaller of the Bonferroni and Scheffe critical values
        bonferroni_value = stats.t.ppf(1 - alpha / (2 * D.shape[0]), n - k_plus_1)
        scheffe_value = np.sqrt((k_plus_1) * stats.f.ppf(1 - alpha, k_plus_1, n - k_plus_1))
        critical_value = min(bonferroni_value, scheffe_value)
    else:
        # Raise an error if an invalid method is specified
        raise ValueError("Invalid method specified. Use 'Bonferroni', 'Scheffe', or 'best'.")

    # Initialize a list to store the confidence intervals
    intervals = []

    # Calculate the confidence intervals for each row in D
    for i in range(D.shape[0]):
        di = D[i]
        # Compute the variance of the prediction
        variance = sigma2 * (di @ np.linalg.inv(X.T @ X) @ di.T)
        # Compute the margin of error
        margin_of_error = critical_value * np.sqrt(variance)
        # Calculate the lower and upper bounds of the confidence interval
        lower_bound = predictions[i] - margin_of_error
        upper_bound = predictions[i] + margin_of_error
        # Append the interval to the list
        intervals.append((lower_bound, upper_bound))
    
    # Convert the list of intervals to a NumPy array
    intervals = np.array(intervals)
    
    # Return the predictions and their corresponding confidence intervals
    return predictions, intervals











        
        
