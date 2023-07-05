import pandas as pd
import numpy as np
import numba as nb
from numba import njit

## ************* counts func ************* ##
# ①pandas only
def pandas_counts(df):
  return df.groupby('category').size()

# ②pandas + numpy
def pandas_np_counts(df):
  return df.groupby('category').apply(lambda x: np.size([*x], axis=0))

# ③numpy only
def grouped_counts_bincount(categories):
  return np.bincount(categories)

# ④numpy + Numba
@njit("i8[:](i4[:],i8[:])", cache=True)
def grouped_counts_bincount_jit(categories):
  return np.bincount(categories)

## ************* sum func ************* ##
# ①pandas only
def pandas_sum(df):
  return df.groupby('category')['value'].sum()

# ②pandas + numpy
def pandas_np_sum(df):
  return df.groupby('category')['value'].apply(lambda x: np.sum([*x], axis=0))

# ③numpy only
def grouped_sum_bincount(values, categorys):
  return np.bincount(categories, values)

# ④numpy + Numba
@njit("f8[:](i4[:], i8[:])", cache=True)
def grouped_sum_bincount_jit(values, categorys):
  return np.bincount(categories, values)

## ************* mean func ************* ##
# ①pandas only
def pandas_mean(df):
  return df.groupby('category')['value'].mean()

# ②pandas + numpy
def pandas_np_mean(df):
  return df.groupby('category')['value'].apply(lambda x: np.mean([*x], axis=0))

# ③numpy only
def grouped_mean(values, categories):
  counts = grouped_counts_bincount(values, categories)
  sums = grouped_sum_bincount(values, categories)
  return sums / counts

# ④numpy + Numba
@njit("f8[:](i4[:], i8[:])", cache=True)
def grouped_mean_jit_bitcount(values, categories):
  counts = grouped_counts_bincount_jit(values, categories)
  sums = grouped_sum_bincount_jit(values, categories)
  return sums / counts

## ************* std func ************* ##
# ①pandas only
def pandas_std(df):
  return df.groupby('category')['value'].std()

# ②pandas + numpy
def pandas_np_std(df):
  return df.groupby('category')['value'].apply(lambda x: np.std([*x], axis=0))

# ③numpy only
def grouped_std_bincount(values, categories):
  counts = grouped_counts_bincount(values, categories)
  sums = grouped_sum_bincount(values, categories)
  means = sums / counts
  return np.sqrt(np.bincount(categories, (means[categories] - values)**2) / counts)

# ④numpy + Numba
@njit("f8[:](i4[:],i8[:])", cache=True)
def grouped_std_bincount_jit(values, categories):
  counts = grouped_counts_bincount_jit(values, categories)
  sums = grouped_sum_bincount_jit(values, categories)
  means = sums / counts
  return np.sqrt(np.bincount(categories, (means[categories] - values)**2) / counts)

## ************* median func ************* ##
# ①pandas only
def pandas_median(df):
  return df.groupby('category')['value'].median()

# ②pandas + numpy
def pandas_np_median(df):
  return df.groupby('category')['value'].apply(lambda x: np.median([*x], axis=0))

# ③numpy only
def grouped_median(values, categories):
    # Calculate the number of groups
    n_groups = np.max(categories) + 1 # categories are sequential numbers from 0 (integers, duplicates allowed)
    # Store the medians for each group
    medians = np.zeros(n_groups, dtype=np.float64)
    # Initialize the start index of the group
    start = 0

    # Process for each group
    for current_group in range(n_groups):
        # Get the end index of the current group
        end = start
        while end < len(categories) and categories[end] == current_group:
            end += 1

        # Get the median of the current group
        grouped_value = values[start:end]
        medians[current_group] = np.median(grouped_value)
        # Update to the start index of the next group
        start = end

    return medians

# ④numpy + Numba
@njit("f8[:](i4[:],i8[:])", cache=True)
def grouped_median_jit(values, categories):
    # Calculate the number of groups
    n_groups = np.max(categories) + 1 # categories are sequential numbers from 0 (integers, duplicates allowed)
    # Store the medians for each group
    medians = np.zeros(n_groups, dtype=np.float64)
    # Initialize the start index of the group
    start = 0

    # Process for each group
    for current_group in range(n_groups):
        # Get the end index of the current group
        end = start
        while end < len(categories) and categories[end] == current_group:
            end += 1

        # Get the median of the current group
        grouped_value = values[start:end]
        medians[current_group] = np.median(grouped_value)
        # Update to the start index of the next group
        start = end

    return medians
  
## ************* min func ************* ##
# ①pandas only
def pandas_min(df):
  return df.groupby('category')['value'].min()

# ②pandas + numpy
def pandas_np_min(df):
  return df.groupby('category')['value'].apply(lambda x: np.min([*x], axis=0))

# ③numpy only
def grouped_min(values, categories):
    # Calculate the total number of groups
    n_groups = np.max(categories) + 1 # categories are sequential numbers from 0 (integers, duplicates allowed)
    # Store the minimum values for each group
    mins = np.zeros(n_groups, dtype=np.float64)
    # Initialize the start index of the group
    start = 0

    ## Process for each group
    for current_group in range(n_groups):
        # Get the end index of the current group
        end = start
        while end < len(categories) and categories[end] == current_group:
            end += 1

        # Get the minimum value of the current group
        grouped_value = values[start:end]
        mins[current_group] = np.min(grouped_value)
        # Update to the start index of the next group
        start = end

    return mins

# ④numpy + Numba
@njit("f8[:](i4[:],i8[:])", cache=True)
def grouped_min_jit(values, categories):
    # Calculate the total number of groups
    n_groups = np.max(categories) + 1 # categories are sequential numbers from 0 (integers, duplicates allowed)
    # Store the minimum values for each group
    mins = np.zeros(n_groups, dtype=np.float64)
    # Initialize the start index of the group
    start = 0

    ## Process for each group
    for current_group in range(n_groups):
        # Get the end index of the current group
        end = start
        while end < len(categories) and categories[end] == current_group:
            end += 1

        # Get the minimum value of the current group
        grouped_value = values[start:end]
        mins[current_group] = np.min(grouped_value)
        # Update to the start index of the next group
        start = end

    return mins

## ************* max func ************* ##
# ①pandas only
def pandas_max(df):
  return df.groupby('category')['value'].max()

# ②pandas + numpy
def pandas_np_max(df):
  return df.groupby('category')['value'].apply(lambda x: np.max([*x], axis=0))

# ③numpy only
def grouped_max(values, categories):
    # Calculate the total number of groups
    n_groups = np.max(categories) + 1 # categories are sequential numbers from 0 (integers, duplicates allowed)
    # Store the maximum values for each group
    maxs = np.zeros(n_groups, dtype=np.float64)
    # Initialize the start index of the group
    start = 0

    # Process for each group
    for current_group in range(n_groups):
        # Get the end index of the current group
        end = start
        while end < len(categories) and categories[end] == current_group:
            end += 1

        # Get the maximum value of the current group
        grouped_value = values[start:end]
        maxs[current_group] = np.max(grouped_value)
        # Update to the start index of the next group
        start = end

    return maxs

# ④numpy + Numba
@njit("f8[:](i4[:],i8[:])", cache=True)
def grouped_max_jit(values, categories):
    # Calculate the total number of groups
    n_groups = np.max(categories) + 1 # categories are sequential numbers from 0 (integers, duplicates allowed)
    # Store the maximum values for each group
    maxs = np.zeros(n_groups, dtype=np.float64)
    # Initialize the start index of the group
    start = 0

    # Process for each group
    for current_group in range(n_groups):
        # Get the end index of the current group
        end = start
        while end < len(categories) and categories[end] == current_group:
            end += 1

        # Get the maximum value of the current group
        grouped_value = values[start:end]
        maxs[current_group] = np.max(grouped_value)
        # Update to the start index of the next group
        start = end

    return maxs


## ************* data ************* ##
# Change the column name of the data frame to any name you wish.
sorted_df = df.sort_values('category')
values = sorted_df['value'].to_numpy()
categories = pd.factorize(sorted_df['category'], sort=True)[0]
