# !pip install benchit
import benchit
import matplotlib.pyplot as plt

# Set the functions to be compared
function_name = 'std'

## ************ Comparison ************ ##
# Generate input data
n_values = 2 ** np.arange(1, 21)
inputs = {}
for n in n_values:
    # sample_indices = np.random.randint(0, len(arr_value), n)
    sample_indices = np.arange(0, n)
    inputs[n] = {
        'pandas_{}'.format(function_name): df.iloc[sample_indices],
        'grouped_{}'.format(function_name): (arr_value[sample_indices], arr_group[sample_indices]),
    }
# pandas_sum takes one argument, so use inputs[n]['pandas_sum'] directly
inputs_for_pandas_function = {n: inputs[n]['pandas_{}'.format(function_name)] for n in n_values}
# grouped_sum and grouped_sum_jit take two arguments, so keep the original format
inputs_for_grouped_function = {n: inputs[n]['grouped_{}'.format(function_name)] for n in n_values}

# Calculation
t_pandas_function = benchit.timings([pandas_std], inputs_for_pandas_function, input_name='Array-length')
t_pandas_np_function = benchit.timings([pandas_np_std], inputs_for_pandas_function, input_name='Array-length')
t_grouped_function = benchit.timings([grouped_std_bincount], inputs_for_grouped_function, multivar=True, input_name='Array-length')
t_grouped_function_jit_b = benchit.timings([grouped_std_bincount_jit], inputs_for_grouped_function, multivar=True, input_name='Array-length')
# t_grouped_function = benchit.timings([grouped_max], inputs_for_grouped_function, multivar=True, input_name='Array-length')
# t_grouped_function_jit = benchit.timings([grouped_max_jit], inputs_for_grouped_function, multivar=True, input_name='Array-length')

# Drawing
# Compare all results on a single graph
# Preparing to feed data to matplotlib.pyplot because it couldn't be summarized in one graph by benchit
# Convert the BenchmarkObj obtained from benchit.timings() to a dataframe
df_pandas_function = t_pandas_function.to_dataframe()
df_pandas_np_function = t_pandas_np_function.to_dataframe()
df_grouped_function = t_grouped_function.to_dataframe()
df_grouped_function_jit_b = t_grouped_function_jit_b.to_dataframe()
# df_grouped_function_jit = t_grouped_function_jit.to_dataframe()

# Plotting with matplotlib.pyplot
fig, ax = plt.subplots()
# Retrieve the x-axis and y-axis values and plot
# Adjust the line width using the linewidth parameter
ax.plot(df_pandas_function.index, df_pandas_function['pandas_{}'.format(function_name)], label='①pandas', linewidth=2.0)
ax.plot(df_pandas_np_function.index, df_pandas_np_function['pandas_np_{}'.format(function_name)], label='②pandas+numpy', linewidth=2.0)
ax.plot(df_grouped_function.index, df_grouped_function['grouped_{}_bincount'.format(function_name)], label='③numpy', linewidth=2.0)
ax.plot(df_grouped_function_jit_b.index, df_grouped_function_jit_b['grouped_{}_bincount_jit'.format(function_name)], label='④numpy+Numba', linewidth=2.0)
# ax.plot(df_grouped_function.index, df_grouped_function['grouped_{}'.format(function_name)], label='③numpy', linewidth=2.0)
# ax.plot(df_grouped_function_jit.index, df_grouped_function_jit['grouped_{}_jit'.format(function_name)], label='④numpy+Numba', linewidth=2.0)

ax.set_xlabel('Array-length')
ax.set_ylabel('Time (s)')
ax.set_xscale('log')
ax.set_yscale('log')
ax.legend()

plt.show()
