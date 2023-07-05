# groupby_with_numpy_func

When using pandas aggregation (groupby sum, mean, std, median, etc.) in preprocessing for feature engineering 
and data analysis, I often felt that it was "slow.
I tried to see if it was actually faster than using pandas, 
and found that there were people who used only numpy instead of pandas. 
I decided to test it with numpy because it is possible to use Numba. 

# Versions confirmed to work

* python：3.10.6
* pandas：2.0.1
* numpy ：1.23.5
* Numba ：0.57.1
* benchit : 0.0.6
* matplotlib : 3.7.1
 
# Installation

```bash
pip install pandas
pip install numpy
pip install Numba
pip install benchit
pip install matplotlib
```
 
# Note
 
This program is MIT licensed.

The 'numpy only' or 'numpy + Numba' functions require that the columns for grouping are factorized and sorted.
If multiple columns are to be used for grouping, the values of the columns to be used for grouping should be stored 
as a combination of values in one column in advance, and the columns should be numericalized (factorized).

The signature type you set for the 'Numba' decorator should be changed to match the data you are using.
 
# Author
 
* amcic
* lustig.schlagzu@gmail.com
 
# License
 
"groupby_with_numpy_func" is under [MIT license](https://en.wikipedia.org/wiki/MIT_License).
