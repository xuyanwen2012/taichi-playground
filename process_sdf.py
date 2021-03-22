import numpy as np
import matplotlib.pyplot as plt

with open('test.npy', 'rb') as f:
    arr = np.load(f)

averaged_arr = np.mean(arr, axis=0)

# Remove outliers based on some deviation
mean = np.mean(averaged_arr)
standard_deviation = np.std(averaged_arr)
distance_from_mean = abs(averaged_arr - mean)
max_deviations = 2
not_outlier = distance_from_mean < max_deviations * standard_deviation
no_outliers = averaged_arr[not_outlier]

print(no_outliers.shape)
print(no_outliers)
print(np.min(no_outliers))
print(np.max(no_outliers))

n, bins, patches = plt.hist(no_outliers.flatten(), 200, alpha=0.75)

plt.savefig('final_out/sdf_100_avg_norm.png')
