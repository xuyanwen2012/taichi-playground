import numpy as np
import matplotlib.pyplot as plt

# Load ready computed results
with open('test.npy', 'rb') as f:
    arr = np.load(f)

averaged_arr = np.mean(arr, axis=0)

# # Remove outliers based on some deviation
mean = np.mean(averaged_arr)
standard_deviation = np.std(averaged_arr)
distance_from_mean = abs(averaged_arr - mean)
max_deviations = 2
# not_outlier = distance_from_mean < max_deviations * standard_deviation
# no_outliers = averaged_arr[not_outlier]

# new_arr = np.zeros((720, 1280))

averaged_arr = np.clip(averaged_arr, 0, 25000)

# averaged_arr /= np.max(np.abs(averaged_arr), axis=0)
# averaged_arr /= np.max(np.abs(averaged_arr), axis=0)
# averaged_arr *= 255

fig, ax = plt.subplots()
im = ax.imshow(np.rot90(averaged_arr))
plt.savefig('final_out/sdf_heat.png')

# plt.show()

# # for i in range(1280):
# #     for j in range(720):
# #         new_arr[i, j] =
#
# #
# # print(no_outliers.shape)
# # print(no_outliers)
# # print(np.min(no_outliers))
# # print(np.max(no_outliers))
# #
# # # n, bins, patches = plt.hist(no_outliers.flatten(), 200, alpha=0.75)
#
# print(averaged_arr.shape)
# print(np.max(averaged_arr))
# print(np.min(averaged_arr))
#
# fig, axs = plt.subplots()
# axs.hist2d(averaged_arr[:, 0], averaged_arr[:, 1], bins=200,
#            range=[[0, 25000], [0, 25000]])
#
# # n, bins, patches = plt.hist2d(averaged_arr[:, 0], averaged_arr[:, 1], 200)
#
# # plt.savefig('final_out/sdf2d_100_avg_norm.png')
# plt.show()
