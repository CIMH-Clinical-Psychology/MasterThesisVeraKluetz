import numpy as np
import timeit

# Initialize a sample windows_power array with random values
n_bands = 10
n_epochs = 144
n_channels = 306
n_windows = 20
mywindows_power = np.random.rand(n_bands, n_epochs, n_channels, n_windows)
#bands, epochs, channels, windows = windows_power.shape


awindows_power = mywindows_power
bwindows_power = mywindows_power
cwindows_power = mywindows_power

# n_epochs, n_bands_channels, n_windows = mywindows_power.shape
# reshaped_windows_power = windows_power.transpose(0,2,1).reshape(n_epochs * n_windows, n_bands_channels)


# Reshape to the desired shape
#areshaped_windows_power = awindows_power.transpose(0, 2, 1).reshape(n_epochs * n_windows, n_bands_channels)
#breshaped_windows_power = bwindows_power.reshape(n_epochs * n_windows, n_bands_channels)
#creshaped_windows_power = cwindows_power.reshape([-1, n_bands_channels])


areshaped_windows_power = awindows_power.transpose(1, 0, 2, 3).reshape(n_epochs, n_bands * n_channels, n_windows)
breshaped_windows_power = bwindows_power.reshape(n_epochs, n_bands * n_channels, n_windows)
creshaped_windows_power = cwindows_power.reshape([n_epochs, -1, n_windows])

print(areshaped_windows_power[0, 0, :])
print(breshaped_windows_power[0, 0, :])
print(creshaped_windows_power[0, 0, :])









# Define the two statements to be benchmarked
stmt1 = '''
reshaped_windows_power = windows_power.transpose(1, 0, 2, 3).reshape(epochs, bands * channels, windows)
'''

stmt2 = '''
reshaped_windows_power = windows_power.reshape([epochs, -1, windows])
'''

# Set up the timing environment
setup = '''
import numpy as np
bands = 5
epochs = 10
channels = 15
windows = 20
windows_power = np.random.rand(bands, epochs, channels, windows)
bands, epochs, channels, windows = windows_power.shape
'''

# Execute the benchmark
time1 = timeit.timeit(stmt1, setup=setup, number=10000)
time2 = timeit.timeit(stmt2, setup=setup, number=10000)


print(f"Time for statement 1: {time1:.6f} seconds")
print(f"Time for statement 2: {time2:.6f} seconds")