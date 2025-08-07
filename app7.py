import torch
import numpy as np
import matplotlib.pyplot as plt

# Parameters for the primary (wealth) signal
primary_frequency = 8  # Brain signal frequency in Hz (alpha wave for example)
primary_amplitude = 3  # Amplitude of the signal (wealth intensity)
phase_shift = np.pi / 6  # Phase shift for simulating wealth dynamics
time_steps = torch.linspace(0, 4 * np.pi, 1000)  # Time steps for the waveform
density_factor = 4  # Density factor to simulate the magnetic wealth effect

# Parameters for the secondary (storage) signal
storage_frequency = 15  # Frequency for the storage signal
storage_amplitude = 1.5  # Amplitude for the storage signal
storage_phase_shift = np.pi / 3  # Phase shift for the storage dynamics
trigger_time = np.pi  # Time when the signal reaches its "destination"

# Function to generate a sine wave
def generate_waveform(time, frequency, amplitude, phase_shift):
    return amplitude * torch.sin(frequency * time + phase_shift)

# Function to encode wealth as a dense magnetic waveform
def encode_magnetic_wealth_waveform(signal, density_factor):
    return signal * density_factor

# Generate the primary brain signal (dense magnetic wealth signal)
primary_signal = generate_waveform(time_steps, primary_frequency, primary_amplitude, phase_shift)

# Encode wealth data into the primary signal
magnetic_wealth_waveform = encode_magnetic_wealth_waveform(primary_signal, density_factor)

# Function to store data with the secondary frequency
def storage_waveform(time, trigger_time, storage_frequency, storage_amplitude, storage_phase_shift):
    # Create a secondary waveform that is activated after a certain time (trigger_time)
    storage_signal = torch.where(
        time >= trigger_time,  # Condition: time greater than trigger_time
        generate_waveform(time, storage_frequency, storage_amplitude, storage_phase_shift),
        torch.zeros_like(time)  # Else, no signal before the trigger
    )
    return storage_signal

# Generate the secondary storage signal that activates after the primary signal reaches its destination
storage_signal = storage_waveform(time_steps, trigger_time, storage_frequency, storage_amplitude, storage_phase_shift)

# Combine the magnetic wealth waveform with the storage signal
combined_signal = magnetic_wealth_waveform + storage_signal

# Visualize the waveforms
plt.figure(figsize=(10, 6))

# Plot the primary dense magnetic wealth waveform
plt.plot(time_steps.numpy(), magnetic_wealth_waveform.numpy(), label="Primary Wealth Signal", color="blue")

# Plot the secondary storage signal
plt.plot(time_steps.numpy(), storage_signal.numpy(), label="Storage Signal", color="green", linestyle="--")

# Plot the combined signal
plt.plot(time_steps.numpy(), combined_signal.numpy(), label="Combined Signal", color="red", alpha=0.7)

plt.title("NeuroSphere")
plt.xlabel("Time")
plt.ylabel("Signal Amplitude")
plt.legend()
plt.grid(True)
plt.show()