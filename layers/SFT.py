import torch.fft as fft
import torch
import torch.nn as nn

class Spectral_Filter_Transform(nn.Module):
    def __init__(self, window_size, k):
        super(Spectral_Filter_Transform, self).__init__()
        self.window_size = window_size  # Size of the Hamming window
        self.k = k  # Number of top frequency components to keep
        self.window = torch.hamming_window(window_size)  # Create the Hamming window

    def forward(self, x):
        # Input shape: x: [Batch, Time, Variate]
        X_fft = fft.rfft(x, dim=1)  # Apply real FFT, shape: [Batch, Freq, Variate]
        filtered_fft = self.filter(X_fft)  # Filter the FFT
        x_filtered = fft.irfft(filtered_fft, dim=1)  # Inverse FFT, shape: [Batch, Time, Variate]

        self.window = self.window.to(x_filtered.device)  # Move the window to the same device as the input
        batch, time, feature = x_filtered.shape  # Extract dimensions

        half_window = self.window_size // 2
        # Reflective padding
        left_padding = x_filtered[:, :half_window, :].flip(dims=[1])
        right_padding = x_filtered[:, -half_window:, :].flip(dims=[1])
        padded_x = torch.cat([left_padding, x_filtered, right_padding], dim=1)  # Padded shape: [Batch, Time + window_size - 1, Variate]

        # Apply window and compute weighted average
        filtered = torch.zeros_like(x_filtered)
        for i in range(time):
            windowed_data = padded_x[:, i:i + self.window_size, :]  # Shape: [Batch, window_size, Variate]
            filtered[:, i, :] = (torch.sum(windowed_data * self.window.view(1, -1, 1), dim=1) / 
                                 torch.sum(self.window))

        return filtered  # Output shape: [Batch, Time, Variate]

    def filter(self, x_fft):
        # Magnitudes of frequency components, shape: [Batch, Freq, Variate]
        magnitudes = x_fft.abs()

        # Top k frequency components
        _, indices = torch.topk(magnitudes, self.k, dim=1, largest=True)

        # Initialize a zero tensor of the same shape as x_fft
        filtered = torch.zeros_like(x_fft)

        batch_size, freq_size, num_features = x_fft.shape
        batch_indices = torch.arange(batch_size).view(-1, 1, 1).to(x_fft.device)
        feature_indices = torch.arange(num_features).view(1, 1, -1).to(x_fft.device)

        # Advanced indexing to place values in the filtered tensor
        filtered[batch_indices, indices, feature_indices] = x_fft[batch_indices, indices, feature_indices]

        return filtered  # Filtered shape: [Batch, Freq, Variate]