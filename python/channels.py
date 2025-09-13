# msprs/channel.py
"""
Channel functions for Additive White Gaussian Noise (AWGN) and other channel models.
"""

import numpy as np


def channel_set_up(eb_no_range, eb):
    """
    Initialize channel configurations.
    """
    # Channel Setup - Use float32 for better memory efficiency
    eb_no_min, eb_no_max, eb_no_step = eb_no_range
    eb_no_db = np.arange(eb_no_min, eb_no_max + eb_no_step, eb_no_step)
    eb_no_linear = 10 ** (eb_no_db / 10)  # type: ignore
    noise_var = eb / (2 * eb_no_linear)
    noise_std = np.sqrt(noise_var)
    return {
        "eb_no_db": eb_no_db,
        "noise_var": noise_var,
        "noise_std": noise_std,
    }


def channelAWGN(signal_length, signal, noise_std):
    """
    Add Additive White Gaussian Noise to signal - Optimized version
    """
    # Pre-allocate noise array and add in-place for memory efficiency
    noise = noise_std * np.random.randn(signal_length)
    return signal + noise


def channelAWGNBatch(signal_batch, noise_std):
    """
    Add AWGN to a batch of signals in parallel

    Parameters
    ----------
    signal_batch : np.ndarray
        Batch of signals (batch_size x signal_length)
    noise_std : float
        Noise standard deviation

    Returns
    -------
    np.ndarray
        Batch of noisy signals
    """
    batch_size, signal_length = signal_batch.shape

    # Generate noise for entire batch
    noise_batch = noise_std * np.random.randn(batch_size, signal_length)

    return signal_batch + noise_batch.astype(signal_batch.dtype)
