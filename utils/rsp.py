#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2024-09-27
# @Author  : Zhaoze Wang
# @Site    : https://github.com/Wangzhaoze/pyradar
# @File    : fft.py
# @IDE     : vscode

"""Radar Signal Processing Module."""
import numpy as np
from typing import Literal
from scipy.fft import fft, fftshift
from typing import Optional, Union

# ######################################################################
# FFT Functions
# ######################################################################

# Constants
speedOfLight = 299792458  # Speed of light in meters/second (default constant)

def windowing(len: int, window_type: Literal['Hamming', 'Blackman', 'Hann']) -> np.ndarray:
    """
    Generate a window function based on the specified type.

    Args:
        len (int): Length of the window.
        window_type (Literal['Hamming', 'Blackman', 'Hann']): Type of the window function.
            - 'Hamming': Hamming window.
            - 'Blackman': Blackman window.
            - 'Hann': Hann window.

    Returns:
        np.ndarray: The generated window function as a 1D numpy array.

    Raises:
        ValueError: If an unsupported window type is provided.
    """
    if window_type == 'Hamming':
        window = np.hamming(len)
    elif window_type == 'Blackman':
        window = np.blackman(len)
    elif window_type == 'Hann':
        window = np.hanning(len)
    else:
        raise ValueError(f"Unsupported window type: {window_type}")

    return window


def range_fft(
    signal: np.ndarray, 
    numRangeBins: Optional[int] = None,
    IdxSamples: int = 1, 
    num_workers: Optional[int] = None,
    window_type: Optional[Literal['Hamming', 'Blackman', 'Hann']] = None
) -> np.ndarray:
    """
    Perform FFT along the range dimension of the ADC cube, with optional windowing.

    Args:
        signal (np.ndarray): Input ADC data cube.
        numRangeBins (Optional[int]): Number of range bins for zero-padding. Defaults to None.
        IdxSamples (int): Index of the range dimension in the input array. Defaults to 1.
        num_workers (Optional[int]): Number of workers for parallel FFT computation. Defaults to None.
        window_type (Optional[Literal['Hamming', 'Blackman', 'Hann']]): Type of window function to apply. Defaults to None.

    Returns:
        np.ndarray: The range FFT spectrum as a numpy array.

    Raises:
        ValueError: If the input signal is not a numpy array.
    """
    if not isinstance(signal, np.ndarray):
        raise ValueError('Input must be a numpy array.')

    # Apply window if specified
    if window_type is not None:
        numADCSamples = signal.shape[IdxSamples]
        window = windowing(numADCSamples, window_type)
        shape = [1] * signal.ndim
        shape[IdxSamples] = numADCSamples
        window = window.reshape(shape)
        signal = signal * window

    range_spectrum = fft(signal, n=numRangeBins, axis=IdxSamples, workers=num_workers)
    return range_spectrum


def doppler_fft(
    signal: np.ndarray, 
    numDopplerBins: Optional[int] = None,
    IdxChirps: int = 2, 
    num_workers: Optional[int] = None,
    window_type: Optional[Literal['Hamming', 'Blackman', 'Hann']] = None
) -> np.ndarray:
    """
    Perform FFT along the Doppler (chirps) dimension of the ADC cube, with optional windowing.

    Args:
        signal (np.ndarray): Input ADC data cube.
        numDopplerBins (Optional[int]): Number of Doppler bins for zero-padding. Defaults to None.
        IdxChirps (int): Index of the Doppler dimension in the input array. Defaults to 2.
        num_workers (Optional[int]): Number of workers for parallel FFT computation. Defaults to None.
        window_type (Optional[Literal['Hamming', 'Blackman', 'Hann']]): Type of window function to apply. Defaults to None.

    Returns:
        np.ndarray: The Doppler FFT spectrum as a numpy array.

    Raises:
        ValueError: If the input signal is not a numpy array.
    """
    if not isinstance(signal, np.ndarray):
        raise ValueError('Input must be a numpy array.')


    # Apply window if specified
    if window_type is not None:
        num_chirps = signal.shape[IdxChirps]
        window = windowing(num_chirps, window_type)
        shape = [1] * signal.ndim
        shape[IdxChirps] = num_chirps
        window = window.reshape(shape)
        signal = signal * window

    doppler_spectrum = fftshift(
        fft(signal, n=numDopplerBins, axis=IdxChirps, workers=num_workers), axes=IdxChirps
    )
    return doppler_spectrum


def angle_fft(
    adc_cube: np.ndarray, 
    numAngleBins: Optional[int] = None,
    IdxVirtualAntennas: int = 0, 
    num_workers: Optional[int] = None,
    window_type: Optional[Literal['Hamming', 'Blackman', 'Hann']] = None
) -> np.ndarray:
    """
    Perform FFT along the virtual antennas (azimuth) dimension of the ADC cube, with optional windowing.

    Args:
        adc_cube (np.ndarray): Input ADC data cube.
        numAngleBins (Optional[int]): Number of angle bins for zero-padding. Defaults to None.
        IdxVirtualAntennas (int): Index of the virtual antennas dimension in the input array. Defaults to 0.
        num_workers (Optional[int]): Number of workers for parallel FFT computation. Defaults to None.
        window_type (Optional[Literal['Hamming', 'Blackman', 'Hann']]): Type of window function to apply. Defaults to None.

    Returns:
        np.ndarray: The angle FFT spectrum as a numpy array.

    Raises:
        ValueError: If the input adc_cube is not a numpy array or does not have three dimensions.
    """
    if not isinstance(adc_cube, np.ndarray):
        raise ValueError('Input must be a numpy array.')
    if adc_cube.ndim != 3:
        raise ValueError('Input array must have exactly three dimensions.')

    # Apply window if specified
    if window_type is not None:
        num_antenna = adc_cube.shape[IdxVirtualAntennas]
        window = windowing(num_antenna, window_type)
        shape = [1, 1, 1]
        shape[IdxVirtualAntennas] = num_antenna
        window = window.reshape(shape)
        adc_cube = adc_cube * window

    azimuth_spectrum = fft(adc_cube, n=numAngleBins, axis=IdxVirtualAntennas, workers=num_workers)
    azimuth_spectrum = fftshift(azimuth_spectrum, axes=IdxVirtualAntennas)
    return azimuth_spectrum


def range_doppler_fft(
    adc_cube: np.ndarray,
    IdxSamples: int = 1,
    IdxChirps: int = 2,
    num_workers: Optional[int] = None,
    range_window_type: Optional[Literal['Hamming', 'Blackman', 'Hann']] = None,
    doppler_window_type: Optional[Literal['Hamming', 'Blackman', 'Hann']] = None,
) -> np.ndarray:
    """
    Perform Range FFT (with optional windowing) followed by Doppler FFT (with optional windowing) to generate a Range-Doppler map.

    Args:
        adc_cube (np.ndarray): Input ADC data cube with three dimensions (virtual antennas, range, chirps).
        IdxSamples (int): Index of the range dimension in the input array. Defaults to 1.
        IdxChirps (int): Index of the Doppler dimension in the input array. Defaults to 2.
        num_workers (Optional[int]): Number of workers for parallel FFT computation. Defaults to None.
        range_window_type (Optional[Literal['Hamming', 'Blackman', 'Hann']]): Type of window function for the range dimension. Defaults to None.
        doppler_window_type (Optional[Literal['Hamming', 'Blackman', 'Hann']]): Type of window function for the Doppler dimension. Defaults to None.

    Returns:
        np.ndarray: The Range-Doppler map as a numpy array.

    Raises:
        ValueError: If the input ADC cube is not a numpy array or does not have three dimensions.
    """
    if not isinstance(adc_cube, np.ndarray):
        raise ValueError('Input must be a numpy array.')
    if adc_cube.ndim != 3:
        raise ValueError('Input array must have exactly three dimensions.')

    range_fft_spectrum = range_fft(
        adc_cube, 
        IdxSamples=IdxSamples, 
        num_workers=num_workers, 
        window_type=range_window_type
    )
    doppler_fft_spectrum = doppler_fft(
        range_fft_spectrum, 
        IdxChirps=IdxChirps, 
        num_workers=num_workers,
        window_type=doppler_window_type
    )
    return doppler_fft_spectrum



# def range_resolution(chirpBandwidth: float) -> float:
#     """
#     Calculate the range resolution of the radar.

#     Parameters:
#         chirpBandwidth (float): Bandwidth of the radar chirp in Hz.

#     Returns:
#         float: Range resolution in meters.

#     Formula:
#         Range resolution = c / (2 * B)
#         Where:
#             c = speed of light (299,792,458 m/s)
#             B = Chirp bandwidth in Hz
#     """
#     return speedOfLight / (2 * chirpBandwidth)


# def range_maximum(adcSampleRate: float, chirpSlope: float) -> float:
#     """
#     Calculate the maximum detectable range of the radar.

#     Parameters:
#         adcSampleRate (float): Analog-to-Digital Converter (ADC) sampling rate in Hz.
#         chirpSlope (float): Chirp slope in Hz/s.

#     Returns:
#         float: Maximum range in meters.

#     Formula:
#         Maximum range = (c * Fs) / (2 * S)
#         Where:
#             c = speed of light (299,792,458 m/s)
#             Fs = ADC sample rate (samples per second)
#             S = Chirp slope (Hz/s)
#     """
#     return speedOfLight * adcSampleRate / (2 * chirpSlope)


# def range_bins(
#     numSamplesPerChirp: int, adcSampleRate: float, chirpSlope: float
# ) -> np.ndarray:
#     """
#     Calculate the range bins for the radar based on maximum range.

#     Parameters:
#         numSamplesPerChirp (int): Number of samples per chirp.
#         adcSampleRate (float): ADC sampling rate in Hz.
#         chirpSlope (float): Chirp slope in Hz/s.

#     Returns:
#         np.ndarray: Array of range bins in meters.

#     Formula:
#         Range axis = linspace(0, Maximum range, numSamplesPerChirp)
#     """
#     # Calculate maximum range
#     range_max = range_maximum(adcSampleRate, chirpSlope)

#     return np.linspace(0, range_max, numSamplesPerChirp)


# def velocity_resolution(
#     carrierFrequency: float, numChirpsPerFrame: int, chirpTime: float
# ) -> float:
#     """
#     Calculate the velocity resolution of the radar.

#     Parameters:
#         carrierFrequency (float): Carrier frequency in Hz.
#         numChirpsPerFrame (int): Number of chirps in one frame.
#         chirpTime (float): Duration of one chirp in seconds.

#     Returns:
#         float: Velocity resolution in m/s.

#     Formula:
#         Velocity resolution = λ / (2 * N * T)
#         Where:
#             λ = wavelength = c / fc (speed of light / carrier frequency)
#             N = number of chirps per frame
#             T = chirp time in seconds
#     """
#     waveLength = speedOfLight / carrierFrequency
#     return waveLength / (2 * numChirpsPerFrame * chirpTime)


# def velocity_maximum(carrierFrequency: float, chirpTime: float) -> float:
#     """
#     Calculate the maximum detectable velocity of the radar.

#     Parameters:
#         carrierFrequency (float): Carrier frequency in Hz.
#         chirpTime (float): Duration of one chirp in seconds.

#     Returns:
#         float: Maximum velocity in m/s.

#     Formula:
#         Maximum velocity = λ / (4 * T)
#         Where:
#             λ = wavelength = c / fc (speed of light / carrier frequency)
#             T = chirp time in seconds
#     """
#     waveLength = speedOfLight / carrierFrequency
#     return waveLength / (4 * chirpTime)


# def velocity_bins(
#     numChirpsPerFrame: int, carrierFrequency: float, chirpTime: float
# ) -> np.ndarray:
#     """
#     Calculate the velocity bins for the radar.

#     Parameters:
#         numChirpsPerFrame (int): Number of chirps in one frame.
#         carrierFrequency (float, optional): Carrier frequency in Hz.
#         chirpTime (float, optional): Duration of one chirp in seconds.

#     Returns:
#         np.ndarray: Array of velocity bins in m/s.

#     Formula:
#         Velocity axis = linspace(-V_max, V_max, numChirpsPerFrame)
#         Where:
#             V_max = Maximum velocity
#     """
#     # Calculate maximum velocity
#     vel_max = velocity_maximum(carrierFrequency, chirpTime)

#     return np.linspace(-vel_max, vel_max, numChirpsPerFrame)


# def azimuth_resolution(
#     carrierFrequency: float, numVirtualAntennas: int, antennaSpacing: float
# ) -> float:
#     """
#     Calculate the azimuth resolution of the radar.

#     Parameters:
#         carrierFrequency (float): Carrier frequency in Hz.
#         numVirtualAntennas (int): Number of virtual antennas in the radar array.
#         antennaSpacing (float): Spacing between antennas in meters.

#     Returns:
#         float: Azimuth resolution in radians.

#     Formula:
#         Azimuth resolution = λ / (N * d)
#         Where:
#             λ = wavelength = c / fc (speed of light / carrier frequency)
#             N = number of virtual antennas
#             d = antenna spacing in meters
#     """
#     waveLength = speedOfLight / carrierFrequency
#     return waveLength / (numVirtualAntennas * antennaSpacing)


# def azimuth_maximum(carrierFrequency: float, antennaSpacing: float) -> float:
#     """
#     Calculate the maximum detectable azimuth angle of the radar.

#     Parameters:
#         carrierFrequency (float): Carrier frequency in Hz.
#         antennaSpacing (float): Spacing between antennas in meters.

#     Returns:
#         float: Maximum azimuth angle in radians.

#     Formula:
#         Maximum azimuth angle = arcsin(λ / (2 * d))
#         Where:
#             λ = wavelength = c / fc (speed of light / carrier frequency)
#             d = antenna spacing in meters
#     """
#     waveLength = speedOfLight / carrierFrequency
#     return np.arcsin(waveLength / (2 * antennaSpacing))


# def azimuth_bins(
#     numAzimuthBins: int, carrierFrequency: float, antennaSpacing: float
# ) -> np.ndarray:
#     """
#     Calculate the azimuth bins for the radar.

#     Parameters:
#         numAzimuthBins (int): Number of samples for azimuth.
#         carrierFrequency (float, optional): Carrier frequency in Hz.
#         antennaSpacing (float, optional): Spacing between antennas in meters.

#     Returns:
#         np.ndarray: Array of azimuth bins in Degrees.

#     Formula:
#         Azimuth axis = linspace(-Azimuth_max, Azimuth_max, numAzimuthBins)
#         Where:
#             Azimuth_max = Maximum azimuth angle
#     """
#     # Calculate maximum azimuth angle
#     azimuth_max = azimuth_maximum(carrierFrequency, antennaSpacing)

#     azimuth_max = np.rad2deg(azimuth_max)

#     # Generate azimuth bins from -azimuth_max to azimuth_max
#     return np.linspace(-azimuth_max, azimuth_max, numAzimuthBins)

