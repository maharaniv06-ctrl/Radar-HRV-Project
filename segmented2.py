import numpy as np
from scipy import signal
from scipy.fft import fft2 
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk, FigureCanvasTkAgg
from scipy.signal import butter, filtfilt, detrend, welch, savgol_filter, find_peaks, medfilt
from scipy import stats
from scipy.interpolate import interp1d
from scipy.signal import welch
from spectrum import arburg
import h5py
import pandas as pd
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from matplotlib.figure import Figure
from matplotlib.widgets import SpanSelector
import traceback
import spectrum
import time
import os
import math
from scipy.stats import pearsonr
from scipy.interpolate import interp1d
from scipy.signal import detrend
from scipy.interpolate import interp1d
from scipy.signal import detrend
from scipy.fft import fft2
from matplotlib.ticker import ScalarFormatter

class EEMDPaperAnalyzer:
    
    def __init__(self):
        self.signals = None
        self.phase_imfs = None
        self.heartrate_imfs = None
        self.imf_freq_results = None
        self.latest_combined_imf = None
        
        self.ecg_data = None
        self.ecg_time = None
        self.ecg_fs = 150  # Default sampling rate
        self.ecg_loaded = False
        
        self.original_timestamps = None
        self.cropped_timestamps = None
        self.cropped_combined_imf = None
        self.is_data_cropped = False
        self.is_segmented = False
        self.seg_start_var = None
        self.seg_end_var = None
        self.seg_info_var = None
        self.hr_imf_vars = []  # Initialize as empty list
        self.use_derivative_var = None
        self.enable_mav_var = None
        self.mav_window_var = None

        # ECG segmentation variables
        self.ecg_original_data = None
        self.ecg_original_time = None
        self.ecg_segmented = None
        self.ecg_segmented_timestamps = None
        self.is_ecg_segmented = False

        # Peak detection variables
        self.peak_detection_signal = None
        self.peak_detection_timestamps = None
        self.detected_radar_peaks = None
        self.radar_peak_times = None
        self.radar_peak_values = None
        self.radar_algo_info = None
        self.detected_ecg_peaks = None
        self.ecg_peak_times = None
        self.ecg_peak_values = None
        self.ecg_algo_info = None
    
        # HRV analysis variables
        self.radar_hrv = None
        self.ecg_hrv = None
        self.hrv_comparison = None
        self.peak_editing_mode = False
        self.manual_radar_peaks = None
        self.edited_radar_peaks = None
        self.edited_radar_values = None
        self.edit_segment_start = 0  # Current viewing window start
        self.edit_segment_duration = 10  # 10 second viewing window
        self.original_radar_peaks = None  # Store original peaks for reset
                

    def enhanced_bandpass_filter(self, data, lowcut, highcut, fs, order=2):
        try:
            # Ensure data is clean
            data = np.array(data, dtype=float)
            
            # Remove any NaN or infinite values
            data = data[np.isfinite(data)]
            
            if len(data) == 0:
                raise ValueError("No valid data after cleaning")
            
            # Apply detrending first
            data = detrend(data)
            
            # Normalize
            data = (data - np.mean(data)) / np.std(data)
            
            # Design filter
            nyq = 0.5 * fs
            if lowcut >= nyq or highcut >= nyq:
                raise ValueError(f"Filter frequencies must be less than Nyquist frequency ({nyq} Hz)")
            
            low = lowcut / nyq
            high = highcut / nyq
            
            # Ensure frequencies are within valid range
            low = max(low, 0.01)  # Minimum frequency
            high = min(high, 0.99)  # Maximum frequency
            
            if low >= high:
                raise ValueError("Low cutoff frequency must be less than high cutoff frequency")
            
            b, a = butter(order, [low, high], btype='band')
            y = filtfilt(b, a, data)
            
            return y
            
        except Exception as e:
            print(f"Warning: Enhanced bandpass filter failed ({e}), using basic filter")
            # Fallback to basic filter
            return self.bandpass_filter(data, lowcut, highcut, fs, order)

    def bandpass_filter(self, data, lowcut, highcut, fs, order=2):
        """Basic bandpass filter"""
        data = detrend(data)
        data = (data - np.mean(data)) / np.std(data)
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        return filtfilt(b, a, data)

    def moving_average(self, data, window_size):
        if len(data) < window_size:
            return data
        return np.convolve(data, np.ones(window_size)/window_size, mode='same')

    def process_radar_signals(self, file_path):
        """Process radar signals with enhanced processing"""
        try:
            with h5py.File(file_path, 'r') as f:
                print("Available datasets:", list(f.keys()))
                print("File attributes:", dict(f.attrs))
                
                I = np.array(f['i_channel'])
                Q = np.array(f['q_channel'])
                timestamps = np.array(f['timestamp'])
                fs = f.attrs['actual_sample_rate']
                print(f"Detected sampling rate: {fs} Hz")

            # Calculate phase using arctangent
            raw_phase = np.unwrap(np.arctan2(Q, I))  # This is what we want to use for EEMD later
            phase = raw_phase.copy()  # This will be processed for HR and BR

            # Apply enhanced processing
            print("Applying enhanced signal processing...")

            # Step 1: Detrending
            phase = detrend(phase)
            # Step 2: Normalization
            phase = phase - np.mean(phase)
            phase = phase / np.std(phase)
            # Step 3: Apply enhanced bandpass filter for heart rate frequency range
            heartrate_signal = self.enhanced_bandpass_filter(phase, 1.0, 6.0, fs, order=2)
            # Step 4: Extract breathing signal (lower frequency range)
            breathing_signal = self.enhanced_bandpass_filter(phase, 0.6, 3.0, fs, order=2)
            # Apply moving average to both signals
            breathing_signal = self.moving_average(breathing_signal, 50)
            heartrate_signal = self.moving_average(heartrate_signal, 50)
            # Calculate heart rate derivative
            dt = np.diff(timestamps)[0]  # Time step
            heartrate_derivative = np.gradient(heartrate_signal, dt)
            # Calculate breathing rate using the enhanced method
            br_freqs, br_psd = welch(breathing_signal, fs, nperseg=int(fs*10))
            br_mask = (br_freqs >= 0.1) & (br_freqs <= 0.4)
            if any(br_mask):
                br_peak_freq = br_freqs[br_mask][np.argmax(br_psd[br_mask])]
                br = br_peak_freq * 60
            else:
                br = 0
            return {
                'I': I,
                'Q': Q,
                'phase': raw_phase,  # This will be used for EEMD - already processed
                'breathing': breathing_signal,
                'heartrate': heartrate_signal,
                'heartrate_derivative': heartrate_derivative,
                'fs': fs,
                'timestamps': timestamps,
                'br': br
            }
        except Exception as e:
            print(f"Error processing signals: {e}")
            raise

    #EEMD
    def emd(self, data, max_iterations=10):
        def find_extrema(x):
            return signal.find_peaks(x)[0], signal.find_peaks(-x)[0]

        def spline_envelope(x, indices):
            if len(indices) < 2:
                return np.zeros_like(x)
            return np.interp(np.arange(len(x)), indices, x[indices])

        def is_imf(x):
            maxima_indices, minima_indices = find_extrema(x)
            if len(maxima_indices) + len(minima_indices) < 2:
                return False
            
            mean_env = np.zeros_like(x)
            if len(maxima_indices) > 1 and len(minima_indices) > 1:
                upper_env = spline_envelope(x, maxima_indices)
                lower_env = spline_envelope(x, minima_indices)
                mean_env = (upper_env + lower_env) / 2
            
            return np.all(np.abs(mean_env) <= 0.1 * np.abs(x))

        imfs = []
        remainder = data.copy()

        while not is_imf(remainder) and len(imfs) < 10:
            current_imf = remainder.copy()
            
            for _ in range(max_iterations):
                maxima_indices, minima_indices = find_extrema(current_imf)
                
                if len(maxima_indices) < 2 or len(minima_indices) < 2:
                    break
                
                upper_envelope = spline_envelope(current_imf, maxima_indices)
                lower_envelope = spline_envelope(current_imf, minima_indices)
                mean_envelope = (upper_envelope + lower_envelope) / 2
                
                previous_imf = current_imf.copy()
                current_imf = current_imf - mean_envelope
                
                if is_imf(current_imf) or np.sum(np.abs(previous_imf - current_imf)) / np.sum(np.abs(previous_imf)) < 0.01:
                    break
            
            imfs.append(current_imf)
            remainder -= current_imf
            
            if np.all(np.abs(remainder) < 1e-10):
                break

        if not np.all(np.abs(remainder) < 1e-10):
            imfs.append(remainder)
        
        return imfs

    def eemd(self, data, num_ensembles=100, noise_amplitude=0.2):
        data_normalized = (data - np.mean(data)) / np.std(data)
        max_imf_count = 10
        ensemble_imfs = [[] for _ in range(max_imf_count)]
        
        for _ in range(num_ensembles):
            noisy_signal = data_normalized + noise_amplitude * np.random.normal(0, 1, len(data_normalized))
            noisy_imfs = self.emd(noisy_signal)
            
            for i, imf in enumerate(noisy_imfs[:max_imf_count]):
                ensemble_imfs[i].append(imf)
        
        final_imfs = []
        for imf_ensemble in ensemble_imfs:
            if imf_ensemble:
                final_imfs.append(np.mean(imf_ensemble, axis=0))
        
        return final_imfs
    
    def analyze_imf_frequencies(self, imfs, fs, num_peaks=3):
        results = []
        
        for i, imf in enumerate(imfs):
            fft_result = np.fft.fft(imf)
            freqs = np.fft.fftfreq(len(imf), 1/fs)
            
            pos_mask = freqs > 0
            freqs_pos = freqs[pos_mask]
            mags = np.abs(fft_result)[pos_mask]
            
            if len(mags) > 0:
                sorted_indices = np.argsort(mags)[::-1]
                top_indices = sorted_indices[:num_peaks]
                top_freqs = freqs_pos[top_indices]
                top_mags = mags[top_indices]
                
                total_energy = np.sum(mags**2)
                energy_percentage = (top_mags**2 / total_energy) * 100
                
                results.append({
                    'imf_num': i+1,
                    'dominant_freqs': top_freqs,
                    'magnitudes': top_mags,
                    'energy_percentage': energy_percentage,
                    'total_energy': total_energy
                })
        return results
    def load_ecg_data(self):
        file_path = filedialog.askopenfilename(
            title="Select ECG CSV File",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if not file_path:
            return False 
        try:
            df = self.load_csv_data(file_path)
            
            # Use first column as ECG data, create time vector
            raw_ecg_data = df.iloc[:, 0].values
            self.ecg_time = np.arange(len(raw_ecg_data)) / self.ecg_fs
            
            cleaned_ecg = self.clean_ecg_data(raw_ecg_data)
            filtered_ecg = self.apply_enhanced_ecg_filtering(cleaned_ecg, self.ecg_fs)
            
            normalized_ecg = (filtered_ecg - np.min(filtered_ecg)) / (np.max(filtered_ecg) - np.min(filtered_ecg))
            self.ecg_data = normalized_ecg * (np.max(cleaned_ecg) - np.min(cleaned_ecg)) + np.min(cleaned_ecg)
            self.ecg_data = (self.ecg_data - np.mean(self.ecg_data)) / np.std(self.ecg_data)
            
            r_peaks = self.detect_r_peaks(filtered_ecg, self.ecg_fs)
            bpm = self.calculate_bpm_from_peaks(r_peaks, self.ecg_fs)
            
            self.ecg_loaded = True
            
            messagebox.showinfo("Success", 
                f"ECG data processed successfully!\n"
                f"Samples: {len(self.ecg_data)}\n"
                f"Sampling Rate: {self.ecg_fs} Hz\n"
                f"Detected R-peaks: {len(r_peaks)}\n"
                f"Estimated BPM: {bpm:.1f}")
            
            if hasattr(self, 'results_text'):
                self.results_text.insert(tk.END, f"\nECG Data Processed:\n")
                self.results_text.insert(tk.END, f"  Samples: {len(self.ecg_data)}\n")
                self.results_text.insert(tk.END, f"  Duration: {self.ecg_time[-1]:.2f} s\n")
                self.results_text.insert(tk.END, f"  Sampling Rate: {self.ecg_fs} Hz\n")
                self.results_text.insert(tk.END, f"  R-peaks detected: {len(r_peaks)}\n")
                self.results_text.insert(tk.END, f"  Estimated BPM: {bpm:.1f}\n\n")
            
            return True
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load ECG data: {str(e)}")
            return False

    def load_csv_data(self, file_path):
        try:
            df = pd.read_csv(file_path)
            if len(df.columns) == 1 or 'Unnamed' in df.columns[0]:
                df = pd.read_csv(file_path, header=None)
        except:
            df = pd.read_csv(file_path, header=None)
        return df

    def clean_ecg_data(self, ecg_values):
        z_scores = stats.zscore(ecg_values)
        threshold = 3.0
        outlier_indices = np.where(np.abs(z_scores) > threshold)[0]
        
        cleaned_data = ecg_values.copy()
        
        for idx in outlier_indices:
            before_points = []
            after_points = []
            
            i = 1
            while len(before_points) < 5 and idx - i >= 0:
                if idx - i not in outlier_indices:
                    before_points.append(ecg_values[idx - i])
                i += 1
            
            i = 1
            while len(after_points) < 5 and idx + i < len(ecg_values):
                if idx + i not in outlier_indices:
                    after_points.append(ecg_values[idx + i])
                i += 1
            
            surrounding_points = before_points + after_points
            if surrounding_points:
                cleaned_data[idx] = np.median(surrounding_points)
        
        return cleaned_data

    def apply_enhanced_ecg_filtering(self, ecg_values, fs=150):
        lowcut, highcut = 0.5, 40.0
        nyquist = 0.5 * fs
        low, high = lowcut / nyquist, highcut / nyquist
        
        b, a = butter(4, [low, high], btype='band')
        filtered_ecg = filtfilt(b, a, ecg_values)
        
        notch_freq = 60.0
        b_notch, a_notch = butter(2, [(notch_freq - 1) / nyquist, (notch_freq + 1) / nyquist], btype='bandstop')
        filtered_ecg = filtfilt(b_notch, a_notch, filtered_ecg)
        
        window_length = int(fs * 0.05)
        if window_length % 2 == 0:
            window_length += 1
        if window_length < 3:
            window_length = 3
        polyorder = 3
        filtered_ecg = savgol_filter(filtered_ecg, window_length, polyorder)
        
        window_size = int(fs * 0.01)
        if window_size < 3:
            window_size = 3
        kernel = np.ones(window_size) / window_size
        smoothed_ecg = np.convolve(filtered_ecg, kernel, mode='same')
        
        return smoothed_ecg

    def detect_r_peaks(self, ecg_values, fs=150):
        distance = int(0.5 * fs)
        all_peaks, properties = find_peaks(ecg_values, distance=distance)
        
        peak_heights = ecg_values[all_peaks]
        top_peaks_count = max(3, int(0.3 * len(peak_heights)))
        top_heights = np.sort(peak_heights)[-top_peaks_count:]
        mean_top_height = np.mean(top_heights)
        
        threshold = 0.6 * mean_top_height
        r_peaks = all_peaks[peak_heights >= threshold]
        
        return r_peaks

    def calculate_bpm_from_peaks(self, r_peaks, fs=150):
        if len(r_peaks) < 2:
            return 0
        
        rr_intervals = np.diff(r_peaks) / fs
        hr_instantaneous = 60 / rr_intervals
        valid_hrs = hr_instantaneous[(hr_instantaneous >= 30) & (hr_instantaneous <= 220)]
        
        if len(valid_hrs) == 0:
            return 0
        
        return np.mean(valid_hrs)

    def synchronize_signals(self, imf_signal, imf_time, ecg_signal, ecg_time):
        # Ensure we have numpy arrays
        imf_signal = np.array(imf_signal)
        imf_time = np.array(imf_time)
        ecg_signal = np.array(ecg_signal)
        ecg_time = np.array(ecg_time)
        
        # Find overlapping time range
        start_time = max(imf_time[0], ecg_time[0])
        end_time = min(imf_time[-1], ecg_time[-1])
        
        if start_time >= end_time:
            print(f"DEBUG: No time overlap - IMF: [{imf_time[0]:.2f}, {imf_time[-1]:.2f}], ECG: [{ecg_time[0]:.2f}, {ecg_time[-1]:.2f}]")
            return None, None, None, None
        
        # Calculate optimal sampling rate (use lower of the two to avoid upsampling)
        imf_fs = (len(imf_signal) - 1) / (imf_time[-1] - imf_time[0])
        ecg_fs = (len(ecg_signal) - 1) / (ecg_time[-1] - ecg_time[0])
        target_fs = min(imf_fs, ecg_fs, self.signals['fs'], self.ecg_fs)
        
        # Create common time vector
        duration = end_time - start_time
        num_samples = int(duration * target_fs)
        common_time = np.linspace(start_time, end_time, num_samples)
        
        # Interpolate both signals to common time base
        try:
            from scipy.interpolate import interp1d
            
            # Create interpolation functions
            imf_interp_func = interp1d(imf_time, imf_signal, kind='linear', 
                                    bounds_error=False, fill_value=0)
            ecg_interp_func = interp1d(ecg_time, ecg_signal, kind='linear', 
                                    bounds_error=False, fill_value=0)
            
            # Interpolate to common time base
            imf_interp = imf_interp_func(common_time)
            ecg_interp = ecg_interp_func(common_time)
            
        except ImportError:
            # Fallback to numpy interpolation
            imf_interp = np.interp(common_time, imf_time, imf_signal)
            ecg_interp = np.interp(common_time, ecg_time, ecg_signal)
        
        # Normalize signals (zero mean, unit variance)
        imf_interp = (imf_interp - np.mean(imf_interp))
        ecg_interp = (ecg_interp - np.mean(ecg_interp))
        
        if np.std(imf_interp) > 0:
            imf_interp = imf_interp / np.std(imf_interp)
        if np.std(ecg_interp) > 0:
            ecg_interp = ecg_interp / np.std(ecg_interp)
        
        return imf_interp, ecg_interp, common_time, target_fs

    def calculate_correlation(self, signal1, signal2):
        correlation = np.corrcoef(signal1, signal2)[0, 1]
        cross_corr = np.correlate(signal1, signal2, mode='full')
        max_corr_idx = np.argmax(np.abs(cross_corr))
        max_correlation = cross_corr[max_corr_idx] / (len(signal1) * np.std(signal1) * np.std(signal2))
        
        return correlation, max_correlation

    def calculate_specific_frequency_percentages(self, freqs, mags, target_freqs, total_energy):
        """Calculate energy percentages at specific frequencies using windowing"""
        freq_data = {}
        
        for target_freq in target_freqs:
            # Define a small window around the target frequency (±0.1 Hz)
            window_size = 0.1
            freq_mask = (freqs >= target_freq - window_size) & (freqs <= target_freq + window_size)
            
            if np.any(freq_mask):
                # Sum energy in the window
                window_energy = np.sum(mags[freq_mask]**2)
                energy_percentage = (window_energy / total_energy) * 100 if total_energy > 0 else 0
                
                # Find the peak magnitude and frequency within the window
                window_mags = mags[freq_mask]
                window_freqs = freqs[freq_mask]
                if len(window_mags) > 0:
                    peak_idx = np.argmax(window_mags)
                    peak_magnitude = window_mags[peak_idx]
                    peak_freq = window_freqs[peak_idx]
                else:
                    peak_magnitude = 0
                    peak_freq = target_freq
            else:
                energy_percentage = 0
                peak_magnitude = 0
                peak_freq = target_freq
            
            freq_data[f'freq_{target_freq:.1f}Hz'] = {
                'target_freq': target_freq,
                'energy_percentage': energy_percentage,
                'peak_magnitude': peak_magnitude,
                'peak_freq': peak_freq,
                'window_size': window_size * 2  # Total window size
            }
        
        return freq_data

    def show_frequency_table_popup(self):
        """Show detailed frequency table in popup window with specific frequency analysis"""
        if not hasattr(self, 'imf_frequency_data') or len(self.imf_frequency_data) == 0:
            messagebox.showwarning("Warning", "No frequency data available. Run FFT analysis first.")
            return
        
        # Create popup window
        popup = tk.Toplevel(self.root)
        popup.title("Detailed Frequency Analysis Table")
        popup.geometry("1400x900")  # Larger window for more data
        
        # Create notebook for different views
        notebook = ttk.Notebook(popup)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Tab 1: Peak Frequencies
        peaks_frame = ttk.Frame(notebook)
        notebook.add(peaks_frame, text="Peak Frequencies")
        self.create_peaks_tab(peaks_frame)
        
        # Tab 2: Energy Distribution
        energy_frame = ttk.Frame(notebook)
        notebook.add(energy_frame, text="Energy Distribution")
        self.create_energy_tab(energy_frame)
        
        # Tab 3: Spectral Statistics
        stats_frame = ttk.Frame(notebook)
        notebook.add(stats_frame, text="Spectral Statistics")
        self.create_stats_tab(stats_frame)
        
        # NEW Tab 4: Specific Frequency Analysis
        specific_frame = ttk.Frame(notebook)
        notebook.add(specific_frame, text="Specific Frequencies")
        self.create_specific_frequencies_tab(specific_frame)
        
        # Add export button
        export_frame = ttk.Frame(popup)
        export_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Button(export_frame, text="Export to CSV", command=self.export_frequency_table_detailed).pack(side=tk.LEFT, padx=5)
        ttk.Button(export_frame, text="Export Specific Frequencies", command=self.export_specific_frequencies_csv).pack(side=tk.LEFT, padx=5)
        ttk.Button(export_frame, text="Close", command=popup.destroy).pack(side=tk.RIGHT, padx=5)
        
        # Add legend
        legend_frame = ttk.LabelFrame(popup, text="Frequency Analysis Information")
        legend_frame.pack(fill=tk.X, padx=10, pady=5)
        
        legend_text = tk.Text(legend_frame, height=5, wrap=tk.WORD)
        legend_text.pack(fill=tk.X, padx=5, pady=5)
        
        legend_content = """Frequency Bands:
    • 0.5-1.0 Hz: Very low frequency components  • 1.0-2.0 Hz: Low frequency components (typical heart rate range)
    • 2.0-3.0 Hz: Medium frequency components    • 3.0-4.0 Hz: High frequency components    • 4.0-8.0 Hz: Very high frequency components

    Specific Frequencies Analysis:
    • Energy percentages calculated using ±0.1 Hz windows around target frequencies (0.8, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 Hz)
    • Peak Magnitude: Highest magnitude within each frequency window
    • Peak Frequency: Exact frequency of the peak within each window

    Spectral Features:
    • Spectral Centroid: Weighted average frequency (center of mass)  • Rolloff 85%: Frequency below which 85% of energy is contained"""
        
        legend_text.insert(tk.END, legend_content)
        legend_text.config(state=tk.DISABLED)

    def create_peaks_tab(self, parent):
        """Create peak frequencies tab"""
        # Create treeview for peak frequencies
        peaks_tree = ttk.Treeview(parent, columns=('IMF', 'Freq1', 'Mag1', 'Freq2', 'Mag2', 'Freq3', 'Mag3', 'Freq4', 'Mag4', 'Freq5', 'Mag5'), show='headings')
        
        # Define headings
        peaks_tree.heading('IMF', text='IMF')
        peaks_tree.heading('Freq1', text='Freq 1 (Hz)')
        peaks_tree.heading('Mag1', text='Magnitude 1')
        peaks_tree.heading('Freq2', text='Freq 2 (Hz)')
        peaks_tree.heading('Mag2', text='Magnitude 2')
        peaks_tree.heading('Freq3', text='Freq 3 (Hz)')
        peaks_tree.heading('Mag3', text='Magnitude 3')
        peaks_tree.heading('Freq4', text='Freq 4 (Hz)')
        peaks_tree.heading('Mag4', text='Magnitude 4')
        peaks_tree.heading('Freq5', text='Freq 5 (Hz)')
        peaks_tree.heading('Mag5', text='Magnitude 5')
        
        # Configure column widths
        for col in peaks_tree['columns']:
            peaks_tree.column(col, width=100, anchor='center')
        
        # Add data
        for data in self.imf_frequency_data:
            freqs = data['peak_frequencies']
            mags = data['peak_magnitudes']
            
            row_data = [f"IMF {data['imf_num']}"]
            for i in range(5):
                if i < len(freqs):
                    row_data.extend([f"{freqs[i]:.4f}", f"{mags[i]:.2e}"])
                else:
                    row_data.extend(["-", "-"])
            
            peaks_tree.insert('', 'end', values=row_data)
        
        # Add scrollbar for peaks
        peaks_scrollbar = ttk.Scrollbar(parent, orient="vertical", command=peaks_tree.yview)
        peaks_tree.configure(yscrollcommand=peaks_scrollbar.set)
        
        peaks_tree.pack(side="left", fill="both", expand=True)
        peaks_scrollbar.pack(side="right", fill="y")

    def create_energy_tab(self, parent):
        """Create energy distribution tab"""
        # Create treeview for energy distribution
        energy_tree = ttk.Treeview(parent, columns=('IMF', 'Band1', 'Band2', 'Band3', 'Band4', 'Band5', 'Total'), show='headings')
        
        # Define headings
        energy_tree.heading('IMF', text='IMF')
        energy_tree.heading('Band1', text='0.5-1.0 Hz (%)')
        energy_tree.heading('Band2', text='1.0-2.0 Hz (%)')
        energy_tree.heading('Band3', text='2.0-3.0 Hz (%)')
        energy_tree.heading('Band4', text='3.0-4.0 Hz (%)')
        energy_tree.heading('Band5', text='4.0-8.0 Hz (%)')
        energy_tree.heading('Total', text='Total Energy')
        
        # Configure column widths
        for col in energy_tree['columns']:
            energy_tree.column(col, width=120, anchor='center')
        
        # Add data
        for data in self.imf_frequency_data:
            row_data = [
                f"IMF {data['imf_num']}",
                f"{data['energy_0_5_1_0_pct']:.1f}",
                f"{data['energy_1_0_2_0_pct']:.1f}",
                f"{data['energy_2_0_3_0_pct']:.1f}",
                f"{data['energy_3_0_4_0_pct']:.1f}",
                f"{data['energy_4_0_8_0_pct']:.1f}",
                f"{data['total_energy']:.2e}"
            ]
            energy_tree.insert('', 'end', values=row_data)
        
        # Add scrollbar for energy
        energy_scrollbar = ttk.Scrollbar(parent, orient="vertical", command=energy_tree.yview)
        energy_tree.configure(yscrollcommand=energy_scrollbar.set)
        
        energy_tree.pack(side="left", fill="both", expand=True)
        energy_scrollbar.pack(side="right", fill="y")

    def create_stats_tab(self, parent):
        """Create spectral statistics tab"""
        # Create treeview for spectral statistics
        stats_tree = ttk.Treeview(parent, columns=('IMF', 'Mean', 'Median', 'Std', 'Bandwidth', 'Centroid', 'Rolloff'), show='headings')
        
        # Define headings
        stats_tree.heading('IMF', text='IMF')
        stats_tree.heading('Mean', text='Mean Freq (Hz)')
        stats_tree.heading('Median', text='Median Freq (Hz)')
        stats_tree.heading('Std', text='Freq Std (Hz)')
        stats_tree.heading('Bandwidth', text='Bandwidth (Hz)')
        stats_tree.heading('Centroid', text='Spectral Centroid (Hz)')
        stats_tree.heading('Rolloff', text='Rolloff 85% (Hz)')
        
        # Configure column widths
        for col in stats_tree['columns']:
            stats_tree.column(col, width=140, anchor='center')
        
        # Add data
        for data in self.imf_frequency_data:
            row_data = [
                f"IMF {data['imf_num']}",
                f"{data['mean_freq']:.4f}",
                f"{data['median_freq']:.4f}",
                f"{data['freq_std']:.4f}",
                f"{data['bandwidth']:.4f}",
                f"{data['spectral_centroid']:.4f}",
                f"{data['spectral_rolloff']:.4f}"
            ]
            stats_tree.insert('', 'end', values=row_data)
        
        # Add scrollbar for stats
        stats_scrollbar = ttk.Scrollbar(parent, orient="vertical", command=stats_tree.yview)
        stats_tree.configure(yscrollcommand=stats_scrollbar.set)
        
        stats_tree.pack(side="left", fill="both", expand=True)
        stats_scrollbar.pack(side="right", fill="y")

    def create_specific_frequencies_tab(self, parent):
        """Create specific frequencies analysis tab"""
        # Create treeview for specific frequency analysis
        specific_tree = ttk.Treeview(parent, columns=('IMF', 'F0_8', 'F1_0', 'F2_0', 'F3_0', 'F4_0', 'F5_0', 'F6_0', 'F7_0', 'F8_0'), show='headings')
        
        # Define headings
        specific_tree.heading('IMF', text='IMF')
        specific_tree.heading('F0_8', text='0.8 Hz (%)')
        specific_tree.heading('F1_0', text='1.0 Hz (%)')
        specific_tree.heading('F2_0', text='2.0 Hz (%)')
        specific_tree.heading('F3_0', text='3.0 Hz (%)')
        specific_tree.heading('F4_0', text='4.0 Hz (%)')
        specific_tree.heading('F5_0', text='5.0 Hz (%)')
        specific_tree.heading('F6_0', text='6.0 Hz (%)')
        specific_tree.heading('F7_0', text='7.0 Hz (%)')
        specific_tree.heading('F8_0', text='8.0 Hz (%)')
        
        # Configure column widths
        for col in specific_tree['columns']:
            specific_tree.column(col, width=90, anchor='center')
        
        # Add data
        target_freqs = [0.8, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        
        for data in self.imf_frequency_data:
            row_data = [f"IMF {data['imf_num']}"]
            
            for target_freq in target_freqs:
                freq_key = f'freq_{target_freq:.1f}Hz'
                if freq_key in data['specific_frequencies']:
                    percentage = data['specific_frequencies'][freq_key]['energy_percentage']
                    row_data.append(f"{percentage:.2f}")
                else:
                    row_data.append("0.00")
            
            specific_tree.insert('', 'end', values=row_data)
        
        # Add scrollbar for specific frequencies
        specific_scrollbar = ttk.Scrollbar(parent, orient="vertical", command=specific_tree.yview)
        specific_tree.configure(yscrollcommand=specific_scrollbar.set)
        
        specific_tree.pack(side="left", fill="both", expand=True)
        specific_scrollbar.pack(side="right", fill="y")
        
        # Add summary statistics below the table
        summary_frame = ttk.LabelFrame(parent, text="Summary Statistics")
        summary_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Calculate and display summary
        if len(self.imf_frequency_data) > 0:
            summary_text = tk.Text(summary_frame, height=8, wrap=tk.WORD)
            summary_text.pack(fill=tk.X, padx=5, pady=5)
            
            # Calculate averages across all IMFs
            target_freqs = [0.8, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
            summary_content = "AVERAGE ENERGY DISTRIBUTION ACROSS ALL SELECTED IMFs:\n"
            summary_content += "=" * 55 + "\n"
            
            for target_freq in target_freqs:
                freq_key = f'freq_{target_freq:.1f}Hz'
                percentages = []
                
                for data in self.imf_frequency_data:
                    if freq_key in data['specific_frequencies']:
                        percentages.append(data['specific_frequencies'][freq_key]['energy_percentage'])
                
                if percentages:
                    avg_percentage = np.mean(percentages)
                    std_percentage = np.std(percentages)
                    max_percentage = np.max(percentages)
                    min_percentage = np.min(percentages)
                    
                    summary_content += f"{target_freq:.1f} Hz: Avg={avg_percentage:.2f}% ± {std_percentage:.2f}%, "
                    summary_content += f"Range=[{min_percentage:.2f}% - {max_percentage:.2f}%]\n"
            
            # Find most dominant frequency overall
            summary_content += "\n" + "=" * 55 + "\n"
            summary_content += "FREQUENCY DOMINANCE ANALYSIS:\n"
            
            freq_totals = {}
            for target_freq in target_freqs:
                freq_key = f'freq_{target_freq:.1f}Hz'
                total = 0
                count = 0
                for data in self.imf_frequency_data:
                    if freq_key in data['specific_frequencies']:
                        total += data['specific_frequencies'][freq_key]['energy_percentage']
                        count += 1
                if count > 0:
                    freq_totals[target_freq] = total / count
            
            if freq_totals:
                sorted_freqs = sorted(freq_totals.items(), key=lambda x: x[1], reverse=True)
                summary_content += f"Most dominant: {sorted_freqs[0][0]:.1f} Hz ({sorted_freqs[0][1]:.2f}% avg)\n"
                summary_content += f"Least dominant: {sorted_freqs[-1][0]:.1f} Hz ({sorted_freqs[-1][1]:.2f}% avg)\n"
                
                # Categorize frequencies
                summary_content += f"\nHigh Energy (>2%): "
                high_energy = [f"{freq:.1f}Hz" for freq, pct in sorted_freqs if pct > 2.0]
                summary_content += ", ".join(high_energy) if high_energy else "None"
                
                summary_content += f"\nMedium Energy (0.5-2%): "
                medium_energy = [f"{freq:.1f}Hz" for freq, pct in sorted_freqs if 0.5 <= pct <= 2.0]
                summary_content += ", ".join(medium_energy) if medium_energy else "None"
                
                summary_content += f"\nLow Energy (<0.5%): "
                low_energy = [f"{freq:.1f}Hz" for freq, pct in sorted_freqs if pct < 0.5]
                summary_content += ", ".join(low_energy) if low_energy else "None"
            
            summary_text.insert(tk.END, summary_content)
            summary_text.config(state=tk.DISABLED)

    def export_specific_frequencies_csv(self):
        """Export specific frequencies analysis to CSV"""
        if not hasattr(self, 'imf_frequency_data') or len(self.imf_frequency_data) == 0:
            messagebox.showwarning("Warning", "No frequency data available.")
            return
        
        # Ask for save location
        file_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            title="Export Specific Frequencies Analysis"
        )
        
        if not file_path:
            return
        
        try:
            # Prepare export data
            export_data = []
            target_freqs = [0.8, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
            
            for data in self.imf_frequency_data:
                row = {'IMF_Number': data['imf_num']}
                
                # Add percentage data for each target frequency
                for target_freq in target_freqs:
                    freq_key = f'freq_{target_freq:.1f}Hz'
                    if freq_key in data['specific_frequencies']:
                        freq_data = data['specific_frequencies'][freq_key]
                        row[f'{target_freq:.1f}Hz_Energy_Percent'] = freq_data['energy_percentage']
                        row[f'{target_freq:.1f}Hz_Peak_Magnitude'] = freq_data['peak_magnitude']
                        row[f'{target_freq:.1f}Hz_Peak_Frequency'] = freq_data['peak_freq']
                    else:
                        row[f'{target_freq:.1f}Hz_Energy_Percent'] = 0
                        row[f'{target_freq:.1f}Hz_Peak_Magnitude'] = 0
                        row[f'{target_freq:.1f}Hz_Peak_Frequency'] = target_freq
                
                export_data.append(row)
            
            # Create DataFrame and save
            df = pd.DataFrame(export_data)
            df.to_csv(file_path, index=False)
            
            messagebox.showinfo("Export Complete", f"Specific frequencies analysis exported to:\n{file_path}")
            
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to export specific frequencies: {str(e)}")

    def export_frequency_table_detailed(self):
        """Export detailed frequency analysis table to CSV - ENHANCED VERSION"""
        if not hasattr(self, 'imf_frequency_data') or len(self.imf_frequency_data) == 0:
            messagebox.showwarning("Warning", "No frequency data available.")
            return
        
        # Ask for save location
        file_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            title="Export Complete Frequency Analysis"
        )
        
        if not file_path:
            return
        
        try:
            # Prepare export data
            export_data = []
            target_freqs = [0.8, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
            
            for data in self.imf_frequency_data:
                # Create row with all frequency data
                row = {
                    'IMF_Number': data['imf_num'],
                    'Primary_Freq_Hz': data['peak_frequencies'][0] if len(data['peak_frequencies']) > 0 else 0,
                    'Secondary_Freq_Hz': data['peak_frequencies'][1] if len(data['peak_frequencies']) > 1 else 0,
                    'Tertiary_Freq_Hz': data['peak_frequencies'][2] if len(data['peak_frequencies']) > 2 else 0,
                    'Fourth_Freq_Hz': data['peak_frequencies'][3] if len(data['peak_frequencies']) > 3 else 0,
                    'Fifth_Freq_Hz': data['peak_frequencies'][4] if len(data['peak_frequencies']) > 4 else 0,
                    'Primary_Magnitude': data['peak_magnitudes'][0] if len(data['peak_magnitudes']) > 0 else 0,
                    'Secondary_Magnitude': data['peak_magnitudes'][1] if len(data['peak_magnitudes']) > 1 else 0,
                    'Tertiary_Magnitude': data['peak_magnitudes'][2] if len(data['peak_magnitudes']) > 2 else 0,
                    'Fourth_Magnitude': data['peak_magnitudes'][3] if len(data['peak_magnitudes']) > 3 else 0,
                    'Fifth_Magnitude': data['peak_magnitudes'][4] if len(data['peak_magnitudes']) > 4 else 0,
                    'Energy_0_5_1_0_Hz_Percent': data['energy_0_5_1_0_pct'],
                    'Energy_1_0_2_0_Hz_Percent': data['energy_1_0_2_0_pct'],
                    'Energy_2_0_3_0_Hz_Percent': data['energy_2_0_3_0_pct'],
                    'Energy_3_0_4_0_Hz_Percent': data['energy_3_0_4_0_pct'],
                    'Energy_4_0_8_0_Hz_Percent': data['energy_4_0_8_0_pct'],
                    'Total_Energy': data['total_energy'],
                    'Mean_Frequency_Hz': data['mean_freq'],
                    'Median_Frequency_Hz': data['median_freq'],
                    'Frequency_Std_Hz': data['freq_std'],
                    'Bandwidth_Hz': data['bandwidth'],
                    'Spectral_Centroid_Hz': data['spectral_centroid'],
                    'Spectral_Rolloff_85_Hz': data['spectral_rolloff']
                }
                
                # Add specific frequency data
                for target_freq in target_freqs:
                    freq_key = f'freq_{target_freq:.1f}Hz'
                    if freq_key in data['specific_frequencies']:
                        freq_data = data['specific_frequencies'][freq_key]
                        row[f'Specific_{target_freq:.1f}Hz_Energy_Percent'] = freq_data['energy_percentage']
                        row[f'Specific_{target_freq:.1f}Hz_Peak_Magnitude'] = freq_data['peak_magnitude']
                        row[f'Specific_{target_freq:.1f}Hz_Peak_Frequency'] = freq_data['peak_freq']
                    else:
                        row[f'Specific_{target_freq:.1f}Hz_Energy_Percent'] = 0
                        row[f'Specific_{target_freq:.1f}Hz_Peak_Magnitude'] = 0
                        row[f'Specific_{target_freq:.1f}Hz_Peak_Frequency'] = target_freq
                
                export_data.append(row)
            
            # Create DataFrame and save
            df = pd.DataFrame(export_data)
            df.to_csv(file_path, index=False)
            
            messagebox.showinfo("Export Complete", f"Complete frequency analysis exported to:\n{file_path}")
            
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to export frequency table: {str(e)}")

    # ADD THESE UTILITY FUNCTIONS AS WELL:

    def calculate_spectral_rolloff(self, freqs, mags, rolloff_percent=0.85):
        """Calculate spectral rolloff frequency"""
        try:
            cumsum_mags = np.cumsum(mags)
            total_mag = cumsum_mags[-1]
            rolloff_idx = np.where(cumsum_mags >= rolloff_percent * total_mag)[0]
            if len(rolloff_idx) > 0:
                return freqs[rolloff_idx[0]]
            else:
                return freqs[-1]
        except:
            return 0

    def update_fft_results_text_simple(self, selected_indices):
        """Update results text with simple summary including specific frequencies"""
        if not hasattr(self, 'eemd_results'):
            return
        
        # Clear previous FFT results
        current_text = self.eemd_results.get(1.0, tk.END)
        lines = current_text.split('\n')
        
        # Remove old FFT analysis section
        new_lines = []
        skip = False
        for line in lines:
            if "FFT ANALYSIS RESULTS:" in line:
                skip = True
            elif line.startswith("="*40) and skip:
                skip = False
                continue
            elif not skip:
                new_lines.append(line)
        
        # Clear and insert updated text
        self.eemd_results.delete(1.0, tk.END)
        self.eemd_results.insert(tk.END, '\n'.join(new_lines))
        
        # Add new FFT analysis summary
        self.eemd_results.insert(tk.END, f"\nFFT ANALYSIS RESULTS:\n")
        self.eemd_results.insert(tk.END, "=" * 40 + "\n")
        
        if not hasattr(self, 'imf_frequency_data') or len(self.imf_frequency_data) == 0:
            self.eemd_results.insert(tk.END, "No IMFs selected for analysis\n\n")
            return
        
        # Quick summary
        self.eemd_results.insert(tk.END, f"Selected IMFs: {len(selected_indices)}\n")
        self.eemd_results.insert(tk.END, f"IMF Numbers: {[i+1 for i in selected_indices]}\n\n")
        
        # Brief frequency summary
        self.eemd_results.insert(tk.END, "Primary Frequencies:\n")
        for data in self.imf_frequency_data:
            if len(data['peak_frequencies']) > 0:
                self.eemd_results.insert(tk.END, 
                    f"IMF {data['imf_num']}: {data['peak_frequencies'][0]:.3f} Hz\n")
        
        # NEW: Add specific frequency highlights
        if len(self.imf_frequency_data) > 0:
            self.eemd_results.insert(tk.END, f"\nHighest Energy at Specific Frequencies:\n")
            target_freqs = [0.8, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
            
            for target_freq in target_freqs:
                max_energy = 0
                best_imf = None
                
                for data in self.imf_frequency_data:
                    freq_key = f'freq_{target_freq:.1f}Hz'
                    if freq_key in data['specific_frequencies']:
                        energy = data['specific_frequencies'][freq_key]['energy_percentage']
                        if energy > max_energy:
                            max_energy = energy
                            best_imf = data['imf_num']
                
                if best_imf and max_energy > 0.1:  # Only show if > 0.1%
                    self.eemd_results.insert(tk.END, f"{target_freq:.1f}Hz: IMF{best_imf} ({max_energy:.2f}%)\n")
        
        self.eemd_results.insert(tk.END, f"\n📊 Click 'Show Frequency Table' for detailed analysis\n")
        self.eemd_results.insert(tk.END, f"📈 'Specific Frequencies' tab shows energy at target frequencies\n")
        self.eemd_results.insert(tk.END, "=" * 40 + "\n")

    def update_fft_results_text(self, message):
        """Simple text update for when no IMFs are selected"""
        if not hasattr(self, 'eemd_results'):
            return
        
        # Clear previous FFT results and add simple message
        current_text = self.eemd_results.get(1.0, tk.END)
        lines = current_text.split('\n')
        
        # Remove old FFT analysis section
        new_lines = []
        skip = False
        for line in lines:
            if "FFT ANALYSIS RESULTS:" in line:
                skip = True
            elif line.startswith("="*40) and skip:
                skip = False
                continue
            elif not skip:
                new_lines.append(line)
        
        # Clear and insert updated text
        self.eemd_results.delete(1.0, tk.END)
        self.eemd_results.insert(tk.END, '\n'.join(new_lines))
        
        # Add simple message
        self.eemd_results.insert(tk.END, f"\nFFT ANALYSIS RESULTS:\n")
        self.eemd_results.insert(tk.END, "=" * 40 + "\n")
        self.eemd_results.insert(tk.END, f"{message}\n")
        self.eemd_results.insert(tk.END, "=" * 40 + "\n")

    # CRITICAL FIX: ADD THIS ALIAS FUNCTION TO PREVENT THE ERROR
    def export_frequency_table(self):
        """Alias for export_frequency_table_detailed to maintain backward compatibility"""
        self.export_frequency_table_detailed()

    #IMF COMBINATION

    def combine_hr_imfs(self):
        """Combine selected heart rate IMFs with optional Moving Average filter"""
        selected_imfs = [self.heartrate_imfs[i] for i in range(len(self.heartrate_imfs)) if self.hr_imf_vars[i].get()]
        
        if not selected_imfs:
            messagebox.showwarning("Warning", "Select at least one IMF")
            return
        
        # Combine IMFs first
        combined_imf_raw = np.sum(selected_imfs, axis=0)

        # Check if derivative should be used instead
        if hasattr(self, 'use_derivative_var') and self.use_derivative_var.get():
            # Use heart rate derivative from main analysis
            if hasattr(self, 'signals') and 'heartrate_derivative' in self.signals:
                combined_imf_raw = self.signals['heartrate_derivative'].copy()
                derivative_used = True
                signal_source = "Heart Rate Derivative"
            else:
                messagebox.showwarning("Warning", "Heart rate derivative not available. Using combined IMFs.")
                derivative_used = False
                signal_source = "Combined IMFs (fallback)"
        else:
            derivative_used = False
            signal_source = "Combined IMFs"

        # Apply Moving Average based on UI settings
        mav_applied = False
        mav_info = "Not Applied"
        
        if hasattr(self, 'enable_mav_var') and self.enable_mav_var.get():
            mav_window = self.mav_window_var.get() if hasattr(self, 'mav_window_var') else 25
            if len(combined_imf_raw) >= mav_window and mav_window > 0:
                self.latest_combined_imf = self.moving_average(combined_imf_raw, mav_window)  # FIXED METHOD NAME
                mav_applied = True
                mav_info = f"Applied (Window={mav_window})"
            else:
                self.latest_combined_imf = combined_imf_raw
                mav_applied = False
                mav_info = "Skipped (invalid window or signal too short)"
        else:
            # MAV is disabled, use raw signal
            self.latest_combined_imf = combined_imf_raw
            mav_applied = False
            mav_info = "Disabled by user"
        
        # Clear and update the plot
        self.ax_combined.clear()
        
        # **FIXED PLOTTING LOGIC - Only show MAV if it's actually applied**
        if derivative_used:
            if mav_applied:
                self.ax_combined.plot(self.signals['timestamps'], combined_imf_raw, 'purple', 
                                    linewidth=1, alpha=0.5, label='Raw HR Derivative')
                self.ax_combined.plot(self.signals['timestamps'], self.latest_combined_imf, 'red', 
                                    linewidth=1.5, label=f'HR Derivative + MAV (Window={mav_window})')
                self.ax_combined.set_title('Heart Rate Derivative Signal with Moving Average')
            else:
                self.ax_combined.plot(self.signals['timestamps'], self.latest_combined_imf, 'red', 
                                    linewidth=1.5, label='HR Derivative (No MAV)')
                self.ax_combined.set_title('Heart Rate Derivative Signal')
        else:
            if mav_applied:
                # Show both raw and filtered
                self.ax_combined.plot(self.signals['timestamps'], combined_imf_raw, 'b-', 
                                    linewidth=1, alpha=0.5, label='Raw Combined IMFs')
                self.ax_combined.plot(self.signals['timestamps'], self.latest_combined_imf, 'g-', 
                                    linewidth=1.5, label=f'MAV Filtered (Window={mav_window})')
                self.ax_combined.set_title('Combined Heart Rate IMFs with Moving Average Filter')
            else:
                # Show only the raw signal
                self.ax_combined.plot(self.signals['timestamps'], self.latest_combined_imf, 'b-', 
                                    linewidth=1.5, label='Combined IMFs (No MAV)')
                self.ax_combined.set_title('Combined Heart Rate IMFs')

        self.ax_combined.legend()
        self.ax_combined.set_xlabel('Time (s)')
        self.ax_combined.set_ylabel('Amplitude')
        self.ax_combined.grid(True)
        
        # Update results text
        selected = [i+1 for i in range(len(self.heartrate_imfs)) if self.hr_imf_vars[i].get()]
        self.eemd_results.delete(1.0, tk.END)

        if derivative_used:
            self.eemd_results.insert(tk.END, f"USING HEART RATE DERIVATIVE SIGNAL\n")
            self.eemd_results.insert(tk.END, f"Source: Main Analysis Tab\n")
            self.eemd_results.insert(tk.END, f"Selected IMFs (ignored): {selected}\n")
        else:
            self.eemd_results.insert(tk.END, f"Combined Heart Rate IMFs: {selected}\n")

        self.eemd_results.insert(tk.END, f"Signal Source: {signal_source}\n")
        self.eemd_results.insert(tk.END, f"Moving Average: {mav_info}\n")
        self.eemd_results.insert(tk.END, f"Signal length: {len(self.latest_combined_imf)} samples\n")
        
        if mav_applied:
            # Calculate MAV effect metrics
            correlation = np.corrcoef(combined_imf_raw, self.latest_combined_imf)[0, 1]
            rmse = np.sqrt(np.mean((combined_imf_raw - self.latest_combined_imf)**2))
            smoothing_effect = (1 - correlation) * 100
            
            self.eemd_results.insert(tk.END, f"MAV Effect - Correlation: {correlation:.4f}\n")
            self.eemd_results.insert(tk.END, f"MAV Effect - RMSE: {rmse:.6f}\n")
            self.eemd_results.insert(tk.END, f"Smoothing Effect: {smoothing_effect:.2f}%\n")
        
        self.eemd_results.insert(tk.END, "\n")
        
        # Update FFT analysis after combining and filtering
        self.plot_imf_fft_analysis()
        
        self.fig_eemd.tight_layout()
        if hasattr(self, 'canvas_eemd'):
            self.canvas_eemd.draw()

    def compare_imfs_ecg(self):
        """Compare combined IMFs with ECG - Uses segmented signals when available"""
        if not self.ecg_loaded:
            messagebox.showwarning("Warning", "Load ECG data first")
            return
        
        if self.latest_combined_imf is None:
            messagebox.showwarning("Warning", "Combine IMFs first")
            return
        
        # Determine which signals to use (segmented or full)
        radar_signal = self.latest_combined_imf
        radar_timestamps = self.get_current_radar_timestamps()
        
        ecg_signal = self.ecg_data
        ecg_timestamps = self.ecg_time
        
        # Check segmentation status and update display info
        radar_status = "Segmented Radar" if (hasattr(self, 'is_segmented') and self.is_segmented) else "Full Radar"
        ecg_status = "Segmented ECG" if (hasattr(self, 'is_ecg_segmented') and self.is_ecg_segmented) else "Full ECG"
        
        # Synchronize signals
        imf_sync, ecg_sync, common_time, sync_fs = self.synchronize_signals(
            radar_signal, radar_timestamps, ecg_signal, ecg_timestamps)
        
        if imf_sync is None:
            messagebox.showerror("Error", 
                f"No time overlap between signals.\n"
                f"Radar range: {radar_timestamps[0]:.1f}s - {radar_timestamps[-1]:.1f}s\n"
                f"ECG range: {ecg_timestamps[0]:.1f}s - {ecg_timestamps[-1]:.1f}s")
            return
        
        # Calculate correlation metrics
        correlation, max_correlation = self.calculate_correlation(imf_sync, ecg_sync)
        
        # Calculate additional comparison metrics
        rmse = np.sqrt(np.mean((imf_sync - ecg_sync)**2))
        mae = np.mean(np.abs(imf_sync - ecg_sync))
        
        # Calculate cross-correlation with lag analysis
        cross_corr = np.correlate(imf_sync, ecg_sync, mode='full')
        lags = np.arange(-len(ecg_sync) + 1, len(imf_sync))
        max_corr_idx = np.argmax(np.abs(cross_corr))
        best_lag = lags[max_corr_idx] / sync_fs  # Convert to seconds
        
        # Clear and plot comparison
        self.ax_imf_compare.clear()
        
        # Plot synchronized signals
        self.ax_imf_compare.plot(common_time, imf_sync, 'g-', label=f'{radar_status} Signal', 
                            alpha=0.8, linewidth=1.5)
        self.ax_imf_compare.plot(common_time, ecg_sync, 'r-', label=f'{ecg_status} Signal', 
                            alpha=0.8, linewidth=1.5)
        
        # Add correlation info to title
        title = f'{radar_status} vs {ecg_status} Comparison (r={correlation:.3f})'
        self.ax_imf_compare.set_title(title)
        self.ax_imf_compare.set_xlabel('Time (s)')
        self.ax_imf_compare.set_ylabel('Normalized Amplitude')
        self.ax_imf_compare.legend()
        self.ax_imf_compare.grid(True, alpha=0.3)
        
        # Add vertical lines to show segment boundaries if segmented
        if hasattr(self, 'is_segmented') and self.is_segmented:
            seg_start = common_time[0]
            seg_end = common_time[-1]
            self.ax_imf_compare.axvline(x=seg_start, color='green', linestyle='--', alpha=0.5, 
                                    label=f'Segment Start ({seg_start:.1f}s)')
            self.ax_imf_compare.axvline(x=seg_end, color='orange', linestyle='--', alpha=0.5, 
                                    label=f'Segment End ({seg_end:.1f}s)')
            self.ax_imf_compare.legend()
        
        # Update results text with comprehensive comparison
        self.eemd_results.insert(tk.END, f"\n" + "="*50 + "\n")
        self.eemd_results.insert(tk.END, f"RADAR-ECG COMPARISON RESULTS\n")
        self.eemd_results.insert(tk.END, f"="*50 + "\n")
        
        # Signal information
        self.eemd_results.insert(tk.END, f"Radar Signal: {radar_status} ({len(radar_signal)} samples)\n")
        self.eemd_results.insert(tk.END, f"ECG Signal: {ecg_status} ({len(ecg_signal)} samples)\n")
        self.eemd_results.insert(tk.END, f"Synchronized Duration: {common_time[-1] - common_time[0]:.2f} seconds\n")
        self.eemd_results.insert(tk.END, f"Sync Sampling Rate: {sync_fs:.1f} Hz\n")
        self.eemd_results.insert(tk.END, f"Synchronized Samples: {len(common_time)}\n\n")
        
        # Correlation metrics
        self.eemd_results.insert(tk.END, f"CORRELATION ANALYSIS:\n")
        self.eemd_results.insert(tk.END, f"Pearson Correlation: {correlation:.4f}\n")
        self.eemd_results.insert(tk.END, f"Max Cross-correlation: {max_correlation:.4f}\n")
        self.eemd_results.insert(tk.END, f"Best Time Lag: {best_lag:.3f} seconds\n\n")
        
        # Error metrics
        self.eemd_results.insert(tk.END, f"ERROR METRICS:\n")
        self.eemd_results.insert(tk.END, f"Root Mean Square Error: {rmse:.4f}\n")
        self.eemd_results.insert(tk.END, f"Mean Absolute Error: {mae:.4f}\n\n")
        
        # Segmentation status
        if hasattr(self, 'is_segmented') and self.is_segmented:
            self.eemd_results.insert(tk.END, f"SEGMENTATION STATUS:\n")
            self.eemd_results.insert(tk.END, f"Radar Segmented: ✓ YES\n")
            if hasattr(self, 'is_ecg_segmented') and self.is_ecg_segmented:
                self.eemd_results.insert(tk.END, f"ECG Segmented: ✓ YES (Synchronized)\n")
            else:
                self.eemd_results.insert(tk.END, f"ECG Segmented: ⚠ NO (Using full ECG)\n")
        else:
            self.eemd_results.insert(tk.END, f"SEGMENTATION STATUS:\n")
            self.eemd_results.insert(tk.END, f"Using Full Signals (No segmentation applied)\n")
        
        # Interpretation
        self.eemd_results.insert(tk.END, f"\nINTERPRETATION:\n")
        if abs(correlation) > 0.7:
            self.eemd_results.insert(tk.END, f"Strong correlation - Signals are well matched\n")
        elif abs(correlation) > 0.4:
            self.eemd_results.insert(tk.END, f"Moderate correlation - Some signal similarity\n")
        else:
            self.eemd_results.insert(tk.END, f"Weak correlation - Limited signal similarity\n")
        
        if abs(best_lag) < 0.1:
            self.eemd_results.insert(tk.END, f"Minimal time lag - Good synchronization\n")
        else:
            self.eemd_results.insert(tk.END, f"Time lag detected - Check synchronization\n")
        
        self.eemd_results.insert(tk.END, f"="*50 + "\n")
        
        # Redraw the plot
        self.fig_eemd.tight_layout()
        if hasattr(self, 'canvas_eemd'):
            self.canvas_eemd.draw()

    def compare_raw_vs_mav(self):
        if not hasattr(self, 'latest_combined_imf'):
            messagebox.showwarning("Warning", "No combined IMF available. Please combine IMFs first.")
            return
        
        # Get selected IMFs and create raw version
        selected_imfs = [self.heartrate_imfs[i] for i in range(len(self.heartrate_imfs)) if self.hr_imf_vars[i].get()]
        if not selected_imfs:
            messagebox.showwarning("Warning", "No IMFs selected")
            return
        
        raw_combined = np.sum(selected_imfs, axis=0)
        mav_window = self.mav_window_var.get() if hasattr(self, 'mav_window_var') else 25
        
        if mav_window <= 0 or len(raw_combined) < mav_window:
            messagebox.showwarning("Warning", "Invalid MAV window size or signal too short")
            return
        
        mav_filtered = self.apply_moving_average(raw_combined, mav_window)
        
        # Calculate comparison metrics
        correlation = np.corrcoef(raw_combined, mav_filtered)[0, 1]
        rmse = np.sqrt(np.mean((raw_combined - mav_filtered)**2))
        snr_raw = np.var(raw_combined) / np.var(raw_combined - mav_filtered) if np.var(raw_combined - mav_filtered) > 0 else float('inf')
        
        # Calculate frequency domain comparison
        fft_raw = np.fft.fft(raw_combined)
        fft_mav = np.fft.fft(mav_filtered)
        freqs = np.fft.fftfreq(len(raw_combined), 1/self.signals['fs'])
        
        # Find dominant frequencies
        pos_mask = freqs > 0
        freqs_pos = freqs[pos_mask]
        mag_raw = np.abs(fft_raw)[pos_mask]
        mag_mav = np.abs(fft_mav)[pos_mask]
        
        # Update results text with detailed comparison
        self.eemd_results.insert(tk.END, f"\n" + "="*40 + "\n")
        self.eemd_results.insert(tk.END, f"RAW vs MAV DETAILED COMPARISON:\n")
        self.eemd_results.insert(tk.END, f"="*40 + "\n")
        self.eemd_results.insert(tk.END, f"MAV Window Size: {mav_window}\n")
        self.eemd_results.insert(tk.END, f"Signal Length: {len(raw_combined)} samples\n")
        self.eemd_results.insert(tk.END, f"Time Domain Correlation: {correlation:.4f}\n")
        self.eemd_results.insert(tk.END, f"Root Mean Square Error: {rmse:.6f}\n")
        self.eemd_results.insert(tk.END, f"Signal-to-Noise Ratio: {snr_raw:.2f}\n")
        self.eemd_results.insert(tk.END, f"Smoothing Effect: {(1-correlation)*100:.2f}%\n")
        
        # Statistical comparison
        raw_std = np.std(raw_combined)
        mav_std = np.std(mav_filtered)
        noise_reduction = ((raw_std - mav_std) / raw_std) * 100 if raw_std > 0 else 0
        
        self.eemd_results.insert(tk.END, f"Raw Signal StdDev: {raw_std:.6f}\n")
        self.eemd_results.insert(tk.END, f"MAV Signal StdDev: {mav_std:.6f}\n")
        self.eemd_results.insert(tk.END, f"Noise Reduction: {noise_reduction:.2f}%\n")
        
        # Frequency domain analysis
        if len(mag_raw) > 0 and len(mag_mav) > 0:
            freq_corr = np.corrcoef(mag_raw, mag_mav)[0, 1]
            
            # Find peak frequencies
            raw_peak_idx = np.argmax(mag_raw)
            mav_peak_idx = np.argmax(mag_mav)
            
            self.eemd_results.insert(tk.END, f"Frequency Domain Correlation: {freq_corr:.4f}\n")
            self.eemd_results.insert(tk.END, f"Raw Peak Frequency: {freqs_pos[raw_peak_idx]:.3f} Hz\n")
            self.eemd_results.insert(tk.END, f"MAV Peak Frequency: {freqs_pos[mav_peak_idx]:.3f} Hz\n")
        
        self.eemd_results.insert(tk.END, f"\n")

#SEGMENTATION

    def apply_segment(self):
        """Apply segmentation to the combined IMF signal and automatically segment ECG"""
        if not hasattr(self, 'latest_combined_imf'):
            messagebox.showwarning("Warning", "No combined IMF available. Combine IMFs first.")
            return
        
        start_time = self.seg_start_var.get()
        end_time = self.seg_end_var.get()

        if start_time >= end_time:
            messagebox.showerror("Error", "Start time must be less than end time")
            return
        
        timestamps = self.signals['timestamps']
        
        if start_time < timestamps[0] or end_time > timestamps[-1]:
            messagebox.showerror("Error", 
                f"Time range must be within signal duration: {timestamps[0]:.1f} - {timestamps[-1]:.1f} sec")
            return
        
        # **FORCE ECG SEGMENTATION FIRST - BEFORE RADAR SEGMENTATION**
        if self.ecg_loaded:
            print(f"DEBUG: Applying ECG segmentation from {start_time:.2f}s to {end_time:.2f}s")
            try:
                success, message = self.apply_ecg_segmentation(start_time, end_time)
                if success:
                    self.eemd_results.insert(tk.END, f"✓ ECG automatically segmented: {message}\n")
                    print(f"DEBUG: ECG segmentation successful - new ECG length: {len(self.ecg_data)}")
                    print(f"DEBUG: ECG new time range: {self.ecg_time[0]:.2f} - {self.ecg_time[-1]:.2f}s")
                else:
                    self.eemd_results.insert(tk.END, f"⚠ ECG segmentation failed: {message}\n")
                    print(f"DEBUG: ECG segmentation failed: {message}")
            except Exception as e:
                self.eemd_results.insert(tk.END, f"⚠ ECG segmentation error: {str(e)}\n")
                print(f"DEBUG: ECG segmentation error: {str(e)}")

        # Now apply radar segmentation
        start_idx = np.argmin(np.abs(timestamps - start_time))
        end_idx = np.argmin(np.abs(timestamps - end_time))
        
        # Store original data if not already stored
        if not hasattr(self, 'original_combined_imf'):
            self.original_combined_imf = self.latest_combined_imf.copy()
            self.original_timestamps = timestamps.copy()
        
        # Apply radar segmentation
        self.latest_combined_imf = self.latest_combined_imf[start_idx:end_idx+1]
        self.segmented_timestamps = timestamps[start_idx:end_idx+1]
        self.is_segmented = True
        
        print(f"DEBUG: Radar segmentation applied - new length: {len(self.latest_combined_imf)}")
        print(f"DEBUG: Radar new time range: {self.segmented_timestamps[0]:.2f} - {self.segmented_timestamps[-1]:.2f}s")
        
        # Update the combined plot
        self.ax_combined.clear()
        self.ax_combined.plot(self.segmented_timestamps, self.latest_combined_imf, 'b-', linewidth=1.5)
        self.ax_combined.set_title('Segmented Combined Phase IMFs')
        self.ax_combined.set_xlabel('Time (s)')
        self.ax_combined.set_ylabel('Amplitude')
        self.ax_combined.grid(True, alpha=0.3)
        self.canvas_eemd.draw()
        
        # Show success with segmentation status
        success_msg = f"Segment applied successfully!\nDuration: {end_time - start_time:.1f} seconds"
        if self.ecg_loaded and hasattr(self, 'is_ecg_segmented') and self.is_ecg_segmented:
            success_msg += "\n✓ ECG and Radar synchronized"
            success_msg += f"\nECG segmented to {len(self.ecg_data)} samples"
        
        messagebox.showinfo("Segmentation Applied", success_msg)

    def get_current_radar_timestamps(self):
        """Get the current radar timestamps (segmented or full)"""
        if hasattr(self, 'is_segmented') and self.is_segmented and hasattr(self, 'segmented_timestamps'):
            return self.segmented_timestamps
        else:
            return self.signals['timestamps']

    def apply_ecg_segmentation(self, start_time, end_time):
        """Apply segmentation to ECG data automatically - DEBUG VERSION"""
        print(f"DEBUG: apply_ecg_segmentation called with {start_time:.2f}s to {end_time:.2f}s")
        
        if not self.ecg_loaded or self.ecg_data is None:
            print("DEBUG: No ECG data loaded")
            return False, "No ECG data loaded"
        
        print(f"DEBUG: Current ECG data length before segmentation: {len(self.ecg_data)}")
        print(f"DEBUG: Current ECG time range before segmentation: {self.ecg_time[0]:.2f} - {self.ecg_time[-1]:.2f}s")
        
        # Store original ECG data if not already stored
        if not hasattr(self, 'ecg_original_data'):
            print("DEBUG: Storing original ECG data")
            self.ecg_original_data = self.ecg_data.copy()
            self.ecg_original_time = self.ecg_time.copy()
            print(f"DEBUG: Original ECG stored - length: {len(self.ecg_original_data)}")
        
        # Always use original data for segmentation
        ecg_timestamps = self.ecg_original_time
        
        if start_time < ecg_timestamps[0] or end_time > ecg_timestamps[-1]:
            error_msg = f"ECG time range: {ecg_timestamps[0]:.1f}s - {ecg_timestamps[-1]:.1f}s"
            print(f"DEBUG: Time range error - {error_msg}")
            return False, error_msg
        
        # Find indices for ECG segmentation
        start_idx = np.argmin(np.abs(ecg_timestamps - start_time))
        end_idx = np.argmin(np.abs(ecg_timestamps - end_time))
        
        print(f"DEBUG: Segmentation indices - start: {start_idx}, end: {end_idx}")
        
        # Apply segmentation
        self.ecg_segmented = self.ecg_original_data[start_idx:end_idx+1]
        self.ecg_segmented_timestamps = ecg_timestamps[start_idx:end_idx+1]
        self.is_ecg_segmented = True
        
        # **CRITICAL**: Immediately replace the main ECG data
        self.ecg_data = self.ecg_segmented.copy()
        self.ecg_time = self.ecg_segmented_timestamps.copy()
        
        print(f"DEBUG: ECG segmentation completed")
        print(f"DEBUG: New ECG data length: {len(self.ecg_data)}")
        print(f"DEBUG: New ECG time range: {self.ecg_time[0]:.2f} - {self.ecg_time[-1]:.2f}s")
        print(f"DEBUG: is_ecg_segmented flag: {self.is_ecg_segmented}")
        
        return True, f"ECG segmented: {len(self.ecg_segmented)} samples"

    def reset_ecg_segmentation(self):
        """Reset ECG to full signal"""
        if hasattr(self, 'ecg_original_data') and self.ecg_original_data is not None:
            self.ecg_data = self.ecg_original_data.copy()
            self.ecg_time = self.ecg_original_time.copy()
            self.is_ecg_segmented = False
            return True
        return False

    def preview_segment(self):
        """Preview the selected segment without applying it"""
        if not hasattr(self, 'latest_combined_imf'):
            messagebox.showwarning("Warning", "No combined IMF available. Combine IMFs first.")
            return
        
        try:
            start_time = self.seg_start_var.get()
            end_time = self.seg_end_var.get()
            
            if start_time >= end_time:
                messagebox.showerror("Error", "Start time must be less than end time")
                return
            
            timestamps = self.signals['timestamps']
            
            if start_time < timestamps[0] or end_time > timestamps[-1]:
                messagebox.showerror("Error", 
                    f"Time range must be within signal duration: {timestamps[0]:.1f} - {timestamps[-1]:.1f} sec")
                return

            # Find indices for the time range
            start_idx = np.argmin(np.abs(timestamps - start_time))
            end_idx = np.argmin(np.abs(timestamps - end_time))
            
            # Update the combined plot to show preview
            self.ax_combined.clear()
            
            # Plot full signal in light color
            self.ax_combined.plot(timestamps, self.latest_combined_imf, 'lightgray', 
                                linewidth=1, label='Full Signal', alpha=0.5)
            
            # Highlight selected segment
            seg_timestamps = timestamps[start_idx:end_idx+1]
            seg_signal = self.latest_combined_imf[start_idx:end_idx+1]
            
            self.ax_combined.plot(seg_timestamps, seg_signal, 'red', 
                                linewidth=2, label=f'Selected Segment ({start_time:.1f}-{end_time:.1f}s)')
            
            # Add vertical lines at segment boundaries
            self.ax_combined.axvline(x=start_time, color='green', linestyle='--', alpha=0.7, label='Start')
            self.ax_combined.axvline(x=end_time, color='orange', linestyle='--', alpha=0.7, label='End')
            
            self.ax_combined.set_title('Segment Preview - Red shows selected range')
            self.ax_combined.set_xlabel('Time (s)')
            self.ax_combined.set_ylabel('Amplitude')
            self.ax_combined.legend()
            self.ax_combined.grid(True, alpha=0.3)
            self.canvas_eemd.draw()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to preview segment: {str(e)}")

    def reset_segment(self):
        """Reset to use the full signal and reset ECG segmentation"""
        if not hasattr(self, 'original_combined_imf'):
            messagebox.showinfo("Info", "No segmentation to reset - using full signal")
            return

        # Reset ECG segmentation if possible
        if self.ecg_loaded:
            try:
                if self.reset_ecg_segmentation():
                    self.eemd_results.insert(tk.END, f"✓ ECG segmentation also reset\n")
                else:
                    self.eemd_results.insert(tk.END, f"⚠ ECG reset failed\n")
            except Exception as e:
                self.eemd_results.insert(tk.END, f"⚠ ECG reset error: {str(e)}\n")

        # Restore original data
        self.latest_combined_imf = self.original_combined_imf.copy()
        self.is_segmented = False
        
        # Re-plot the full signal
        self.ax_combined.clear()
        self.ax_combined.plot(self.original_timestamps, self.latest_combined_imf, 'b-', linewidth=1.5)
        self.ax_combined.set_title('Combined Phase IMFs - Full Signal Restored')
        self.ax_combined.set_xlabel('Time (s)')
        self.ax_combined.set_ylabel('Amplitude')
        self.ax_combined.grid(True, alpha=0.3)
        self.canvas_eemd.draw()
        
        messagebox.showinfo("Reset Complete", "Full signal and ECG restored")

    def update_default_range(self):
        """Update the default time range based on current signal"""
        if not hasattr(self, 'signals') or self.signals is None:
            messagebox.showwarning("Warning", "No signal loaded")
            return
        
        timestamps = self.signals['timestamps']
        duration = timestamps[-1] - timestamps[0]
        
        # Set reasonable defaults
        self.seg_start_var.set(timestamps[0])
        
        # Set end time to either 30 seconds or full duration, whichever is smaller
        default_duration = min(30.0, duration * 0.1)
        end_time = min(timestamps[0] + default_duration, timestamps[-1])
        self.seg_end_var.set(end_time)

    def check_segmentation_status(self):
        """Debug function to check segmentation status"""
        print("=== SEGMENTATION STATUS DEBUG ===")
        print(f"Radar segmented: {hasattr(self, 'is_segmented') and self.is_segmented}")
        if hasattr(self, 'is_segmented') and self.is_segmented:
            print(f"Radar segment length: {len(self.latest_combined_imf)}")
            print(f"Radar time range: {self.segmented_timestamps[0]:.2f} - {self.segmented_timestamps[-1]:.2f}s")
        
        print(f"ECG loaded: {self.ecg_loaded}")
        print(f"ECG segmented: {hasattr(self, 'is_ecg_segmented') and self.is_ecg_segmented}")
        if self.ecg_loaded:
            print(f"ECG current length: {len(self.ecg_data)}")
            print(f"ECG time range: {self.ecg_time[0]:.2f} - {self.ecg_time[-1]:.2f}s")
            if hasattr(self, 'ecg_original_data'):
                print(f"ECG original length: {len(self.ecg_original_data)}")
        
        print("=================================")

    #PEAK DETECTION
    def sensors_peak_detection_algorithm(self, signal, timestamps, fs):
        if len(signal) < 10:
            return np.array([]), np.array([]), np.array([]), {}
        
        # Step 1: Signal conditioning - remove DC offset
        signal = signal - np.mean(signal)
        
        # Step 2: Zero-crossing method to find initial peaks
        zero_crossings = []
        for i in range(1, len(signal)-1):
            if (signal[i-1] < 0 < signal[i+1]) or (signal[i-1] > 0 > signal[i+1]):
                zero_crossings.append(i)
        
        if len(zero_crossings) == 0:
            return np.array([]), np.array([]), np.array([]), {}
        
        # Find positive peaks near zero crossings
        initial_peaks = []
        initial_magnitudes = []
        
        for zc in zero_crossings:
            # Look for local maximum in small window around zero crossing
            window_start = max(0, zc - 5)
            window_end = min(len(signal), zc + 6)
            window = signal[window_start:window_end]
            
            if len(window) > 0:
                local_max_idx = np.argmax(window)
                actual_idx = window_start + local_max_idx
                
                # Only consider positive peaks
                if signal[actual_idx] > 0:
                    initial_peaks.append(actual_idx)
                    initial_magnitudes.append(abs(signal[actual_idx]))
        
        if len(initial_peaks) == 0:
            return np.array([]), np.array([]), np.array([]), {}
        
        initial_peaks = np.array(initial_peaks)
        initial_magnitudes = np.array(initial_magnitudes)
        
        # Step 3: Calculate threshold voltage (Vth) - Equation 1 from paper
        n = len(initial_magnitudes)
        vth = (1.0 / (2.0 * n)) * np.sum(initial_magnitudes / np.sqrt(2))
        
        # Step 4: Apply threshold filtering
        valid_mask = initial_magnitudes >= vth
        filtered_peaks = initial_peaks[valid_mask]
        filtered_magnitudes = initial_magnitudes[valid_mask]
        
        if len(filtered_peaks) == 0:
            return np.array([]), np.array([]), np.array([]), {}
        
        # Step 5: Time interval validation (Tth)
        # Normal heart rate: 60-90 BPM, so intervals should be 0.67-1.0 seconds
        min_interval = 0.5  # 120 BPM max
        max_interval = 2.0  # 30 BPM min
        
        final_peaks = []
        final_magnitudes = []
        
        if len(filtered_peaks) > 0:
            final_peaks.append(filtered_peaks[0])
            final_magnitudes.append(filtered_magnitudes[0])
        
        for i in range(1, len(filtered_peaks)):
            current_peak = filtered_peaks[i]
            previous_peak = final_peaks[-1]
            
            time_interval = (current_peak - previous_peak) / fs
            
            if time_interval >= min_interval:
                if time_interval <= max_interval:
                    # Normal interval - accept peak
                    final_peaks.append(current_peak)
                    final_magnitudes.append(filtered_magnitudes[i])
                else:
                    # Interval too large - try to restore previous peak if available
                    # Look for peaks in the gap that were filtered out
                    gap_start = previous_peak
                    gap_end = current_peak
                    
                    # Check if any initial peaks fell in this gap
                    gap_peaks = initial_peaks[(initial_peaks > gap_start) & (initial_peaks < gap_end)]
                    if len(gap_peaks) > 0:
                        # Find the peak with highest magnitude in the gap
                        gap_magnitudes = [abs(signal[p]) for p in gap_peaks]
                        best_gap_peak = gap_peaks[np.argmax(gap_magnitudes)]
                        
                        # Add the restored peak
                        final_peaks.append(best_gap_peak)
                        final_magnitudes.append(max(gap_magnitudes))
                    
                    # Also add current peak
                    final_peaks.append(current_peak)
                    final_magnitudes.append(filtered_magnitudes[i])
            # If interval too small, skip this peak
        
        final_peaks = np.array(final_peaks)
        final_magnitudes = np.array(final_magnitudes)
        
        # Convert to times and values
        peak_times = timestamps[final_peaks] if len(final_peaks) > 0 else np.array([])
        peak_values = signal[final_peaks] if len(final_peaks) > 0 else np.array([])
        
        # Calculate heart rate
        if len(final_peaks) >= 2:
            duration = timestamps[-1] - timestamps[0]
            heart_rate = (len(final_peaks) / duration) * 60
        else:
            heart_rate = 0
        
        algorithm_info = {
            'initial_peaks': len(initial_peaks),
            'after_threshold': len(filtered_peaks),
            'final_peaks': len(final_peaks),
            'threshold_voltage': vth,
            'heart_rate_bpm': heart_rate,
            'signal_duration': timestamps[-1] - timestamps[0],
            'sampling_rate': fs
        }
        
        return final_peaks, peak_times, peak_values, algorithm_info

    def detect_ecg_r_peaks_advanced(self, ecg_signal, timestamps, fs):
        if len(ecg_signal) < 10:
            return np.array([]), np.array([]), np.array([]), {}
        
        from scipy.signal import butter, filtfilt, find_peaks
        
        # Apply bandpass filter (0.5-40 Hz for ECG)
        nyq = fs / 2
        low = 0.5 / nyq
        high = 40.0 / nyq
        
        if low < 0.01: low = 0.01
        if high > 0.99: high = 0.99
        
        b, a = butter(4, [low, high], btype='band')
        filtered_ecg = filtfilt(b, a, ecg_signal)
        
        # STEP 1: Find ALL peaks directly in filtered ECG signal (not integrated!)
        min_distance = int(0.5 * fs)  # Minimum 0.5 seconds between peaks
        all_peaks, properties = find_peaks(filtered_ecg, distance=min_distance)
        
        if len(all_peaks) == 0:
            return np.array([]), np.array([]), np.array([]), {}
        
        # STEP 2: Get peak heights from filtered ECG
        peak_heights = filtered_ecg[all_peaks]
        
        # STEP 3: Calculate adaptive threshold based on top 30% peaks
        top_peaks_count = max(3, int(0.3 * len(peak_heights)))  # minimum 3 peaks
        top_heights = np.sort(peak_heights)[-top_peaks_count:]
        mean_top_height = np.mean(top_heights)
        
        # STEP 4: Apply threshold (60% of mean top height)
        adaptive_threshold = 0.6 * mean_top_height
        r_peaks = all_peaks[peak_heights >= adaptive_threshold]
        
        if len(r_peaks) == 0:
            return np.array([]), np.array([]), np.array([]), {}
        
        # Convert to times and values
        peak_times = timestamps[r_peaks]
        peak_values = ecg_signal[r_peaks]  # Use original signal for display
        
        # Calculate heart rate
        if len(r_peaks) >= 2:
            rr_intervals = np.diff(peak_times)
            avg_rr = np.mean(rr_intervals)
            heart_rate = 60.0 / avg_rr if avg_rr > 0 else 0
        else:
            heart_rate = 0
        
        algorithm_info = {
            'detected_peaks': len(r_peaks),
            'heart_rate_bpm': heart_rate,
            'signal_duration': timestamps[-1] - timestamps[0],
            'sampling_rate': fs,
            'method': 'Adaptive Threshold (ECG_PLOT_TEST_2 method)'
        }
        
        return r_peaks, peak_times, peak_values, algorithm_info

    def compare_peak_detection_results(self, radar_peaks, radar_times, ecg_peaks, ecg_times):
        """
        Compare radar and ECG peak detection results
        """
        if len(radar_peaks) == 0 or len(ecg_peaks) == 0:
            return {
                'accuracy': 0,
                'sensitivity': 0,
                'precision': 0,
                'time_correlation': 0,
                'heart_rate_difference': 0,
                'synchronized_peaks': 0
            }
        
        # Calculate heart rates
        radar_duration = radar_times[-1] - radar_times[0] if len(radar_times) > 1 else 1
        ecg_duration = ecg_times[-1] - ecg_times[0] if len(ecg_times) > 1 else 1
        
        radar_hr = (len(radar_peaks) / radar_duration) * 60
        ecg_hr = (len(ecg_peaks) / ecg_duration) * 60
        
        # Calculate accuracy using equation from paper
        hr_difference = abs(ecg_hr - radar_hr)
        accuracy = ((ecg_hr - hr_difference) / ecg_hr) * 100 if ecg_hr > 0 else 0
        
        # Find synchronized peaks (within ±0.2 seconds)
        tolerance = 0.2  # seconds
        synchronized_count = 0
        
        for radar_time in radar_times:
            # Find closest ECG peak
            time_differences = np.abs(ecg_times - radar_time)
            min_diff = np.min(time_differences)
            
            if min_diff <= tolerance:
                synchronized_count += 1
        
        # Calculate sensitivity and precision
        sensitivity = synchronized_count / len(ecg_peaks) if len(ecg_peaks) > 0 else 0
        precision = synchronized_count / len(radar_peaks) if len(radar_peaks) > 0 else 0
        
        # Time correlation
        if len(radar_times) > 1 and len(ecg_times) > 1:
            # Calculate RR intervals
            radar_rr = np.diff(radar_times)
            ecg_rr = np.diff(ecg_times)
            
            if len(radar_rr) > 0 and len(ecg_rr) > 0:
                # Match similar length arrays
                min_length = min(len(radar_rr), len(ecg_rr))
                radar_rr = radar_rr[:min_length]
                ecg_rr = ecg_rr[:min_length]
                
                if min_length > 1:
                    time_correlation = np.corrcoef(radar_rr, ecg_rr)[0, 1]
                else:
                    time_correlation = 0
            else:
                time_correlation = 0
        else:
            time_correlation = 0
        
        return {
            'accuracy': max(0, accuracy),
            'sensitivity': sensitivity * 100,
            'precision': precision * 100,
            'time_correlation': time_correlation if not np.isnan(time_correlation) else 0,
            'heart_rate_difference': hr_difference,
            'synchronized_peaks': synchronized_count,
            'radar_heart_rate': radar_hr,
            'ecg_heart_rate': ecg_hr,
            'radar_peak_count': len(radar_peaks),
            'ecg_peak_count': len(ecg_peaks)
        }
    
    def load_signal_for_peak_detection(self):
        """Load the combined IMF signal from EEMD tab for peak detection"""
        if not hasattr(self, 'latest_combined_imf') or self.latest_combined_imf is None:
            messagebox.showwarning("Warning", 
                "No combined IMF signal available.\n"
                "Go to EEMD tab and combine IMFs first.")
            return
        
        # Use segmented signal and timestamps if available, otherwise use full signal
        if hasattr(self, 'is_segmented') and self.is_segmented and hasattr(self, 'segmented_timestamps'):
            self.peak_detection_signal = self.latest_combined_imf.copy()
            self.peak_detection_timestamps = self.segmented_timestamps.copy()
            signal_status = "✓ Using segmented radar signal"
        else:
            self.peak_detection_signal = self.latest_combined_imf.copy()
            self.peak_detection_timestamps = self.signals['timestamps'].copy()
            signal_status = "Using full radar signal"
        
        # Update sampling rate
        duration = self.peak_detection_timestamps[-1] - self.peak_detection_timestamps[0]
        actual_fs = (len(self.peak_detection_signal) - 1) / duration
        self.peak_fs_var.set(int(actual_fs))
        
        # Update status
        status_text = f"Signal loaded: {len(self.peak_detection_signal)} samples, FS: {int(actual_fs)}Hz"
        self.peak_signal_status.set(status_text)
        
        # Plot raw signal
        self.ax_signal.clear()
        self.ax_signal.plot(self.peak_detection_timestamps, self.peak_detection_signal, 'b-', linewidth=0.8)
        
        if hasattr(self, 'is_segmented') and self.is_segmented:
            title = 'Segmented Radar Signal for Peak Detection ✓'
        else:
            title = 'Full Radar Signal for Peak Detection'
        
        self.ax_signal.set_title(title)
        self.ax_signal.set_xlabel('Time (s)')
        self.ax_signal.set_ylabel('Amplitude')
        self.ax_signal.grid(True, alpha=0.3)
        
        # **ADD ECG STATUS CHECK**
        ecg_status_text = ""
        if self.ecg_loaded:
            print(f"DEBUG: ECG status check - ECG loaded: True")
            print(f"DEBUG: ECG current length: {len(self.ecg_data)}")
            print(f"DEBUG: ECG time range: {self.ecg_time[0]:.2f} - {self.ecg_time[-1]:.2f}s")
            print(f"DEBUG: is_ecg_segmented: {hasattr(self, 'is_ecg_segmented') and self.is_ecg_segmented}")
            
            if hasattr(self, 'is_ecg_segmented') and self.is_ecg_segmented:
                ecg_status_text = f"ECG: Segmented ({len(self.ecg_data)} samples, {self.ecg_time[0]:.1f}-{self.ecg_time[-1]:.1f}s)"
            else:
                ecg_status_text = f"ECG: Full signal ({len(self.ecg_data)} samples)"
        else:
            ecg_status_text = "ECG: Not loaded"
        
        # Update results
        self.peak_results.delete(1.0, tk.END)
        self.peak_results.insert(tk.END, f"SIGNAL LOADED FOR PEAK DETECTION\n")
        self.peak_results.insert(tk.END, f"=" * 35 + "\n")
        self.peak_results.insert(tk.END, f"Radar: {signal_status}\n")
        self.peak_results.insert(tk.END, f"Samples: {len(self.peak_detection_signal)}\n")
        self.peak_results.insert(tk.END, f"Duration: {duration:.2f} seconds\n")
        self.peak_results.insert(tk.END, f"Sampling Rate: {int(actual_fs)} Hz\n")
        self.peak_results.insert(tk.END, f"Time Range: {self.peak_detection_timestamps[0]:.2f} - {self.peak_detection_timestamps[-1]:.2f} s\n\n")
        self.peak_results.insert(tk.END, f"{ecg_status_text}\n\n")
        self.canvas_peaks.draw()
        self.initialize_edit_mode_variables()
        messagebox.showinfo("Signal Loaded", f"Ready for peak detection!\n{signal_status}\n{ecg_status_text}")

    def force_ecg_segmentation_for_peak_detection(self):
        """Force ECG to use segmented data matching radar segmentation - ERROR SAFE VERSION"""
        if not self.ecg_loaded:
            return False, "No ECG loaded"
        
        # Check if radar is segmented
        if not (hasattr(self, 'is_segmented') and self.is_segmented):
            return False, "Radar not segmented"
        
        # Check if we have valid segmented timestamps
        if not hasattr(self, 'segmented_timestamps') or self.segmented_timestamps is None:
            return False, "No segmented radar timestamps available"
        
        # Get radar segment time range
        radar_start_time = self.segmented_timestamps[0]
        radar_end_time = self.segmented_timestamps[-1]
        
        print(f"DEBUG: Forcing ECG segmentation to match radar: {radar_start_time:.2f}s - {radar_end_time:.2f}s")
        
        # Check ECG data availability and validity
        if not hasattr(self, 'ecg_data') or self.ecg_data is None:
            return False, "ECG data is None"
        
        if not hasattr(self, 'ecg_time') or self.ecg_time is None:
            return False, "ECG time data is None"
        
        # Use original ECG data if available, otherwise use current
        if hasattr(self, 'ecg_original_data') and self.ecg_original_data is not None:
            source_ecg = self.ecg_original_data
            source_time = self.ecg_original_time
            print(f"DEBUG: Using original ECG data - length: {len(source_ecg)}")
        else:
            # Store current as original and use it
            self.ecg_original_data = self.ecg_data.copy()
            self.ecg_original_time = self.ecg_time.copy()
            source_ecg = self.ecg_data
            source_time = self.ecg_time
            print(f"DEBUG: Using current ECG data as original - length: {len(source_ecg)}")
        
        # Final check that source_time is not None
        if source_time is None:
            return False, "ECG time data is None - cannot segment"
        
        print(f"DEBUG: ECG source time range: {source_time[0]:.2f}s - {source_time[-1]:.2f}s")
        
        # Check if radar time range is within ECG time range
        if radar_start_time < source_time[0] or radar_end_time > source_time[-1]:
            return False, f"Radar time range ({radar_start_time:.2f}-{radar_end_time:.2f}s) outside ECG range ({source_time[0]:.2f}-{source_time[-1]:.2f}s)"
        
        # Find ECG indices for the radar time range
        start_idx = np.argmin(np.abs(source_time - radar_start_time))
        end_idx = np.argmin(np.abs(source_time - radar_end_time))
        
        print(f"DEBUG: ECG segmentation indices: {start_idx} to {end_idx}")
        
        # Apply segmentation
        self.ecg_data = source_ecg[start_idx:end_idx+1].copy()
        self.ecg_time = source_time[start_idx:end_idx+1].copy()
        self.is_ecg_segmented = True
        
        print(f"DEBUG: ECG segmented successfully - new length: {len(self.ecg_data)}")
        print(f"DEBUG: ECG new time range: {self.ecg_time[0]:.2f} - {self.ecg_time[-1]:.2f}s")
        
        return True, f"ECG segmented to match radar: {len(self.ecg_data)} samples"

    def validate_ecg_data(self):
        """Validate ECG data integrity"""
        print("=== ECG DATA VALIDATION ===")
        print(f"ECG loaded flag: {self.ecg_loaded}")
        
        if not self.ecg_loaded:
            print("ECG not loaded")
            return False
        
        print(f"ECG data exists: {hasattr(self, 'ecg_data') and self.ecg_data is not None}")
        print(f"ECG time exists: {hasattr(self, 'ecg_time') and self.ecg_time is not None}")
        
        if hasattr(self, 'ecg_data') and self.ecg_data is not None:
            print(f"ECG data length: {len(self.ecg_data)}")
            print(f"ECG data type: {type(self.ecg_data)}")
        else:
            print("ECG data is None!")
            return False
        
        if hasattr(self, 'ecg_time') and self.ecg_time is not None:
            print(f"ECG time length: {len(self.ecg_time)}")
            print(f"ECG time range: {self.ecg_time[0]:.2f} - {self.ecg_time[-1]:.2f}s")
            print(f"ECG time type: {type(self.ecg_time)}")
        else:
            print("ECG time is None!")
            return False
        
        if hasattr(self, 'ecg_original_data'):
            print(f"ECG original data exists: {self.ecg_original_data is not None}")
            if self.ecg_original_data is not None:
                print(f"ECG original length: {len(self.ecg_original_data)}")
        else:
            print("No ECG original data stored")
        
        print("========================")
        return True

    def detect_radar_peaks(self):
        """Detect peaks in radar signal using sensors algorithm"""
        if not hasattr(self, 'peak_detection_signal') or self.peak_detection_signal is None:
            messagebox.showwarning("Warning", "Load signal first")
            return
        
        fs = self.peak_fs_var.get()
        
        # Apply sensors peak detection algorithm
        peak_indices, peak_times, peak_values, algo_info = self.sensors_peak_detection_algorithm(
            self.peak_detection_signal, self.peak_detection_timestamps, fs)
        
        if len(peak_indices) == 0:
            messagebox.showwarning("Warning", "No peaks detected in radar signal")
            return
        
        # Store results
        self.detected_radar_peaks = peak_indices
        self.radar_peak_times = peak_times
        self.radar_peak_values = peak_values
        self.radar_algo_info = algo_info
        
        # Plot results
        self.ax_radar_peaks.clear()
        self.ax_radar_peaks.plot(self.peak_detection_timestamps, self.peak_detection_signal, 'b-', 
                                linewidth=0.8, alpha=0.7, label='Radar Signal')
        self.ax_radar_peaks.plot(peak_times, peak_values, 'ro', markersize=6, 
                                label=f'Detected Peaks ({len(peak_indices)})')
        
        self.ax_radar_peaks.set_title(f'Radar Peak Detection - {len(peak_indices)} peaks, HR: {algo_info["heart_rate_bpm"]:.1f} BPM')
        self.ax_radar_peaks.set_xlabel('Time (s)')
        self.ax_radar_peaks.set_ylabel('Amplitude')
        self.ax_radar_peaks.legend()
        self.ax_radar_peaks.grid(True, alpha=0.3)
        
        # Update results
        self.peak_results.insert(tk.END, f"RADAR PEAK DETECTION RESULTS\n")
        self.peak_results.insert(tk.END, f"=" * 35 + "\n")
        self.peak_results.insert(tk.END, f"Algorithm: Sensors Paper Method\n")
        self.peak_results.insert(tk.END, f"Initial peaks (zero-crossing): {algo_info['initial_peaks']}\n")
        self.peak_results.insert(tk.END, f"After threshold filter: {algo_info['after_threshold']}\n")
        self.peak_results.insert(tk.END, f"Final detected peaks: {algo_info['final_peaks']}\n")
        self.peak_results.insert(tk.END, f"Threshold voltage (Vth): {algo_info['threshold_voltage']:.4f}\n")
        self.peak_results.insert(tk.END, f"Heart Rate: {algo_info['heart_rate_bpm']:.1f} BPM\n")
        self.peak_results.insert(tk.END, f"Signal duration: {algo_info['signal_duration']:.2f} s\n\n")
        
        self.canvas_peaks.draw()

    def detect_ecg_peaks(self):
        """Detect R-peaks in ECG signal - with proper error handling"""
        if not self.ecg_loaded:
            messagebox.showwarning("Warning", "Load ECG data first")
            return
        
        # Validate ECG data first
        if not self.validate_ecg_data():
            messagebox.showerror("Error", "ECG data is not properly loaded. Please load ECG data again.")
            return
        
        # Try to force ECG segmentation if radar is segmented
        if hasattr(self, 'is_segmented') and self.is_segmented:
            success, message = self.force_ecg_segmentation_for_peak_detection()
            if success:
                print(f"DEBUG: ECG segmentation successful - {message}")
                ecg_status = "Segmented ECG (Matched to Radar)"
            else:
                print(f"DEBUG: ECG segmentation failed - {message}")
                ecg_status = "Full ECG (Segmentation Failed)"
        else:
            ecg_status = "Full ECG"
        
        # Use current ECG data
        ecg_signal = self.ecg_data
        ecg_timestamps = self.ecg_time
        
        print(f"DEBUG: Using {ecg_status}")
        print(f"DEBUG: ECG signal length: {len(ecg_signal)}, time range: {ecg_timestamps[0]:.2f}-{ecg_timestamps[-1]:.2f}s")
        
        # Apply ECG peak detection
        try:
            peak_indices, peak_times, peak_values, algo_info = self.detect_ecg_r_peaks_advanced(
                ecg_signal, ecg_timestamps, self.ecg_fs)
        except Exception as e:
            messagebox.showerror("Error", f"ECG peak detection failed: {str(e)}")
            return
        
        if len(peak_indices) == 0:
            messagebox.showwarning("Warning", "No R-peaks detected in ECG signal")
            return
        
        # Store results
        self.detected_ecg_peaks = peak_indices
        self.ecg_peak_times = peak_times
        self.ecg_peak_values = peak_values
        self.ecg_algo_info = algo_info
        
        # Plot results
        self.ax_ecg_peaks.clear()
        self.ax_ecg_peaks.plot(ecg_timestamps, ecg_signal, 'g-', 
                            linewidth=0.8, alpha=0.7, label='ECG Signal')
        self.ax_ecg_peaks.plot(peak_times, peak_values, 'ro', markersize=6, 
                            label=f'R-Peaks ({len(peak_indices)})')
        
        self.ax_ecg_peaks.set_title(f'ECG R-Peak Detection ({ecg_status}) - {len(peak_indices)} peaks, HR: {algo_info["heart_rate_bpm"]:.1f} BPM')
        self.ax_ecg_peaks.set_xlabel('Time (s)')
        self.ax_ecg_peaks.set_ylabel('Amplitude')
        self.ax_ecg_peaks.legend()
        self.ax_ecg_peaks.grid(True, alpha=0.3)
        
        # Update results
        self.peak_results.insert(tk.END, f"ECG PEAK DETECTION RESULTS\n")
        self.peak_results.insert(tk.END, f"=" * 35 + "\n")
        self.peak_results.insert(tk.END, f"Status: {ecg_status}\n")
        self.peak_results.insert(tk.END, f"Detected R-peaks: {algo_info['detected_peaks']}\n")
        self.peak_results.insert(tk.END, f"Heart Rate: {algo_info['heart_rate_bpm']:.1f} BPM\n")
        self.peak_results.insert(tk.END, f"Signal duration: {algo_info['signal_duration']:.2f} s\n")
        self.peak_results.insert(tk.END, f"Time range: {ecg_timestamps[0]:.2f} - {ecg_timestamps[-1]:.2f} s\n\n")
        
        self.canvas_peaks.draw()

    def compare_peak_results(self):
        """Compare radar and ECG peak detection results - FORCES segmented view"""
        if not hasattr(self, 'detected_radar_peaks') or not hasattr(self, 'detected_ecg_peaks'):
            messagebox.showwarning("Warning", "Detect peaks in both signals first")
            return
        
        # **FORCE ECG SEGMENTATION AGAIN** to ensure comparison uses segmented data
        if hasattr(self, 'is_segmented') and self.is_segmented and self.ecg_loaded:
            self.force_ecg_segmentation_for_peak_detection()
        
        # Compare results
        comparison = self.compare_peak_detection_results(
            self.detected_radar_peaks, self.radar_peak_times,
            self.detected_ecg_peaks, self.ecg_peak_times)
        
        # Plot comparison - use current data (now definitely segmented)
        self.ax_comparison.clear()
        
        # Get radar data (from peak detection)
        radar_signal = self.peak_detection_signal
        radar_timestamps = self.peak_detection_timestamps
        
        # Get ECG data (now forced to be segmented)
        ecg_signal = self.ecg_data
        ecg_timestamps = self.ecg_time
        
        # Determine status
        radar_status = "Segmented Radar" if (hasattr(self, 'is_segmented') and self.is_segmented) else "Full Radar"
        ecg_status = "Segmented ECG" if (hasattr(self, 'is_segmented') and self.is_segmented) else "Full ECG"
        
        print(f"DEBUG: Comparison using {radar_status} vs {ecg_status}")
        print(f"DEBUG: Radar: {len(radar_signal)} samples, {radar_timestamps[0]:.2f}-{radar_timestamps[-1]:.2f}s")
        print(f"DEBUG: ECG: {len(ecg_signal)} samples, {ecg_timestamps[0]:.2f}-{ecg_timestamps[-1]:.2f}s")
        
        # Normalize signals
        radar_norm = (radar_signal - np.mean(radar_signal)) / np.std(radar_signal)
        ecg_norm = (ecg_signal - np.mean(ecg_signal)) / np.std(ecg_signal)
        
        # Plot both signals
        self.ax_comparison.plot(radar_timestamps, radar_norm, 'b-', 
                            alpha=0.6, linewidth=1, label=f'{radar_status}')
        self.ax_comparison.plot(ecg_timestamps, ecg_norm, 'g-', 
                            alpha=0.6, linewidth=1, label=f'{ecg_status}')
        
        # Mark detected peaks
        radar_peak_values_norm = radar_norm[self.detected_radar_peaks]
        ecg_peak_values_norm = ecg_norm[self.detected_ecg_peaks]
        
        self.ax_comparison.plot(self.radar_peak_times, radar_peak_values_norm, 'ro', 
                            markersize=8, label=f'Radar Peaks ({len(self.detected_radar_peaks)})')
        self.ax_comparison.plot(self.ecg_peak_times, ecg_peak_values_norm, 'gs', 
                            markersize=8, label=f'ECG Peaks ({len(self.detected_ecg_peaks)})')
        
        self.ax_comparison.set_title(f'{radar_status} vs {ecg_status} Comparison - Accuracy: {comparison["accuracy"]:.1f}%')
        self.ax_comparison.set_xlabel('Time (s)')
        self.ax_comparison.set_ylabel('Normalized Amplitude')
        self.ax_comparison.legend()
        self.ax_comparison.grid(True, alpha=0.3)
        
        # Update results
        self.peak_results.insert(tk.END, f"RADAR vs ECG COMPARISON\n")
        self.peak_results.insert(tk.END, f"=" * 35 + "\n")
        self.peak_results.insert(tk.END, f"Comparison: {radar_status} vs {ecg_status}\n")
        self.peak_results.insert(tk.END, f"Accuracy (Paper Eq. 6): {comparison['accuracy']:.2f}%\n")
        self.peak_results.insert(tk.END, f"Sensitivity: {comparison['sensitivity']:.2f}%\n")
        self.peak_results.insert(tk.END, f"Precision: {comparison['precision']:.2f}%\n")
        self.peak_results.insert(tk.END, f"HR Difference: {comparison['heart_rate_difference']:.1f} BPM\n")
        self.peak_results.insert(tk.END, f"Radar HR: {comparison['radar_heart_rate']:.1f} BPM\n")
        self.peak_results.insert(tk.END, f"ECG HR: {comparison['ecg_heart_rate']:.1f} BPM\n\n")
        
        self.canvas_peaks.draw()
        
        messagebox.showinfo("Comparison Complete", 
            f"Comparison: {radar_status} vs {ecg_status}\n"
            f"Accuracy: {comparison['accuracy']:.1f}%\n"
            f"HR Difference: {comparison['heart_rate_difference']:.1f} BPM")

    def get_current_ecg_for_peak_detection(self):
        """Get current ECG data for peak detection (segmented or full)"""
        if hasattr(self, 'is_ecg_segmented') and self.is_ecg_segmented and hasattr(self, 'ecg_segmented'):
            return self.ecg_segmented, self.ecg_segmented_timestamps, "Segmented ECG"
        else:
            return self.ecg_data, self.ecg_time, "Full ECG"

    def export_peak_data(self):
        """Export peak detection data to CSV"""
        if not hasattr(self, 'detected_radar_peaks') or not hasattr(self, 'detected_ecg_peaks'):
            messagebox.showwarning("Warning", "No peak data to export. Run detection first.")
            return
        
        file_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            title="Export Peak Detection Data"
        )
        
        if not file_path:
            return
        
        try:
            import pandas as pd
            
            # Create export data
            export_data = {
                'Radar_Peak_Times': pd.Series(self.radar_peak_times),
                'Radar_Peak_Values': pd.Series(self.radar_peak_values),
                'ECG_Peak_Times': pd.Series(self.ecg_peak_times),
                'ECG_Peak_Values': pd.Series(self.ecg_peak_values)
            }
            
            df = pd.DataFrame(export_data)
            df.to_csv(file_path, index=False)
            
            messagebox.showinfo("Export Complete", f"Peak data exported to:\n{file_path}")
            
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to export data: {str(e)}")

    def save_peak_plots(self):
        """Save peak detection plots"""
        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("PDF files", "*.pdf"), ("All files", "*.*")],
            title="Save Peak Detection Plots"
        )
        
        if not file_path:
            return
        
        try:
            self.fig_peaks.savefig(file_path, dpi=300, bbox_inches='tight')
            messagebox.showinfo("Save Complete", f"Plots saved to:\n{file_path}")
        except Exception as e:
            messagebox.showerror("Save Error", f"Failed to save plots: {str(e)}")

    #HRV
    def calculate_hrv_from_peaks(self, peak_times, signal_name="Signal"):
        """Calculate HRV metrics from detected peak times (without pNN50)"""
        if len(peak_times) < 2:
            return None
        
        # Calculate RR intervals (peak-to-peak intervals in milliseconds)
        rr_intervals = np.diff(peak_times) * 1000  # Convert to milliseconds
        
        if len(rr_intervals) < 2:
            return None
        
        # Remove outliers (RR intervals outside 300-2000ms range)
        valid_mask = (rr_intervals >= 300) & (rr_intervals <= 2000)
        rr_clean = rr_intervals[valid_mask]
        
        if len(rr_clean) < 2:
            return None
        
        # Time domain HRV metrics
        mean_rr = np.mean(rr_clean)
        sdnn = np.std(rr_clean, ddof=1)  # Standard deviation of RR intervals
        
        # RMSSD - Root Mean Square of Successive Differences
        if len(rr_clean) > 1:
            successive_diffs = np.diff(rr_clean)
            rmssd = np.sqrt(np.mean(successive_diffs**2))
        else:
            rmssd = 0
        
        # Heart rate statistics
        heart_rates = 60000 / rr_clean  # Convert to BPM
        mean_hr = np.mean(heart_rates)
        std_hr = np.std(heart_rates)
        
        # Geometric measures
        rr_tri_index = len(rr_clean) / np.max(np.histogram(rr_clean, bins=50)[0]) if len(rr_clean) > 0 else 0
        
        return {
            'signal_name': signal_name,
            'rr_intervals': rr_clean,
            'rr_times': peak_times[1:len(rr_clean)+1],  # Time points for RR intervals
            'valid_rr_count': len(rr_clean),
            'total_peaks': len(peak_times),
            'mean_rr': mean_rr,
            'sdnn': sdnn,
            'rmssd': rmssd,
            'mean_hr': mean_hr,
            'std_hr': std_hr,
            'min_rr': np.min(rr_clean),
            'max_rr': np.max(rr_clean),
            'rr_tri_index': rr_tri_index,
            'duration': peak_times[-1] - peak_times[0]
        }

    def calculate_hrv_comparison_metrics(self, radar_hrv, ecg_hrv):
        """Calculate comparison metrics between radar and ECG HRV (without pNN50)"""
        if radar_hrv is None or ecg_hrv is None:
            return None
        
        # Compare key metrics
        metrics_comparison = {}
        
        # Mean RR interval comparison
        rr_correlation = 0
        if len(radar_hrv['rr_intervals']) > 1 and len(ecg_hrv['rr_intervals']) > 1:
            # Match lengths for correlation
            min_length = min(len(radar_hrv['rr_intervals']), len(ecg_hrv['rr_intervals']))
            radar_rr_sync = radar_hrv['rr_intervals'][:min_length]
            ecg_rr_sync = ecg_hrv['rr_intervals'][:min_length]
            
            if min_length > 1:
                rr_correlation = np.corrcoef(radar_rr_sync, ecg_rr_sync)[0, 1]
                if np.isnan(rr_correlation):
                    rr_correlation = 0
        
        metrics_comparison = {
            'rr_correlation': rr_correlation,
            'mean_rr_diff': abs(radar_hrv['mean_rr'] - ecg_hrv['mean_rr']),
            'sdnn_diff': abs(radar_hrv['sdnn'] - ecg_hrv['sdnn']),
            'rmssd_diff': abs(radar_hrv['rmssd'] - ecg_hrv['rmssd']),
            'hr_diff': abs(radar_hrv['mean_hr'] - ecg_hrv['mean_hr']),
            'sdnn_ratio': radar_hrv['sdnn'] / ecg_hrv['sdnn'] if ecg_hrv['sdnn'] > 0 else 0,
            'rmssd_ratio': radar_hrv['rmssd'] / ecg_hrv['rmssd'] if ecg_hrv['rmssd'] > 0 else 0,
        }
        return metrics_comparison

    def analyze_hrv_from_peaks(self):
        """Analyze HRV from both radar and ECG detected peaks with RMSSD and RMSE"""
        print("DEBUG: analyze_hrv_from_peaks called")
        
        # Check if peaks exist
        if not hasattr(self, 'detected_radar_peaks') or not hasattr(self, 'detected_ecg_peaks'):
            print("DEBUG: Missing detected peaks")
            messagebox.showwarning("Warning", "Detect peaks in both signals first")
            return
        
        if not hasattr(self, 'radar_peak_times') or not hasattr(self, 'ecg_peak_times'):
            print("DEBUG: Missing peak times")
            messagebox.showwarning("Warning", "Peak times not available. Run peak detection first.")
            return
        
        print(f"DEBUG: Radar peaks: {len(self.radar_peak_times) if self.radar_peak_times is not None else 0}")
        print(f"DEBUG: ECG peaks: {len(self.ecg_peak_times) if self.ecg_peak_times is not None else 0}")
        
        try:
            # Calculate HRV for radar signal
            print("DEBUG: Calculating radar HRV")
            self.radar_hrv = self.calculate_hrv_from_peaks(self.radar_peak_times, "Radar")
            print(f"DEBUG: Radar HRV result: {self.radar_hrv is not None}")
            
            # Calculate HRV for ECG signal  
            print("DEBUG: Calculating ECG HRV")
            self.ecg_hrv = self.calculate_hrv_from_peaks(self.ecg_peak_times, "ECG")
            print(f"DEBUG: ECG HRV result: {self.ecg_hrv is not None}")
            
            if self.radar_hrv is None and self.ecg_hrv is None:
                print("DEBUG: Both HRV calculations failed")
                messagebox.showerror("Error", "Unable to calculate HRV from either signal")
                return
            
            if self.radar_hrv is None:
                print("DEBUG: Radar HRV calculation failed")
                messagebox.showwarning("Warning", "Unable to calculate radar HRV")
            
            if self.ecg_hrv is None:
                print("DEBUG: ECG HRV calculation failed")
                messagebox.showwarning("Warning", "Unable to calculate ECG HRV")
            
            # ============= CALCULATE RMSSD AND RMSE =============
            print("DEBUG: Calculating RMSSD for both signals")
            radar_rmssd_result = self.calculate_rmssd_from_peaks(self.radar_peak_times)
            ecg_rmssd_result = self.calculate_rmssd_from_peaks(self.ecg_peak_times)
            
            # Calculate RMSE between radar and ECG RR intervals
            rmse_result = None
            if self.radar_hrv and self.ecg_hrv:
                rmse_result = self.calculate_rr_intervals_rmse(
                    self.radar_hrv['rr_intervals'], 
                    self.ecg_hrv['rr_intervals']
                )
            # ====================================================
            
            # Calculate comparison metrics
            print("DEBUG: Calculating comparison metrics")
            self.hrv_comparison = self.calculate_hrv_comparison_metrics(self.radar_hrv, self.ecg_hrv)
            print(f"DEBUG: Comparison metrics: {self.hrv_comparison is not None}")
            
            # Plot HRV comparison
            print("DEBUG: Plotting HRV comparison")
            self.plot_hrv_comparison()
            
            # Display regular HRV results
            print("DEBUG: Displaying HRV results")
            self.display_hrv_results()
            
            # Display detailed SDNN and RMSSD analysis
            print("DEBUG: Displaying detailed SDNN/RMSSD results")
            self.display_detailed_sdnn_rmssd_results()
            
            # ============= DISPLAY RMSSD AND RMSE RESULTS =============
            self.peak_results.insert(tk.END, f"\n{'='*60}\n")
            self.peak_results.insert(tk.END, f"HRV RMSSD ANALYSIS:\n")
            self.peak_results.insert(tk.END, f"{'='*60}\n")
            
            if radar_rmssd_result:
                self.peak_results.insert(tk.END, f"\nRadar HRV:\n")
                self.peak_results.insert(tk.END, f"  RMSSD: {radar_rmssd_result['rmssd']:.2f} ms\n")
                self.peak_results.insert(tk.END, f"  Valid RR intervals: {radar_rmssd_result['valid_intervals']}/{radar_rmssd_result['total_intervals']}\n")
            
            if ecg_rmssd_result:
                self.peak_results.insert(tk.END, f"\nECG HRV:\n")
                self.peak_results.insert(tk.END, f"  RMSSD: {ecg_rmssd_result['rmssd']:.2f} ms\n")
                self.peak_results.insert(tk.END, f"  Valid RR intervals: {ecg_rmssd_result['valid_intervals']}/{ecg_rmssd_result['total_intervals']}\n")
            
            if radar_rmssd_result and ecg_rmssd_result:
                rmssd_diff = abs(radar_rmssd_result['rmssd'] - ecg_rmssd_result['rmssd'])
                rmssd_error_percent = (rmssd_diff / ecg_rmssd_result['rmssd']) * 100 if ecg_rmssd_result['rmssd'] > 0 else 0
                
                self.peak_results.insert(tk.END, f"\nRMSSD Comparison:\n")
                self.peak_results.insert(tk.END, f"  Absolute Difference: {rmssd_diff:.2f} ms\n")
                self.peak_results.insert(tk.END, f"  Relative Error: {rmssd_error_percent:.1f}%\n")
            
            if rmse_result:
                self.peak_results.insert(tk.END, f"\n{'='*60}\n")
                self.peak_results.insert(tk.END, f"RR INTERVALS RMSE (Radar vs ECG):\n")
                self.peak_results.insert(tk.END, f"{'='*60}\n")
                self.peak_results.insert(tk.END, f"  RMSE: {rmse_result['rmse']:.2f} ms\n")
                self.peak_results.insert(tk.END, f"  MAE (Mean Absolute Error): {rmse_result['mae']:.2f} ms\n")
                self.peak_results.insert(tk.END, f"  Compared intervals: {rmse_result['n_compared']}\n")
                self.peak_results.insert(tk.END, f"  Correlation: {rmse_result['correlation']:.4f}\n")
                
                # Interpretation
                if rmse_result['rmse'] < 10:
                    quality = "Excellent"
                elif rmse_result['rmse'] < 20:
                    quality = "Good"
                elif rmse_result['rmse'] < 50:
                    quality = "Moderate"
                else:
                    quality = "Poor"
                
                self.peak_results.insert(tk.END, f"  Quality: {quality}\n")
            
            self.peak_results.insert(tk.END, f"{'='*60}\n\n")
            # ===========================================================
            
            print("DEBUG: HRV analysis completed successfully")
            
        except Exception as e:
            print(f"DEBUG: Error in HRV analysis: {e}")
            import traceback
            traceback.print_exc()
            messagebox.showerror("Error", f"HRV analysis failed: {str(e)}")

    def plot_hrv_comparison(self):
        """Plot HRV comparison between radar and ECG with error handling"""
        print("DEBUG: plot_hrv_comparison called")
        
        try:
            self.ax_hrv.clear()
            
            plots_added = 0
            
            # Plot RR intervals for radar
            if self.radar_hrv is not None and len(self.radar_hrv['rr_intervals']) > 0:
                print(f"DEBUG: Plotting radar RR intervals: {len(self.radar_hrv['rr_intervals'])}")
                self.ax_hrv.plot(self.radar_hrv['rr_times'], self.radar_hrv['rr_intervals'], 
                                'bo-', markersize=4, linewidth=1, alpha=0.7, 
                                label=f"Radar RR ({len(self.radar_hrv['rr_intervals'])} intervals)")
                
                # Add mean line for radar
                self.ax_hrv.axhline(y=self.radar_hrv['mean_rr'], color='blue', linestyle='--', 
                                alpha=0.5, label=f"Radar Mean: {self.radar_hrv['mean_rr']:.1f}ms")
                plots_added += 1
            else:
                print("DEBUG: No radar HRV data to plot")
            
            # Plot RR intervals for ECG
            if self.ecg_hrv is not None and len(self.ecg_hrv['rr_intervals']) > 0:
                print(f"DEBUG: Plotting ECG RR intervals: {len(self.ecg_hrv['rr_intervals'])}")
                self.ax_hrv.plot(self.ecg_hrv['rr_times'], self.ecg_hrv['rr_intervals'], 
                                'ro-', markersize=4, linewidth=1, alpha=0.7,
                                label=f"ECG RR ({len(self.ecg_hrv['rr_intervals'])} intervals)")
                
                # Add mean line for ECG
                self.ax_hrv.axhline(y=self.ecg_hrv['mean_rr'], color='red', linestyle='--', 
                                alpha=0.5, label=f"ECG Mean: {self.ecg_hrv['mean_rr']:.1f}ms")
                plots_added += 1
            else:
                print("DEBUG: No ECG HRV data to plot")
            
            if plots_added == 0:
                # No data to plot
                self.ax_hrv.text(0.5, 0.5, 'No HRV data available\nCheck peak detection results', 
                                ha='center', va='center', transform=self.ax_hrv.transAxes)
                print("DEBUG: No HRV data available for plotting")
            else:
                print(f"DEBUG: Added {plots_added} HRV plots")
            
            self.ax_hrv.set_title('HRV Analysis: RR Intervals Comparison')
            self.ax_hrv.set_xlabel('Time (s)')
            self.ax_hrv.set_ylabel('RR Interval (ms)')
            self.ax_hrv.legend()
            self.ax_hrv.grid(True, alpha=0.3)
            
            # Force canvas redraw
            print("DEBUG: Drawing canvas")
            self.canvas_peaks.draw()
            
        except Exception as e:
            print(f"DEBUG: Error in plot_hrv_comparison: {e}")
            import traceback
            traceback.print_exc()
            
            # Show error on plot
            self.ax_hrv.clear()
            self.ax_hrv.text(0.5, 0.5, f'Error plotting HRV:\n{str(e)}', 
                            ha='center', va='center', transform=self.ax_hrv.transAxes)
            self.canvas_peaks.draw()

    def plot_sdnn_rmssd_comparison(self):
        """Create a bar chart comparing SDNN and RMSSD values"""
        if self.radar_hrv is None or self.ecg_hrv is None:
            messagebox.showwarning("Warning", "Run HRV analysis first")
            return
        
        try:
            import matplotlib.pyplot as plt
            
            # Create a new figure for SDNN/RMSSD comparison
            fig_comparison = plt.figure(figsize=(10, 6))
            
            # SDNN comparison
            ax1 = fig_comparison.add_subplot(1, 2, 1)
            signals = ['Radar', 'ECG']
            sdnn_values = [self.radar_hrv['sdnn'], self.ecg_hrv['sdnn']]
            colors = ['blue', 'red']
            
            bars1 = ax1.bar(signals, sdnn_values, color=colors, alpha=0.7)
            ax1.set_title('SDNN Comparison')
            ax1.set_ylabel('SDNN (ms)')
            ax1.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, value in zip(bars1, sdnn_values):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                        f'{value:.1f}', ha='center', va='bottom')
            
            # RMSSD comparison
            ax2 = fig_comparison.add_subplot(1, 2, 2)
            rmssd_values = [self.radar_hrv['rmssd'], self.ecg_hrv['rmssd']]
            
            bars2 = ax2.bar(signals, rmssd_values, color=colors, alpha=0.7)
            ax2.set_title('RMSSD Comparison')
            ax2.set_ylabel('RMSSD (ms)')
            ax2.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, value in zip(bars2, rmssd_values):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                        f'{value:.1f}', ha='center', va='bottom')
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"DEBUG: Error in plot_sdnn_rmssd_comparison: {e}")
            messagebox.showerror("Error", f"Failed to create SDNN/RMSSD comparison chart: {str(e)}")

    def display_hrv_results(self):
        """Display HRV analysis results in the text area (without pNN50)"""
        try:
            self.peak_results.insert(tk.END, f"\nHRV ANALYSIS RESULTS\n")
            self.peak_results.insert(tk.END, f"=" * 40 + "\n")
            
            # Radar HRV results
            if self.radar_hrv is not None:
                self.peak_results.insert(tk.END, f"RADAR HRV METRICS:\n")
                self.peak_results.insert(tk.END, f"Valid RR intervals: {self.radar_hrv['valid_rr_count']} / {self.radar_hrv['total_peaks']} peaks\n")
                self.peak_results.insert(tk.END, f"Mean RR: {self.radar_hrv['mean_rr']:.1f} ± {self.radar_hrv['sdnn']:.1f} ms\n")
                self.peak_results.insert(tk.END, f"SDNN: {self.radar_hrv['sdnn']:.1f} ms\n")
                self.peak_results.insert(tk.END, f"RMSSD: {self.radar_hrv['rmssd']:.1f} ms\n")
                self.peak_results.insert(tk.END, f"Mean HR: {self.radar_hrv['mean_hr']:.1f} ± {self.radar_hrv['std_hr']:.1f} BPM\n")
                self.peak_results.insert(tk.END, f"RR Range: {self.radar_hrv['min_rr']:.1f} - {self.radar_hrv['max_rr']:.1f} ms\n\n")
            
            # ECG HRV results
            if self.ecg_hrv is not None:
                self.peak_results.insert(tk.END, f"ECG HRV METRICS:\n")
                self.peak_results.insert(tk.END, f"Valid RR intervals: {self.ecg_hrv['valid_rr_count']} / {self.ecg_hrv['total_peaks']} peaks\n")
                self.peak_results.insert(tk.END, f"Mean RR: {self.ecg_hrv['mean_rr']:.1f} ± {self.ecg_hrv['sdnn']:.1f} ms\n")
                self.peak_results.insert(tk.END, f"SDNN: {self.ecg_hrv['sdnn']:.1f} ms\n")
                self.peak_results.insert(tk.END, f"RMSSD: {self.ecg_hrv['rmssd']:.1f} ms\n")
                self.peak_results.insert(tk.END, f"Mean HR: {self.ecg_hrv['mean_hr']:.1f} ± {self.ecg_hrv['std_hr']:.1f} BPM\n")
                self.peak_results.insert(tk.END, f"RR Range: {self.ecg_hrv['min_rr']:.1f} - {self.ecg_hrv['max_rr']:.1f} ms\n\n")
            
            # Comparison metrics
            if self.hrv_comparison is not None:
                self.peak_results.insert(tk.END, f"HRV COMPARISON METRICS:\n")
                self.peak_results.insert(tk.END, f"RR Correlation: {self.hrv_comparison['rr_correlation']:.3f}\n")
                self.peak_results.insert(tk.END, f"Mean RR Difference: {self.hrv_comparison['mean_rr_diff']:.1f} ms\n")
                self.peak_results.insert(tk.END, f"SDNN Difference: {self.hrv_comparison['sdnn_diff']:.1f} ms\n")
                self.peak_results.insert(tk.END, f"RMSSD Difference: {self.hrv_comparison['rmssd_diff']:.1f} ms\n")
                self.peak_results.insert(tk.END, f"Heart Rate Difference: {self.hrv_comparison['hr_diff']:.1f} BPM\n")
                self.peak_results.insert(tk.END, f"SDNN Ratio (Radar/ECG): {self.hrv_comparison['sdnn_ratio']:.2f}\n")
                self.peak_results.insert(tk.END, f"RMSSD Ratio (Radar/ECG): {self.hrv_comparison['rmssd_ratio']:.2f}\n\n")
                
                # Interpretation
                if self.hrv_comparison['rr_correlation'] > 0.7:
                    interpretation = "Strong HRV correlation"
                elif self.hrv_comparison['rr_correlation'] > 0.4:
                    interpretation = "Moderate HRV correlation"
                else:
                    interpretation = "Weak HRV correlation"
                
                self.peak_results.insert(tk.END, f"INTERPRETATION: {interpretation}\n")
                self.peak_results.insert(tk.END, f"=" * 40 + "\n")
        
        except Exception as e:
            print(f"DEBUG: Error in display_hrv_results: {e}")
            self.peak_results.insert(tk.END, f"Error displaying HRV results: {str(e)}\n")

    def display_detailed_sdnn_rmssd_results(self):
        """Display detailed SDNN and RMSSD analysis results"""
        try:
            self.peak_results.insert(tk.END, f"\nDETAILED SDNN & RMSSD ANALYSIS\n")
            self.peak_results.insert(tk.END, f"=" * 45 + "\n")
            
            # Radar detailed analysis
            if self.radar_hrv is not None:
                radar_detailed = self.calculate_detailed_sdnn_rmssd(
                    self.radar_hrv['rr_intervals'], "Radar")
                
                if radar_detailed:
                    self.peak_results.insert(tk.END, f"RADAR SIGNAL:\n")
                    self.peak_results.insert(tk.END, f"  SDNN: {radar_detailed['sdnn']:.2f} ms ({radar_detailed['sdnn_category']})\n")
                    self.peak_results.insert(tk.END, f"  SDNN CV: {radar_detailed['sdnn_cv']:.1f}%\n")
                    self.peak_results.insert(tk.END, f"  RMSSD: {radar_detailed['rmssd']:.2f} ms ({radar_detailed['rmssd_category']})\n")
                    self.peak_results.insert(tk.END, f"  RMSSD %: {radar_detailed['rmssd_percentage']:.1f}%\n")
                    self.peak_results.insert(tk.END, f"  Based on {radar_detailed['rr_count']} RR intervals\n\n")
            
            # ECG detailed analysis
            if self.ecg_hrv is not None:
                ecg_detailed = self.calculate_detailed_sdnn_rmssd(
                    self.ecg_hrv['rr_intervals'], "ECG")
                
                if ecg_detailed:
                    self.peak_results.insert(tk.END, f"ECG SIGNAL:\n")
                    self.peak_results.insert(tk.END, f"  SDNN: {ecg_detailed['sdnn']:.2f} ms ({ecg_detailed['sdnn_category']})\n")
                    self.peak_results.insert(tk.END, f"  SDNN CV: {ecg_detailed['sdnn_cv']:.1f}%\n")
                    self.peak_results.insert(tk.END, f"  RMSSD: {ecg_detailed['rmssd']:.2f} ms ({ecg_detailed['rmssd_category']})\n")
                    self.peak_results.insert(tk.END, f"  RMSSD %: {ecg_detailed['rmssd_percentage']:.1f}%\n")
                    self.peak_results.insert(tk.END, f"  Based on {ecg_detailed['rr_count']} RR intervals\n\n")
            
            # Comparison if both available
            if self.radar_hrv is not None and self.ecg_hrv is not None:
                radar_detailed = self.calculate_detailed_sdnn_rmssd(self.radar_hrv['rr_intervals'], "Radar")
                ecg_detailed = self.calculate_detailed_sdnn_rmssd(self.ecg_hrv['rr_intervals'], "ECG")
                
                if radar_detailed and ecg_detailed:
                    sdnn_agreement = abs(radar_detailed['sdnn'] - ecg_detailed['sdnn'])
                    rmssd_agreement = abs(radar_detailed['rmssd'] - ecg_detailed['rmssd'])
                    
                    sdnn_percent_diff = (sdnn_agreement / ecg_detailed['sdnn']) * 100 if ecg_detailed['sdnn'] > 0 else 0
                    rmssd_percent_diff = (rmssd_agreement / ecg_detailed['rmssd']) * 100 if ecg_detailed['rmssd'] > 0 else 0
                    
                    self.peak_results.insert(tk.END, f"RADAR vs ECG COMPARISON:\n")
                    self.peak_results.insert(tk.END, f"  SDNN Difference: {sdnn_agreement:.2f} ms ({sdnn_percent_diff:.1f}%)\n")
                    self.peak_results.insert(tk.END, f"  RMSSD Difference: {rmssd_agreement:.2f} ms ({rmssd_percent_diff:.1f}%)\n")
                    
                    # Agreement assessment
  
                    '''if sdnn_percent_diff < 10:
                        sdnn_agreement_level = "Excellent"
                    elif sdnn_percent_diff < 20:
                        sdnn_agreement_level = "Good"
                    elif sdnn_percent_diff < 30:
                        sdnn_agreement_level = "Moderate"
                    else:
                        sdnn_agreement_level = "Poor"
                    
                    if rmssd_percent_diff < 10:
                        rmssd_agreement_level = "Excellent"
                    elif rmssd_percent_diff < 20:
                        rmssd_agreement_level = "Good"
                    elif rmssd_percent_diff < 30:
                        rmssd_agreement_level = "Moderate"
                    else:
                        rmssd_agreement_level = "Poor"'''''
           
                    
                    #self.peak_results.insert(tk.END, f"  SDNN Agreement: {sdnn_agreement_level}\n")
                    #self.peak_results.insert(tk.END, f"  RMSSD Agreement: {rmssd_agreement_level}\n")
                    
                    # Category agreement
                    category_match = (radar_detailed['sdnn_category'] == ecg_detailed['sdnn_category'] and 
                                    radar_detailed['rmssd_category'] == ecg_detailed['rmssd_category'])
                    
                    if category_match:
                        self.peak_results.insert(tk.END, f"  Clinical Categories: MATCH\n")
                    else:
                        self.peak_results.insert(tk.END, f"  Clinical Categories: DIFFERENT\n")
            
            self.peak_results.insert(tk.END, f"=" * 45 + "\n")
            
        except Exception as e:
            print(f"DEBUG: Error in display_detailed_sdnn_rmssd_results: {e}")
            self.peak_results.insert(tk.END, f"Error displaying detailed SDNN/RMSSD results: {str(e)}\n")

    def calculate_detailed_sdnn_rmssd(self, rr_intervals, signal_name):
        """Calculate detailed SDNN and RMSSD analysis with interpretation"""
        if len(rr_intervals) < 2:
            return None
        
        # SDNN calculation
        mean_rr = np.mean(rr_intervals)
        sdnn = np.std(rr_intervals, ddof=1)  # Sample standard deviation
        
        # RMSSD calculation
        successive_diffs = np.diff(rr_intervals)
        rmssd = np.sqrt(np.mean(successive_diffs**2))
        
        # Additional SDNN analysis
        sdnn_coefficient_variation = (sdnn / mean_rr) * 100 if mean_rr > 0 else 0
        
        # Additional RMSSD analysis
        rmssd_percentage = (rmssd / mean_rr) * 100 if mean_rr > 0 else 0
        
        # Classification based on clinical ranges
        sdnn_category = "Unknown"
        if sdnn > 50:
            sdnn_category = "Normal"
        elif sdnn > 25:
            sdnn_category = "Reduced"
        else:
            sdnn_category = "Severely Reduced"
        
        rmssd_category = "Unknown"
        if rmssd > 35:
            rmssd_category = "Normal"
        elif rmssd > 15:
            rmssd_category = "Reduced"
        else:
            rmssd_category = "Severely Reduced"
        
        return {
            'signal_name': signal_name,
            'sdnn': sdnn,
            'sdnn_cv': sdnn_coefficient_variation,
            'sdnn_category': sdnn_category,
            'rmssd': rmssd,
            'rmssd_percentage': rmssd_percentage,
            'rmssd_category': rmssd_category,
            'mean_rr': mean_rr,
            'rr_count': len(rr_intervals)
        }

    def export_hrv_data(self):
        """Export HRV analysis data to CSV"""
        if not hasattr(self, 'radar_hrv') or not hasattr(self, 'ecg_hrv'):
            messagebox.showwarning("Warning", "No HRV data to export. Run HRV analysis first.")
            return
        
        file_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            title="Export HRV Analysis Data"
        )
        
        if not file_path:
            return
        
        try:
            import pandas as pd
            
            # Create export data
            export_data = []
            
            # Add radar data
            if self.radar_hrv is not None:
                for i, (time, rr) in enumerate(zip(self.radar_hrv['rr_times'], self.radar_hrv['rr_intervals'])):
                    export_data.append({
                        'Signal_Type': 'Radar',
                        'RR_Index': i+1,
                        'Time_s': time,
                        'RR_Interval_ms': rr
                    })
            
            # Add ECG data
            if self.ecg_hrv is not None:
                for i, (time, rr) in enumerate(zip(self.ecg_hrv['rr_times'], self.ecg_hrv['rr_intervals'])):
                    export_data.append({
                        'Signal_Type': 'ECG',
                        'RR_Index': i+1,
                        'Time_s': time,
                        'RR_Interval_ms': rr
                    })
            
            df = pd.DataFrame(export_data)
            df.to_csv(file_path, index=False)
            
            # Also save summary metrics
            summary_file = file_path.replace('.csv', '_summary.csv')
            
            summary_data = []
            if self.radar_hrv is not None:
                summary_data.append({
                    'Signal': 'Radar',
                    'Mean_RR_ms': self.radar_hrv['mean_rr'],
                    'SDNN_ms': self.radar_hrv['sdnn'],
                    'RMSSD_ms': self.radar_hrv['rmssd'],
                    'Mean_HR_BPM': self.radar_hrv['mean_hr'],
                    'Valid_RR_Count': self.radar_hrv['valid_rr_count']
                })
            
            if self.ecg_hrv is not None:
                summary_data.append({
                    'Signal': 'ECG',
                    'Mean_RR_ms': self.ecg_hrv['mean_rr'],
                    'SDNN_ms': self.ecg_hrv['sdnn'],
                    'RMSSD_ms': self.ecg_hrv['rmssd'],
                    'Mean_HR_BPM': self.ecg_hrv['mean_hr'],
                    'Valid_RR_Count': self.ecg_hrv['valid_rr_count']
                })
            
            df_summary = pd.DataFrame(summary_data)
            df_summary.to_csv(summary_file, index=False)
            
            messagebox.showinfo("Export Complete", f"HRV data exported to:\n{file_path}\n{summary_file}")
            
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to export HRV data: {str(e)}")

    def toggle_edit_mode(self):
        """Toggle peak editing mode on/off - DEBUG VERSION"""
        print("DEBUG: toggle_edit_mode called")
        print(f"DEBUG: Current peak_edit_mode state: {getattr(self, 'peak_edit_mode', 'NOT SET')}")
        
        # Check if radar peaks exist
        if not hasattr(self, 'detected_radar_peaks'):
            print("DEBUG: No detected_radar_peaks attribute")
            messagebox.showwarning("Warning", "Detect radar peaks first")
            return
        
        if self.detected_radar_peaks is None:
            print("DEBUG: detected_radar_peaks is None")
            messagebox.showwarning("Warning", "Detect radar peaks first")
            return
        
        print(f"DEBUG: Found {len(self.detected_radar_peaks)} radar peaks")
        
        # Check if signal is segmented (optional - remove this check if you want to allow editing on full signal)
        if not hasattr(self, 'is_segmented'):
            print("DEBUG: No is_segmented attribute")
        elif not self.is_segmented:
            print("DEBUG: Signal not segmented")
            # Make this a warning instead of blocking
            result = messagebox.askyesno("Continue?", 
                "Signal is not segmented. Continue with full signal editing?")
            if not result:
                return
        
        # Initialize peak_edit_mode if it doesn't exist
        if not hasattr(self, 'peak_edit_mode'):
            print("DEBUG: Initializing peak_edit_mode to False")
            self.peak_edit_mode = False
        
        # Toggle the mode
        self.peak_edit_mode = not self.peak_edit_mode
        print(f"DEBUG: Toggled peak_edit_mode to: {self.peak_edit_mode}")
        
        try:
            if self.peak_edit_mode:
                # Entering edit mode
                print("DEBUG: Entering edit mode")
                self.edit_mode_var.set("Edit Mode: ON")
                
                # Store original peaks for restoration if needed
                self.original_radar_peaks = self.detected_radar_peaks.copy()
                self.original_radar_peak_times = self.radar_peak_times.copy()
                
                # Create editable copies
                self.edited_radar_peaks = self.detected_radar_peaks.copy()
                self.edited_radar_peak_times = self.radar_peak_times.copy()
                
                # Calculate segments
                if hasattr(self, 'peak_detection_timestamps') and self.peak_detection_timestamps is not None:
                    signal_duration = self.peak_detection_timestamps[-1] - self.peak_detection_timestamps[0]
                    self.total_edit_segments = int(np.ceil(signal_duration / self.edit_segment_duration))
                    self.edit_segment_index = 0
                    print(f"DEBUG: Calculated {self.total_edit_segments} segments")
                    
                    self.update_segment_display()
                else:
                    print("DEBUG: No peak_detection_timestamps available")
                    messagebox.showwarning("Warning", "No signal timestamps available")
                    self.peak_edit_mode = False
                    self.edit_mode_var.set("Edit Mode: OFF")
                    return
                
                messagebox.showinfo("Edit Mode", 
                    "Edit mode enabled!\n\n"
                    "Click on comparison plot:\n"
                    "• Click to add peaks\n"
                    "• Click near existing peaks to remove them\n"
                    "• Use Prev/Next buttons to navigate")
                
            else:
                # Exiting edit mode
                print("DEBUG: Exiting edit mode")
                self.edit_mode_var.set("Edit Mode: OFF")
                self.edit_segment_info.set("Segment: - / -")
                messagebox.showinfo("Edit Mode", "Edit mode disabled")
            
            # Refresh the comparison plot
            try:
                self.refresh_comparison_plot()
            except Exception as e:
                print(f"DEBUG: Error refreshing comparison plot: {e}")
                
        except Exception as e:
            print(f"DEBUG: Error in toggle_edit_mode: {e}")
            messagebox.showerror("Error", f"Error toggling edit mode: {str(e)}")
            # Reset to off state
            self.peak_edit_mode = False
            self.edit_mode_var.set("Edit Mode: OFF")
    
    def initialize_edit_mode_variables(self):
        """Initialize edit mode variables if they don't exist"""
        if not hasattr(self, 'peak_edit_mode'):
            self.peak_edit_mode = False
        
        if not hasattr(self, 'edit_segment_start'):
            self.edit_segment_start = 0
        
        if not hasattr(self, 'edit_segment_duration'):
            self.edit_segment_duration = 10
        
        if not hasattr(self, 'edit_segment_index'):
            self.edit_segment_index = 0
        
        if not hasattr(self, 'total_edit_segments'):
            self.total_edit_segments = 0
        
        print("DEBUG: Edit mode variables initialized")

    def previous_edit_segment(self):
        """Move to previous 10-second segment"""
        if not self.peak_edit_mode:
            messagebox.showwarning("Warning", "Enable edit mode first")
            return
        
        if self.edit_segment_index > 0:
            self.edit_segment_index -= 1
            self.update_segment_display()
            self.refresh_comparison_plot()

    def next_edit_segment(self):
        """Move to next 10-second segment"""
        if not self.peak_edit_mode:
            messagebox.showwarning("Warning", "Enable edit mode first")
            return
        
        if self.edit_segment_index < self.total_edit_segments - 1:
            self.edit_segment_index += 1
            self.update_segment_display()
            self.refresh_comparison_plot()

    def update_segment_display(self):
        """Update segment information display"""
        if self.peak_edit_mode:
            start_time = self.peak_detection_timestamps[0] + (self.edit_segment_index * self.edit_segment_duration)
            end_time = min(start_time + self.edit_segment_duration, self.peak_detection_timestamps[-1])
            self.edit_segment_start = start_time
            
            self.edit_segment_info.set(f"Segment: {self.edit_segment_index + 1} / {self.total_edit_segments}")
            
            # Update results with current segment info
            current_text = self.peak_results.get("end-20c", "end")
            if "EDIT MODE:" not in current_text:
                self.peak_results.insert(tk.END, f"\nEDIT MODE: {start_time:.1f}s - {end_time:.1f}s\n")

    def on_plot_click(self, event):
        """Handle mouse clicks on the comparison plot for peak editing"""
        if not self.peak_edit_mode or event.inaxes != self.ax_comparison:
            return
        
        if event.button != 1:  # Only left clicks
            return
        
        click_time = event.xdata
        if click_time is None:
            return
        
        # Check if click is within current edit segment
        segment_start = self.edit_segment_start
        segment_end = segment_start + self.edit_segment_duration
        
        if not (segment_start <= click_time <= segment_end):
            messagebox.showinfo("Edit Info", f"Click within current segment: {segment_start:.1f}s - {segment_end:.1f}s")
            return
        
        # Check if clicking near an existing peak (to remove it)
        tolerance = 0.2  # seconds
        peak_to_remove = None
        
        for i, peak_time in enumerate(self.edited_radar_peak_times):
            if abs(peak_time - click_time) <= tolerance:
                peak_to_remove = i
                break
        
        if peak_to_remove is not None:
            # Remove peak
            self.edited_radar_peak_times = np.delete(self.edited_radar_peak_times, peak_to_remove)
            self.edited_radar_peaks = np.delete(self.edited_radar_peaks, peak_to_remove)
            print(f"DEBUG: Removed peak at {click_time:.2f}s")
        else:
            # Add new peak
            # Find corresponding index in signal
            click_idx = np.argmin(np.abs(self.peak_detection_timestamps - click_time))
            
            # Insert peak at correct position (maintaining time order)
            insert_pos = np.searchsorted(self.edited_radar_peak_times, click_time)
            
            self.edited_radar_peak_times = np.insert(self.edited_radar_peak_times, insert_pos, click_time)
            self.edited_radar_peaks = np.insert(self.edited_radar_peaks, insert_pos, click_idx)
            print(f"DEBUG: Added peak at {click_time:.2f}s")
        
        # Refresh the plot
        self.refresh_comparison_plot()

    def refresh_comparison_plot(self):
        """Refresh the comparison plot with current edit state"""
        if not self.peak_edit_mode:
            # Normal comparison plot
            self.compare_peak_results()
            return
        
        # Edit mode comparison plot - show only current segment
        self.ax_comparison.clear()
        
        # Get current segment bounds
        segment_start = self.edit_segment_start
        segment_end = segment_start + self.edit_segment_duration
        
        # Get data indices for current segment
        radar_start_idx = np.argmin(np.abs(self.peak_detection_timestamps - segment_start))
        radar_end_idx = np.argmin(np.abs(self.peak_detection_timestamps - segment_end))
        
        ecg_start_idx = np.argmin(np.abs(self.ecg_time - segment_start))
        ecg_end_idx = np.argmin(np.abs(self.ecg_time - segment_end))
        
        # Get segment data
        radar_segment_times = self.peak_detection_timestamps[radar_start_idx:radar_end_idx+1]
        radar_segment_signal = self.peak_detection_signal[radar_start_idx:radar_end_idx+1]
        
        ecg_segment_times = self.ecg_time[ecg_start_idx:ecg_end_idx+1]
        ecg_segment_signal = self.ecg_data[ecg_start_idx:ecg_end_idx+1]
        
        # Normalize signals
        radar_norm = (radar_segment_signal - np.mean(radar_segment_signal)) / np.std(radar_segment_signal)
        ecg_norm = (ecg_segment_signal - np.mean(ecg_segment_signal)) / np.std(ecg_segment_signal)
        
        # Plot segment signals
        self.ax_comparison.plot(radar_segment_times, radar_norm, 'b-', 
                            alpha=0.6, linewidth=1, label='Radar (Segment)')
        self.ax_comparison.plot(ecg_segment_times, ecg_norm, 'g-', 
                            alpha=0.6, linewidth=1, label='ECG (Segment)')
        
        # Plot edited peaks within segment
        segment_radar_peaks = self.edited_radar_peak_times[
            (self.edited_radar_peak_times >= segment_start) & 
            (self.edited_radar_peak_times <= segment_end)]
        
        if len(segment_radar_peaks) > 0:
            # Get normalized values for peaks
            radar_peak_values_norm = []
            for peak_time in segment_radar_peaks:
                peak_idx = np.argmin(np.abs(radar_segment_times - peak_time))
                if peak_idx < len(radar_norm):
                    radar_peak_values_norm.append(radar_norm[peak_idx])
                else:
                    radar_peak_values_norm.append(0)
            
            self.ax_comparison.plot(segment_radar_peaks, radar_peak_values_norm, 'ro', 
                                markersize=8, label=f'Edited Radar Peaks ({len(segment_radar_peaks)})')
        
        # Plot ECG peaks in segment
        segment_ecg_peaks = self.ecg_peak_times[
            (self.ecg_peak_times >= segment_start) & 
            (self.ecg_peak_times <= segment_end)]
        
        if len(segment_ecg_peaks) > 0:
            ecg_peak_values_norm = []
            for peak_time in segment_ecg_peaks:
                peak_idx = np.argmin(np.abs(ecg_segment_times - peak_time))
                if peak_idx < len(ecg_norm):
                    ecg_peak_values_norm.append(ecg_norm[peak_idx])
                else:
                    ecg_peak_values_norm.append(0)
            
            self.ax_comparison.plot(segment_ecg_peaks, ecg_peak_values_norm, 'gs', 
                                markersize=8, label=f'ECG Peaks ({len(segment_ecg_peaks)})')
        
        self.ax_comparison.set_xlim(segment_start, segment_end)
        self.ax_comparison.set_title(f'Peak Editing: Segment {self.edit_segment_index + 1}/{self.total_edit_segments} ({segment_start:.1f}s - {segment_end:.1f}s)')
        self.ax_comparison.set_xlabel('Time (s)')
        self.ax_comparison.set_ylabel('Normalized Amplitude')
        self.ax_comparison.legend()
        self.ax_comparison.grid(True, alpha=0.3)
        
        self.canvas_peaks.draw()

    def update_after_editing(self):
        """Update all calculations and plots after peak editing - ENHANCED VERSION with RMSSD"""
        if not self.peak_edit_mode:
            messagebox.showinfo("Info", "Not in edit mode")
            return
        
        if not hasattr(self, 'edited_radar_peaks') or self.edited_radar_peaks is None:
            messagebox.showwarning("Warning", "No edited peaks available")
            return
        
        print("DEBUG: update_after_editing called")
        print(f"DEBUG: Edited peaks count: {len(self.edited_radar_peaks)}")
        
        try:
            # Update the main peak detection results with edited peaks
            self.detected_radar_peaks = self.edited_radar_peaks.copy()
            self.radar_peak_times = self.edited_radar_peak_times.copy()
            
            # Ensure peak values are correctly updated
            if hasattr(self, 'peak_detection_signal') and self.peak_detection_signal is not None:
                # Make sure indices are within bounds
                valid_indices = self.detected_radar_peaks < len(self.peak_detection_signal)
                self.detected_radar_peaks = self.detected_radar_peaks[valid_indices]
                self.radar_peak_times = self.radar_peak_times[valid_indices]
                
                self.radar_peak_values = self.peak_detection_signal[self.detected_radar_peaks]
            else:
                print("DEBUG: No peak_detection_signal available")
                messagebox.showwarning("Warning", "Signal data not available")
                return
            
            print(f"DEBUG: Final radar peaks: {len(self.detected_radar_peaks)}")
            print(f"DEBUG: Final radar peak times: {len(self.radar_peak_times)}")
            
            # Calculate heart rate
            if len(self.radar_peak_times) > 1:
                rr_intervals = np.diff(self.radar_peak_times)
                heart_rate = 60 / np.mean(rr_intervals)
            else:
                heart_rate = 0
            
            # Store algorithm info
            self.radar_algo_info = {
                'heart_rate_bpm': heart_rate,
                'num_peaks': len(self.detected_radar_peaks)
            }
              
            # Update radar peaks plot
            self.update_radar_peaks_plot()
            
            # Update comparison plot if ECG is available
            if hasattr(self, 'ecg_data') and self.ecg_data is not None:
                self.refresh_comparison_plot()
            
            # Update results display
            self.peak_results.insert(tk.END, f"\n{'='*45}\n")
            self.peak_results.insert(tk.END, f"PEAKS UPDATED (EDITED):\n")
            self.peak_results.insert(tk.END, f"Original peaks: {len(self.original_radar_peaks)}\n")
            self.peak_results.insert(tk.END, f"Edited peaks: {len(self.detected_radar_peaks)}\n")
            self.peak_results.insert(tk.END, f"Change: {len(self.detected_radar_peaks) - len(self.original_radar_peaks):+d}\n")
            self.peak_results.insert(tk.END, f"Heart Rate: {heart_rate:.1f} BPM\n")
            self.peak_results.insert(tk.END, f"{'='*45}\n\n")

            messagebox.showinfo("Update Complete",
            f"Peaks updated!\n"
            f"Total radar peaks: {len(self.original_radar_peaks)} → {len(self.detected_radar_peaks)}\n"
            f"HR: {heart_rate:.1f} BPM\n"
            f"Click 'Analyze HRV' to update HRV calculations")
            
        except Exception as e:
            print(f"DEBUG: Error in update_after_editing: {e}")
            import traceback
            traceback.print_exc()
            messagebox.showerror("Error", f"Update failed: {str(e)}")

    def update_radar_peaks_plot(self):
        """Update the radar peaks plot with edited peaks"""
        self.ax_radar_peaks.clear()
        self.ax_radar_peaks.plot(self.peak_detection_timestamps, self.peak_detection_signal, 'b-', 
                                linewidth=0.8, alpha=0.7, label='Radar Signal')
        self.ax_radar_peaks.plot(self.radar_peak_times, self.radar_peak_values, 'ro', markersize=6, 
                                label=f'Edited Peaks ({len(self.detected_radar_peaks)})')
        
        self.ax_radar_peaks.set_title(f'Radar Peak Detection (EDITED) - {len(self.detected_radar_peaks)} peaks, HR: {self.radar_algo_info["heart_rate_bpm"]:.1f} BPM')
        self.ax_radar_peaks.set_xlabel('Time (s)')
        self.ax_radar_peaks.set_ylabel('Amplitude')
        self.ax_radar_peaks.legend()
        self.ax_radar_peaks.grid(True, alpha=0.3)
        
        self.canvas_peaks.draw()
    
    def reset_peak_editing(self):
        """Reset peaks to original detected state"""
        if not hasattr(self, 'original_radar_peaks'):
            messagebox.showinfo("Info", "No original peaks to restore")
            return
        
        result = messagebox.askyesno("Reset Peaks", 
            "Reset all edited peaks to original detection?\nThis will lose all manual edits.")
        
        if result:
            self.detected_radar_peaks = self.original_radar_peaks.copy()
            self.radar_peak_times = self.original_radar_peak_times.copy()
            self.edited_radar_peaks = self.original_radar_peaks.copy()
            self.edited_radar_peak_times = self.original_radar_peak_times.copy()
            
            self.update_after_editing()
            messagebox.showinfo("Reset Complete", "Peaks reset to original detection")

    #GUI
    def create_scrollable_tab(self, master, content_func):
        """Create a scrollable tab with navigation toolbar"""
        frame = tk.Frame(master)
        frame.pack(fill=tk.BOTH, expand=True)
        
        canvas = tk.Canvas(frame)
        scrollbar = tk.Scrollbar(frame, orient=tk.VERTICAL, command=canvas.yview)
        scrollable_frame = tk.Frame(canvas)
        canvas.configure(yscrollcommand=scrollbar.set)
        
        content_func(scrollable_frame)
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        scrollable_frame.bind("<Configure>", 
            lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        def on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        canvas.bind_all("<MouseWheel>", on_mousewheel)

    def create_main_analysis_tab(self, parent):
        """Create main analysis tab with scrollable interface and toolbar"""
        def create_main_content(scrollable_frame):
            # Create figure
            fig = Figure(figsize=(15, 25), dpi=80)
            gs = fig.add_gridspec(7, 1, height_ratios=[1, 1, 2, 1, 1, 1, 1])
            axs = [fig.add_subplot(g) for g in gs]
            
            time = self.signals['timestamps']
            
            axs[0].plot(time, self.signals['I'], label='I', linewidth=1)
            axs[0].plot(time, self.signals['Q'], label='Q', linewidth=1)
            axs[0].set_title('Raw I/Q Signals')
            axs[0].set_xlabel('Time (s)')
            axs[0].set_ylabel('Amplitude')
            axs[0].legend()
            axs[0].grid(True)
            
            axs[1].plot(time, self.signals['phase'], linewidth=1)
            axs[1].set_title('Phase Signal (normalized)')
            axs[1].set_xlabel('Time (s)')
            axs[1].set_ylabel('Phase')
            axs[1].grid(True)
            
            self.plot_imfs_custom(axs[2], time, self.phase_imfs, 'Phase Signal EEMD Decomposition')
            
            axs[3].plot(time, self.signals['breathing'], linewidth=1)
            axs[3].set_title(f'Breathing Signal (0.6-3.0 Hz) - BR: {self.signals["br"]:.1f} breaths/min')
            axs[3].set_xlabel('Time (s)')
            axs[3].set_ylabel('Amplitude')
            axs[3].grid(True)
            
            # Plot heart rate signal
            if 'heartrate' in self.signals:
                axs[4].plot(time, self.signals['heartrate'], linewidth=1)
                axs[4].set_title('Heart Rate Signal (2.0-6.0 Hz)')
                axs[4].set_xlabel('Time (s)')
                axs[4].set_ylabel('Amplitude')
                axs[4].grid(True)
            else:
                axs[4].axis('off')
            
            # Plot heart rate derivative
            if 'heartrate_derivative' in self.signals:
                axs[5].plot(time, self.signals['heartrate_derivative'], 'r-', linewidth=1)
                axs[5].set_title('Heart Rate Signal Derivative (d/dt)')
                axs[5].set_xlabel('Time (s)')
                axs[5].set_ylabel('Rate of Change (V/s)')
                axs[5].grid(True)
            else:
                axs[5].axis('off')
            
            # Enhanced processing summary
            axs[6].text(0.5, 0.5, 
                f'Enhanced Signal Processing Applied:\n'
                f'Breathing Rate: {self.signals["br"]:.1f} breaths/min\n'
                f'Sampling Rate: {self.signals["fs"]:.1f} Hz\n'
                f'Duration: {self.signals["timestamps"][-1]-self.signals["timestamps"][0]:.1f} seconds\n'
                f'Processing: Detrending → Normalization → Enhanced Bandpass Filtering\n'
                f'Heart Rate Range: 2.0-6.0 Hz, Breathing Range: 0.6-3.0 Hz\n'
                f'Moving Average Applied (Window: 50)\n'
                f'Phase signal ready for EEMD decomposition', 
                horizontalalignment='center', 
                verticalalignment='center', 
                transform=axs[6].transAxes,
                bbox=dict(facecolor='wheat', alpha=0.5))
            axs[6].axis('off')
            
            fig.tight_layout()
            
            # Create canvas and toolbar
            canvas = FigureCanvasTkAgg(fig, master=scrollable_frame)
            canvas.draw()
            
            # Add toolbar for zoom, pan, save functionality
            toolbar_frame = ttk.Frame(scrollable_frame)
            toolbar_frame.pack(side=tk.TOP, fill=tk.X)
            
            toolbar = NavigationToolbar2Tk(canvas, toolbar_frame)
            toolbar.update()
            
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        self.create_scrollable_tab(parent, create_main_content)

    # 4. MODIFIED FUNCTION - Complete create_hr_eemd_tab() function
    def create_hr_eemd_tab(self, parent):
        def create_hr_eemd_content(scrollable_frame):
            main_frame = ttk.Frame(scrollable_frame)
            main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            left_panel = ttk.LabelFrame(main_frame, text="EEMD Controls", width=250)
            left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10, expand=False)
            left_panel.pack_propagate(False)
            
            # IMF selection
            imf_frame = ttk.LabelFrame(left_panel, text="Select Heart Rate IMFs")
            imf_frame.pack(fill=tk.X, padx=5, pady=5)
            
            self.hr_imf_vars = []
            for i in range(len(self.heartrate_imfs)):
                var = tk.BooleanVar()
                ttk.Checkbutton(imf_frame, text=f"HR IMF {i+1}", variable=var).pack(anchor=tk.W, padx=5)
                self.hr_imf_vars.append(var)
            
            # Signal source selection
            source_frame = ttk.LabelFrame(left_panel, text="Signal Source Selection")
            source_frame.pack(fill=tk.X, padx=5, pady=5)

            self.use_derivative_var = tk.BooleanVar(value=False)
            ttk.Checkbutton(source_frame, text="Use Heart Rate Derivative", 
                        variable=self.use_derivative_var).pack(anchor=tk.W, padx=5, pady=2)

            ttk.Label(source_frame, text="If unchecked: Uses combined IMFs\nIf checked: Uses HR derivative from Main tab", 
                    font=("Arial", 8), foreground="gray").pack(padx=5, pady=2)

            # Separator
            ttk.Separator(left_panel, orient='horizontal').pack(fill=tk.X, padx=5, pady=5)
            
            # MAV controls
            mav_frame = ttk.LabelFrame(left_panel, text="Moving Average Filter")
            mav_frame.pack(fill=tk.X, padx=5, pady=5)
            
            self.enable_mav_var = tk.BooleanVar(value=True)
            ttk.Checkbutton(mav_frame, text="Apply Moving Average", 
                        variable=self.enable_mav_var).pack(anchor=tk.W, padx=5, pady=2)
            
            mav_window_frame = ttk.Frame(mav_frame)
            mav_window_frame.pack(fill=tk.X, padx=5, pady=2)
            ttk.Label(mav_window_frame, text="Window Size:").pack(side=tk.LEFT)
            self.mav_window_var = tk.IntVar(value=25)
            ttk.Entry(mav_window_frame, textvariable=self.mav_window_var, width=10).pack(side=tk.RIGHT)
            
            # MAV info
            info_label = ttk.Label(mav_frame, text="Recommended: 15-35 samples", 
                                font=("Arial", 8), foreground="gray")
            info_label.pack(padx=5, pady=1)

            # **ADD SEGMENTATION CONTROLS HERE**
            # Segmentation Controls
            seg_frame = ttk.LabelFrame(left_panel, text="Signal Segmentation")
            seg_frame.pack(fill=tk.X, padx=5, pady=5)

            # Time range inputs
            time_range_frame = ttk.Frame(seg_frame)
            time_range_frame.pack(fill=tk.X, padx=5, pady=2)

            ttk.Label(time_range_frame, text="From (sec):").pack(side=tk.LEFT)
            self.seg_start_var = tk.DoubleVar(value=0.0)
            ttk.Entry(time_range_frame, textvariable=self.seg_start_var, width=8).pack(side=tk.LEFT, padx=2)

            ttk.Label(time_range_frame, text="To (sec):").pack(side=tk.LEFT, padx=(10,0))
            self.seg_end_var = tk.DoubleVar(value=30.0)
            ttk.Entry(time_range_frame, textvariable=self.seg_end_var, width=8).pack(side=tk.LEFT, padx=2)

            # Add update button for convenience
            ttk.Button(time_range_frame, text="Set Range", 
                    command=self.update_default_range).pack(side=tk.LEFT, padx=5)

            # Segmentation buttons
            seg_button_frame = ttk.Frame(seg_frame)
            seg_button_frame.pack(fill=tk.X, padx=5, pady=2)

            ttk.Button(seg_button_frame, text="Preview Segment", 
                    command=self.preview_segment).pack(fill=tk.X, pady=1)
            ttk.Button(seg_button_frame, text="Apply Segment", 
                    command=self.apply_segment).pack(fill=tk.X, pady=1)
            ttk.Button(seg_button_frame, text="Reset to Full Signal", 
                    command=self.reset_segment).pack(fill=tk.X, pady=1)

            # Segment info
            self.seg_info_var = tk.StringVar(value="Full signal")
            ttk.Label(seg_frame, textvariable=self.seg_info_var, 
                    font=("Arial", 8), foreground="blue").pack(padx=5, pady=2)
            
            # Button frame
            button_frame = ttk.Frame(left_panel)
            button_frame.pack(fill=tk.X, padx=5, pady=5)

            ttk.Button(button_frame, text="Combine IMFs", command=self.combine_hr_imfs).pack(fill=tk.X, padx=5, pady=2)
            ttk.Button(button_frame, text="Update FFT Analysis", command=self.plot_imf_fft_analysis).pack(fill=tk.X, padx=5, pady=2)
            ttk.Button(button_frame, text="Show Frequency Table", command=self.show_frequency_table_popup).pack(fill=tk.X, padx=5, pady=2)

            # Separator
            ttk.Separator(left_panel, orient='horizontal').pack(fill=tk.X, padx=5, pady=5)

            # ECG Comparison frame  
            ecg_frame = ttk.LabelFrame(left_panel, text="ECG Comparison")
            ecg_frame.pack(fill=tk.X, padx=5, pady=5)

            ttk.Button(ecg_frame, text="Load ECG Data", command=self.load_ecg_data).pack(fill=tk.X, padx=5, pady=2)
            ttk.Button(ecg_frame, text="Compare with ECG", command=self.compare_imfs_ecg).pack(fill=tk.X, padx=5, pady=2)
            
            # Results
            results_frame = ttk.LabelFrame(left_panel, text="EEMD Results")
            results_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
            
            self.eemd_results = tk.Text(results_frame, width=40, height=15)
            self.eemd_results.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
            
            right_panel = ttk.Frame(main_frame)
            right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            # Plots
            self.fig_eemd = Figure(figsize=(9, 12))
            self.ax_imfs = self.fig_eemd.add_subplot(4, 1, 1)
            self.ax_combined = self.fig_eemd.add_subplot(4, 1, 2)
            self.ax_imf_fft = self.fig_eemd.add_subplot(4, 1, 3)
            self.ax_imf_compare = self.fig_eemd.add_subplot(4, 1, 4)
            
            # Plot IMFs initially
            self.plot_imfs_custom(self.ax_imfs, self.signals['timestamps'], self.heartrate_imfs, 'Heart Rate EEMD')
            
            # Add toolbar for EEMD analysis
            eemd_toolbar_frame = ttk.Frame(right_panel)
            eemd_toolbar_frame.pack(side=tk.TOP, fill=tk.X)
            
            canvas = FigureCanvasTkAgg(self.fig_eemd, right_panel)
            toolbar = NavigationToolbar2Tk(canvas, eemd_toolbar_frame)
            toolbar.update()
            
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            self.canvas_eemd = canvas
            
            # Plot initial FFT analysis AFTER canvas is created
            self.plot_imf_fft_analysis()
        
        self.create_scrollable_tab(parent, create_hr_eemd_content)

    def create_peak_detection_tab(self, parent):
        def create_peak_detection_content(scrollable_frame):
            main_frame = ttk.Frame(scrollable_frame)
            main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            # Left panel for controls
            left_panel = ttk.LabelFrame(main_frame, text="Peak Detection Controls", width=300)
            left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10, expand=False)
            left_panel.pack_propagate(False)
            
            # Signal source status
            status_frame = ttk.LabelFrame(left_panel, text="Signal Status")
            status_frame.pack(fill=tk.X, padx=5, pady=5)
            
            self.peak_signal_status = tk.StringVar(value="No signal loaded")
            ttk.Label(status_frame, textvariable=self.peak_signal_status, 
                    font=("Arial", 9), foreground="blue").pack(padx=5, pady=3)
            
            ttk.Button(status_frame, text="Load Signal from EEMD", 
                    command=self.load_signal_for_peak_detection).pack(fill=tk.X, padx=5, pady=2)
            
            # Algorithm parameters
            params_frame = ttk.LabelFrame(left_panel, text="Detection Parameters")
            params_frame.pack(fill=tk.X, padx=5, pady=5)
            
            # Sampling rate
            fs_frame = ttk.Frame(params_frame)
            fs_frame.pack(fill=tk.X, padx=5, pady=2)
            ttk.Label(fs_frame, text="Sampling Rate (Hz):").pack(side=tk.LEFT)
            self.peak_fs_var = tk.IntVar(value=250)
            ttk.Entry(fs_frame, textvariable=self.peak_fs_var, width=8).pack(side=tk.RIGHT)
            
            # Detection buttons
            detect_frame = ttk.LabelFrame(left_panel, text="Peak Detection")
            detect_frame.pack(fill=tk.X, padx=5, pady=5)
            
            ttk.Button(detect_frame, text="Detect Radar Peaks", 
                    command=self.detect_radar_peaks).pack(fill=tk.X, padx=5, pady=2)
            ttk.Button(detect_frame, text="Detect ECG Peaks", 
                    command=self.detect_ecg_peaks).pack(fill=tk.X, padx=5, pady=2)
            ttk.Button(detect_frame, text="Compare Results", 
                    command=self.compare_peak_results).pack(fill=tk.X, padx=5, pady=2)
            
            # Peak Editing Controls
            edit_frame = ttk.LabelFrame(left_panel, text="Radar Peak Editing")
            edit_frame.pack(fill=tk.X, padx=5, pady=5)
            
            # Edit mode toggle button
            self.edit_mode_var = tk.StringVar(value="Edit Mode: OFF")
            self.edit_mode_button = ttk.Button(edit_frame, textvariable=self.edit_mode_var, 
                                            command=self.toggle_edit_mode)
            self.edit_mode_button.pack(fill=tk.X, padx=5, pady=2)
            
            # Segment navigation buttons
            nav_frame = ttk.Frame(edit_frame)
            nav_frame.pack(fill=tk.X, padx=5, pady=2)
            
            ttk.Button(nav_frame, text="◀ Prev 10s", 
                    command=self.previous_edit_segment).pack(side=tk.LEFT, padx=2)
            ttk.Button(nav_frame, text="Next 10s ▶", 
                    command=self.next_edit_segment).pack(side=tk.RIGHT, padx=2)
            
            # Segment information display
            self.edit_segment_info = tk.StringVar(value="Segment: - / -")
            ttk.Label(edit_frame, textvariable=self.edit_segment_info, 
                    font=("Arial", 8), foreground="blue").pack(padx=5, pady=2)
            
            # Edit instructions text
            edit_instructions = tk.Text(edit_frame, height=4, width=35, wrap=tk.WORD)
            edit_instructions.pack(padx=5, pady=2)
            edit_instructions.insert(tk.END, "Edit Instructions:\n• Enable Edit Mode first\n• Click on comparison plot to add peaks\n• Click near existing peaks to remove them\n• Use Prev/Next for 10s navigation")
            edit_instructions.config(state=tk.DISABLED, font=("Arial", 8))
            
            # Edit control buttons
            edit_buttons_frame = ttk.Frame(edit_frame)
            edit_buttons_frame.pack(fill=tk.X, padx=5, pady=2)
            
            ttk.Button(edit_buttons_frame, text="Update Results", 
                    command=self.update_after_editing).pack(fill=tk.X, pady=1)
            ttk.Button(edit_buttons_frame, text="Reset to Original", 
                    command=self.reset_peak_editing).pack(fill=tk.X, pady=1)
            ttk.Button(edit_frame, text="Reset to Original", 
                    command=self.reset_peak_editing).pack(fill=tk.X, padx=5, pady=2)
            
            # HRV Analysis Section
            hrv_frame = ttk.LabelFrame(left_panel, text="HRV Analysis")
            hrv_frame.pack(fill=tk.X, padx=5, pady=5)
            
            ttk.Button(hrv_frame, text="Analyze HRV", 
                    command=self.analyze_hrv_from_peaks).pack(fill=tk.X, padx=5, pady=2)
            ttk.Button(hrv_frame, text="Export HRV Data", 
                    command=self.export_hrv_data).pack(fill=tk.X, padx=5, pady=2)
            
            # Export buttons
            export_frame = ttk.LabelFrame(left_panel, text="Export Results")
            export_frame.pack(fill=tk.X, padx=5, pady=5)
            
            ttk.Button(export_frame, text="Export Peak Data", 
                    command=self.export_peak_data).pack(fill=tk.X, padx=5, pady=2)
            ttk.Button(export_frame, text="Save Peak Plots", 
                    command=self.save_peak_plots).pack(fill=tk.X, padx=5, pady=2)
            
            # Results text area
            results_frame = ttk.LabelFrame(left_panel, text="Analysis Results")
            results_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
            
            self.peak_results = tk.Text(results_frame, width=35, height=20, wrap=tk.WORD)
            peak_scrollbar = ttk.Scrollbar(results_frame, command=self.peak_results.yview)
            self.peak_results.config(yscrollcommand=peak_scrollbar.set)
            
            self.peak_results.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            peak_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            
            # Right panel for plots
            right_panel = ttk.Frame(main_frame)
            right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            # Create figure with 5 subplots (including HRV plot)
            self.fig_peaks = Figure(figsize=(9, 15))
            
            # Create subplots
            self.ax_signal = self.fig_peaks.add_subplot(5, 1, 1)         # Raw signal
            self.ax_radar_peaks = self.fig_peaks.add_subplot(5, 1, 2)    # Radar peaks
            self.ax_ecg_peaks = self.fig_peaks.add_subplot(5, 1, 3)      # ECG peaks  
            self.ax_comparison = self.fig_peaks.add_subplot(5, 1, 4)     # Comparison (EDITABLE)
            self.ax_hrv = self.fig_peaks.add_subplot(5, 1, 5)            # HRV Analysis
            
            # Add toolbar for navigation and zoom
            peaks_toolbar_frame = ttk.Frame(right_panel)
            peaks_toolbar_frame.pack(side=tk.TOP, fill=tk.X)
            
            canvas = FigureCanvasTkAgg(self.fig_peaks, right_panel)
            toolbar = NavigationToolbar2Tk(canvas, peaks_toolbar_frame)
            toolbar.update()
            
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            self.canvas_peaks = canvas
            
            # Connect mouse click events for peak editing
            self.canvas_peaks.mpl_connect('button_press_event', self.on_plot_click)
            
            # Initialize all plots
            self.initialize_peak_plots()
            
        # Create the scrollable tab
        self.create_scrollable_tab(parent, create_peak_detection_content)

    def create_hrv_feature_extraction_tab(self, parent):
        """Create HRV Feature Extraction tab with same layout as EEMD and Peak Detection tabs"""
        
        def create_hrv_feature_content(scrollable_frame):
            main_frame = ttk.Frame(scrollable_frame)
            main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            # Left panel for controls (same width as other tabs)
            left_panel = ttk.LabelFrame(main_frame, text="HRV Feature Extraction Controls", width=350)
            left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10, expand=False)
            left_panel.pack_propagate(False)
            
            # Feature Information
            info_frame = ttk.LabelFrame(left_panel, text="Feature Information")
            info_frame.pack(fill=tk.X, padx=5, pady=5)

            info_label = ttk.Label(info_frame, 
                    text="Extract 19 HRV features:\n"
                         "• 2 PSD (LF Power, HF Power)\n"
                         "• 6 Bispectral (P1, P2, H1-H4)\n"
                         "• 4 Non-linear (SD1, SD2, SD1/SD2, SamEn)\n"
                         "• 7 Mei Paper (HR, RRmean, RRstd, HankDist_RRI,\n"
                         "  HankDist_dRR, CCM, AFE)\n"
                         "\nBased on Mei et al. AF Detection paper", 
                    justify=tk.LEFT, foreground="darkblue", font=("Arial", 8))
            info_label.pack(padx=5, pady=5)
            
            # Data Source Selection
            source_frame = ttk.LabelFrame(left_panel, text="HRV Data Source")
            source_frame.pack(fill=tk.X, padx=5, pady=5)
            
            # Status display
            self.hrv_source_status = tk.StringVar(value="No peaks detected yet")
            ttk.Label(source_frame, textvariable=self.hrv_source_status, 
                    font=("Arial", 8), foreground="blue").pack(padx=5, pady=2)
            
            ttk.Button(source_frame, text="Load from Peak Detection Tab", 
                    command=self.load_peaks_for_hrv_features).pack(fill=tk.X, padx=5, pady=2)
            
            # Feature Extraction Controls
            extract_frame = ttk.LabelFrame(left_panel, text="Feature Extraction")
            extract_frame.pack(fill=tk.X, padx=5, pady=5)
            
            ttk.Button(extract_frame, text="Extract ECG HRV Features", 
                    command=lambda: self.extract_hrv_features_for_signal("ECG")).pack(fill=tk.X, padx=5, pady=2)
            ttk.Button(extract_frame, text="Extract Radar HRV Features", 
                    command=lambda: self.extract_hrv_features_for_signal("Radar")).pack(fill=tk.X, padx=5, pady=2)
            ttk.Button(extract_frame, text="Extract Both & Compare", 
                    command=self.extract_both_hrv_features).pack(fill=tk.X, padx=5, pady=2)
            
            # Results Display Controls
            results_control_frame = ttk.LabelFrame(left_panel, text="Results Display")
            results_control_frame.pack(fill=tk.X, padx=5, pady=5)
            
            ttk.Button(results_control_frame, text="Show Feature Table", 
                    command=self.show_hrv_features_table).pack(fill=tk.X, padx=5, pady=2)
            ttk.Button(results_control_frame, text="Copy Features to Clipboard", 
                    command=self.copy_hrv_features_to_clipboard).pack(fill=tk.X, padx=5, pady=2)
            ttk.Button(results_control_frame, text="Clear Results", 
                    command=self.clear_hrv_feature_results).pack(fill=tk.X, padx=5, pady=2)
            
            # Results Text Area (similar to EEMD tab)
            results_frame = ttk.LabelFrame(left_panel, text="HRV Feature Results")
            results_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
            
            # Text widget with scrollbar
            text_scroll_frame = ttk.Frame(results_frame)
            text_scroll_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
            
            self.hrv_features_text = tk.Text(text_scroll_frame, width=40, height=20, 
                                        font=("Consolas", 9))
            hrv_scrollbar = ttk.Scrollbar(text_scroll_frame, orient="vertical", 
                                        command=self.hrv_features_text.yview)
            self.hrv_features_text.configure(yscrollcommand=hrv_scrollbar.set)
            
            self.hrv_features_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            hrv_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            
            # Right panel for plots - simplified 3x2 layout
            right_panel = ttk.Frame(main_frame)
            right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            # Create figure with clean 3x2 layout
            self.fig_hrv_features = Figure(figsize=(9, 12))
            gs = self.fig_hrv_features.add_gridspec(3, 2, height_ratios=[1, 1, 1], 
                                                width_ratios=[1, 1], hspace=0.35, wspace=0.25)
            
            # Row 1: PSD plots (ECG, Radar)
            self.ax_ecg_psd = self.fig_hrv_features.add_subplot(gs[0, 0])
            self.ax_radar_psd = self.fig_hrv_features.add_subplot(gs[0, 1])
            
            # Row 2: Bispectrum plots (ECG, Radar)
            self.ax_ecg_bispectrum = self.fig_hrv_features.add_subplot(gs[1, 0])
            self.ax_radar_bispectrum = self.fig_hrv_features.add_subplot(gs[1, 1])
            
            # Row 3: Poincaré plots (ECG, Radar)
            self.ax_ecg_poincare = self.fig_hrv_features.add_subplot(gs[2, 0])
            self.ax_radar_poincare = self.fig_hrv_features.add_subplot(gs[2, 1])
            
            # Initialize empty plots
            self.initialize_hrv_feature_plots()
            
            # Canvas and toolbar (same pattern as other tabs)
            hrv_toolbar_frame = ttk.Frame(right_panel)
            hrv_toolbar_frame.pack(side=tk.TOP, fill=tk.X)
            
            self.canvas_hrv_features = FigureCanvasTkAgg(self.fig_hrv_features, right_panel)
            toolbar = NavigationToolbar2Tk(self.canvas_hrv_features, hrv_toolbar_frame)
            toolbar.update()
            
            self.canvas_hrv_features.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Create the scrollable tab using the same pattern
        self.create_scrollable_tab(parent, create_hrv_feature_content)

    def initialize_hrv_feature_plots(self):
        """Initialize empty HRV feature plots - simplified 3x2 layout"""
        plots_info = [
            (self.ax_ecg_psd, 'ECG Power Spectral Density\n(Extract ECG features first)'),
            (self.ax_radar_psd, 'Radar Power Spectral Density\n(Extract Radar features first)'),
            (self.ax_ecg_bispectrum, 'ECG Bispectrum |B(f1,f2)|\n(Extract ECG features first)'),
            (self.ax_radar_bispectrum, 'Radar Bispectrum |B(f1,f2)|\n(Extract Radar features first)'),
            (self.ax_ecg_poincare, 'ECG Poincaré Plot\n(Extract ECG features first)'),
            (self.ax_radar_poincare, 'Radar Poincaré Plot\n(Extract Radar features first)')
        ]
        
        for ax, text in plots_info:
            ax.text(0.5, 0.5, text, ha='center', va='center', 
                transform=ax.transAxes, fontsize=10,
                bbox=dict(boxstyle='round', alpha=0.8, facecolor='lightblue'))
            ax.set_title(text.split('\n')[0])
        
        self.fig_hrv_features.tight_layout()
        if hasattr(self, 'canvas_hrv_features'):
            self.canvas_hrv_features.draw()

    def load_peaks_for_hrv_features(self):
        """Load peak detection results for HRV feature extraction"""
        try:
            # Check if peak detection has been run
            if not hasattr(self, 'radar_peak_times') or self.radar_peak_times is None:
                messagebox.showwarning("Warning", 
                                    "No radar peaks found. Please run radar peak detection first.")
                return
                
            if not hasattr(self, 'ecg_peak_times') or self.ecg_peak_times is None:
                messagebox.showwarning("Warning", 
                                    "No ECG peaks found. Please run ECG peak detection first.")
                return
            
            # Store the peak times for HRV analysis
            self.hrv_radar_peaks = self.radar_peak_times.copy()
            self.hrv_ecg_peaks = self.ecg_peak_times.copy()
            
            # Update status
            radar_count = len(self.hrv_radar_peaks) if self.hrv_radar_peaks is not None else 0
            ecg_count = len(self.hrv_ecg_peaks) if self.hrv_ecg_peaks is not None else 0
            
            status_text = f"Loaded: {radar_count} Radar peaks, {ecg_count} ECG peaks"
            self.hrv_source_status.set(status_text)
            
            # Display in results
            self.hrv_features_text.delete(1.0, tk.END)
            self.hrv_features_text.insert(tk.END, f"HRV FEATURE EXTRACTION DATA LOADED\n")
            self.hrv_features_text.insert(tk.END, f"=" * 50 + "\n\n")
            self.hrv_features_text.insert(tk.END, f"Radar Peaks: {radar_count}\n")
            self.hrv_features_text.insert(tk.END, f"ECG Peaks: {ecg_count}\n\n")
            self.hrv_features_text.insert(tk.END, f"Ready for HRV feature extraction.\n")
            self.hrv_features_text.insert(tk.END, f"Click 'Extract ECG/Radar HRV Features' to continue.\n\n")
            
            messagebox.showinfo("Success", "Peak data loaded successfully for HRV feature extraction!")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load peak data: {str(e)}")

    def extract_hrv_features_for_signal(self, signal_type):
        """Extract HRV features for specified signal type (ECG or Radar) with enhanced debugging"""
        try:
            # Check if peaks are loaded
            if not hasattr(self, 'hrv_radar_peaks') or not hasattr(self, 'hrv_ecg_peaks'):
                messagebox.showwarning("Warning", "Please load peaks from Peak Detection tab first.")
                return
            
            if signal_type == "ECG":
                peak_times = self.hrv_ecg_peaks
                if peak_times is None or len(peak_times) < 2:
                    messagebox.showwarning("Warning", "Insufficient ECG peaks for HRV analysis.")
                    return
            else:  # Radar
                peak_times = self.hrv_radar_peaks
                if peak_times is None or len(peak_times) < 2:
                    messagebox.showwarning("Warning", "Insufficient Radar peaks for HRV analysis.")
                    return
            
            print(f"\n🔬 Processing {signal_type} peaks for HRV analysis...")
            print(f"   Total peaks: {len(peak_times)}")
            
            # Calculate RR intervals from peak times
            rr_intervals = []
            for i in range(len(peak_times) - 1):
                rr_interval = (peak_times[i+1] - peak_times[i]) * 1000  # Convert to milliseconds
                rr_intervals.append(rr_interval)
            
            rr_intervals = np.array(rr_intervals)
            print(f"   Raw RR intervals: {len(rr_intervals)}")
            print(f"   RR range: {np.min(rr_intervals):.1f} - {np.max(rr_intervals):.1f} ms")
            
            # Validate and filter RR intervals
            # Remove physiologically impossible RR intervals (outside 300-2000 ms range)
            valid_mask = (rr_intervals >= 300) & (rr_intervals <= 2000)
            valid_rr = rr_intervals[valid_mask]
            
            if len(valid_rr) < 10:
                messagebox.showwarning("Warning", 
                                    f"Insufficient valid RR intervals for {signal_type} HRV analysis. "
                                    f"Found {len(valid_rr)} valid intervals (need at least 10).")
                return
            
            print(f"   Valid RR intervals: {len(valid_rr)} (removed {len(rr_intervals) - len(valid_rr)} outliers)")
            print(f"   Valid RR stats: mean={np.mean(valid_rr):.1f}ms, std={np.std(valid_rr):.1f}ms")
            
            # Extract features
            print(f"\n🔬 Extracting HRV features for {signal_type}...")
            features = self.extract_all_hrv_features_from_rr(valid_rr)
            
            if features is None:
                messagebox.showerror("Error", f"Failed to extract {signal_type} HRV features")
                return
            
            # Store results
            if signal_type == "ECG":
                self.ecg_hrv_features = features
                self.ecg_rr_intervals = valid_rr
            else:
                self.radar_hrv_features = features  
                self.radar_rr_intervals = valid_rr
            
            # Debug axes availability
            self.debug_hrv_axes()
            # Update plots and results display
            print(f"\n📊 Updating plots...")
            self.update_hrv_feature_plots(signal_type, valid_rr, features)
            self.display_hrv_feature_results(signal_type, features, valid_rr)
            
            messagebox.showinfo("Success", f"{signal_type} HRV features extracted successfully!")
            print(f"\n✅ {signal_type} HRV EXTRACTION COMPLETE")
            
        except Exception as e:
            error_msg = f"Failed to extract {signal_type} HRV features: {str(e)}"
            print(f"❌ ERROR: {error_msg}")
            import traceback
            traceback.print_exc()
            messagebox.showerror("Error", error_msg)

    def extract_both_hrv_features(self):
        """Extract HRV features for both ECG and Radar, then compare"""
        try:
            # Extract both
            self.extract_hrv_features_for_signal("ECG")
            self.extract_hrv_features_for_signal("Radar")
            
            # Check if both extractions were successful
            if hasattr(self, 'ecg_hrv_features') and hasattr(self, 'radar_hrv_features'):
                self.create_feature_comparison_plots()
                self.display_feature_comparison_results()
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to extract and compare features: {str(e)}")

    def extract_all_hrv_features_from_rr(self, rr_intervals):
        """Enhanced HRV feature extraction including Mei et al. paper features"""
        try:
            print(f"🔍 Extracting ENHANCED HRV features from {len(rr_intervals)} RR intervals")
            print(f"   RR range: {np.min(rr_intervals):.1f} - {np.max(rr_intervals):.1f} ms")
            print(f"   RR mean: {np.mean(rr_intervals):.1f} ± {np.std(rr_intervals):.1f} ms")
            
            # 1. ORIGINAL SPECTRAL FEATURES (2) - Use your existing method
            print("   Computing spectral features using AR model...")
            freqs, psd, lf_power, hf_power = self.calculate_psd_features(rr_intervals)
            
            # 2. ORIGINAL BISPECTRAL FEATURES (6) - Use your existing method
            print("   Computing bispectral features using third-order cumulants...")
            bispectral_features = self.calculate_bispectral_features(rr_intervals)
            
            # 3. ORIGINAL NON-LINEAR FEATURES (4) - Use your CORRECT function names
            print("   Computing Poincaré features...")
            poincare_features = self.calculate_poincare_features_hrv(rr_intervals)
            
            print("   Computing Sample Entropy...")
            sample_entropy = self.calculate_sample_entropy(rr_intervals)
            
            # 4. NEW MEI ET AL. PAPER FEATURES (7)
            print(f"\n📊 Extracting Mei et al. paper features...")
            mei_features = self.extract_hrv_features_mei_paper(rr_intervals)
            
            if mei_features is None:
                print("⚠ Failed to extract Mei paper features, using zeros")
                mei_features = {
                    'HR': 0, 'RRmean': 0, 'RRstd': 0,
                    'HankDist_RRI': 0, 'HankDist_dRR': 0,
                    'CCM': 0, 'AFE': 0
                }
            
            # Combine all features (19 total)
            features = {
                # Original spectral features (2)
                'LF_power': lf_power,
                'HF_power': hf_power,
                
                # Original bispectral features (6)
                'P1': bispectral_features['P1'],
                'P2': bispectral_features['P2'],
                'H1': bispectral_features['H1'],
                'H2': bispectral_features['H2'],
                'H3': bispectral_features['H3'],
                'H4': bispectral_features['H4'],
                
                # Original non-linear features (4)
                'SD1': poincare_features['SD1'],
                'SD2': poincare_features['SD2'],
                'SD1_SD2_ratio': poincare_features['SD1_SD2_ratio'],
                'SamEn': sample_entropy,
                
                # New Mei paper features (7)
                'HR': mei_features['HR'],
                'RRmean': mei_features['RRmean'],
                'RRstd': mei_features['RRstd'],
                'HankDist_RRI': mei_features['HankDist_RRI'],
                'HankDist_dRR': mei_features['HankDist_dRR'],
                'CCM': mei_features['CCM'],
                'AFE': mei_features['AFE']
            }
            
            # CRITICAL: Clean features to prevent NaN/inf values
            features = self._validate_and_clean_features(features)
            
            # IMPORTANT: Store frequencies and PSD for plotting
            features['frequencies'] = freqs
            features['psd'] = psd
            
            print(f"✅ Total features extracted: 19 (12 original + 7 Mei paper)")
            print(f"✅ Plot data stored: {len(freqs)} frequency points, {len(psd)} PSD values")
            
            return features
            
        except Exception as e:
            print(f"❌ Error extracting enhanced HRV features: {e}")
            import traceback
            traceback.print_exc()
            return None

    def extract_hrv_features_mei_paper(self, rr_intervals, sampling_freq=250):
        """
        Extract Mei et al. paper HRV features (7 features total)
        Based on: Detection_Automatic_Atrial_Fibrillation_Detection_Based_on_Heart_Rate_Variability_and_Spectral_Features.pdf
        """
        if len(rr_intervals) < 4:
            print("⚠ Need at least 4 RR intervals for Mei paper features")
            return None
        
        try:
            print(f"🔍 Extracting Mei et al. HRV features from {len(rr_intervals)} RR intervals")
            
            # Convert RR intervals from ms to seconds for calculations
            rr_sec = np.array(rr_intervals) / 1000.0
            N = len(rr_intervals) + 1  # Number of R peaks = RR intervals + 1
            
            # 1. TIME-DOMAIN FEATURES
            # HR (Heart Rate) = 60 * N * f / L (beats/min)
            # Assuming total length L based on sum of RR intervals
            total_length_sec = np.sum(rr_sec)
            HR = 60.0 * N / total_length_sec
            
            # RRmean = average RRI (in ms)
            RRmean = np.mean(rr_intervals)
            
            # RRstd = standard deviation RRI (in ms) 
            RRstd = np.std(rr_intervals, ddof=1)
            
            print(f"   ✓ Time-domain: HR={HR:.1f} bpm, RRmean={RRmean:.1f}ms, RRstd={RRstd:.1f}ms")
            
            # 2. HANKEL MATRIX BASED FEATURES
            # Calculate HankDist for both RRI and dRR
            HankDist_RRI = self._calculate_hankel_distance(rr_intervals)
            
            # Calculate dRR (first order difference)
            dRR = np.diff(rr_intervals)  # dRRi = RRi+1 - RRi
            HankDist_dRR = self._calculate_hankel_distance(dRR) if len(dRR) >= 3 else 0.0
            
            print(f"   ✓ Hankel features: HankDist_RRI={HankDist_RRI:.4f}, HankDist_dRR={HankDist_dRR:.4f}")
            
            # 3. COMPLEX CORRELATION MEASURE (CCM)
            CCM = self._calculate_ccm(rr_intervals)
            print(f"   ✓ CCM: {CCM:.4f}")
            
            # 4. AFE (ATRIAL FIBRILLATION EVIDENCE)
            AFE = self._calculate_afe(dRR) if len(dRR) >= 2 else 0.0
            print(f"   ✓ AFE: {AFE:.4f}")
            
            # Return feature dictionary
            features = {
                # Time-domain features
                'HR': HR,
                'RRmean': RRmean, 
                'RRstd': RRstd,
                
                # Hankel-matrix based features
                'HankDist_RRI': HankDist_RRI,
                'HankDist_dRR': HankDist_dRR,
                
                # Complex correlation measure
                'CCM': CCM,
                
                # Atrial fibrillation evidence
                'AFE': AFE
            }
            
            print(f"✅ Mei et al. HRV features extracted successfully!")
            return features
            
        except Exception as e:
            print(f"❌ Error extracting Mei et al. features: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _calculate_hankel_distance(self, time_series):
        try:
            series = np.array(time_series)
            n = len(series)
            
            if n < 3:
                return 0.0
                
            # Create Hankel matrix (equation 2 in paper)
            # For computational efficiency, use smaller matrix if series is long
            matrix_size = min(n, 32)  # Limit size for computational efficiency
            if n > matrix_size:
                # Take first matrix_size elements
                series = series[:matrix_size]
                n = matrix_size
                
            # Build Hankel matrix H (circular Hankel as in equation 2)
            H = np.zeros((n, n), dtype=complex)
            for i in range(n):
                for j in range(n):
                    H[i, j] = series[(i + j) % n]
            
            # Diagonalize Hankel matrix: H = U* D U (equation 3)
            eigenvalues, eigenvectors = np.linalg.eigh(H)
            
            # Perform DFT on eigenvectors (as described in paper)
            U_fft = np.fft.fft(eigenvectors, axis=0)
            
            # Find non-zero entries and calculate HankDist
            hankel_dist = 0.0
            n_half = n // 2
            
            for i in range(n_half):
                # Find row indices of significant entries in each column
                col = U_fft[:, i]
                significant_indices = np.where(np.abs(col) > 0.1 * np.max(np.abs(col)))[0]
                
                if len(significant_indices) > 0:
                    # Use the first significant index as Ii
                    Ii = significant_indices[0]
                    hankel_dist += abs(Ii - i)
            
            # Normalize according to equation (4)
            if n_half > 0:
                hankel_dist = (4.0 / n_half) * hankel_dist
            
            return hankel_dist
            
        except Exception as e:
            print(f"Warning: Hankel distance calculation failed: {e}")
            return 0.0

    def _calculate_ccm(self, rr_intervals):
        """
        Calculate Complex Correlation Measure (CCM)
        Based on equations (6) and (7) in Mei et al. paper
        """
        try:
            rr = np.array(rr_intervals)
            
            if len(rr) < 4:  # Need at least 4 RR intervals for 3 consecutive triplets
                return 0.0
            
            # Calculate SD1 and SD2 for normalization (needed for equation 7)
            # Standard Poincaré plot descriptors
            rr_n = rr[:-1]
            rr_n1 = rr[1:]
            
            # SD1: short-term variability
            diff_rr = rr_n1 - rr_n
            SD1 = np.std(diff_rr) / np.sqrt(2)
            
            # SD2: long-term variability  
            sum_rr = rr_n1 + rr_n
            SD2 = np.std(sum_rr) / np.sqrt(2)
            
            if SD1 == 0 or SD2 == 0:
                return 0.0
            
            # Calculate oriented areas A(i) using determinant (equation 6)
            total_area = 0.0
            K = len(rr) - 3  # Number of consecutive triplets
            
            for i in range(K):
                # Create matrix for determinant calculation
                # A(i) = det([[RRi, RRi+1, 1], [RRi+1, RRi+2, 1], [RRi+2, RRi+3, 1]])
                matrix = np.array([
                    [rr[i], rr[i+1], 1],
                    [rr[i+1], rr[i+2], 1], 
                    [rr[i+2], rr[i+3], 1]
                ])
                
                area = np.linalg.det(matrix)
                total_area += area
            
            # Calculate CCM using equation (7)
            CCM = total_area / (K * np.pi * SD1 * SD2)
            
            return CCM
            
        except Exception as e:
            print(f"Warning: CCM calculation failed: {e}")
            return 0.0

    def _calculate_afe(self, dRR):
        """
        Calculate AFE (Atrial Fibrillation Evidence) from dRR Poincaré plot
        Based on dRR irregularity measure from the paper
        """
        try:
            if len(dRR) < 2:
                return 0.0
                
            # Create Poincaré plot of dRR: dRR[i] vs dRR[i+1]
            dRR_n = dRR[:-1]
            dRR_n1 = dRR[1:]
            
            # Calculate SD1 and SD2 for dRR Poincaré plot
            diff_dRR = dRR_n1 - dRR_n
            sum_dRR = dRR_n1 + dRR_n
            
            SD1_dRR = np.std(diff_dRR) / np.sqrt(2)
            SD2_dRR = np.std(sum_dRR) / np.sqrt(2)
            
            # AFE is based on the ratio and irregularity of dRR
            # Higher values indicate more irregular (AF-like) patterns
            if SD2_dRR > 0:
                AFE = SD1_dRR / SD2_dRR
            else:
                AFE = 0.0
                
            # Add irregularity component based on variance
            irregularity = np.var(dRR) / (np.mean(np.abs(dRR)) + 1e-6)
            AFE = AFE * (1 + irregularity)
            
            return AFE
            
        except Exception as e:
            print(f"Warning: AFE calculation failed: {e}")
            return 0.0

    def calculate_psd_features(self, rr_intervals):
        """Calculate PSD features using EXACT AR method from Preprocessing.py (since you have spectrum library)"""
        try:
            from spectrum import arburg
            
            # EXACT same parameters and preprocessing as Preprocessing.py
            fs = 4  # Same resampling frequency
            
            # Resample (EXACT same method as Preprocessing.py)
            rr_interp = np.interp(np.linspace(0, len(rr_intervals), len(rr_intervals)*fs), 
                                np.arange(len(rr_intervals)), rr_intervals)  # keep in ms
            rr_interp = rr_interp - np.mean(rr_interp)  # remove DC offset only
            
            # AR modeling with order 16 (EXACT same as Preprocessing.py)
            order = 16
            ar_coeffs, e, _ = arburg(rr_interp, order)
            ar_coeffs = np.array(ar_coeffs)
            
            # Calculate PSD from AR coefficients (EXACT same method as Preprocessing.py)
            nfft = 512
            freqs = np.linspace(0, fs/2, nfft)
            psd = np.zeros_like(freqs)
            for i, f in enumerate(freqs):
                omega = np.exp(-2j * np.pi * f / fs)
                psd[i] = e / np.abs(np.polyval(ar_coeffs[::-1], omega))**2
            psd = np.real(psd)  # no normalization
            psd = psd / 100.0  # same scaling as Preprocessing.py
            
            # Power features (EXACT same bands as Preprocessing.py)
            lf_band = (freqs >= 0.04) & (freqs < 0.15)  # Note: < 0.15 (not <=)
            hf_band = (freqs >= 0.15) & (freqs <= 0.4)
            lf_power = np.trapz(psd[lf_band], freqs[lf_band]) if np.any(lf_band) else 0
            hf_power = np.trapz(psd[hf_band], freqs[hf_band]) if np.any(hf_band) else 0
            
            # Peak frequencies (same as Preprocessing.py)
            peak_lf = freqs[lf_band][np.argmax(psd[lf_band])] if np.any(lf_band) else 0
            peak_hf = freqs[hf_band][np.argmax(psd[hf_band])] if np.any(hf_band) else 0
            
            print(f"Spectral features — LF: {lf_power:.4f}, HF: {hf_power:.4f}, Peak LF: {peak_lf:.4f} Hz, Peak HF: {peak_hf:.4f} Hz")
            
            return freqs, psd, lf_power, hf_power
            
        except Exception as e:
            print(f"Error in PSD calculation: {e}")
            return np.array([0]), np.array([0]), 0, 0

    def calculate_bispectral_features(self, rr_intervals):
        """Calculate bispectral features using EXACT method from Preprocessing.py"""
        try:
            # STEP 1: Compute bispectrum using third-order cumulant method (EXACT same as Preprocessing.py)
            self.compute_bispectrum_exact(rr_intervals)
            
            # STEP 2: Extract features using EXACT same method as Preprocessing.py
            B = self.last_bispectrum
            f1 = self.last_f1
            f2 = self.last_f2
            
            # Create frequency meshgrid
            F1, F2 = np.meshgrid(f1, f2)
            
            # Define triangular region Ω: f1 > f2 AND f1 + f2 < 1 (EXACT same as Preprocessing.py)
            omega_mask = (F1 > F2) & (F1 + F2 < 1.0)
            B_omega = B.copy()
            B_omega[~omega_mask] = 0
            
            # Extract values in omega region
            omega_values = B_omega[omega_mask]
            omega_values = omega_values[omega_values > 1e-12]  # Remove very small values
            
            if len(omega_values) == 0:
                print("Warning: No valid omega values for bispectral analysis")
                return {'P1': 0, 'P2': 0, 'H1': 0, 'H2': 0, 'H3': 0, 'H4': 0}
            
            # P1: Normalized bispectral entropy (EXACT formula from Preprocessing.py)
            p_n = omega_values / np.sum(omega_values)
            P1 = -np.sum(p_n * np.log(p_n + 1e-12))
            
            # P2: Normalized squared bispectral entropy (EXACT formula from Preprocessing.py)
            omega_squared = omega_values ** 2
            q_n = omega_squared / np.sum(omega_squared)
            P2 = -np.sum(q_n * np.log(q_n + 1e-12))
            
            # H1: Sum of log amplitudes in omega region (EXACT formula from Preprocessing.py)
            H1 = np.sum(np.log(omega_values + 1e-12))
            
            # H2-H4: Diagonal analysis (EXACT same method as Preprocessing.py)
            f_diag = f1
            valid_diag_idx = np.where(f_diag < 0.5)[0]  # Same frequency limit as Preprocessing.py
            
            if len(valid_diag_idx) == 0:
                return {'P1': P1, 'P2': P2, 'H1': H1, 'H2': 0, 'H3': 0, 'H4': 0}
            
            diagonal_values = B[valid_diag_idx, valid_diag_idx]
            diag_valid = diagonal_values[diagonal_values > 1e-12]
            
            if len(diag_valid) == 0:
                return {'P1': P1, 'P2': P2, 'H1': H1, 'H2': 0, 'H3': 0, 'H4': 0}
            
            # H2-H4 calculation
            log_diag = np.log(diag_valid + 1e-12)
            H2 = np.sum(log_diag)
            
            N_diag = len(diag_valid)
            k = np.arange(1, N_diag + 1)
            H3 = np.sum(k * log_diag) / N_diag if N_diag > 0 else 0
            H4 = np.sum((k - H3) ** 2 * log_diag) / N_diag if N_diag > 0 else 0
            
            print(f"Bispectral features — P1: {P1:.4f}, P2: {P2:.4f}, H1: {H1:.4f}, H2: {H2:.4f}, H3: {H3:.4f}, H4: {H4:.4f}")
            
            return {'P1': P1, 'P2': P2, 'H1': H1, 'H2': H2, 'H3': H3, 'H4': H4}
            
        except Exception as e:
            print(f"Error in bispectral analysis: {e}")
            return {'P1': 0, 'P2': 0, 'H1': 0, 'H2': 0, 'H3': 0, 'H4': 0}

    def compute_bispectrum_exact(self, rr_intervals):
        """Compute bispectrum using EXACT method from Preprocessing.py with enhanced debugging"""
        try:
            # STEP 1: Preprocessing (EXACT same as Preprocessing.py) but handle small datasets
            x = rr_intervals - np.mean(rr_intervals)
            N = len(x)
            
            # Adjust max_lag for small datasets (removed minimum threshold)
            max_lag = min(50, N // 2)
            if max_lag < 5:  # If very small, use minimum viable lag
                max_lag = max(5, N // 4) if N >= 10 else max(2, N // 3)
            
            print(f"   Bispectrum computation: N={N}, max_lag={max_lag}")
            print(f"   Input signal stats: mean={np.mean(rr_intervals):.2f}, std={np.std(rr_intervals):.2f}")
            print(f"   Preprocessed signal stats: mean={np.mean(x):.6f}, std={np.std(x):.2f}")
            
            # STEP 2: Build third-order cumulant matrix C (EXACT same algorithm as Preprocessing.py)
            C = np.zeros((max_lag, max_lag))
            computed_elements = 0
            
            for m in range(max_lag):
                for n in range(max_lag):
                    if m + n < N:
                        # Third-order cumulant computation
                        available_length = N - m - n
                        if available_length > 0:
                            C[m, n] = np.mean(x[:available_length] * x[m:available_length + m] * x[m + n:available_length + m + n])
                            computed_elements += 1
            
            print(f"   Cumulant matrix: {C.shape}, computed {computed_elements}/{max_lag*max_lag} elements")
            print(f"   Cumulant stats: min={np.min(C):.6f}, max={np.max(C):.6f}, std={np.std(C):.6f}")
            
            # Check for meaningful cumulant values
            if np.all(np.abs(C) < 1e-10):
                print(f"   WARNING: All cumulant values are very small (< 1e-10)")
                # For very small datasets, try alternative approach
                if N < 20:
                    print(f"   Trying simplified approach for small dataset (N={N})")
                    # Use smaller lag windows for tiny datasets
                    max_lag = max(2, N // 5)
                    C = np.zeros((max_lag, max_lag))
                    for m in range(max_lag):
                        for n in range(max_lag):
                            if m + n < N:
                                available_length = N - m - n
                                if available_length > 0:
                                    C[m, n] = np.mean(x[:available_length] * x[m:available_length + m] * x[m + n:available_length + m + n])
            
            # STEP 3: Apply FFT2 to get bispectrum (EXACT same as Preprocessing.py)
            from scipy.fft import fft2
            
            # Apply FFT2 to cumulant matrix
            B_complex = fft2(C)
            B = np.abs(B_complex)  # Take magnitude
            
            # Create frequency axes
            f1 = np.linspace(0, 1, B.shape[0])
            f2 = np.linspace(0, 1, B.shape[1])
            
            print(f"   Bispectrum computed: {B.shape}")
            print(f"   Bispectrum stats: min={np.min(B):.8f}, max={np.max(B):.8f}")
            print(f"   Non-zero elements: {np.sum(B > 1e-12)}/{B.size}")
            
            # Store for plotting and feature extraction (same as Preprocessing.py)
            self.last_bispectrum = B
            self.last_f1 = f1
            self.last_f2 = f2
            
            # Quality checks (adjusted for small datasets)
            if np.max(B) == 0:
                print(f"   WARNING: Bispectrum is all zeros! (But continuing anyway)")
                return True  # Continue even with zero bispectrum
            
            if np.sum(B > np.max(B) * 0.1) < 5:  # Reduced threshold for small datasets
                print(f"   NOTE: Few significant bispectrum values (expected for small datasets)")
            
            # Check dynamic range
            dynamic_range = np.max(B) / np.median(B[B > 0]) if np.any(B > 0) else 0
            print(f"   Dynamic range: {dynamic_range:.2f}")
            
            return True
            
        except Exception as e:
            print(f"   ERROR computing bispectrum: {e}")
            import traceback
            traceback.print_exc()
            
            # Set to None to indicate failure
            self.last_bispectrum = None
            self.last_f1 = None
            self.last_f2 = None
            return False

    def calculate_poincare_features_hrv(self, rr_intervals):
        """Calculate Poincaré plot features"""
        try:
            if len(rr_intervals) < 2:
                return {'SD1': 0, 'SD2': 0, 'SD1_SD2_ratio': 0}
            
            # Create Poincaré plot data
            rr_n = rr_intervals[:-1]
            rr_n1 = rr_intervals[1:]
            
            # Calculate SD1 and SD2
            diff = rr_n1 - rr_n
            SD1 = np.std(diff) / np.sqrt(2)
            
            sum_vals = rr_n1 + rr_n  
            SD2 = np.std(sum_vals) / np.sqrt(2)
            
            SD1_SD2_ratio = SD1 / SD2 if SD2 != 0 else 0
            
            return {'SD1': SD1, 'SD2': SD2, 'SD1_SD2_ratio': SD1_SD2_ratio}
            
        except Exception as e:
            print(f"Error in Poincaré analysis: {e}")
            return {'SD1': 0, 'SD2': 0, 'SD1_SD2_ratio': 0}

    def calculate_sample_entropy(self, rr_intervals, m=2, r_factor=0.2):
        """Calculate Sample Entropy with adaptive tolerance and detailed debugging"""
        try:
            def sample_entropy_adaptive(signal, m, r_factor):
                signal = np.array(signal, dtype=float)
                n = len(signal)
                
                print(f"\n=== SAMPLE ENTROPY DEBUGGING ===")
                print(f"Signal length: {n}")
                print(f"Signal range: [{np.min(signal):.1f}, {np.max(signal):.1f}]")
                print(f"Signal mean: {np.mean(signal):.1f}, std: {np.std(signal):.1f}")
                print(f"First 10 values: {signal[:10]}")
                
                if n < 10:  # Very permissive minimum
                    print(f"Signal too short: {n} < 10")
                    return 0.0
                
                signal_std = np.std(signal)
                if signal_std == 0:
                    print("Zero standard deviation - signal is constant")
                    return 0.0
                
                # Try multiple r_factor values if needed
                r_factors_to_try = [r_factor, r_factor * 2, r_factor * 5, 0.1, 0.15, 0.25, 0.3]
                
                for r_test in r_factors_to_try:
                    print(f"\n--- Trying r_factor = {r_test} ---")
                    r = r_test * signal_std
                    print(f"Tolerance r = {r:.6f} (r_factor * std)")
                    
                    def _phi(m_val):
                        patterns = np.array([signal[i:i+m_val] for i in range(n - m_val + 1)])
                        n_patterns = len(patterns)
                        matches = np.zeros(n_patterns)
                        
                        for i in range(n_patterns):
                            template = patterns[i]
                            # Calculate maximum absolute difference for each pattern
                            max_diffs = np.max(np.abs(patterns - template), axis=1)
                            # Count matches within tolerance (exclude self-match at i)
                            matches[i] = np.sum(max_diffs <= r) - 1
                        
                        total_matches = np.sum(matches)
                        phi = total_matches / n_patterns
                        
                        print(f"  m={m_val}: patterns={n_patterns}, total_matches={total_matches}, phi={phi:.6f}")
                        return phi, total_matches
                    
                    try:
                        phi_m, matches_m = _phi(m)
                        phi_m1, matches_m1 = _phi(m + 1)
                        
                        print(f"  Results: phi_m={phi_m:.6f}, phi_m1={phi_m1:.6f}")
                        print(f"  Matches: m={matches_m}, m+1={matches_m1}")
                        
                        # Check if we have enough matches to calculate
                        if phi_m > 0 and phi_m1 > 0:
                            sam_en = -np.log(phi_m1 / phi_m)
                            
                            if not (np.isnan(sam_en) or np.isinf(sam_en)) and sam_en >= 0:
                                print(f"  SUCCESS: Sample Entropy = {sam_en:.6f}")
                                print("=== END DEBUGGING ===\n")
                                return sam_en
                            else:
                                print(f"  Invalid result: {sam_en}")
                        else:
                            print(f"  Insufficient matches (phi_m={phi_m}, phi_m1={phi_m1})")
                            
                    except Exception as e:
                        print(f"  Error with r_factor {r_test}: {e}")
                        continue
                
                print("All r_factor values failed")
                print("=== END DEBUGGING ===\n")
                return 0.0
            
            # Validate input
            if len(rr_intervals) < 10:
                print(f"Input too short: {len(rr_intervals)} RR intervals")
                return 0.0
            
            # Check for reasonable RR interval values
            if np.std(rr_intervals) < 1.0:
                print(f"Warning: Very low RR variability (std={np.std(rr_intervals):.3f}ms)")
            
            result = sample_entropy_adaptive(rr_intervals, m=m, r_factor=r_factor)
            return result
            
        except Exception as e:
            print(f"ERROR in Sample Entropy: {e}")
            import traceback
            traceback.print_exc()
            return 0.0

    def _validate_and_clean_features(self, features):
        """Clean features dictionary by replacing NaN/inf values with 0"""
        cleaned_features = {}
        
        for key, value in features.items():
            if isinstance(value, (int, float, np.number)):
                if np.isnan(value) or np.isinf(value):
                    print(f"⚠️ Warning: {key} has invalid value {value}, setting to 0")
                    cleaned_features[key] = 0.0
                else:
                    cleaned_features[key] = float(value)
            else:
                cleaned_features[key] = value
                
        return cleaned_features

    def extract_hrv_features_mei_paper(self, rr_intervals, sampling_freq=250):
        if len(rr_intervals) < 4:
            print("❌ Need at least 4 RR intervals for Mei paper features")
            return None
        
        try:
            print(f"🔍 Extracting Mei et al. HRV features from {len(rr_intervals)} RR intervals")
            
            # Convert RR intervals from ms to seconds for calculations
            rr_sec = np.array(rr_intervals) / 1000.0
            N = len(rr_intervals) + 1  # Number of R peaks = RR intervals + 1
            
            # 1. TIME-DOMAIN FEATURES
            # HR (Heart Rate) = 60 * N * f / L (beats/min)
            # Assuming total length L based on sum of RR intervals
            total_length_sec = np.sum(rr_sec)
            HR = 60.0 * N / total_length_sec
            
            # RRmean = average RRI (in ms)
            RRmean = np.mean(rr_intervals)
            
            # RRstd = standard deviation RRI (in ms) 
            RRstd = np.std(rr_intervals, ddof=1)
            
            print(f"   ✓ Time-domain: HR={HR:.1f} bpm, RRmean={RRmean:.1f}ms, RRstd={RRstd:.1f}ms")
            
            # 2. HANKEL MATRIX BASED FEATURES
            # Calculate HankDist for both RRI and dRR
            HankDist_RRI = self._calculate_hankel_distance(rr_intervals)
            
            # Calculate dRR (first order difference)
            dRR = np.diff(rr_intervals)  # dRRi = RRi+1 - RRi
            HankDist_dRR = self._calculate_hankel_distance(dRR) if len(dRR) >= 3 else 0.0
            
            print(f"   ✓ Hankel features: HankDist_RRI={HankDist_RRI:.4f}, HankDist_dRR={HankDist_dRR:.4f}")
            
            # 3. COMPLEX CORRELATION MEASURE (CCM)
            CCM = self._calculate_ccm(rr_intervals)
            print(f"   ✓ CCM: {CCM:.4f}")
            
            # 4. AFE (ATRIAL FIBRILLATION EVIDENCE)
            AFE = self._calculate_afe(dRR) if len(dRR) >= 2 else 0.0
            print(f"   ✓ AFE: {AFE:.4f}")
            
            # Return feature dictionary
            features = {
                # Time-domain features
                'HR': HR,
                'RRmean': RRmean, 
                'RRstd': RRstd,
                
                # Hankel-matrix based features
                'HankDist_RRI': HankDist_RRI,
                'HankDist_dRR': HankDist_dRR,
                
                # Complex correlation measure
                'CCM': CCM,
                
                # Atrial fibrillation evidence
                'AFE': AFE
            }
            
            print(f"✅ Mei et al. HRV features extracted successfully!")
            return features
            
        except Exception as e:
            print(f"❌ Error extracting Mei et al. features: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _calculate_hankel_distance(self, time_series):
        """
        Calculate HankDist using Hankel matrix eigenspectrum analysis
        Based on equation (4) in Mei et al. paper:
        HankDist = (4/n/2) * Σ|Ii - i|
        """
        try:
            series = np.array(time_series)
            n = len(series)
            
            if n < 3:
                return 0.0
                
            # Create Hankel matrix (equation 2 in paper)
            # For computational efficiency, use smaller matrix if series is long
            matrix_size = min(n, 32)  # Limit size for computational efficiency
            if n > matrix_size:
                # Take first matrix_size elements
                series = series[:matrix_size]
                n = matrix_size
                
            # Build Hankel matrix H
            H = np.zeros((n, n), dtype=complex)
            for i in range(n):
                for j in range(n):
                    H[i, j] = series[(i + j) % n]  # Circular Hankel as in equation (2)
            
            # Diagonalize Hankel matrix: H = U* D U (equation 3)
            eigenvalues, eigenvectors = np.linalg.eigh(H)
            
            # Perform DFT on eigenvectors (as described in paper)
            U_fft = np.fft.fft(eigenvectors, axis=0)
            
            # Find non-zero entries and calculate HankDist
            hankel_dist = 0.0
            n_half = n // 2
            
            for i in range(n_half):
                # Find row indices of significant entries in each column
                col = U_fft[:, i]
                significant_indices = np.where(np.abs(col) > 0.1 * np.max(np.abs(col)))[0]
                
                if len(significant_indices) > 0:
                    # Use the first significant index as Ii
                    Ii = significant_indices[0]
                    hankel_dist += abs(Ii - i)
            
            # Normalize according to equation (4)
            if n_half > 0:
                hankel_dist = (4.0 / n_half) * hankel_dist
            
            return hankel_dist
            
        except Exception as e:
            print(f"Warning: Hankel distance calculation failed: {e}")
            return 0.0


    def _calculate_ccm(self, rr_intervals):
        """
        Calculate Complex Correlation Measure (CCM)
        Based on equations (6) and (7) in Mei et al. paper
        """
        try:
            rr = np.array(rr_intervals)
            
            if len(rr) < 4:  # Need at least 4 RR intervals for 3 consecutive triplets
                return 0.0
            
            # Calculate SD1 and SD2 for normalization (needed for equation 7)
            # Standard Poincaré plot descriptors
            rr_n = rr[:-1]
            rr_n1 = rr[1:]
            
            # SD1: short-term variability
            diff_rr = rr_n1 - rr_n
            SD1 = np.std(diff_rr) / np.sqrt(2)
            
            # SD2: long-term variability  
            sum_rr = rr_n1 + rr_n
            SD2 = np.std(sum_rr) / np.sqrt(2)
            
            if SD1 == 0 or SD2 == 0:
                return 0.0
            
            # Calculate oriented areas A(i) using determinant (equation 6)
            total_area = 0.0
            K = len(rr) - 3  # Number of consecutive triplets
            
            for i in range(K):
                # Create matrix for determinant calculation
                # A(i) = det([[RRi, RRi+1, 1], [RRi+1, RRi+2, 1], [RRi+2, RRi+3, 1]])
                matrix = np.array([
                    [rr[i], rr[i+1], 1],
                    [rr[i+1], rr[i+2], 1], 
                    [rr[i+2], rr[i+3], 1]
                ])
                
                area = np.linalg.det(matrix)
                total_area += area
            
            # Calculate CCM using equation (7)
            CCM = total_area / (K * np.pi * SD1 * SD2)
            
            return CCM
            
        except Exception as e:
            print(f"Warning: CCM calculation failed: {e}")
            return 0.0

    def calculate_rmssd_from_peaks(self, peak_times):
        """Calculate RMSSD from peak times"""
        if len(peak_times) < 2:
            return None
        
        # Calculate RR intervals in milliseconds
        rr_intervals = np.diff(peak_times) * 1000
        
        if len(rr_intervals) < 2:
            return None
        
        # Remove outliers (300-2000ms range)
        valid_mask = (rr_intervals >= 300) & (rr_intervals <= 2000)
        rr_clean = rr_intervals[valid_mask]
        
        if len(rr_clean) < 2:
            return None
        
        # RMSSD calculation
        successive_diffs = np.diff(rr_clean)
        rmssd = np.sqrt(np.mean(successive_diffs**2))
        
        return {
            'rmssd': rmssd,
            'valid_intervals': len(rr_clean),
            'total_intervals': len(rr_intervals)
        }
    
    def calculate_rr_intervals_rmse(self, radar_rr, ecg_rr):
        """Calculate RMSE between radar and ECG RR intervals"""
        # Synchronize lengths (use minimum length)
        min_length = min(len(radar_rr), len(ecg_rr))
        
        if min_length < 2:
            return None
        
        radar_sync = radar_rr[:min_length]
        ecg_sync = ecg_rr[:min_length]
        
        # Calculate RMSE
        squared_errors = (radar_sync - ecg_sync) ** 2
        rmse = np.sqrt(np.mean(squared_errors))
        
        # Calculate MAE for additional context
        mae = np.mean(np.abs(radar_sync - ecg_sync))
        
        # Calculate correlation
        correlation = np.corrcoef(radar_sync, ecg_sync)[0, 1] if min_length > 1 else 0
        if np.isnan(correlation):
            correlation = 0
        
        return {
            'rmse': rmse,
            'mae': mae,
            'n_compared': min_length,
            'correlation': correlation
        }

    def _calculate_afe(self, dRR):
        """
        Calculate AFE (Atrial Fibrillation Evidence) from dRR Poincaré plot
        Note: The paper mentions AFE is extracted from Poincaré plot of dRR
        but doesn't give exact formula. Implementing based on typical AF detection measures.
        """
        try:
            if len(dRR) < 2:
                return 0.0
                
            dRR = np.array(dRR)
            
            # Create Poincaré plot of dRR (dRR[n] vs dRR[n+1])
            dRR_n = dRR[:-1]
            dRR_n1 = dRR[1:]
            
            if len(dRR_n) == 0:
                return 0.0
            
            # AFE measure: Based on dispersion and irregularity of dRR Poincaré plot
            # This is a common approach for AF detection from RR differences
            
            # Method 1: Standard deviation of dRR (higher for AF)
            dRR_std = np.std(dRR)
            
            # Method 2: Poincaré plot dispersion measure
            # Distance from identity line y=x in dRR Poincaré plot
            distances_from_identity = np.abs(dRR_n1 - dRR_n) / np.sqrt(2)
            poincare_dispersion = np.mean(distances_from_identity)
            
            # Method 3: Coefficient of variation of dRR
            dRR_mean = np.mean(np.abs(dRR))
            cv_dRR = dRR_std / dRR_mean if dRR_mean != 0 else 0
            
            # Combine measures (weighted combination)
            AFE = 0.5 * poincare_dispersion + 0.3 * dRR_std + 0.2 * cv_dRR
            
            return AFE
            
        except Exception as e:
            print(f"Warning: AFE calculation failed: {e}")
            return 0.0

    def update_hrv_feature_plots(self, signal_type, rr_intervals, features):
        """Update plots for the specified signal type - using original sophisticated styling"""
        try:
            if signal_type == "ECG":
                psd_ax = self.ax_ecg_psd
                poincare_ax = self.ax_ecg_poincare
                bispectrum_ax = self.ax_ecg_bispectrum
                color = 'blue'
            else:
                psd_ax = self.ax_radar_psd
                poincare_ax = self.ax_radar_poincare  
                bispectrum_ax = self.ax_radar_bispectrum
                color = 'red'
            
            # 1. PSD plot (styled like Preprocessing.py)
            psd_ax.clear()
            if 'frequencies' in features and 'psd' in features:
                frequencies = features['frequencies']
                psd = features['psd']
                
                # Main PSD plot
                psd_ax.plot(frequencies, psd, color=color, linewidth=1.5, alpha=0.8)
                psd_ax.fill_between(frequencies, psd, alpha=0.3, color=color)
                
                # Highlight LF and HF bands (exactly like Preprocessing.py)
                lf_band = (frequencies >= 0.04) & (frequencies < 0.15)
                hf_band = (frequencies >= 0.15) & (frequencies <= 0.4)
                
                if np.any(lf_band):
                    psd_ax.fill_between(frequencies[lf_band], psd[lf_band], 
                                    alpha=0.5, color='red', label=f'LF: {features["LF_power"]:.4f}')
                if np.any(hf_band):
                    psd_ax.fill_between(frequencies[hf_band], psd[hf_band], 
                                    alpha=0.5, color='green', label=f'HF: {features["HF_power"]:.4f}')
                
                # Set limits and formatting like Preprocessing.py
                psd_ax.set_xlim(0, 0.5)
                psd_max = np.max(psd)
                psd_ax.set_ylim(0, psd_max * 1.2)  # Dynamic scaling like Preprocessing.py
                
                # Format y-axis (same as Preprocessing.py)
                from matplotlib.ticker import ScalarFormatter
                psd_ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
                psd_ax.ticklabel_format(style='plain', axis='y')
                
                psd_ax.legend(fontsize=9)
            
            psd_ax.set_title(f'{signal_type} Power Spectral Density (16-order AR Model)', fontsize=12, fontweight='bold')
            psd_ax.set_xlabel('Frequency (Hz)')
            psd_ax.set_ylabel('Power Spectral Density (s²/Hz)')
            psd_ax.grid(True, alpha=0.3)
            
            # 2. Poincaré plot (styled like Preprocessing.py)
            poincare_ax.clear()
            if len(rr_intervals) > 1:
                rr_n = rr_intervals[:-1]
                rr_n1 = rr_intervals[1:]
                
                # Scatter plot with color coding like Preprocessing.py
                scatter = poincare_ax.scatter(rr_n, rr_n1, alpha=0.6, s=15, color=color)
                
                # Calculate RRm (mean of all RR intervals) as per paper
                RRm = np.mean(rr_intervals)
                
                # Draw reference lines as specified in Preprocessing.py
                min_rr = min(min(rr_n), min(rr_n1))
                max_rr = max(max(rr_n), max(rr_n1))
                
                # Line 1: y = x (identity line)
                poincare_ax.plot([min_rr, max_rr], [min_rr, max_rr], 'k--', alpha=0.7, 
                            linewidth=2, label='y = x')
                
                # Line 2: y = -x + 2*RRm (as specified in paper/Preprocessing.py)
                x_line = np.linspace(min_rr, max_rr, 100)
                y_line = -x_line + 2 * RRm
                poincare_ax.plot(x_line, y_line, 'r--', alpha=0.7, linewidth=2, 
                            label=f'y = -x + 2RRm')
                
                # Calculate SD1 and SD2 as per paper equations (same as Preprocessing.py)
                distances_id = np.abs(rr_n1 - rr_n) / np.sqrt(2)
                SD1 = np.std(distances_id)
                
                distances_2 = np.abs(rr_n1 + rr_n - 2*RRm) / np.sqrt(2)
                SD2 = np.std(distances_2)
                
                # Add SD1/SD2 info with better formatting
                poincare_ax.text(0.05, 0.95, 
                            f"SD1: {SD1:.2f} ms\nSD2: {SD2:.2f} ms\nRatio: {SD1/SD2 if SD2 != 0 else 0:.3f}\nRRm: {RRm:.1f} ms", 
                            transform=poincare_ax.transAxes, verticalalignment='top', fontsize=9,
                            bbox=dict(boxstyle='round', alpha=0.8, facecolor='white', edgecolor='gray'))
                
                poincare_ax.legend(fontsize=8, loc='lower right')
            
            poincare_ax.set_title(f'{signal_type} Poincaré Plot', fontsize=12, fontweight='bold')
            poincare_ax.set_xlabel('RRn (ms)')
            poincare_ax.set_ylabel('RRn+1 (ms)')
            poincare_ax.grid(True, alpha=0.3)
            
            # CRITICAL: Force square aspect ratio for Poincaré plot
            poincare_ax.set_aspect('equal', adjustable='box')
            
            # 3. Bispectrum plot - MAKE IT SQUARE AND EQUAL
            bispectrum_ax.clear()
            self.plot_bispectrum_for_signal(bispectrum_ax, rr_intervals, signal_type)
            
            # Force square aspect ratio for bispectrum
            bispectrum_ax.set_aspect('equal')
            
            self.fig_hrv_features.tight_layout()
            self.canvas_hrv_features.draw()
            
        except Exception as e:
            print(f"Error updating plots for {signal_type}: {e}")

    def debug_hrv_axes(self):
        """Debug function to check which HRV axes are available"""
        print("=== HRV AXES DEBUG ===")
        axes_to_check = [
            'ax_ecg_psd', 'ax_radar_psd',
            'ax_ecg_bispectrum', 'ax_radar_bispectrum', 
            'ax_ecg_poincare', 'ax_radar_poincare',
            'canvas_hrv_features'
        ]
        
        for axis_name in axes_to_check:
            if hasattr(self, axis_name):
                axis_obj = getattr(self, axis_name)
                print(f"✓ {axis_name}: {type(axis_obj)}")
            else:
                print(f"❌ {axis_name}: NOT FOUND")
        
        print("=====================")

    def plot_bispectrum_for_signal(self, ax, rr_intervals, signal_type):
        """Plot bispectrum using EXACT method from Preprocessing.py (full bispectrum, no triangular mask for display)"""
        try:
            # Check if bispectrum data is available from feature extraction
            if (hasattr(self, 'last_bispectrum') and hasattr(self, 'last_f1') and hasattr(self, 'last_f2') and
                self.last_bispectrum is not None and self.last_f1 is not None and self.last_f2 is not None):
                
                # Use stored bispectrum data (computed during feature extraction)
                B = self.last_bispectrum
                f1_axis = self.last_f1
                f2_axis = self.last_f2
                
                print(f"Using stored bispectrum for {signal_type}: shape={B.shape}, range=[{np.min(B):.6f}, {np.max(B):.6f}]")
            else:
                # Compute fresh bispectrum if not available
                print(f"Computing fresh bispectrum for {signal_type} plotting...")
                self.compute_bispectrum_exact(rr_intervals)
                
                if self.last_bispectrum is None:
                    raise Exception("Failed to compute bispectrum")
                    
                B = self.last_bispectrum
                f1_axis = self.last_f1
                f2_axis = self.last_f2

            if B.size > 0:
                F1, F2 = np.meshgrid(f1_axis, f2_axis)
                
                # CRITICAL: Use FULL bispectrum for plotting (no triangular mask!)
                # This is the key difference - Preprocessing.py plots the full bispectrum
                print(f"Plotting full bispectrum (no triangular mask) for {signal_type}")
                
                # EXACT level determination as Preprocessing.py
                vmin = np.percentile(B, 5)
                vmax = np.percentile(B, 95)
                levels = np.linspace(vmin, vmax, 20)
                
                print(f"Bispectrum plot levels for {signal_type}: [{vmin:.6f}, {vmax:.6f}]")
                
                # CHOOSE PLOTTING STYLE: Try vibrant version first (like kim_findpeaks_method.py)
                # Style 1: Vibrant version (contour + contourf)
                try:
                    # Main contour lines
                    contour = ax.contour(F1, F2, B, levels=levels, cmap='jet', linewidths=0.8)
                    
                    # Filled contour for vibrant effect  
                    filled_contour = ax.contourf(F1, F2, B, levels=levels, cmap='jet', alpha=0.6)
                    
                    # Add colorbar (vibrant style)
                    try:
                        cbar = self.fig_hrv_features.colorbar(filled_contour, ax=ax, fraction=0.046, pad=0.04)
                        cbar.set_label('|B(f1,f2)|', fontsize=10)
                    except Exception as e:
                        print(f"Colorbar warning for {signal_type}: {e}")
                    
                    # Grid styling (vibrant version)
                    ax.grid(True, linestyle='--', alpha=0.4, color='white', linewidth=1)
                    
                except Exception as e:
                    print(f"Vibrant plotting failed for {signal_type}, trying simple version: {e}")
                    
                    # Fallback: Simple version (contour only, like original Preprocessing.py)
                    contour = ax.contour(F1, F2, B, levels=levels, cmap='jet', linewidths=0.8)
                    
                    # Add colorbar (simple style)
                    try:
                        cbar = self.fig_hrv_features.colorbar(contour, ax=ax, fraction=0.046, pad=0.04)
                        cbar.set_label('|B(f1,f2)|', fontsize=10)
                    except:
                        pass
                    
                    # Grid styling (simple version)
                    ax.grid(True, linestyle='--', alpha=0.4)
                
            else:
                ax.text(0.5, 0.5, f'No bispectrum data for {signal_type}', ha='center', va='center', 
                    transform=ax.transAxes, fontsize=12,
                    bbox=dict(boxstyle='round', alpha=0.8, facecolor='lightblue'))

            # EXACT styling as Preprocessing.py with SQUARE aspect
            ax.set_xlabel('f1 (Normalized Frequency)', fontsize=11, fontweight='bold')
            ax.set_ylabel('f2 (Normalized Frequency)', fontsize=11, fontweight='bold')
            ax.set_title(f'{signal_type} |B(f1,f2)| Contour Plot', fontsize=12, fontweight='bold', pad=20)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            
            # CRITICAL: Force square aspect ratio - this prevents cropping
            ax.set_aspect('equal', adjustable='box')  # Force square with box adjustment

        except Exception as e:
            print(f"BISPECTRUM PLOTTING ERROR for {signal_type}: {e}")
            ax.clear()
            ax.text(0.5, 0.5, f'{signal_type} Bispectrum Error:\n{str(e)}\n\nTrying to recompute...', 
                ha='center', va='center', transform=ax.transAxes, color='red', fontsize=10,
                bbox=dict(boxstyle='round', alpha=0.8, facecolor='lightcoral'))
            
            # Try to recompute fresh bispectrum as last resort
            try:
                print(f"Last resort: Computing fresh bispectrum for {signal_type}")
                x = rr_intervals - np.mean(rr_intervals)
                N = len(x)
                max_lag = min(50, N // 2)
                
                if max_lag < 10:
                    ax.text(0.5, 0.3, f'Not enough data for bispectrum\n(need more RR intervals)', 
                        ha='center', va='center', transform=ax.transAxes, color='orange', fontsize=10)
                    return
                
                C = np.zeros((max_lag, max_lag))
                for m in range(max_lag):
                    for n in range(max_lag):
                        if m + n < N:
                            C[m, n] = np.mean(x[:N - m - n] * x[m:N - n] * x[m + n:])
                
                from scipy.fft import fft2
                B = np.abs(fft2(C))
                f1 = np.linspace(0, 1, B.shape[0])
                f2 = np.linspace(0, 1, B.shape[1])
                
                if B.size > 0 and np.max(B) > 0:
                    F1, F2 = np.meshgrid(f1, f2)
                    vmin = np.percentile(B, 10)  # Less aggressive percentile
                    vmax = np.percentile(B, 90)
                    levels = np.linspace(vmin, vmax, 15)
                    
                    ax.contourf(F1, F2, B, levels=levels, cmap='jet', alpha=0.7)
                    ax.set_title(f'{signal_type} |B(f1,f2)| (Recomputed)', fontsize=11)
                    ax.set_xlim(0, 1)
                    ax.set_ylim(0, 1)
                    ax.grid(True, alpha=0.3)
                    print(f"Successfully recomputed bispectrum for {signal_type}")
                else:
                    ax.text(0.5, 0.3, f'Recomputation failed\n(zero bispectrum values)', 
                        ha='center', va='center', transform=ax.transAxes, color='red', fontsize=10)
                    
            except Exception as e2:
                print(f"Recomputation also failed for {signal_type}: {e2}")
                ax.text(0.5, 0.3, f'All bispectrum methods failed\nfor {signal_type}', 
                    ha='center', va='center', transform=ax.transAxes, color='red', fontsize=10)

    def extract_both_hrv_features(self):
        """Extract HRV features for both ECG and Radar"""
        try:
            # Extract both
            self.extract_hrv_features_for_signal("ECG")
            self.extract_hrv_features_for_signal("Radar")
            
            # Display combined results without correlation analysis
            if hasattr(self, 'ecg_hrv_features') and hasattr(self, 'radar_hrv_features'):
                self.display_both_feature_results()
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to extract and compare features: {str(e)}")

    def display_both_feature_results(self):
        """Display results for both ECG and Radar features - ENHANCED with 19 features"""
        try:
            if not (hasattr(self, 'ecg_hrv_features') and hasattr(self, 'radar_hrv_features')):
                return
            
            self.hrv_features_text.delete(1.0, tk.END)
            
            # ===============================
            # RADAR FEATURES FIRST
            # ===============================
            text = "RADAR HRV FEATURE EXTRACTION RESULTS (ENHANCED)\n"
            text += "=" * 60 + "\n\n"
            
            radar_features = self.radar_hrv_features
            
            # Radar - RR intervals info (if available)
            if hasattr(self, 'radar_rr_intervals'):
                rr = self.radar_rr_intervals
                text += f"RR INTERVALS INFO:\n"
                text += f"Total intervals: {len(rr)}\n"
                text += f"Mean RR: {np.mean(rr):.1f} ms\n"
                text += f"Std RR: {np.std(rr):.1f} ms\n"
                text += f"Range: {np.min(rr):.1f} - {np.max(rr):.1f} ms\n\n"
            
            # Radar - All feature categories (12 original + 7 new = 19 total)
            text += "SPECTRAL FEATURES (2):\n"
            text += "-" * 30 + "\n"
            text += f"LF Power: {radar_features['LF_power']:.8f}\n"
            text += f"HF Power: {radar_features['HF_power']:.8f}\n\n"
            
            text += "BISPECTRAL FEATURES (6):\n"
            text += "-" * 30 + "\n"
            text += f"P1 (Norm. Bisp. Entropy): {radar_features['P1']:.8f}\n"
            text += f"P2 (Norm. Bisp. Sq. Entropy): {radar_features['P2']:.8f}\n"
            text += f"H1 (Sum Amplitudes): {radar_features['H1']:.8f}\n"
            text += f"H2 (Sum Log Diag): {radar_features['H2']:.8f}\n"
            text += f"H3 (1st Order Moment): {radar_features['H3']:.8f}\n"
            text += f"H4 (2nd Order Moment): {radar_features['H4']:.8f}\n\n"
            
            text += "NON-LINEAR FEATURES (4):\n"
            text += "-" * 30 + "\n"
            text += f"SD1: {radar_features['SD1']:.8f} ms\n"
            text += f"SD2: {radar_features['SD2']:.8f} ms\n"
            text += f"SD1/SD2 Ratio: {radar_features['SD1_SD2_ratio']:.8f}\n"
            text += f"Sample Entropy: {radar_features['SamEn']:.8f}\n\n"
            
            # NEW MEI PAPER FEATURES (7) - Add these to display
            text += "MEI ET AL. PAPER FEATURES (7):\n"
            text += "-" * 30 + "\n"
            text += f"HR (Heart Rate): {radar_features.get('HR', 0):.2f} bpm\n"
            text += f"RRmean (Mean RR): {radar_features.get('RRmean', 0):.2f} ms\n"
            text += f"RRstd (RR Std Dev): {radar_features.get('RRstd', 0):.2f} ms\n"
            text += f"HankDist_RRI: {radar_features.get('HankDist_RRI', 0):.8f}\n"
            text += f"HankDist_dRR: {radar_features.get('HankDist_dRR', 0):.8f}\n"
            text += f"CCM (Complex Corr): {radar_features.get('CCM', 0):.8f}\n"
            text += f"AFE (AF Evidence): {radar_features.get('AFE', 0):.8f}\n\n"
            
            # MOST IMPORTANT: Excel format for Radar (19 features)
            text += "RADAR FEATURE VECTOR (for Excel copy-paste - ENHANCED):\n"
            text += "-" * 60 + "\n"
            radar_vector = [
                radar_features['LF_power'], radar_features['HF_power'],
                radar_features['P1'], radar_features['P2'], radar_features['H1'], radar_features['H2'], 
                radar_features['H3'], radar_features['H4'],
                radar_features['SD1'], radar_features['SD2'], radar_features['SD1_SD2_ratio'], radar_features['SamEn'],
                # NEW FEATURES
                radar_features.get('HR', 0), radar_features.get('RRmean', 0), radar_features.get('RRstd', 0),
                radar_features.get('HankDist_RRI', 0), radar_features.get('HankDist_dRR', 0), 
                radar_features.get('CCM', 0), radar_features.get('AFE', 0)
            ]
            text += "\t".join([f"{v:.10f}" for v in radar_vector]) + "\n\n"
            
            text += f"RADAR Total features extracted: 19\n\n"
            
            # ===============================  
            # ECG FEATURES AFTER RADAR
            # ===============================
            text += "\n" + "="*80 + "\n\n"
            text += "ECG HRV FEATURE EXTRACTION RESULTS (ENHANCED)\n"
            text += "=" * 60 + "\n\n"
            
            ecg_features = self.ecg_hrv_features
            
            # ECG - RR intervals info (if available)  
            if hasattr(self, 'ecg_rr_intervals'):
                rr = self.ecg_rr_intervals
                text += f"RR INTERVALS INFO:\n"
                text += f"Total intervals: {len(rr)}\n"
                text += f"Mean RR: {np.mean(rr):.1f} ms\n"
                text += f"Std RR: {np.std(rr):.1f} ms\n"
                text += f"Range: {np.min(rr):.1f} - {np.max(rr):.1f} ms\n\n"
            
            # ECG - All feature categories (same format as Radar)
            text += "SPECTRAL FEATURES (2):\n"
            text += "-" * 30 + "\n"
            text += f"LF Power: {ecg_features['LF_power']:.8f}\n"
            text += f"HF Power: {ecg_features['HF_power']:.8f}\n\n"
            
            text += "BISPECTRAL FEATURES (6):\n"
            text += "-" * 30 + "\n"
            text += f"P1 (Norm. Bisp. Entropy): {ecg_features['P1']:.8f}\n"
            text += f"P2 (Norm. Bisp. Sq. Entropy): {ecg_features['P2']:.8f}\n"
            text += f"H1 (Sum Amplitudes): {ecg_features['H1']:.8f}\n"
            text += f"H2 (Sum Log Diag): {ecg_features['H2']:.8f}\n"
            text += f"H3 (1st Order Moment): {ecg_features['H3']:.8f}\n"
            text += f"H4 (2nd Order Moment): {ecg_features['H4']:.8f}\n\n"
            
            text += "NON-LINEAR FEATURES (4):\n"
            text += "-" * 30 + "\n"
            text += f"SD1: {ecg_features['SD1']:.8f} ms\n"
            text += f"SD2: {ecg_features['SD2']:.8f} ms\n"
            text += f"SD1/SD2 Ratio: {ecg_features['SD1_SD2_ratio']:.8f}\n"
            text += f"Sample Entropy: {ecg_features['SamEn']:.8f}\n\n"
            
            # NEW MEI PAPER FEATURES (7) for ECG
            text += "MEI ET AL. PAPER FEATURES (7):\n"
            text += "-" * 30 + "\n"
            text += f"HR (Heart Rate): {ecg_features.get('HR', 0):.2f} bpm\n"
            text += f"RRmean (Mean RR): {ecg_features.get('RRmean', 0):.2f} ms\n"
            text += f"RRstd (RR Std Dev): {ecg_features.get('RRstd', 0):.2f} ms\n"
            text += f"HankDist_RRI: {ecg_features.get('HankDist_RRI', 0):.8f}\n"
            text += f"HankDist_dRR: {ecg_features.get('HankDist_dRR', 0):.8f}\n"
            text += f"CCM (Complex Corr): {ecg_features.get('CCM', 0):.8f}\n"
            text += f"AFE (AF Evidence): {ecg_features.get('AFE', 0):.8f}\n\n"
            
            # MOST IMPORTANT: Excel format for ECG (19 features)
            text += "ECG FEATURE VECTOR (for Excel copy-paste - ENHANCED):\n"
            text += "-" * 60 + "\n"
            ecg_vector = [
                ecg_features['LF_power'], ecg_features['HF_power'],
                ecg_features['P1'], ecg_features['P2'], ecg_features['H1'], ecg_features['H2'], 
                ecg_features['H3'], ecg_features['H4'],
                ecg_features['SD1'], ecg_features['SD2'], ecg_features['SD1_SD2_ratio'], ecg_features['SamEn'],
                # NEW FEATURES  
                ecg_features.get('HR', 0), ecg_features.get('RRmean', 0), ecg_features.get('RRstd', 0),
                ecg_features.get('HankDist_RRI', 0), ecg_features.get('HankDist_dRR', 0),
                ecg_features.get('CCM', 0), ecg_features.get('AFE', 0)
            ]
            text += "\t".join([f"{v:.10f}" for v in ecg_vector]) + "\n\n"
            
            text += f"ECG Total features extracted: 19\n\n"
            
            # ===============================
            # EXCEL HEADERS AND SUMMARY  
            # ===============================
            text += "\n" + "="*80 + "\n"
            text += "EXCEL HEADERS AND COMPARISON SUMMARY\n"
            text += "="*80 + "\n\n"
            
            # Headers for Excel (19 features)
            text += "HEADERS (for Excel - ENHANCED 19 features):\n"
            text += "-" * 45 + "\n"
            headers = ['LF_power', 'HF_power', 'P1', 'P2', 'H1', 'H2', 'H3', 'H4', 
                    'SD1', 'SD2', 'SD1_SD2_ratio', 'SamEn',
                    'HR', 'RRmean', 'RRstd', 'HankDist_RRI', 'HankDist_dRR', 'CCM', 'AFE']
            text += "\t".join(headers) + "\n\n"
            
            # Comparison table (19 features)
            text += f"{'Feature':<18} {'Radar':<15} {'ECG':<15} {'Difference':<15}\n"
            text += "-" * 65 + "\n"
            for i, name in enumerate(headers):
                radar_val = radar_vector[i]
                ecg_val = ecg_vector[i]
                diff = abs(ecg_val - radar_val)
                text += f"{name:<18} {radar_val:<15.6f} {ecg_val:<15.6f} {diff:<15.6f}\n"
            
            # RMSE calculation (19 features)
            rmse = np.sqrt(np.mean((np.array(ecg_vector) - np.array(radar_vector))**2))
            text += f"\nROOT MEAN SQUARE ERROR (19 features): {rmse:.6f}\n"
            text += f"Both signals: 19 features each (12 original + 7 Mei paper)\n"
            
            self.hrv_features_text.insert(tk.END, text)
            
        except Exception as e:
            print(f"Error displaying both feature results: {e}")
            import traceback
            traceback.print_exc()

    def display_hrv_feature_results(self, signal_type, features, rr_intervals):
        """Display HRV feature results in the text box with all new features"""
        try:
            # Clear and add new results
            self.hrv_features_text.delete(1.0, tk.END)
            
            text = f"{signal_type} HRV FEATURE EXTRACTION RESULTS\n"
            text += "=" * 60 + "\n\n"
            
            # RR intervals info
            text += f"RR INTERVALS INFO:\n"
            text += f"Total intervals: {len(rr_intervals)}\n"
            text += f"Mean RR: {np.mean(rr_intervals):.1f} ms\n"
            text += f"Std RR: {np.std(rr_intervals):.1f} ms\n"
            text += f"Range: {np.min(rr_intervals):.1f} - {np.max(rr_intervals):.1f} ms\n\n"
            
            # Spectral features
            text += "SPECTRAL FEATURES (2):\n"
            text += "-" * 30 + "\n"
            text += f"LF Power: {features['LF_power']:.8f}\n"
            text += f"HF Power: {features['HF_power']:.8f}\n\n"
            
            # Bispectral features
            text += "BISPECTRAL FEATURES (6):\n"
            text += "-" * 30 + "\n"
            text += f"P1 (Norm. Bisp. Entropy): {features['P1']:.8f}\n"
            text += f"P2 (Norm. Bisp. Sq. Entropy): {features['P2']:.8f}\n"
            text += f"H1 (Sum Amplitudes): {features['H1']:.8f}\n"
            text += f"H2 (Sum Log Diag): {features['H2']:.8f}\n"
            text += f"H3 (1st Order Moment): {features['H3']:.8f}\n"
            text += f"H4 (2nd Order Moment): {features['H4']:.8f}\n\n"
            
            # Non-linear features
            text += "NON-LINEAR FEATURES (4):\n"
            text += "-" * 30 + "\n"
            text += f"SD1: {features['SD1']:.8f} ms\n"
            text += f"SD2: {features['SD2']:.8f} ms\n"
            text += f"SD1/SD2 Ratio: {features['SD1_SD2_ratio']:.8f}\n"
            text += f"Sample Entropy: {features['SamEn']:.8f}\n\n"
            
            # NEW MEI PAPER FEATURES (7)
            text += "MEI ET AL. PAPER FEATURES (7):\n"
            text += "-" * 30 + "\n"
            text += f"HR (Heart Rate): {features.get('HR', 0):.2f} bpm\n"
            text += f"RRmean (Mean RR): {features.get('RRmean', 0):.2f} ms\n"
            text += f"RRstd (RR Std Dev): {features.get('RRstd', 0):.2f} ms\n"
            text += f"HankDist_RRI: {features.get('HankDist_RRI', 0):.8f}\n"
            text += f"HankDist_dRR: {features.get('HankDist_dRR', 0):.8f}\n"
            text += f"CCM (Complex Corr): {features.get('CCM', 0):.8f}\n"
            text += f"AFE (AF Evidence): {features.get('AFE', 0):.8f}\n\n"
            
            # FEATURE VECTOR for copy-paste (19 features total)
            text += f"{signal_type} FEATURE VECTOR (for Excel copy-paste - 19 features):\n"
            text += "-" * 60 + "\n"
            feature_vector = [
                # Original 12 features
                features['LF_power'], features['HF_power'],
                features['P1'], features['P2'], features['H1'], features['H2'], 
                features['H3'], features['H4'],
                features['SD1'], features['SD2'], features['SD1_SD2_ratio'], features['SamEn'],
                # New 7 features  
                features.get('HR', 0), features.get('RRmean', 0), features.get('RRstd', 0),
                features.get('HankDist_RRI', 0), features.get('HankDist_dRR', 0),
                features.get('CCM', 0), features.get('AFE', 0)
            ]
            text += "\t".join([f"{val:.8f}" for val in feature_vector]) + "\n\n"
            
            text += f"Total features extracted: 19 (12 original + 7 Mei paper)\n"
            
            self.hrv_features_text.insert(tk.END, text)
            
        except Exception as e:
            print(f"Error displaying feature results: {e}")
            import traceback
            traceback.print_exc()

    def display_feature_comparison_results(self):
        """Display comparison results for both ECG and Radar features - removed correlation analysis"""
        try:
            # This function is now replaced by display_both_feature_results()
            # Call the simpler version instead
            self.display_both_feature_results()
            
        except Exception as e:
            print(f"Error displaying comparison results: {e}")

    def show_hrv_features_table(self):
        """Show HRV features in a popup table window"""
        try:
            # Check if features are available
            if not (hasattr(self, 'ecg_hrv_features') or hasattr(self, 'radar_hrv_features')):
                messagebox.showwarning("Warning", "No HRV features extracted yet.")
                return
            
            # Create popup window
            table_window = tk.Toplevel(self.root)
            table_window.title("HRV Features Table")
            table_window.geometry("700x600")
            
            # Create treeview for table display
            columns = ['Feature', 'ECG', 'Radar', 'Difference']
            tree = ttk.Treeview(table_window, columns=columns, show='headings', height=15)
            
            # Define column headings
            for col in columns:
                tree.heading(col, text=col)
                tree.column(col, width=160, anchor='center')
            
            # Add data
            feature_names = ['LF_power', 'HF_power', 'P1', 'P2', 'H1', 'H2', 'H3', 'H4', 
                            'SD1', 'SD2', 'SD1_SD2_ratio', 'SamEn']
            
            for name in feature_names:
                ecg_val = self.ecg_hrv_features.get(name, 0) if hasattr(self, 'ecg_hrv_features') else 0
                radar_val = self.radar_hrv_features.get(name, 0) if hasattr(self, 'radar_hrv_features') else 0
                diff = abs(ecg_val - radar_val)
                
                tree.insert('', 'end', values=[
                    name, 
                    f"{ecg_val:.8f}", 
                    f"{radar_val:.8f}", 
                    f"{diff:.8f}"
                ])
            
            # Add scrollbar
            scrollbar = ttk.Scrollbar(table_window, orient="vertical", command=tree.yview)
            tree.configure(yscrollcommand=scrollbar.set)
            
            # Pack elements
            tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)
            scrollbar.pack(side=tk.RIGHT, fill=tk.Y, pady=10)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to create features table: {str(e)}")

    def copy_hrv_features_to_clipboard(self):
        """Copy enhanced HRV features to clipboard (19 features)"""
        try:
            if not (hasattr(self, 'ecg_hrv_features') or hasattr(self, 'radar_hrv_features')):
                messagebox.showwarning("Warning", "No HRV features extracted yet.")
                return
            
            # All 19 feature names (12 original + 7 new)
            feature_names = ['LF_power', 'HF_power', 'P1', 'P2', 'H1', 'H2', 'H3', 'H4', 
                            'SD1', 'SD2', 'SD1_SD2_ratio', 'SamEn',
                            'HR', 'RRmean', 'RRstd', 'HankDist_RRI', 'HankDist_dRR', 'CCM', 'AFE']
            
            # Create clipboard text for Excel
            if hasattr(self, 'ecg_hrv_features') and hasattr(self, 'radar_hrv_features'):
                # Both available
                clipboard_text = "Feature\tRadar\tECG\n"
                for name in feature_names:
                    radar_val = self.radar_hrv_features.get(name, 0)
                    ecg_val = self.ecg_hrv_features.get(name, 0)
                    clipboard_text += f"{name}\t{radar_val:.10f}\t{ecg_val:.10f}\n"
            elif hasattr(self, 'radar_hrv_features'):
                # Only Radar
                clipboard_text = "Feature\tRadar\n"
                for name in feature_names:
                    radar_val = self.radar_hrv_features.get(name, 0)
                    clipboard_text += f"{name}\t{radar_val:.10f}\n"
            else:
                # Only ECG
                clipboard_text = "Feature\tECG\n"
                for name in feature_names:
                    ecg_val = self.ecg_hrv_features.get(name, 0)
                    clipboard_text += f"{name}\t{ecg_val:.10f}\n"
            
            # Copy to clipboard
            self.root.clipboard_clear()
            self.root.clipboard_append(clipboard_text)
            
            messagebox.showinfo("Success", "Enhanced HRV features (19 total) copied to clipboard!")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to copy to clipboard: {str(e)}")

    def clear_hrv_feature_results(self):
        """Clear all HRV feature results and plots"""
        try:
            # Clear stored results
            if hasattr(self, 'ecg_hrv_features'):
                delattr(self, 'ecg_hrv_features')
            if hasattr(self, 'radar_hrv_features'):
                delattr(self, 'radar_hrv_features')
            if hasattr(self, 'ecg_rr_intervals'):
                delattr(self, 'ecg_rr_intervals') 
            if hasattr(self, 'radar_rr_intervals'):
                delattr(self, 'radar_rr_intervals')
            
            # Clear text display
            self.hrv_features_text.delete(1.0, tk.END)
            self.hrv_features_text.insert(tk.END, "HRV feature results cleared.\nExtract features to see new results.")
            
            # Reset plots
            self.initialize_hrv_feature_plots()
            
            # Reset status
            self.hrv_source_status.set("Ready for new feature extraction")
            
            messagebox.showinfo("Success", "HRV feature results cleared.")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to clear results: {str(e)}")

    def initialize_peak_plots(self):
        """Initialize empty peak detection plots including HRV plot"""
        self.ax_signal.text(0.5, 0.5, 'Load signal from EEMD tab first', 
                        ha='center', va='center', transform=self.ax_signal.transAxes)
        self.ax_signal.set_title('Signal for Peak Detection')
        
        self.ax_radar_peaks.text(0.5, 0.5, 'Radar peaks will appear here', 
                                ha='center', va='center', transform=self.ax_radar_peaks.transAxes)
        self.ax_radar_peaks.set_title('Radar Peak Detection (Sensors Algorithm)')
        
        self.ax_ecg_peaks.text(0.5, 0.5, 'ECG peaks will appear here', 
                            ha='center', va='center', transform=self.ax_ecg_peaks.transAxes)
        self.ax_ecg_peaks.set_title('ECG R-Peak Detection')
        
        self.ax_comparison.text(0.5, 0.5, 'Peak comparison will appear here', 
                            ha='center', va='center', transform=self.ax_comparison.transAxes)
        self.ax_comparison.set_title('Radar vs ECG Peak Comparison')
        
        # NEW: Initialize HRV plot
        self.ax_hrv.text(0.5, 0.5, 'HRV analysis will appear here\nDetect peaks first, then click "Analyze HRV"', 
                        ha='center', va='center', transform=self.ax_hrv.transAxes)
        self.ax_hrv.set_title('HRV Analysis: RR Intervals Comparison')
        
        self.fig_peaks.tight_layout()
        if hasattr(self, 'canvas_peaks'):
            self.canvas_peaks.draw()

    def plot_imfs_custom(self, ax, time, imfs, title):
        num_imfs = len(imfs)
        max_abs_val = max(np.max(np.abs(imf)) for imf in imfs)
        
        for i, imf in enumerate(imfs):
            normalized_imf = imf / max_abs_val
            offset = i * 2
            ax.plot(time, normalized_imf + offset, label=f'IMF {i+1}', linewidth=1)
        
        ax.set_title(title)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Normalized IMFs')
        ax.legend(loc='upper right')
        ax.grid(True)
        
        return ax

    def plot_imf_fft_analysis(self):
        """Plot FFT analysis for selected heart rate IMFs only with specific frequency analysis"""
        self.ax_imf_fft.clear()
        
        if not hasattr(self, 'heartrate_imfs') or len(self.heartrate_imfs) == 0:
            self.ax_imf_fft.text(0.5, 0.5, 'No Heart Rate IMFs available', 
                                ha='center', va='center', transform=self.ax_imf_fft.transAxes)
            return
        
        # Get selected IMFs only
        selected_imfs = []
        selected_indices = []
        for i in range(len(self.heartrate_imfs)):
            if hasattr(self, 'hr_imf_vars') and i < len(self.hr_imf_vars) and self.hr_imf_vars[i].get():
                selected_imfs.append(self.heartrate_imfs[i])
                selected_indices.append(i)
        
        if len(selected_imfs) == 0:
            self.ax_imf_fft.text(0.5, 0.5, 'No IMFs selected\nPlease select IMFs to analyze', 
                                ha='center', va='center', transform=self.ax_imf_fft.transAxes)
            self.update_fft_results_text("No IMFs selected for FFT analysis")
            return
        
        # Get sampling rate
        fs = self.signals['fs']
        
        # Color map for selected IMFs
        colors = plt.cm.tab10(np.linspace(0, 1, len(selected_imfs)))
        
        # Frequency range for analysis (0.5 to 8 Hz)
        freq_min, freq_max = 0.5, 8.0
        
        self.imf_frequency_data = []  # Store for popup table
        
        for idx, (imf, original_idx) in enumerate(zip(selected_imfs, selected_indices)):
            # Compute FFT
            fft_result = np.fft.fft(imf)
            freqs = np.fft.fftfreq(len(imf), 1/fs)
            
            # Take positive frequencies only
            pos_mask = (freqs > 0) & (freqs <= freq_max) & (freqs >= freq_min)
            freqs_pos = freqs[pos_mask]
            mags = np.abs(fft_result)[pos_mask]
            
            if len(mags) > 0:
                # Normalize magnitude for better visualization
                mags_normalized = mags / np.max(mags)
                
                # Plot FFT
                self.ax_imf_fft.plot(freqs_pos, mags_normalized, 
                                color=colors[idx], linewidth=2, 
                                label=f'IMF {original_idx+1}', alpha=0.8)
                
                # Find dominant frequencies (top 5)
                top_indices = np.argsort(mags)[-5:][::-1]  # Top 5 indices
                top_freqs = freqs_pos[top_indices]
                top_mags = mags[top_indices]
                top_mags_norm = mags_normalized[top_indices]
                
                # Calculate energy in different frequency bands
                band1_mask = (freqs_pos >= 0.5) & (freqs_pos < 1.0)   # 0.5-1.0 Hz
                band2_mask = (freqs_pos >= 1.0) & (freqs_pos < 2.0)   # 1.0-2.0 Hz
                band3_mask = (freqs_pos >= 2.0) & (freqs_pos < 3.0)   # 2.0-3.0 Hz
                band4_mask = (freqs_pos >= 3.0) & (freqs_pos < 4.0)   # 3.0-4.0 Hz
                band5_mask = (freqs_pos >= 4.0) & (freqs_pos <= 8.0)  # 4.0-8.0 Hz
                
                energy_band1 = np.sum(mags[band1_mask]**2) if np.any(band1_mask) else 0
                energy_band2 = np.sum(mags[band2_mask]**2) if np.any(band2_mask) else 0
                energy_band3 = np.sum(mags[band3_mask]**2) if np.any(band3_mask) else 0
                energy_band4 = np.sum(mags[band4_mask]**2) if np.any(band4_mask) else 0
                energy_band5 = np.sum(mags[band5_mask]**2) if np.any(band5_mask) else 0
                total_energy = np.sum(mags**2)
                
                # NEW: Calculate energy percentages at specific frequencies
                specific_freqs = [0.8, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
                specific_freq_data = self.calculate_specific_frequency_percentages(
                    freqs_pos, mags, specific_freqs, total_energy)
                
                # Store detailed information for popup table
                imf_data = {
                    'imf_num': original_idx + 1,
                    'peak_frequencies': top_freqs.tolist(),
                    'peak_magnitudes': top_mags.tolist(),
                    'peak_magnitudes_norm': top_mags_norm.tolist(),
                    'energy_0_5_1_0_Hz': energy_band1,
                    'energy_1_0_2_0_Hz': energy_band2,
                    'energy_2_0_3_0_Hz': energy_band3,
                    'energy_3_0_4_0_Hz': energy_band4,
                    'energy_4_0_8_0_Hz': energy_band5,
                    'total_energy': total_energy,
                    'energy_0_5_1_0_pct': (energy_band1/total_energy)*100 if total_energy > 0 else 0,
                    'energy_1_0_2_0_pct': (energy_band2/total_energy)*100 if total_energy > 0 else 0,
                    'energy_2_0_3_0_pct': (energy_band3/total_energy)*100 if total_energy > 0 else 0,
                    'energy_3_0_4_0_pct': (energy_band4/total_energy)*100 if total_energy > 0 else 0,
                    'energy_4_0_8_0_pct': (energy_band5/total_energy)*100 if total_energy > 0 else 0,
                    'mean_freq': np.mean(freqs_pos),
                    'median_freq': np.median(freqs_pos),
                    'freq_std': np.std(freqs_pos),
                    'bandwidth': np.max(freqs_pos) - np.min(freqs_pos) if len(freqs_pos) > 1 else 0,
                    'spectral_centroid': np.sum(freqs_pos * mags) / np.sum(mags) if np.sum(mags) > 0 else 0,
                    'spectral_rolloff': self.calculate_spectral_rolloff(freqs_pos, mags, 0.85),
                    # NEW: Add specific frequency data
                    'specific_frequencies': specific_freq_data
                }
                self.imf_frequency_data.append(imf_data)
                
                # Mark top 2 dominant frequencies on plot
                for i, (freq, mag_norm) in enumerate(zip(top_freqs[:2], top_mags_norm[:2])):
                    self.ax_imf_fft.plot(freq, mag_norm, 'o', color=colors[idx], 
                                    markersize=8, markeredgecolor='black', markeredgewidth=1)
                    # Add annotation for the highest peak only
                    if i == 0:
                        self.ax_imf_fft.annotate(f'{freq:.3f}Hz', 
                                            xy=(freq, mag_norm), xytext=(5, 5),
                                            textcoords='offset points', fontsize=8,
                                            bbox=dict(boxstyle='round,pad=0.3', facecolor=colors[idx], alpha=0.7))
        
        # Add vertical lines for specific frequencies
        specific_freqs = [0.8, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        colors_lines = ['red', 'orange', 'green', 'blue', 'purple', 'brown', 'pink', 'gray', 'olive']
        
        for freq, color in zip(specific_freqs, colors_lines):
            if freq_min <= freq <= freq_max:
                self.ax_imf_fft.axvline(x=freq, color=color, linestyle=':', alpha=0.6, linewidth=1)
        
        # Add legend for frequency lines
        self.ax_imf_fft.axvline(x=1.0, color='gray', linestyle='--', alpha=0.5, label='Reference Frequencies')
        
        self.ax_imf_fft.set_xlabel('Frequency (Hz)')
        self.ax_imf_fft.set_ylabel('Normalized Magnitude')
        self.ax_imf_fft.set_title(f'FFT Analysis of Selected Heart Rate IMFs ({len(selected_imfs)} selected)')
        self.ax_imf_fft.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        self.ax_imf_fft.grid(True, alpha=0.3)
        self.ax_imf_fft.set_xlim([freq_min, freq_max])
        
        # Update results text with simple summary
        self.update_fft_results_text_simple(selected_indices)
        
        # Redraw the canvas
        self.fig_eemd.tight_layout()
        if hasattr(self, 'canvas_eemd'):
            self.canvas_eemd.draw()

    def create_main_interface(self):
        self.root = tk.Tk()
        self.root.title("EEMD Radar Signal Analysis - Simplified Version")
        self.root.geometry("1800x1000")
        
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill=tk.BOTH, expand=True)
        
        # Main Analysis Tab (Keep as is)
        main_tab = ttk.Frame(notebook)
        notebook.add(main_tab, text="Main Analysis")
        self.create_main_analysis_tab(main_tab)
        
        # EEMD Tab with Segmentation
        eemd_tab = ttk.Frame(notebook)
        notebook.add(eemd_tab, text="EEMD Analysis")
        self.create_hr_eemd_tab(eemd_tab)
        
        # Peak Detection Tab
        peak_detection_tab = ttk.Frame(notebook)
        notebook.add(peak_detection_tab, text="Peak Detection")
        self.create_peak_detection_tab(peak_detection_tab)
        
        # NEW: HRV Feature Extraction Tab
        hrv_feature_tab = ttk.Frame(notebook)
        notebook.add(hrv_feature_tab, text="HRV Feature Extraction")
        self.create_hrv_feature_extraction_tab(hrv_feature_tab)
        
        self.root.mainloop()

    #RUN
    def run_analysis(self, file_path):
        try:
            print("Processing radar signals...")
            self.signals = self.process_radar_signals(file_path)
            
            print("Computing EEMD decomposition...")
            self.phase_imfs = self.eemd(self.signals['phase'])
            
            print("Computing heart rate EEMD...")
            self.heartrate_imfs = self.eemd(self.signals['heartrate'])
            
            print("Analyzing IMF frequencies...")
            self.imf_freq_results = self.analyze_imf_frequencies(self.phase_imfs, self.signals['fs'])
            
            print("Creating user interface...")
            self.create_main_interface()
            
        except Exception as e:
            print(f"Error in analysis: {e}")
            traceback.print_exc()

def main():
    try:
        default_file = 'AFsub_SE_4_20250529_093531_downsampled.h5'
        
        if not os.path.exists(default_file):
            print(f"Default file '{default_file}' not found.")
            print("Please select your HDF5 data file...")
            
            root = tk.Tk()
            root.withdraw()
            
            file_path = filedialog.askopenfilename(
                title="Select HDF5 Radar Data File",
                filetypes=[("HDF5 files", "*.h5"), ("All files", "*.*")]
            )
            
            root.destroy()
            
            if not file_path:
                print("No file selected. Exiting...")
                return
        else:
            file_path = default_file
        
        print(f"Processing file: {file_path}")
        
        analyzer = EEMDPaperAnalyzer()
        analyzer.run_analysis(file_path)
        
    except Exception as e:
        print(f"Error in main: {e}")
        traceback.print_exc()
        
        try:
            root = tk.Tk()
            root.withdraw()
            messagebox.showerror("Error", f"Analysis failed: {str(e)}")
            root.destroy()
        except:
            pass
        

if __name__ == "__main__":
    main()