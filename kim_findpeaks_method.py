import numpy as np
from scipy import signal
from scipy.fft import fft2 
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk, FigureCanvasTkAgg
from scipy.signal import butter, filtfilt, detrend, welch, savgol_filter, find_peaks, medfilt
from scipy import stats
from scipy.interpolate import interp1d
import h5py
import pandas as pd
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from matplotlib.figure import Figure
from matplotlib.widgets import SpanSelector
import traceback
import time
import os
import math

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
        
        # Paper method peak detection results
        self.paper_peaks = None
        self.paper_analysis_results = None


        # HRV Peak Editing variables
        self.hrv_editing_mode = False
        self.manual_radar_peaks = None
        self.hrv_edit_segment_start = 0
        self.hrv_edit_segment_duration = 30  # seconds
        self.edited_radar_peaks = None
        self.edited_radar_values = None

        self.latest_edited_rr_intervals = None
        self.latest_edited_rr_times = None
        self.edited_peaks_timestamp = None
        
        # HRV Feature extraction variables
        self.hrv_features_extracted = None
        self.hrv_rr_intervals = None

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
        start_time = max(imf_time[0], ecg_time[0])
        end_time = min(imf_time[-1], ecg_time[-1])
        
        if start_time >= end_time:
            return None, None, None, None
        
        common_time = np.linspace(start_time, end_time, int((end_time - start_time) * min(self.signals['fs'], self.ecg_fs)))
        
        imf_interp = np.interp(common_time, imf_time, imf_signal)
        ecg_interp = np.interp(common_time, ecg_time, ecg_signal)
        
        imf_interp = (imf_interp - np.mean(imf_interp)) / np.std(imf_interp)
        ecg_interp = (ecg_interp - np.mean(ecg_interp)) / np.std(ecg_interp)
        
        return imf_interp, ecg_interp, common_time, min(self.signals['fs'], self.ecg_fs)

    def calculate_correlation(self, signal1, signal2):
        correlation = np.corrcoef(signal1, signal2)[0, 1]
        cross_corr = np.correlate(signal1, signal2, mode='full')
        max_corr_idx = np.argmax(np.abs(cross_corr))
        max_correlation = cross_corr[max_corr_idx] / (len(signal1) * np.std(signal1) * np.std(signal2))
        
        return correlation, max_correlation

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
            
            # ADD THIS - Signal source selection
            source_frame = ttk.LabelFrame(left_panel, text="Signal Source Selection")
            source_frame.pack(fill=tk.X, padx=5, pady=5)

            self.use_derivative_var = tk.BooleanVar(value=False)
            ttk.Checkbutton(source_frame, text="Use Heart Rate Derivative", 
                        variable=self.use_derivative_var).pack(anchor=tk.W, padx=5, pady=2)

            ttk.Label(source_frame, text="If unchecked: Uses combined IMFs\nIf checked: Uses HR derivative from Main tab", 
                    font=("Arial", 8), foreground="gray").pack(padx=5, pady=2)

            # Separator
            ttk.Separator(left_panel, orient='horizontal').pack(fill=tk.X, padx=5, pady=5)
            # ADDED - MAV controls
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
            
            # MAV info and controls
            mav_info_frame = ttk.Frame(mav_frame)
            mav_info_frame.pack(fill=tk.X, padx=5, pady=2)
            
            info_label = ttk.Label(mav_frame, text="Recommended: 15-35 samples", 
                                font=("Arial", 8), foreground="gray")
            info_label.pack(padx=5, pady=1)
            
            ttk.Button(mav_frame, text="Compare Raw vs MAV", 
                    command=self.compare_raw_vs_mav).pack(fill=tk.X, padx=5, pady=2)
            
            # Button frame
            button_frame = ttk.Frame(left_panel)
            button_frame.pack(fill=tk.X, padx=5, pady=5)

            ttk.Button(button_frame, text="Combine IMFs", command=self.combine_hr_imfs).pack(fill=tk.X, padx=5, pady=2)
            ttk.Button(button_frame, text="Update FFT Analysis", command=self.plot_imf_fft_analysis).pack(fill=tk.X, padx=5, pady=2)

            # NEW: Show frequency table button (instead of inline table)
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
            
            # Plots - Updated to remove empty ECG plot and add FFT analysis
            self.fig_eemd = Figure(figsize=(9, 12))
            self.ax_imfs = self.fig_eemd.add_subplot(4, 1, 1)
            self.ax_combined = self.fig_eemd.add_subplot(4, 1, 2)
            self.ax_imf_fft = self.fig_eemd.add_subplot(4, 1, 3)  # FFT analysis
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
    
    # FINAL COMPLETE FIX - ADD ALL THESE FUNCTIONS TO EEMDPaperAnalyzer CLASS
    # AND REMOVE ANY OLD REFERENCES TO export_frequency_table

    # ENHANCED VERSION - REPLACE THE EXISTING FUNCTIONS WITH THESE:

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

    def create_paper_method_tab(self, parent):
        """Create Paper Method tab implementing steps 2-7 with all improvements"""
        def create_paper_content(scrollable_frame):
            main_frame = ttk.Frame(scrollable_frame)
            main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            left_panel = ttk.LabelFrame(main_frame, text="Paper Method Controls", width=300)  # Wider for new controls
            left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10, expand=False)
            left_panel.pack_propagate(False)
            
            # Information frame
            info_frame = ttk.LabelFrame(left_panel, text="Algorithm Information")
            info_frame.pack(fill=tk.X, expand=False, padx=5, pady=5)
            
            info_label = ttk.Label(info_frame, 
                                text="Steps 2-7 from:\n'Peak Detection Algorithm for\nVital Sign Detection Using\nDoppler Radar Sensors'\n(Kim et al., Sensors 2019)\n\nWith Improvements Applied", 
                                justify=tk.CENTER, foreground="darkblue")
            info_label.pack(padx=5, pady=5)
            
            # Algorithm parameters frame
            params_frame = ttk.LabelFrame(left_panel, text="Algorithm Parameters")
            params_frame.pack(fill=tk.X, expand=False, padx=5, pady=5)
            
            # Time window for mHR calculation (Step 6)
            window_frame = ttk.Frame(params_frame)
            window_frame.pack(fill=tk.X, padx=5, pady=2)
            ttk.Label(window_frame, text="Time Window T (s):").pack(side=tk.LEFT)
            self.time_window_var = tk.DoubleVar(value=60.0)
            window_entry = ttk.Entry(window_frame, textvariable=self.time_window_var, width=10)
            window_entry.pack(side=tk.RIGHT)
            
            # Maximum threshold Tth (Step 5) - UPDATED DEFAULT TO 1.2
            tth_frame = ttk.Frame(params_frame)
            tth_frame.pack(fill=tk.X, padx=5, pady=2)
            ttk.Label(tth_frame, text="Max Interval Tth (s):").pack(side=tk.LEFT)
            self.tth_var = tk.DoubleVar(value=1.2)  # CHANGED from 1.5 to 1.2
            tth_entry = ttk.Entry(tth_frame, textvariable=self.tth_var, width=10)
            tth_entry.pack(side=tk.RIGHT)
            
            # NEW: Threshold scaling factor (Improvement 1)
            alpha_frame = ttk.Frame(params_frame)
            alpha_frame.pack(fill=tk.X, padx=5, pady=2)
            ttk.Label(alpha_frame, text="Threshold Alpha:").pack(side=tk.LEFT)
            self.alpha_var = tk.DoubleVar(value=0.6)
            alpha_entry = ttk.Entry(alpha_frame, textvariable=self.alpha_var, width=10)
            alpha_entry.pack(side=tk.RIGHT)
            
            # NEW: Peak polarity selection (Improvement 2)
            polarity_frame = ttk.Frame(params_frame)
            polarity_frame.pack(fill=tk.X, padx=5, pady=2)
            ttk.Label(polarity_frame, text="Peak Polarity:").pack(side=tk.LEFT)
            self.polarity_var = tk.StringVar(value="both")
            polarity_combo = ttk.Combobox(polarity_frame, textvariable=self.polarity_var, 
                                        values=["both", "positive", "negative"], width=8)
            polarity_combo.pack(side=tk.RIGHT)
            
            # NEW: Restoration threshold factor (Improvement 3)
            restore_frame = ttk.Frame(params_frame)
            restore_frame.pack(fill=tk.X, padx=5, pady=2)
            ttk.Label(restore_frame, text="Restore Factor:").pack(side=tk.LEFT)
            self.restore_factor_var = tk.DoubleVar(value=0.5)
            restore_entry = ttk.Entry(restore_frame, textvariable=self.restore_factor_var, width=10)
            restore_entry.pack(side=tk.RIGHT)
            
            # Sampling rate for calculations
            fs_frame = ttk.Frame(params_frame)
            fs_frame.pack(fill=tk.X, padx=5, pady=2)
            ttk.Label(fs_frame, text="Sampling Rate (Hz):").pack(side=tk.LEFT)
            self.fs_display_var = tk.StringVar(value="Auto-detect")
            fs_label = ttk.Label(fs_frame, textvariable=self.fs_display_var)
            fs_label.pack(side=tk.RIGHT)
            
            # Buttons frame
            button_frame = ttk.Frame(left_panel)
            button_frame.pack(fill=tk.X, expand=False, padx=5, pady=10)
            
            detect_button = ttk.Button(button_frame, text="Run Algorithm", command=self.run_paper_algorithm)
            detect_button.pack(fill=tk.X, padx=5, pady=5)
            
            save_button = ttk.Button(button_frame, text="Save Results", command=self.save_paper_results)
            save_button.pack(fill=tk.X, padx=5, pady=5)
            
            # Results display
            results_frame = ttk.LabelFrame(left_panel, text="Algorithm Results")
            results_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
            
            self.paper_results_text = tk.Text(results_frame, width=45, height=25)
            paper_scrollbar = ttk.Scrollbar(results_frame, orient=tk.VERTICAL, command=self.paper_results_text.yview)
            self.paper_results_text.configure(yscrollcommand=paper_scrollbar.set)
            
            self.paper_results_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
            paper_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            
            # Right panel for plots
            right_panel = ttk.Frame(main_frame)
            right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            # Create figure with subplots for each step
            self.fig_paper = Figure(figsize=(9, 12))  # Tingkatkan tinggi dari 10 ke 12
            gs_paper = self.fig_paper.add_gridspec(7, 1, height_ratios=[1, 1, 1, 1, 1, 1, 1], hspace=0.9)
            
            self.ax_step1 = self.fig_paper.add_subplot(gs_paper[0])  # Input signal
            self.ax_step2 = self.fig_paper.add_subplot(gs_paper[1])  # Zero crossings & candidates
            self.ax_step3 = self.fig_paper.add_subplot(gs_paper[2])  # RMS threshold
            self.ax_step4 = self.fig_paper.add_subplot(gs_paper[3])  # Filtered peaks
            self.ax_step5 = self.fig_paper.add_subplot(gs_paper[4])  # Interval check
            self.ax_step6 = self.fig_paper.add_subplot(gs_paper[5])  # mHR calculation
            self.ax_step7 = self.fig_paper.add_subplot(gs_paper[6])  # Final results
            
            # Show placeholder text
            for i, ax in enumerate([self.ax_step1, self.ax_step2, self.ax_step3, self.ax_step4, 
                                   self.ax_step5, self.ax_step6, self.ax_step7]):
                ax.text(0.5, 0.5, f'Step {i+1}: Combine Heart Rate IMFs\nin EEMD tab then click "Run Algorithm"', 
                        ha='center', va='center', transform=ax.transAxes)
            
            self.canvas_paper = FigureCanvasTkAgg(self.fig_paper, master=right_panel)
            
            # Add toolbar
            paper_toolbar_frame = ttk.Frame(right_panel)
            paper_toolbar_frame.pack(side=tk.TOP, fill=tk.X)
            
            paper_toolbar = NavigationToolbar2Tk(self.canvas_paper, paper_toolbar_frame)
            paper_toolbar.update()
            
            self.canvas_paper.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        self.create_scrollable_tab(parent, create_paper_content)
    
    def synchronize_peak_times(self, radar_peak_times, ecg_peak_times, max_time_diff=2.0):
        # Find overlapping time range
        radar_start, radar_end = radar_peak_times[0], radar_peak_times[-1]
        ecg_start, ecg_end = ecg_peak_times[0], ecg_peak_times[-1]
        
        overlap_start = max(radar_start, ecg_start)
        overlap_end = min(radar_end, ecg_end)
        
        if overlap_end <= overlap_start:
            print("Warning: No time overlap between radar and ECG data")
            return radar_peak_times, ecg_peak_times, 0.0
        
        # Filter peaks to overlapping time range
        radar_mask = (radar_peak_times >= overlap_start) & (radar_peak_times <= overlap_end)
        ecg_mask = (ecg_peak_times >= overlap_start) & (ecg_peak_times <= overlap_end)
        
        radar_sync = radar_peak_times[radar_mask]
        ecg_sync = ecg_peak_times[ecg_mask]
        
        # Estimate time offset using cross-correlation of peak count histograms
        if len(radar_sync) > 5 and len(ecg_sync) > 5:
            # Create time bins for histogram-based correlation
            time_bins = np.linspace(overlap_start, overlap_end, 100)
            
            radar_hist, _ = np.histogram(radar_sync, bins=time_bins)
            ecg_hist, _ = np.histogram(ecg_sync, bins=time_bins)
            
            # Cross-correlation to find time offset
            correlation = np.correlate(radar_hist, ecg_hist, mode='full')
            offset_bins = correlation.argmax() - (len(ecg_hist) - 1)
            
            # Convert bin offset to time offset
            bin_width = (overlap_end - overlap_start) / (len(time_bins) - 1)
            time_offset = offset_bins * bin_width
            
            # Apply offset correction if reasonable
            if abs(time_offset) <= max_time_diff:
                radar_sync = radar_sync - time_offset
            else:
                time_offset = 0.0
                print(f"Warning: Calculated time offset {time_offset:.3f}s exceeds maximum {max_time_diff}s, not applied")
        else:
            time_offset = 0.0
            print("Warning: Insufficient peaks for time offset calculation")
        
        return radar_sync, ecg_sync, time_offset
    
    #
    def run_paper_algorithm(self):
        """Run the complete paper algorithm steps 2-7 with wave consolidation and adaptive RR validation"""
        if self.latest_combined_imf is None:
            messagebox.showwarning("Warning", "No combined Heart Rate IMF available. Please combine IMFs first in the Heart Rate EEMD tab.")
            return
        
        # Clear results
        self.paper_results_text.delete(1.0, tk.END)
        for ax in [self.ax_step1, self.ax_step2, self.ax_step3, self.ax_step4, 
                self.ax_step5, self.ax_step6, self.ax_step7]:
            ax.clear()
        
        try:
            # Get input signal and parameters
            signal = self.latest_combined_imf.copy()
            time_data = self.signals['timestamps']
            fs = self.signals['fs']
            self.fs_display_var.set(f"{fs:.1f}")
            time_window = self.time_window_var.get()
            tth_seconds = self.tth_var.get()
            alpha = self.alpha_var.get()
            polarity = self.polarity_var.get()
            restore_factor = self.restore_factor_var.get()
            
            # Check signal source
            if hasattr(self, 'use_derivative_var') and self.use_derivative_var.get():
                signal_source_info = "Heart Rate Derivative (from Main Analysis)"
            else:
                signal_source_info = "Combined Heart Rate IMFs"
            
            # Store parameters
            self.paper_results_text.insert(tk.END, "PAPER ALGORITHM IMPLEMENTATION\n")
            self.paper_results_text.insert(tk.END, "=" * 50 + "\n")
            self.paper_results_text.insert(tk.END, "Reference: Kim et al., Sensors 2019, 19, 1575\n")
            self.paper_results_text.insert(tk.END, "Steps 2-7 with Wave Consolidation + Adaptive RR Validation\n\n")
            self.paper_results_text.insert(tk.END, f"SIGNAL SOURCE: {signal_source_info}\n")
            
            self.paper_results_text.insert(tk.END, f"PARAMETERS:\n")
            self.paper_results_text.insert(tk.END, f"Time Window T: {time_window} s\n")
            self.paper_results_text.insert(tk.END, f"Max Interval Tth: {tth_seconds} s\n")
            self.paper_results_text.insert(tk.END, f"Sampling Rate: {fs} Hz\n")
            self.paper_results_text.insert(tk.END, f"Signal Length: {len(signal)} samples\n\n")
            
            # Step 1: Plot input signal
            if hasattr(self, 'use_derivative_var') and self.use_derivative_var.get():
                signal_color = 'red'
                signal_title = 'Step 1: Heart Rate Derivative Signal'
            else:
                signal_color = 'blue'
                signal_title = 'Step 1: Combined IMF Signal'

            self.ax_step1.plot(time_data, signal, color=signal_color, linewidth=1)
            self.ax_step1.axhline(y=0, color='k', linestyle='--', alpha=0.5)
            self.ax_step1.set_title(signal_title)
            self.ax_step1.set_xlabel('Time (s)')
            self.ax_step1.set_ylabel('Amplitude')
            self.ax_step1.grid(True)
            
            # Step 2: Zero-Crossing Detection and Candidate Peak Finding
            zero_crossings, candidate_peaks, candidate_values = self.step2_zero_crossing_detection(signal, polarity)
            
            self.paper_results_text.insert(tk.END, f"STEP 2 RESULTS:\n")
            self.paper_results_text.insert(tk.END, f"Zero crossings found: {len(zero_crossings)}\n")
            self.paper_results_text.insert(tk.END, f"Candidate peaks found: {len(candidate_peaks)}\n")
            self.paper_results_text.insert(tk.END, f"Peak polarity: {polarity}\n\n")
            
            # Plot Step 2
            self.ax_step2.plot(time_data, signal, 'b-', linewidth=1, alpha=0.7, label='Signal')
            self.ax_step2.plot(time_data[zero_crossings], signal[zero_crossings], 'ko', 
                            markersize=3, alpha=0.6, label=f'Zero Crossings ({len(zero_crossings)})')
            self.ax_step2.plot(time_data[candidate_peaks], candidate_values, 'ro', 
                            markersize=5, label=f'Candidate Peaks ({len(candidate_peaks)})')
            self.ax_step2.axhline(y=0, color='k', linestyle='--', alpha=0.5)
            self.ax_step2.set_title('Step 2: Zero-Crossing Detection & Candidate Peaks')
            self.ax_step2.set_xlabel('Time (s)')
            self.ax_step2.set_ylabel('Amplitude')
            self.ax_step2.legend()
            self.ax_step2.grid(True)
            
            # Step 3: Compute RMS-based Threshold
            vth = self.step3_compute_threshold(candidate_values, alpha)
            
            self.paper_results_text.insert(tk.END, f"STEP 3 RESULTS:\n")
            self.paper_results_text.insert(tk.END, f"Computed threshold Vth: {vth:.6f}\n")
            self.paper_results_text.insert(tk.END, f"Alpha scaling factor: {alpha}\n\n")
            
            # Plot Step 3
            self.ax_step3.plot(time_data, signal, 'b-', linewidth=1, alpha=0.7, label='Signal')
            self.ax_step3.plot(time_data[candidate_peaks], candidate_values, 'ro', 
                            markersize=5, alpha=0.7, label=f'Candidate Peaks')
            self.ax_step3.axhline(y=vth, color='red', linestyle='-', linewidth=2, 
                                label=f'Threshold Vth = {vth:.4f}')
            self.ax_step3.axhline(y=-vth, color='red', linestyle='-', linewidth=2, alpha=0.7)
            self.ax_step3.axhline(y=0, color='k', linestyle='--', alpha=0.5)
            
            above_threshold = np.abs(candidate_values) >= vth
            self.ax_step3.plot(time_data[candidate_peaks[above_threshold]], 
                            candidate_values[above_threshold], 'go', 
                            markersize=6, label=f'Above Threshold ({np.sum(above_threshold)})')
            
            self.ax_step3.set_title('Step 3: RMS-based Threshold Calculation')
            self.ax_step3.set_xlabel('Time (s)')
            self.ax_step3.set_ylabel('Amplitude')
            self.ax_step3.legend()
            self.ax_step3.grid(True)
            
            # Step 4: Filter by Threshold + Wave Consolidation
            filtered_peaks, filtered_values = self.step4_filter_by_threshold(candidate_peaks, candidate_values, vth)
            consolidated_peaks, consolidated_values = self.step4b_consolidate_waves(filtered_peaks, filtered_values, time_data, fs)
            
            # NEW Step 4C: Adaptive RR Interval Validation

            final_peaks, final_values, rr_correction_stats = self.step4c_comprehensive_rr_correction_ultra_adaptive(
                consolidated_peaks, consolidated_values, candidate_peaks, candidate_values, time_data, signal)
            # Set for compatibility with existing code
            restored_count = rr_correction_stats.get('total_added', 0)
            
            self.paper_results_text.insert(tk.END, f"STEP 4 RESULTS:\n")
            self.paper_results_text.insert(tk.END, f"Peaks before filtering: {len(candidate_peaks)}\n")
            self.paper_results_text.insert(tk.END, f"After threshold filtering: {len(filtered_peaks)}\n")
            self.paper_results_text.insert(tk.END, f"After wave consolidation: {len(consolidated_peaks)}\n")
            self.paper_results_text.insert(tk.END, f"After comprehensive correction: {len(final_peaks)}\n")
            self.paper_results_text.insert(tk.END, f"Peaks removed (too close): {rr_correction_stats.get('total_removed', 0)}\n")
            self.paper_results_text.insert(tk.END, f"Peaks added (gaps filled): {rr_correction_stats.get('total_added', 0)}\n")
            
            # Plot Step 4 with all stages
            self.ax_step4.plot(time_data, signal, 'b-', linewidth=1, alpha=0.7, label='Signal')
            self.ax_step4.plot(time_data[filtered_peaks], filtered_values, 'orange', marker='o',
                            markersize=4, alpha=0.6, linestyle='None', label=f'Threshold Filtered ({len(filtered_peaks)})')
            self.ax_step4.plot(time_data[consolidated_peaks], consolidated_values, 'cyan', marker='s',
                            markersize=5, alpha=0.7, linestyle='None', label=f'Wave Consolidated ({len(consolidated_peaks)})')
            self.ax_step4.plot(time_data[final_peaks], final_values, 'go', 
                            markersize=6, label=f'Comprehensive Corrected ({len(final_peaks)})')
            self.ax_step4.axhline(y=0, color='k', linestyle='--', alpha=0.5)
            self.ax_step4.set_title('Step 4: Threshold + Wave Consolidation + Adaptive RR Validation')
            self.ax_step4.set_xlabel('Time (s)')
            self.ax_step4.set_ylabel('Amplitude')
            self.ax_step4.legend()
            self.ax_step4.grid(True)
            
            # Continue with Step 5 using RR-validated peaks
            f'''inal_peaks, final_values, restored_count, restoration_stats = self.step5_interval_check(
                rr_validated_peaks, rr_validated_values, candidate_peaks, candidate_values, 
                time_data, tth_seconds, vth, restore_factor)'''
                
            #
            self.paper_results_text.insert(tk.END, f"COMPREHENSIVE RR CORRECTION:\n")
            # Safe access to handle both old and new stat formats
            baseline_rr = rr_correction_stats.get('rr_mean', rr_correction_stats.get('baseline_rr', 0.8))
            iterations = rr_correction_stats.get('iterations', 5)
            removed = rr_correction_stats.get('total_removed', 0)
            added = rr_correction_stats.get('total_added', 0)

            self.paper_results_text.insert(tk.END, f"Baseline RR: {baseline_rr:.3f}s\n")
            self.paper_results_text.insert(tk.END, f"Iterations: {iterations}\n") 
            self.paper_results_text.insert(tk.END, f"Removed (close): {removed}\n")
            self.paper_results_text.insert(tk.END, f"Added (gaps): {added}\n")
            self.paper_results_text.insert(tk.END, f"Final peaks: {len(final_peaks)}\n\n")
            # Plot Step 5 (now shows comprehensive correction results)
            self.ax_step5.plot(time_data, signal, 'b-', linewidth=1, alpha=0.7, label='Signal')
            self.ax_step5.plot(time_data[consolidated_peaks], consolidated_values, 'cyan', marker='s',
                            markersize=5, alpha=0.7, linestyle='None', label='Before Correction')
            self.ax_step5.plot(time_data[final_peaks], final_values, 'ro', 
                            markersize=6, label=f'After Comprehensive Correction ({len(final_peaks)})')
            self.ax_step5.axhline(y=0, color='k', linestyle='--', alpha=0.5)
            self.ax_step5.set_title('Comprehensive RR Interval Correction')
            self.ax_step5.set_xlabel('Time (s)')
            self.ax_step5.set_ylabel('Amplitude')
            self.ax_step5.legend()
            self.ax_step5.grid(True)
            
            # Step 6: Compute Mean Heart Rate
            mhr_values, mhr_times = self.step6_compute_mhr(final_peaks, time_data, time_window)
            
            self.paper_results_text.insert(tk.END, f"STEP 6 RESULTS:\n")
            self.paper_results_text.insert(tk.END, f"mHR calculations: {len(mhr_values)}\n")
            if len(mhr_values) > 0:
                self.paper_results_text.insert(tk.END, f"Mean mHR: {np.mean(mhr_values):.1f} BPM\n")
                self.paper_results_text.insert(tk.END, f"mHR range: {np.min(mhr_values):.1f} - {np.max(mhr_values):.1f} BPM\n\n")
            
            # Plot Step 6
            if len(mhr_values) > 0:
                self.ax_step6.plot(mhr_times, mhr_values, 'b.-', markersize=6, linewidth=2, label='mHR')
                self.ax_step6.axhline(y=60, color='gray', linestyle='--', alpha=0.5)
                self.ax_step6.axhline(y=100, color='gray', linestyle='--', alpha=0.5)
                self.ax_step6.fill_between(mhr_times, 60, 100, alpha=0.1, color='green', label='Normal Range')
                self.ax_step6.legend()
            
            self.ax_step6.set_title('Step 6: Mean Heart Rate (mHR) Calculation')
            self.ax_step6.set_xlabel('Time (s)')
            self.ax_step6.set_ylabel('Heart Rate (BPM)')
            self.ax_step6.grid(True)
            self.ax_step6.set_ylim([40, 120])
            
            # Step 7: Output Results
            self.step7_output_results(final_peaks, final_values, time_data, mhr_values, mhr_times)
            
            # Plot Step 7
            self.ax_step7.plot(time_data, signal, 'b-', linewidth=1, alpha=0.7, label='Input Signal')
            self.ax_step7.plot(time_data[final_peaks], final_values, 'ro', 
                            markersize=8, label=f'Final Peaks ({len(final_peaks)})')
            
            for i, (peak_idx, peak_val) in enumerate(zip(final_peaks[:5], final_values[:5])):
                self.ax_step7.annotate(f'P{i+1}', 
                                    xy=(time_data[peak_idx], peak_val),
                                    xytext=(5, 10), textcoords='offset points',
                                    fontsize=8, 
                                    bbox=dict(facecolor='yellow', alpha=0.7))
            
            self.ax_step7.axhline(y=0, color='k', linestyle='--', alpha=0.5)
            self.ax_step7.set_title('Step 7: Final Results with Adaptive RR Validation')
            self.ax_step7.set_xlabel('Time (s)')
            self.ax_step7.set_ylabel('Amplitude')
            self.ax_step7.legend()
            self.ax_step7.grid(True)
            
            # Store results with RR validation info
            self.paper_analysis_results = {
                'final_peaks': final_peaks,
                'final_values': final_values,
                'peak_times': time_data[final_peaks],
                'mhr_values': mhr_values,
                'mhr_times': mhr_times,
                'mean_mhr': np.mean(mhr_values) if len(mhr_values) > 0 else 0,
                
                # Store all intermediate results
                'step3_vth': vth,
                'step3_candidate_peaks': candidate_peaks,
                'step3_candidate_values': candidate_values,
                
                'step4_peaks': filtered_peaks,
                'step4_values': filtered_values,
                'step4_peak_times': time_data[filtered_peaks],
                
                'step4b_consolidated_peaks': consolidated_peaks,
                'step4b_consolidated_values': consolidated_values,
                'step4b_consolidated_peak_times': time_data[consolidated_peaks],
                
                # NEW: Store RR validation results
                'step4c_comprehensive_peaks': final_peaks,
                'step4c_comprehensive_values': final_values,
                'step4c_comprehensive_peak_times': time_data[final_peaks],
                'step4c_comprehensive_stats': rr_correction_stats,
                
                'step5_peaks': final_peaks,
                'step5_values': final_values,
                'step5_peak_times': time_data[final_peaks],
                'step5_restored_count': restored_count,
                
                'step6_mhr_values': mhr_values,
                'step6_mhr_times': mhr_times,
                'step6_time_window': time_window,
                
                'parameters': {
                    'time_window': time_window,
                    'tth_seconds': tth_seconds,
                    'alpha': alpha,
                    'polarity': polarity,
                    'restore_factor': restore_factor,
                    'vth': vth,
                    'sampling_rate': fs,
                    'wave_consolidation_applied': True,
                    'adaptive_rr_validation_applied': True
                }
            }
            
            self.fig_paper.tight_layout()
            self.canvas_paper.draw()
            # PERBAIKAN KRITIS: Simpan hasil ke paper_analysis_results
            self.paper_analysis_results = {
                'final_peaks': final_peaks,
                'final_values': final_values,
                'peak_times': time_data[final_peaks],
                'mhr_values': mhr_values,
                'mhr_times': mhr_times,
                'signal_source': signal_source_info,
                'parameters': {
                    'time_window': time_window,
                    'tth_seconds': tth_seconds,
                    'alpha': alpha,
                    'polarity': polarity,
                    'fs': fs
                },
                # Hasil setiap step untuk referensi
                'step2_candidate_peaks': candidate_peaks,
                'step2_candidate_values': candidate_values,
                'step3_threshold': vth,
                'step4_filtered_peaks': filtered_peaks,
                'step4_consolidated_peaks': consolidated_peaks,
                'step5_corrected_peaks': final_peaks,
                'step5_correction_stats': rr_correction_stats
            }
        except Exception as e:
            messagebox.showerror("Error", f"Algorithm failed: {str(e)}")
            print(f"Algorithm error: {e}")
            traceback.print_exc()

    def step2_zero_crossing_detection(self, signal, polarity="both"):
        """Step 2: Zero-crossing detection and candidate peak finding with polarity selection"""
        zero_crossings = []
        candidate_peaks = []
        candidate_values = []

        # Find zero crossings
        for i in range(1, len(signal)):
            if (signal[i - 1] <= 0 and signal[i] > 0) or (signal[i - 1] >= 0 and signal[i] < 0):
                zero_crossings.append(i)

        zero_crossings = np.array(zero_crossings)

        # Between each pair of zero crossings, find local extrema
        for i in range(len(zero_crossings) - 1):
            start_idx = zero_crossings[i]
            end_idx = zero_crossings[i + 1]

            if end_idx - start_idx > 1:
                segment = signal[start_idx:end_idx + 1]
                
                # Find both maxima and minima
                max_val = np.max(segment)
                min_val = np.min(segment)
                max_idx = np.argmax(segment)
                min_idx = np.argmin(segment)
                
                # Determine which peaks to include based on polarity setting
                peaks_to_add = []
                
                if polarity in ["both", "positive"] and max_val > 0:
                    # Add positive peak
                    peak_idx = start_idx + max_idx
                    peaks_to_add.append((peak_idx, max_val))
                
                if polarity in ["both", "negative"] and min_val < 0:
                    # Add negative peak (use absolute value for consistency)
                    peak_idx = start_idx + min_idx
                    peaks_to_add.append((peak_idx, abs(min_val)))
                
                # If both positive and negative peaks in same segment, choose the larger magnitude
                if polarity == "both" and len(peaks_to_add) == 2:
                    if abs(max_val) >= abs(min_val):
                        peaks_to_add = [(start_idx + max_idx, max_val)]
                    else:
                        peaks_to_add = [(start_idx + min_idx, abs(min_val))]
                
                # Add selected peaks
                for peak_idx, peak_val in peaks_to_add:
                    candidate_peaks.append(peak_idx)
                    candidate_values.append(peak_val)

        return zero_crossings, np.array(candidate_peaks), np.array(candidate_values)
        
    def step3_compute_threshold(self, candidate_values, alpha=0.5):
        """Step 3: Compute RMS-based threshold with scaling factor"""
        if len(candidate_values) == 0:
            return 0
        
        n = len(candidate_values)
        # Vth = α × (1/2n) * Σ|Vmag(i)|/√2
        vth_base = (1.0 / (2.0 * n)) * np.sum(np.abs(candidate_values) / np.sqrt(2))
        vth = alpha * vth_base
        
        return vth

    def step4_filter_by_threshold(self, candidate_peaks, candidate_values, vth):
        """Step 4: Filter candidate peaks by threshold"""
        # Keep peaks with magnitude >= vth
        mask = np.abs(candidate_values) >= vth
        filtered_peaks = candidate_peaks[mask]
        filtered_values = candidate_values[mask]
        
        return filtered_peaks, filtered_values

    #
    def step5_interval_check(self, filtered_peaks, filtered_values, candidate_peaks, candidate_values, 
                         time_data, tth_seconds, vth, restore_factor=0.5):
        if len(filtered_peaks) < 2:
            return filtered_peaks, filtered_values, 0
        
        # STEP 1: Calculate adaptive RR constraints for missed peak detection
        rr_mean, restoration_stats = self.calculate_rr_mean_for_restoration(filtered_peaks, time_data)
        missed_peak_threshold = 1.5 * rr_mean  # If gap > 1.5 × RR_mean, look for missed peak
        
        # STEP 2: Original Step 5 logic (preserved exactly)
        final_peaks = []
        final_values = []
        restored_count = 0
        
        final_peaks.append(filtered_peaks[0])
        final_values.append(filtered_values[0])
        
        for i in range(1, len(filtered_peaks)):
            current_peak = filtered_peaks[i]
            current_value = filtered_values[i]
            previous_peak = final_peaks[-1]
            
            # Calculate time interval
            time_interval = time_data[current_peak] - time_data[previous_peak]
            
            # ORIGINAL LOGIC: Check against Tth threshold (Kim et al. method)
            if time_interval > tth_seconds:
                # Interval too large - try to restore a candidate peak between them
                between_mask = (candidate_peaks > previous_peak) & (candidate_peaks < current_peak)
                between_candidates = candidate_peaks[between_mask]
                between_values = candidate_values[between_mask]
                
                if len(between_candidates) > 0:
                    # Filter candidates by magnitude threshold: must be > restore_factor × Vth
                    restore_threshold = restore_factor * vth
                    valid_candidates_mask = np.abs(between_values) > restore_threshold
                    
                    if np.any(valid_candidates_mask):
                        valid_candidates = between_candidates[valid_candidates_mask]
                        valid_values = between_values[valid_candidates_mask]
                        
                        # Among valid candidates, find the one with largest magnitude
                        best_idx = np.argmax(np.abs(valid_values))
                        restored_peak = valid_candidates[best_idx]
                        restored_value = valid_values[best_idx]
                        
                        final_peaks.append(restored_peak)
                        final_values.append(restored_value)
                        restored_count += 1
            
            # NEW LOGIC: Adaptive missed peak restoration
            # Check if we should look for missed peaks even when interval < tth_seconds
            elif time_interval > missed_peak_threshold:
                # Gap suggests a missed peak - attempt adaptive restoration
                missed_peak_result = self.restore_missed_peak_adaptive(
                    previous_peak, current_peak, candidate_peaks, candidate_values, 
                    time_data, rr_mean, vth)
                
                if missed_peak_result['peak_restored']:
                    final_peaks.append(missed_peak_result['restored_peak'])
                    final_values.append(missed_peak_result['restored_value'])
                    restored_count += 1
            
            # Always add the current peak after any restoration
            final_peaks.append(current_peak)
            final_values.append(current_value)
        
        # STEP 3: Enhanced statistics including adaptive restoration info
        enhanced_restoration_stats = {
            'original_restored_count': restored_count,
            'rr_mean_used': rr_mean,
            'missed_peak_threshold': missed_peak_threshold,
            'restoration_method': 'enhanced_adaptive',
            'rr_stability': restoration_stats
        }
        
        return np.array(final_peaks), np.array(final_values), restored_count, enhanced_restoration_stats

    def step6_compute_mhr(self, final_peaks, time_data, time_window):
        """Step 6: Compute mean heart rate (mHR)"""
        if len(final_peaks) < 2:
            return np.array([]), np.array([])
        
        mhr_values = []
        mhr_times = []
        
        total_duration = time_data[-1] - time_data[0]
        
        # Sliding window approach
        current_time = time_data[0] + time_window / 2
        
        while current_time + time_window / 2 <= time_data[-1]:
            window_start = current_time - time_window / 2
            window_end = current_time + time_window / 2
            
            # Count peaks in this window
            window_mask = (time_data[final_peaks] >= window_start) & (time_data[final_peaks] <= window_end)
            nd = np.sum(window_mask)
            
            if nd >= 2:  # Need at least 2 peaks for meaningful HR
                # mHR = (nd / T) × 60
                mhr = (nd / time_window) * 60
                mhr_values.append(mhr)
                mhr_times.append(current_time)
            
            current_time += time_window / 4  # Move window by quarter window
        
        return np.array(mhr_values), np.array(mhr_times)

    def step7_output_results(self, final_peaks, final_values, time_data, mhr_values, mhr_times):
        """Step 7: Output results"""
        self.paper_results_text.insert(tk.END, f"STEP 7 - FINAL RESULTS:\n")
        self.paper_results_text.insert(tk.END, "=" * 30 + "\n")
        
        self.paper_results_text.insert(tk.END, f"Total valid peaks: {len(final_peaks)}\n")
        self.paper_results_text.insert(tk.END, f"Peak detection rate: {len(final_peaks)/(time_data[-1]-time_data[0]):.2f} peaks/s\n")
        
        if len(mhr_values) > 0:
            self.paper_results_text.insert(tk.END, f"Mean Heart Rate: {np.mean(mhr_values):.1f} ± {np.std(mhr_values):.1f} BPM\n")
            self.paper_results_text.insert(tk.END, f"mHR Range: {np.min(mhr_values):.1f} - {np.max(mhr_values):.1f} BPM\n")
        else:
            self.paper_results_text.insert(tk.END, f"Mean Heart Rate: Not calculable (insufficient peaks)\n")
        
        self.paper_results_text.insert(tk.END, f"\nFIRST 10 PEAKS:\n")
        self.paper_results_text.insert(tk.END, f"{'#':<3} {'Index':<6} {'Time(s)':<8} {'Value':<10}\n")
        self.paper_results_text.insert(tk.END, "-" * 30 + "\n")
        
        for i in range(min(10, len(final_peaks))):
            peak_idx = final_peaks[i]
            peak_time = time_data[peak_idx]
            peak_val = final_values[i]
            self.paper_results_text.insert(tk.END, f"{i+1:<3} {peak_idx:<6} {peak_time:<8.3f} {peak_val:<10.4f}\n")

    def save_paper_results(self):
        """Save paper method results"""
        if not hasattr(self, 'paper_analysis_results'):
            messagebox.showwarning("Warning", "No results to save. Please run algorithm first.")
            return
        
        # Save figure
        fig_file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("All files", "*.*")],
            title="Save Paper Algorithm Results"
        )
        if fig_file_path:
            self.fig_paper.savefig(fig_file_path, dpi=300, bbox_inches='tight')
        
        # Save data
        data_file_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            title="Save Peak Data"
        )
        if data_file_path:
            results = self.paper_analysis_results
            df = pd.DataFrame({
                'Peak_Value': results['final_values']
            })
            
            # Add mHR data if available
            if len(results['mhr_values']) > 0:
                max_len = max(len(results['final_peaks']), len(results['mhr_values']))
                
                # Pad shorter arrays with NaN
                peak_data_padded = np.full(max_len, np.nan)
                peak_time_padded = np.full(max_len, np.nan)
                peak_value_padded = np.full(max_len, np.nan)
                mhr_time_padded = np.full(max_len, np.nan)
                mhr_value_padded = np.full(max_len, np.nan)
                
                peak_data_padded[:len(results['final_peaks'])] = results['final_peaks']
                peak_time_padded[:len(results['peak_times'])] = results['peak_times']
                peak_value_padded[:len(results['final_values'])] = results['final_values']
                mhr_time_padded[:len(results['mhr_times'])] = results['mhr_times']
                mhr_value_padded[:len(results['mhr_values'])] = results['mhr_values']
                
                df = pd.DataFrame({
                    'Peak_Index': peak_data_padded,
                    'Peak_Time_s': peak_time_padded,
                    'Peak_Value': peak_value_padded,
                    'mHR_Time_s': mhr_time_padded,
                    'mHR_BPM': mhr_value_padded
                })
            
            df.to_csv(data_file_path, index=False)
            messagebox.showinfo("Save Complete", f"Results saved to:\n{fig_file_path}\n{data_file_path}")

    def reset_paper_parameters(self):
        """Reset paper method parameters to default values"""
        self.time_window_var.set(60.0)
        self.tth_var.set(1.5)

    def apply_paper_method(self):
        """Legacy method - redirect to new implementation"""
        self.run_paper_algorithm()

    def save_paper_analysis(self):
        """Legacy method - redirect to new implementation"""
        self.save_paper_results()

    #
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
        if hasattr(self, 'enable_mav_var') and self.enable_mav_var.get():
            mav_window = self.mav_window_var.get() if hasattr(self, 'mav_window_var') else 25
            if len(combined_imf_raw) >= mav_window and mav_window > 0:
                self.latest_combined_imf = self.apply_moving_average(combined_imf_raw, mav_window)
                mav_applied = True
                mav_info = f"Applied (Window={mav_window})"
            else:
                self.latest_combined_imf = combined_imf_raw
                mav_applied = False
                mav_info = "Skipped (invalid window or signal too short)"
        else:
            # Default MAV window for backward compatibility
            mav_window = 25
            if len(combined_imf_raw) >= mav_window:
                self.latest_combined_imf = self.apply_moving_average(combined_imf_raw, mav_window)
                mav_applied = True
                mav_info = f"Applied (Default Window={mav_window})"
            else:
                self.latest_combined_imf = combined_imf_raw
                mav_applied = False
                mav_info = "Skipped (signal too short)"
        
        # Clear and update the plot
        self.ax_combined.clear()
        
        # Plot both raw combined and MAV-filtered for comparison
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
                self.ax_combined.plot(self.signals['timestamps'], combined_imf_raw, 'b-', 
                                    linewidth=1, alpha=0.5, label='Raw Combined IMFs')
                self.ax_combined.plot(self.signals['timestamps'], self.latest_combined_imf, 'g-', 
                                    linewidth=1.5, label=f'MAV Filtered (Window={mav_window})')
                self.ax_combined.set_title('Combined Heart Rate IMFs with Moving Average Filter')
            else:
                self.ax_combined.plot(self.signals['timestamps'], self.latest_combined_imf, 'g-', 
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
        """Compare combined IMFs with ECG"""
        if not self.ecg_loaded:
            messagebox.showwarning("Warning", "Load ECG data first")
            return
        
        if self.latest_combined_imf is None:
            messagebox.showwarning("Warning", "Combine IMFs first")
            return
        
        imf_sync, ecg_sync, common_time, _ = self.synchronize_signals(
            self.latest_combined_imf, self.signals['timestamps'], self.ecg_data, self.ecg_time)
        
        if imf_sync is None:
            messagebox.showerror("Error", "No time overlap")
            return
        
        correlation, max_correlation = self.calculate_correlation(imf_sync, ecg_sync)
        
        self.ax_imf_compare.clear()
        
        self.ax_imf_compare.plot(common_time, imf_sync, 'g-', label='Combined IMFs', alpha=0.7)
        self.ax_imf_compare.plot(common_time, ecg_sync, 'r-', label='ECG', alpha=0.7)
        self.ax_imf_compare.set_title(f'IMFs vs ECG')
        self.ax_imf_compare.set_xlabel('Time (s)')
        self.ax_imf_compare.set_ylabel('Normalized Amplitude')
        self.ax_imf_compare.legend()
        self.ax_imf_compare.grid(True)
        
        self.eemd_results.insert(tk.END, f"IMF-ECG Correlation Results:\n")
        self.eemd_results.insert(tk.END, f"Pearson Correlation: {correlation:.3f}\n")
        self.eemd_results.insert(tk.END, f"Max Cross-correlation: {max_correlation:.3f}\n")
        self.eemd_results.insert(tk.END, f"Synchronized duration: {common_time[-1] - common_time[0]:.2f} s\n\n")
        
        self.fig_eemd.tight_layout()
        if hasattr(self, 'canvas_eemd'):
            self.canvas_eemd.draw()

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

    #
    def create_hrv_analysis_tab(self, parent):
        def create_hrv_content(scrollable_frame):
            main_frame = ttk.Frame(scrollable_frame)
            main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            left_panel = ttk.LabelFrame(main_frame, text="HRV Analysis Controls", width=320)  # Wider for new controls
            left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10, expand=False)
            left_panel.pack_propagate(False)
            
            # Information frame
            info_frame = ttk.LabelFrame(left_panel, text="HRV Information")
            info_frame.pack(fill=tk.X, expand=False, padx=5, pady=5)
            
            info_label = ttk.Label(info_frame, 
                                text="Heart Rate Variability Analysis\nCompares Radar vs ECG HRV\nWith/Without Moving Average", 
                                justify=tk.CENTER, foreground="darkblue")
            info_label.pack(padx=5, pady=5)
            
            # NEW: MAV Controls
            mav_frame = ttk.LabelFrame(left_panel, text="Moving Average Filter")
            mav_frame.pack(fill=tk.X, expand=False, padx=5, pady=5)
            
            self.hrv_enable_mav_var = tk.BooleanVar(value=False)
            mav_check = ttk.Checkbutton(mav_frame, text="Apply MAV to RR intervals", 
                                    variable=self.hrv_enable_mav_var)
            mav_check.pack(anchor=tk.W, padx=5, pady=2)
            
            mav_window_frame = ttk.Frame(mav_frame)
            mav_window_frame.pack(fill=tk.X, padx=5, pady=2)
            ttk.Label(mav_window_frame, text="MAV Window Size:").pack(side=tk.LEFT)
            self.hrv_mav_window_var = tk.IntVar(value=5)
            mav_entry = ttk.Entry(mav_window_frame, textvariable=self.hrv_mav_window_var, width=10)
            mav_entry.pack(side=tk.RIGHT)
            
            # HRV Parameters frame (existing code)
            params_frame = ttk.LabelFrame(left_panel, text="HRV Parameters")
            params_frame.pack(fill=tk.X, expand=False, padx=5, pady=5)
            
            # Window size for HRV calculation
            window_frame = ttk.Frame(params_frame)
            window_frame.pack(fill=tk.X, padx=5, pady=2)
            ttk.Label(window_frame, text="HRV Window (s):").pack(side=tk.LEFT)
            self.hrv_window_var = tk.DoubleVar(value=60.0)
            window_entry = ttk.Entry(window_frame, textvariable=self.hrv_window_var, width=10)
            window_entry.pack(side=tk.RIGHT)
            
            # Overlap percentage
            overlap_frame = ttk.Frame(params_frame)
            overlap_frame.pack(fill=tk.X, padx=5, pady=2)
            ttk.Label(overlap_frame, text="Window Overlap (%):").pack(side=tk.LEFT)
            self.hrv_overlap_var = tk.DoubleVar(value=50.0)
            overlap_entry = ttk.Entry(overlap_frame, textvariable=self.hrv_overlap_var, width=10)
            overlap_entry.pack(side=tk.RIGHT)
            
            # Minimum RR interval filter
            min_rr_frame = ttk.Frame(params_frame)
            min_rr_frame.pack(fill=tk.X, padx=5, pady=2)
            ttk.Label(min_rr_frame, text="Min RR (ms):").pack(side=tk.LEFT)
            self.min_rr_var = tk.DoubleVar(value=300.0)
            min_rr_entry = ttk.Entry(min_rr_frame, textvariable=self.min_rr_var, width=10)
            min_rr_entry.pack(side=tk.RIGHT)
            
            # Maximum RR interval filter
            max_rr_frame = ttk.Frame(params_frame)
            max_rr_frame.pack(fill=tk.X, padx=5, pady=2)
            ttk.Label(max_rr_frame, text="Max RR (ms):").pack(side=tk.LEFT)
            self.max_rr_var = tk.DoubleVar(value=1500.0)
            max_rr_entry = ttk.Entry(max_rr_frame, textvariable=self.max_rr_var, width=10)
            max_rr_entry.pack(side=tk.RIGHT)
            
            # Buttons frame
            button_frame = ttk.Frame(left_panel)
            button_frame.pack(fill=tk.X, padx=5, pady=5)
            self.calculate_hrv_button = ttk.Button(button_frame, text="Calculate HRV", command=self.calculate_hrv_analysis)
            self.calculate_hrv_button.pack(fill=tk.X, padx=5, pady=5)
            # Separator
            ttk.Separator(left_panel, orient='horizontal').pack(fill=tk.X, padx=5, pady=5)

            # ECG Comparison frame  
            ecg_frame = ttk.LabelFrame(left_panel, text="ECG Comparison")
            ecg_frame.pack(fill=tk.X, padx=5, pady=5)

            ttk.Button(ecg_frame, text="Load ECG Data", command=self.load_ecg_data).pack(fill=tk.X, padx=5, pady=2)
            ttk.Button(ecg_frame, text="Compare with ECG", command=self.compare_imfs_ecg).pack(fill=tk.X, padx=5, pady=2)
            
            # NEW: Separate export buttons
            export_frame = ttk.LabelFrame(button_frame, text="Export Results")
            export_frame.pack(fill=tk.X, padx=5, pady=5)
            
            export_raw_button = ttk.Button(export_frame, text="Export Raw HRV", command=self.export_raw_hrv)
            export_raw_button.pack(fill=tk.X, padx=5, pady=2)
            
            export_mav_button = ttk.Button(export_frame, text="Export MAV HRV", command=self.export_mav_hrv)
            export_mav_button.pack(fill=tk.X, padx=5, pady=2)

            export_edited_button = ttk.Button(export_frame, text="Export Edited HRV", command=self.export_edited_hrv)
            export_edited_button.pack(fill=tk.X, padx=5, pady=2)
            
            # Results display
            results_frame = ttk.LabelFrame(left_panel, text="HRV Results")
            results_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
            # TAMBAHKAN SETELAH results_frame.pack()
            # Peak Editing Controls
            editing_frame = ttk.LabelFrame(left_panel, text="Radar Peak Editing")
            editing_frame.pack(fill=tk.X, expand=False, padx=5, pady=5)

            # Navigation buttons
            nav_frame = ttk.Frame(editing_frame)
            nav_frame.pack(fill=tk.X, padx=5, pady=2)

            self.hrv_prev_btn = ttk.Button(nav_frame, text="⏮️ Prev 30s", command=self.prev_segment_hrv)
            self.hrv_prev_btn.pack(side=tk.LEFT, padx=2)

            self.hrv_next_btn = ttk.Button(nav_frame, text="⏭️ Next 30s", command=self.next_segment_hrv) 
            self.hrv_next_btn.pack(side=tk.LEFT, padx=2)

            # Initially disabled
            self.hrv_prev_btn.config(state='disabled')
            self.hrv_next_btn.config(state='disabled')

            # Edit controls
            edit_controls_frame = ttk.Frame(editing_frame)
            edit_controls_frame.pack(fill=tk.X, padx=5, pady=2)

            self.edit_radar_btn = ttk.Button(edit_controls_frame, text="Edit Radar Peaks", 
                                            command=self.edit_radar_peaks)
            self.edit_radar_btn.pack(fill=tk.X, padx=5, pady=2)
            self.edit_radar_btn.config(state='disabled')

            self.update_hrv_btn = ttk.Button(edit_controls_frame, text="Update HRV (Edited)", 
                                        command=self.update_hrv_with_edited_peaks_safe)
            self.update_hrv_btn.pack(fill=tk.X, padx=5, pady=2)
            self.update_hrv_btn.config(state='disabled')

            self.exit_edit_btn = ttk.Button(edit_controls_frame, text="Exit Editing", 
                                            command=self.exit_editing_mode)
            self.exit_edit_btn.pack(fill=tk.X, padx=5, pady=2)
            self.exit_edit_btn.config(state='disabled')

            # Info label
            self.hrv_edit_info = ttk.Label(editing_frame, text="Run HRV analysis first to enable editing", 
                                        font=("Arial", 8), foreground="gray")
            self.hrv_edit_info.pack(padx=5, pady=2)
            self.hrv_results_text = tk.Text(results_frame, width=50, height=25)
            hrv_scrollbar = ttk.Scrollbar(results_frame, orient=tk.VERTICAL, command=self.hrv_results_text.yview)
            self.hrv_results_text.configure(yscrollcommand=hrv_scrollbar.set)
            
            self.hrv_results_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
            hrv_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            
            # Right panel for plots
            right_panel = ttk.Frame(main_frame)
            right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            # Create figure with subplots for HRV analysis
            # Create figure with subplots for HRV analysis
            # Create figure with subplots for HRV analysis
            self.fig_hrv = Figure(figsize=(9, 14))
            gs_hrv = self.fig_hrv.add_gridspec(7, 1, height_ratios=[1, 1, 1, 1, 1, 1, 1.2], hspace=0.9)

            self.ax_rr_radar = self.fig_hrv.add_subplot(gs_hrv[0])          # Radar RR intervals
            self.ax_rr_ecg = self.fig_hrv.add_subplot(gs_hrv[1])            # ECG RR intervals  
            self.ax_rr_compare = self.fig_hrv.add_subplot(gs_hrv[2])        # RR intervals comparison
            self.ax_hrv_radar = self.fig_hrv.add_subplot(gs_hrv[3])         # Radar HRV parameters
            self.ax_hrv_ecg = self.fig_hrv.add_subplot(gs_hrv[4])           # ECG HRV parameters
            self.ax_hrv_compare = self.fig_hrv.add_subplot(gs_hrv[5])       # HRV comparison
            self.ax_peak_compare = self.fig_hrv.add_subplot(gs_hrv[6])      # Peak comparison & editing (paling bawah)

            # Show placeholder text
            for i, ax in enumerate([self.ax_rr_radar, self.ax_rr_ecg, self.ax_rr_compare, 
                                self.ax_hrv_radar, self.ax_hrv_ecg, self.ax_hrv_compare, self.ax_peak_compare]):
                if i == 6:  # Peak comparison plot (yang akan bisa diedit)
                    ax.text(0.5, 0.5, f'Peak Comparison & Editing\nRun Paper Method Algorithm and Load ECG Data\nthen click "Calculate HRV"', 
                            ha='center', va='center', transform=ax.transAxes)
                else:
                    ax.text(0.5, 0.5, f'Run Paper Method Algorithm\nand Load ECG Data\nthen click "Calculate HRV"', 
                            ha='center', va='center', transform=ax.transAxes)
            
            self.canvas_hrv = FigureCanvasTkAgg(self.fig_hrv, master=right_panel)
            
            # Add toolbar
            hrv_toolbar_frame = ttk.Frame(right_panel)
            hrv_toolbar_frame.pack(side=tk.TOP, fill=tk.X)
            
            hrv_toolbar = NavigationToolbar2Tk(self.canvas_hrv, hrv_toolbar_frame)
            hrv_toolbar.update()
            
            self.canvas_hrv.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        self.create_scrollable_tab(parent, create_hrv_content)

    def plot_peak_detection_comparison(self):
        self.ax_peak_compare.clear()  # GANTI dari ax_peak_compare
        
        if not hasattr(self, 'paper_analysis_results') or self.paper_analysis_results is None:
            self.ax_peak_compare.text(0.5, 0.5, 'No radar peaks available\nRun Paper Method first', 
                                        ha='center', va='center', transform=self.ax_peak_compare.transAxes)
            return
        
        if not hasattr(self, 'paper_analysis_results') or self.paper_analysis_results is None:
            self.ax_peak_compare.text(0.5, 0.5, 'No radar peaks available\nRun Paper Method first', 
                                    ha='center', va='center', transform=self.ax_peak_compare.transAxes)
            return
        
        if not self.ecg_loaded:
            self.ax_peak_compare.text(0.5, 0.5, 'No ECG data available\nLoad ECG data first', 
                                    ha='center', va='center', transform=self.ax_peak_compare.transAxes)
            return
        
        try:
            # Get radar peak data (use final peaks by default)
            radar_peaks = self.paper_analysis_results['final_peaks']
            radar_signal = self.latest_combined_imf
            radar_times = self.signals['timestamps']
            radar_peak_times = radar_times[radar_peaks]
            radar_peak_values = radar_signal[radar_peaks]
            
            # Get ECG peak data
            ecg_peaks = self.detect_r_peaks(self.ecg_data, self.ecg_fs)
            ecg_peak_times = self.ecg_time[ecg_peaks]
            ecg_peak_values = self.ecg_data[ecg_peaks]
            
            # Find overlapping time range for display
            radar_start, radar_end = radar_times[0], radar_times[-1]
            ecg_start, ecg_end = self.ecg_time[0], self.ecg_time[-1]
            
            overlap_start = max(radar_start, ecg_start)
            overlap_end = min(radar_end, ecg_end)
            
            if overlap_end <= overlap_start:
                self.ax_peak_compare.text(0.5, 0.5, 'No time overlap between\nradar and ECG data', 
                                        ha='center', va='center', transform=self.ax_peak_compare.transAxes)
                return
            
            # Filter to overlapping range
            radar_mask = (radar_times >= overlap_start) & (radar_times <= overlap_end)
            ecg_mask = (self.ecg_time >= overlap_start) & (self.ecg_time <= overlap_end)
            
            radar_peak_mask = (radar_peak_times >= overlap_start) & (radar_peak_times <= overlap_end)
            ecg_peak_mask = (ecg_peak_times >= overlap_start) & (ecg_peak_times <= overlap_end)
            
            # Normalize signals for better visualization
            radar_signal_norm = (radar_signal - np.mean(radar_signal)) / np.std(radar_signal)
            ecg_signal_norm = (self.ecg_data - np.mean(self.ecg_data)) / np.std(self.ecg_data)
            
            # Offset ECG signal for dual-axis display
            ecg_offset = 3.0
            ecg_signal_offset = ecg_signal_norm + ecg_offset
            ecg_peak_values_offset = ecg_peak_values / np.std(self.ecg_data) + ecg_offset
            
            # Plot signals
            self.ax_peak_compare.plot(radar_times[radar_mask], radar_signal_norm[radar_mask], 
                                    'b-', linewidth=1, alpha=0.7, label='Radar Signal (Normalized)')
            self.ax_peak_compare.plot(self.ecg_time[ecg_mask], ecg_signal_offset[ecg_mask], 
                                    'r-', linewidth=1, alpha=0.7, label='ECG Signal (Normalized + Offset)')
            
            # Plot detected peaks
            self.ax_peak_compare.plot(radar_peak_times[radar_peak_mask], 
                                    radar_peak_values[radar_peak_mask] / np.std(radar_signal), 
                                    'bo', markersize=6, alpha=0.8, 
                                    label=f'Radar Peaks ({np.sum(radar_peak_mask)})')
            self.ax_peak_compare.plot(ecg_peak_times[ecg_peak_mask], 
                                    ecg_peak_values_offset[ecg_peak_mask], 
                                    'ro', markersize=6, alpha=0.8, 
                                    label=f'ECG R-Peaks ({np.sum(ecg_peak_mask)})')
            
            # Add separation line
            self.ax_peak_compare.axhline(y=ecg_offset/2, color='gray', linestyle=':', alpha=0.5)
            
            # Calculate and display peak rates
            overlap_duration = overlap_end - overlap_start
            radar_rate = np.sum(radar_peak_mask) / overlap_duration * 60  # BPM
            ecg_rate = np.sum(ecg_peak_mask) / overlap_duration * 60      # BPM
            
            self.ax_peak_compare.set_title(f'Peak Detection Comparison\n'
                                        f'Radar: {radar_rate:.1f} BPM, ECG: {ecg_rate:.1f} BPM '
                                        f'(Δ: {abs(radar_rate - ecg_rate):.1f} BPM)')
            self.ax_peak_compare.set_xlabel('Time (s)')
            self.ax_peak_compare.set_ylabel('Normalized Amplitude')
            self.ax_peak_compare.legend(loc='upper right')
            self.ax_peak_compare.grid(True, alpha=0.3)
            
            # Set x-axis to focus on overlapping range
            self.ax_peak_compare.set_xlim([overlap_start, overlap_end])
            
            # Add text annotations for signal identification
            self.ax_peak_compare.text(0.02, 0.15, 'Radar Signal', transform=self.ax_peak_compare.transAxes,
                                    bbox=dict(facecolor='blue', alpha=0.3, pad=2), fontsize=9)
            self.ax_peak_compare.text(0.02, 0.85, 'ECG Signal', transform=self.ax_peak_compare.transAxes,
                                    bbox=dict(facecolor='red', alpha=0.3, pad=2), fontsize=9)
            
        except Exception as e:
            self.ax_peak_compare.text(0.5, 0.5, f'Peak comparison failed:\n{str(e)}', 
                                        ha='center', va='center', transform=self.ax_peak_compare.transAxes)
            print(f"Peak comparison error: {e}")

    def calculate_rr_intervals(self, peak_indices, timestamps):
        """Calculate RR intervals from peak indices"""
        if len(peak_indices) < 2:
            return np.array([]), np.array([])
        
        # Calculate RR intervals in milliseconds
        rr_intervals = []
        rr_times = []
        
        for i in range(1, len(peak_indices)):
            rr_interval = (timestamps[peak_indices[i]] - timestamps[peak_indices[i-1]]) * 1000  # Convert to ms
            rr_time = timestamps[peak_indices[i-1]]  # Time of the first peak in the interval
            
            rr_intervals.append(rr_interval)
            rr_times.append(rr_time)
        
        return np.array(rr_intervals), np.array(rr_times)

    def filter_rr_intervals(self, rr_intervals, rr_times, min_rr=300, max_rr=1500):
        """Filter RR intervals to remove outliers"""
        valid_mask = (rr_intervals >= min_rr) & (rr_intervals <= max_rr)
        return rr_intervals[valid_mask], rr_times[valid_mask]

    def calculate_hrv_parameters(self, rr_intervals, window_size_s=60, overlap_percent=50, fs=1000):
        """Calculate time-domain HRV parameters in sliding windows"""
        if len(rr_intervals) < 5:
            return {}, [], []
        
        # Convert window size to number of RR intervals (approximate)
        avg_rr = np.mean(rr_intervals) / 1000  # Average RR in seconds
        window_size_intervals = max(5, int(window_size_s / avg_rr))
        overlap_intervals = int(window_size_intervals * overlap_percent / 100)
        
        # Sliding window calculation
        hrv_metrics = {
            'SDNN': [],
            'RMSSD': [],
            'pNN50': [],
            'HR_mean': [],
            'times': []
        }
        
        step = max(1, window_size_intervals - overlap_intervals)
        
        for i in range(0, len(rr_intervals) - window_size_intervals + 1, step):
            window_rr = rr_intervals[i:i + window_size_intervals]
            
            if len(window_rr) >= 5:
                # SDNN: Standard deviation of RR intervals
                sdnn = np.std(window_rr)
                
                # RMSSD: Root mean square of successive differences
                successive_diffs = np.diff(window_rr)
                rmssd = np.sqrt(np.mean(successive_diffs**2))
                
                # pNN50: Percentage of successive RR intervals that differ by > 50ms
                pnn50 = (np.sum(np.abs(successive_diffs) > 50) / len(successive_diffs)) * 100
                
                # Mean heart rate
                hr_mean = 60000 / np.mean(window_rr)  # Convert from ms to BPM
                
                # Time point (middle of window)
                time_point = i + window_size_intervals // 2
                
                hrv_metrics['SDNN'].append(sdnn)
                hrv_metrics['RMSSD'].append(rmssd)
                hrv_metrics['pNN50'].append(pnn50)
                hrv_metrics['HR_mean'].append(hr_mean)
                hrv_metrics['times'].append(time_point)
        
        # Convert to numpy arrays
        for key in hrv_metrics:
            hrv_metrics[key] = np.array(hrv_metrics[key])
        
        # Overall statistics
        overall_stats = {
            'SDNN_mean': np.mean(hrv_metrics['SDNN']) if len(hrv_metrics['SDNN']) > 0 else 0,
            'SDNN_std': np.std(hrv_metrics['SDNN']) if len(hrv_metrics['SDNN']) > 0 else 0,
            'RMSSD_mean': np.mean(hrv_metrics['RMSSD']) if len(hrv_metrics['RMSSD']) > 0 else 0,
            'RMSSD_std': np.std(hrv_metrics['RMSSD']) if len(hrv_metrics['RMSSD']) > 0 else 0,
            'pNN50_mean': np.mean(hrv_metrics['pNN50']) if len(hrv_metrics['pNN50']) > 0 else 0,
            'HR_mean': np.mean(hrv_metrics['HR_mean']) if len(hrv_metrics['HR_mean']) > 0 else 0,
            'HR_std': np.std(hrv_metrics['HR_mean']) if len(hrv_metrics['HR_mean']) > 0 else 0
        }
        
        return overall_stats, hrv_metrics, window_rr
    
    # GANTI FUNGSI calculate_hrv_analysis() YANG ADA dengan yang ini:

    def calculate_hrv_analysis(self):
        """
        Calculate HRV analysis dengan style vibrant dan perbaikan variable definition
        """
        if not hasattr(self, 'paper_analysis_results') or self.paper_analysis_results is None:
            messagebox.showwarning("Warning", "No paper method results available. Run paper algorithm first.")
            return
        
        if not self.ecg_loaded:
            messagebox.showwarning("Warning", "No ECG data loaded. Please load ECG data first.")
            return
        
        # Clear results
        self.hrv_results_text.delete(1.0, tk.END)
        for ax in [self.ax_rr_radar, self.ax_rr_ecg, self.ax_rr_compare, 
                self.ax_hrv_radar, self.ax_hrv_ecg, self.ax_hrv_compare, self.ax_peak_compare]:
            ax.clear()
        
        try:
            # Get parameters - PERBAIKAN: Pastikan semua variable terdefinisi
            window_size = self.hrv_window_var.get()
            overlap_percent = self.hrv_overlap_var.get()  # PERBAIKAN: overlap_percent bukan overlap
            min_rr = self.min_rr_var.get()
            max_rr = self.max_rr_var.get()
            enable_mav = self.hrv_enable_mav_var.get()
            mav_window = self.hrv_mav_window_var.get()
            
            self.hrv_results_text.insert(tk.END, "🎨 HRV ANALYSIS WITH VIBRANT VISUALIZATION\n")
            self.hrv_results_text.insert(tk.END, "=" * 55 + "\n")
            self.hrv_results_text.insert(tk.END, f"Window Size: {window_size} s\n")
            self.hrv_results_text.insert(tk.END, f"Overlap: {overlap_percent}%\n")
            self.hrv_results_text.insert(tk.END, f"RR Filter: {min_rr}-{max_rr} ms\n")
            self.hrv_results_text.insert(tk.END, f"MAV Applied: {enable_mav}\n")
            if enable_mav:
                self.hrv_results_text.insert(tk.END, f"MAV Window: {mav_window}\n")
            self.hrv_results_text.insert(tk.END, "\n")
            
            # Plot peak detection comparison first
            self.plot_peak_detection_comparison()
            
            # 1. Calculate RR intervals from radar peaks
            radar_peaks = self.paper_analysis_results['final_peaks']
            radar_times = self.signals['timestamps']
            radar_rr, radar_rr_times = self.calculate_rr_intervals(radar_peaks, radar_times)
            radar_rr_filtered, radar_rr_times_filtered = self.filter_rr_intervals(
                radar_rr, radar_rr_times, min_rr, max_rr)
            
            # 2. Calculate RR intervals from ECG peaks  
            ecg_peaks = self.detect_r_peaks(self.ecg_data, self.ecg_fs)
            ecg_rr, ecg_rr_times = self.calculate_rr_intervals(ecg_peaks, self.ecg_time)
            ecg_rr_filtered, ecg_rr_times_filtered = self.filter_rr_intervals(
                ecg_rr, ecg_rr_times, min_rr, max_rr)
            
            # 3. Apply MAV if enabled
            if enable_mav:
                radar_rr_mav = self.apply_moving_average(radar_rr_filtered, mav_window)
                ecg_rr_mav = self.apply_moving_average(ecg_rr_filtered, mav_window)
            else:
                radar_rr_mav = radar_rr_filtered
                ecg_rr_mav = ecg_rr_filtered
            
            # 4. Calculate HRV parameters - PERBAIKAN: Definisi variable yang benar
            radar_stats_raw, radar_hrv_raw, _ = self.calculate_hrv_parameters(
                radar_rr_filtered, window_size, overlap_percent)
            ecg_stats_raw, ecg_hrv_raw, _ = self.calculate_hrv_parameters(
                ecg_rr_filtered, window_size, overlap_percent)
            
            radar_stats_mav, radar_hrv_mav, _ = self.calculate_hrv_parameters(
                radar_rr_mav, window_size, overlap_percent)
            ecg_stats_mav, ecg_hrv_mav, _ = self.calculate_hrv_parameters(
                ecg_rr_mav, window_size, overlap_percent)
            
            # 5. Extract vibrant features for radar
            try:
                # Spectral features
                freqs, psd, lf_power, hf_power, peak_lf, peak_hf = self.ar_psd_burg(radar_rr_filtered)
                
                # Bispectral features dengan style vibrant
                bispectral_features = self.bispectrum_features_vibrant(radar_rr_filtered)
                
                # Non-linear features
                poincare_features = self.calculate_poincare_features(radar_rr_filtered)
                sample_entropy = self.sample_entropy(radar_rr_filtered)
                
                # Combine features
                radar_features = {
                    'LF_power': lf_power,
                    'HF_power': hf_power,
                    'peak_LF': peak_lf,
                    'peak_HF': peak_hf,
                    'frequencies': freqs,
                    'psd': psd,
                    **bispectral_features,
                    **poincare_features,
                    'SamEn': sample_entropy
                }
                
            except Exception as e:
                print(f"Error extracting radar features: {e}")
                radar_features = {}
            
            # 6. Extract features for ECG
            try:
                ecg_freqs, ecg_psd, ecg_lf_power, ecg_hf_power, ecg_peak_lf, ecg_peak_hf = self.ar_psd_burg(ecg_rr_filtered)
                ecg_bispectral = self.bispectrum_features_vibrant(ecg_rr_filtered)
                ecg_poincare = self.calculate_poincare_features(ecg_rr_filtered)
                ecg_sample_entropy = self.sample_entropy(ecg_rr_filtered)
                
                ecg_features = {
                    'LF_power': ecg_lf_power,
                    'HF_power': ecg_hf_power,
                    'peak_LF': ecg_peak_lf,
                    'peak_HF': ecg_peak_hf,
                    **ecg_bispectral,
                    **ecg_poincare,
                    'SamEn': ecg_sample_entropy
                }
                
            except Exception as e:
                print(f"Error extracting ECG features: {e}")
                ecg_features = {}
            
            # 7. Create vibrant visualization
            visualization_results = {
                'radar_rr': radar_rr_filtered,
                'ecg_rr': ecg_rr_filtered,
                'radar_features': radar_features,
                'ecg_features': ecg_features,
                'radar_name': "Radar (Paper Method)",
                'ecg_name': "ECG R-Peaks"
            }
            
            self.create_vibrant_hrv_visualization(visualization_results)
            
            # 8. Plot comparison visualizations (existing style)
            self.plot_rr_intervals_with_mav(radar_rr_filtered, radar_rr_times_filtered, radar_rr_mav,
                                        ecg_rr_filtered, ecg_rr_times_filtered, ecg_rr_mav, enable_mav)
            
            self.plot_hrv_parameters_with_mav(radar_hrv_raw, radar_hrv_mav, 
                                            ecg_hrv_raw, ecg_hrv_mav, enable_mav)
            
            # 9. PERBAIKAN: Store results dengan variable yang benar
            self.hrv_results_raw = {
                'radar_rr': radar_rr_filtered,
                'ecg_rr': ecg_rr_filtered,
                'radar_hrv': radar_hrv_raw,
                'ecg_hrv': ecg_hrv_raw,
                'radar_stats': radar_stats_raw,
                'ecg_stats': ecg_stats_raw,
                'radar_features': radar_features,
                'ecg_features': ecg_features,
                'radar_name': "Radar (Paper Method)",
                'ecg_name': "ECG R-Peaks",
                'parameters': {
                    'window_size': window_size,
                    'overlap_percent': overlap_percent,  # PERBAIKAN: Konsisten dengan nama variable
                    'min_rr': min_rr,
                    'max_rr': max_rr
                }
            }
            
            # Store MAV results if enabled
            if enable_mav:
                self.hrv_results_mav = {
                    'radar_rr': radar_rr_mav,
                    'ecg_rr': ecg_rr_mav,
                    'radar_hrv': radar_hrv_mav,
                    'ecg_hrv': ecg_hrv_mav,
                    'radar_stats': radar_stats_mav,
                    'ecg_stats': ecg_stats_mav,
                    'mav_window': mav_window,
                    'radar_name': f"Radar MAV (Window: {mav_window})",
                    'ecg_name': f"ECG MAV (Window: {mav_window})",
                    'parameters': {
                        'window_size': window_size,
                        'overlap_percent': overlap_percent,
                        'min_rr': min_rr,
                        'max_rr': max_rr,
                        'mav_window': mav_window
                    }
                }
            
            # 10. Display results
            self.display_vibrant_hrv_results(radar_stats_raw, ecg_stats_raw, 
                                            radar_features, ecg_features)
            
            # Update progress and finish
            self.fig_hrv.tight_layout()
            self.canvas_hrv.draw()
            
            print(f"✅ HRV Analysis completed with vibrant visualization!")
            print(f"📊 Radar RR intervals: {len(radar_rr_filtered)}")
            print(f"🎯 ECG RR intervals: {len(ecg_rr_filtered)}")
            if enable_mav:
                print(f"🔄 MAV applied with window: {mav_window}")
            
        except Exception as e:
            messagebox.showerror("Error", f"HRV analysis failed: {str(e)}")
            print(f"HRV analysis error: {e}")
            import traceback
            traceback.print_exc()

    def display_vibrant_hrv_results(self, radar_stats, ecg_stats, radar_features, ecg_features):
        """
        Display HRV results dengan style yang vibrant dan informatif
        """
        self.hrv_results_text.insert(tk.END, f"🎨 VIBRANT HRV ANALYSIS RESULTS\n")
        self.hrv_results_text.insert(tk.END, "=" * 55 + "\n\n")
        
        # Time domain features
        self.hrv_results_text.insert(tk.END, f"📊 TIME DOMAIN FEATURES:\n")
        self.hrv_results_text.insert(tk.END, f"-" * 30 + "\n")
        self.hrv_results_text.insert(tk.END, f"RADAR SDNN: {radar_stats['SDNN_mean']:.2f} ± {radar_stats['SDNN_std']:.2f} ms\n")
        self.hrv_results_text.insert(tk.END, f"ECG SDNN:   {ecg_stats['SDNN_mean']:.2f} ± {ecg_stats['SDNN_std']:.2f} ms\n")
        self.hrv_results_text.insert(tk.END, f"RADAR RMSSD: {radar_stats['RMSSD_mean']:.2f} ± {radar_stats['RMSSD_std']:.2f} ms\n")
        self.hrv_results_text.insert(tk.END, f"ECG RMSSD:   {ecg_stats['RMSSD_mean']:.2f} ± {ecg_stats['RMSSD_std']:.2f} ms\n\n")
        
        # Spectral features
        if 'LF_power' in radar_features and 'LF_power' in ecg_features:
            self.hrv_results_text.insert(tk.END, f"🌊 SPECTRAL FEATURES:\n")
            self.hrv_results_text.insert(tk.END, f"-" * 30 + "\n")
            self.hrv_results_text.insert(tk.END, f"RADAR LF Power: {radar_features['LF_power']:.6f}\n")
            self.hrv_results_text.insert(tk.END, f"ECG LF Power:   {ecg_features['LF_power']:.6f}\n")
            self.hrv_results_text.insert(tk.END, f"RADAR HF Power: {radar_features['HF_power']:.6f}\n")
            self.hrv_results_text.insert(tk.END, f"ECG HF Power:   {ecg_features['HF_power']:.6f}\n\n")
        
        # Bispectral features
        if 'P1' in radar_features and 'P1' in ecg_features:
            self.hrv_results_text.insert(tk.END, f"🎭 BISPECTRAL FEATURES:\n")
            self.hrv_results_text.insert(tk.END, f"-" * 30 + "\n")
            for feature in ['P1', 'P2', 'H1', 'H2', 'H3', 'H4']:
                self.hrv_results_text.insert(tk.END, f"RADAR {feature}: {radar_features.get(feature, 0):.4f}\n")
                self.hrv_results_text.insert(tk.END, f"ECG {feature}:   {ecg_features.get(feature, 0):.4f}\n")
            self.hrv_results_text.insert(tk.END, "\n")
        
        # Non-linear features
        if 'SD1' in radar_features and 'SD1' in ecg_features:
            self.hrv_results_text.insert(tk.END, f"🌀 NON-LINEAR FEATURES:\n")
            self.hrv_results_text.insert(tk.END, f"-" * 30 + "\n")
            self.hrv_results_text.insert(tk.END, f"RADAR SD1: {radar_features['SD1']:.2f} ms\n")
            self.hrv_results_text.insert(tk.END, f"ECG SD1:   {ecg_features['SD1']:.2f} ms\n")
            self.hrv_results_text.insert(tk.END, f"RADAR SD2: {radar_features['SD2']:.2f} ms\n")
            self.hrv_results_text.insert(tk.END, f"ECG SD2:   {ecg_features['SD2']:.2f} ms\n")
            self.hrv_results_text.insert(tk.END, f"RADAR SD1/SD2: {radar_features['SD1_SD2_ratio']:.4f}\n")
            self.hrv_results_text.insert(tk.END, f"ECG SD1/SD2:   {ecg_features['SD1_SD2_ratio']:.4f}\n")
            self.hrv_results_text.insert(tk.END, f"RADAR SamEn: {radar_features['SamEn']:.4f}\n")
            self.hrv_results_text.insert(tk.END, f"ECG SamEn:   {ecg_features['SamEn']:.4f}\n\n")
        
        self.hrv_results_text.insert(tk.END, "✨ Visualization updated with vibrant plots! ✨\n")
        self.hrv_results_text.insert(tk.END, "🎨 Features ready for export to CSV\n")


        # TAMBAHKAN IMPORT INI DI BAGIAN ATAS kim_findpeaks_method.py (jika belum ada):
    from scipy.fft import fft2
    from scipy.ndimage import gaussian_filter
    from matplotlib.ticker import ScalarFormatter

    # TAMBAHKAN FUNGSI-FUNGSI UTILITY INI:

    def ar_psd_burg(self, rr_intervals, fs=4):
        """
        AR PSD using Burg method - adapted dari Preprocessing.py
        """
        try:
            # Resample untuk stabilitas
            rr_interp = np.interp(np.linspace(0, len(rr_intervals), len(rr_intervals)*fs), 
                                np.arange(len(rr_intervals)), rr_intervals)
            rr_interp = rr_interp - np.mean(rr_interp)  # Remove DC offset

            # AR modeling dengan order 16 (standard)
            order = 16
            
            # Simple AR using numpy (fallback if spectrum library not available)
            # Yule-Walker equations untuk AR coefficients
            def yule_walker_ar(data, order):
                """Simple Yule-Walker AR estimation"""
                N = len(data)
                r = np.correlate(data, data, mode='full')
                r = r[N-1:]  # Take positive lags only
                r = r[:order+1]
                
                # Build Toeplitz matrix
                R = np.array([[r[abs(i-j)] for j in range(order)] for i in range(order)])
                b = -r[1:order+1]
                
                try:
                    ar_coeffs = np.linalg.solve(R, b)
                    ar_coeffs = np.concatenate(([1], ar_coeffs))  # Add a0 = 1
                    e = r[0] + np.sum(ar_coeffs[1:] * r[1:order+1])  # Prediction error
                    return ar_coeffs, e
                except:
                    # Fallback
                    return np.ones(order+1), 1.0
            
            ar_coeffs, e = yule_walker_ar(rr_interp, order)

            # Compute PSD
            nfft = 512
            freqs = np.linspace(0, fs/2, nfft)
            psd = np.zeros_like(freqs)
            
            for i, f in enumerate(freqs):
                omega = np.exp(-2j * np.pi * f / fs)
                # AR PSD formula: H(z) = sigma^2 / |A(z)|^2
                A_omega = np.polyval(ar_coeffs[::-1], omega)  # Reverse for polyval
                psd[i] = e / np.abs(A_omega)**2
            
            psd = np.real(psd)
            psd = psd / 100.0  # Light scaling

            # Power in frequency bands
            lf_band = (freqs >= 0.04) & (freqs < 0.15)
            hf_band = (freqs >= 0.15) & (freqs <= 0.4)
            
            lf_power = np.trapz(psd[lf_band], freqs[lf_band]) if np.any(lf_band) else 0
            hf_power = np.trapz(psd[hf_band], freqs[hf_band]) if np.any(hf_band) else 0

            # Peak frequencies
            peak_lf = freqs[lf_band][np.argmax(psd[lf_band])] if np.any(lf_band) and np.sum(lf_band) > 0 else 0
            peak_hf = freqs[hf_band][np.argmax(psd[hf_band])] if np.any(hf_band) and np.sum(hf_band) > 0 else 0

            print(f"✨ Spectral features — LF: {lf_power:.4f}, HF: {hf_power:.4f}, Peak LF: {peak_lf:.4f} Hz, Peak HF: {peak_hf:.4f} Hz")
            return freqs, psd, lf_power, hf_power, peak_lf, peak_hf

        except Exception as e:
            print(f"Error in AR PSD: {e}")
            # Return fallback values
            freqs = np.linspace(0, 2, 100)
            psd = np.ones_like(freqs) * 0.01
            return freqs, psd, 0.01, 0.01, 0.1, 0.25

    def sample_entropy(self, signal, m=2, r_factor=0.2):
        """
        Sample Entropy calculation - adapted dari Preprocessing.py
        """
        try:
            if len(signal) < 10:
                return 0
            
            N = len(signal)
            r = r_factor * np.std(signal)
            
            def _maxdist(xi, xj, m):
                return max([abs(ua - va) for ua, va in zip(xi, xj)])
            
            def _phi(m):
                patterns = np.array([signal[i:i+m] for i in range(N-m+1)])
                C = np.zeros(N-m+1)
                
                for i in range(N-m+1):
                    template = patterns[i]
                    for j in range(N-m+1):
                        if i != j:  # Exclude self-matches
                            if _maxdist(template, patterns[j], m) <= r:
                                C[i] += 1
                
                phi = np.mean(C) / (N - m) if (N - m) > 0 else 0
                return phi
            
            phi_m = _phi(m)
            phi_m1 = _phi(m + 1)
            
            if phi_m == 0 or phi_m1 == 0:
                return 0
            
            return -np.log(phi_m1 / phi_m)
            
        except Exception as e:
            print(f"Error in sample entropy: {e}")
            return 0

    def export_radar_rr_only_vibrant(self):
        """
        Export hanya radar RR intervals dengan style yang vibrant
        """
        if not hasattr(self, 'hrv_results_raw') or not self.hrv_results_raw:
            messagebox.showwarning("Warning", "No radar RR data available. Calculate HRV first.")
            return
        
        file_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")],
            title="Save Radar RR Intervals Only (Vibrant Analysis)",
            initialfile="radar_rr_intervals_vibrant.csv"
        )
        
        if file_path:
            try:
                radar_rr = self.hrv_results_raw['radar_rr']
                
                # Create enhanced dataframe with metadata
                timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
                
                df = pd.DataFrame({
                    'Radar_RR_ms': radar_rr,
                    'Analysis_Type': ['Vibrant_HRV'] * len(radar_rr),
                    'Export_Timestamp': [timestamp] * len(radar_rr),
                    'Index': range(len(radar_rr))
                })
                
                df.to_csv(file_path, index=False)
                
                # Enhanced success message
                messagebox.showinfo("🎨 Vibrant Export Complete", 
                    f"✨ Radar RR intervals exported successfully!\n\n"
                    f"📁 File: {os.path.basename(file_path)}\n"
                    f"📊 Intervals: {len(radar_rr)}\n"
                    f"🎯 Mean RR: {np.mean(radar_rr):.1f} ms\n"
                    f"📈 STD: {np.std(radar_rr):.1f} ms\n"
                    f"🕒 Export time: {timestamp}")
                
            except Exception as e:
                messagebox.showerror("Export Error", f"Failed to export: {str(e)}")

    def export_vibrant_hrv_features(self):
        """
        Export all HRV features dengan format yang comprehensive
        """
        if not hasattr(self, 'hrv_results_raw') or not self.hrv_results_raw:
            messagebox.showwarning("Warning", "No HRV features available. Calculate HRV first.")
            return
        
        file_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")],
            title="Save All HRV Features (Vibrant Analysis)",
            initialfile="hrv_features_vibrant_complete.csv"
        )
        
        if file_path:
            try:
                results = self.hrv_results_raw
                radar_features = results.get('radar_features', {})
                ecg_features = results.get('ecg_features', {})
                
                # Create comprehensive feature export
                feature_data = {
                    # Metadata
                    'Analysis_Type': 'Vibrant_HRV_Complete',
                    'Export_Timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                    'Radar_RR_Count': len(results.get('radar_rr', [])),
                    'ECG_RR_Count': len(results.get('ecg_rr', [])),
                    
                    # Spectral Features
                    'Radar_LF_Power': radar_features.get('LF_power', 0),
                    'Radar_HF_Power': radar_features.get('HF_power', 0),
                    'ECG_LF_Power': ecg_features.get('LF_power', 0),
                    'ECG_HF_Power': ecg_features.get('HF_power', 0),
                    
                    # Bispectral Features
                    'Radar_P1': radar_features.get('P1', 0),
                    'Radar_P2': radar_features.get('P2', 0),
                    'Radar_H1': radar_features.get('H1', 0),
                    'Radar_H2': radar_features.get('H2', 0),
                    'Radar_H3': radar_features.get('H3', 0),
                    'Radar_H4': radar_features.get('H4', 0),
                    'ECG_P1': ecg_features.get('P1', 0),
                    'ECG_P2': ecg_features.get('P2', 0),
                    'ECG_H1': ecg_features.get('H1', 0),
                    'ECG_H2': ecg_features.get('H2', 0),
                    'ECG_H3': ecg_features.get('H3', 0),
                    'ECG_H4': ecg_features.get('H4', 0),
                    
                    # Non-linear Features
                    'Radar_SD1': radar_features.get('SD1', 0),
                    'Radar_SD2': radar_features.get('SD2', 0),
                    'Radar_SD1_SD2_ratio': radar_features.get('SD1_SD2_ratio', 0),
                    'Radar_SamEn': radar_features.get('SamEn', 0),
                    'ECG_SD1': ecg_features.get('SD1', 0),
                    'ECG_SD2': ecg_features.get('SD2', 0),
                    'ECG_SD1_SD2_ratio': ecg_features.get('SD1_SD2_ratio', 0),
                    'ECG_SamEn': ecg_features.get('SamEn', 0),
                }
                
                # Save as single row
                df = pd.DataFrame([feature_data])
                df.to_csv(file_path, index=False)
                
                # Success message
                messagebox.showinfo("🎨 Complete Features Export", 
                    f"✨ All HRV features exported successfully!\n\n"
                    f"📁 File: {os.path.basename(file_path)}\n"
                    f"🎯 Features: 24 total features\n"
                    f"📊 Radar + ECG comparison\n"
                    f"🎭 Includes: Spectral, Bispectral, Non-linear\n"
                    f"🕒 Export time: {time.strftime('%H:%M:%S')}")
                
            except Exception as e:
                messagebox.showerror("Export Error", f"Failed to export features: {str(e)}")

    # UPGRADE EXISTING EXPORT FUNCTIONS
    def export_raw_hrv_vibrant(self):
        """
        Enhanced export function dengan vibrant feedback
        """
        print("[DEBUG] 🎨 Vibrant Raw HRV export button clicked")
        if not hasattr(self, 'hrv_results_raw') or not self.hrv_results_raw:
            messagebox.showwarning("Warning", "No raw HRV results to export. Calculate HRV first.")
            return
        
        file_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            title="Save Raw HRV Results (Vibrant Analysis)",
            initialfile="hrv_raw_results_vibrant.csv"
        )
        
        if file_path:
            try:
                results = self.hrv_results_raw
                
                if not all(key in results for key in ['ecg_rr', 'radar_rr']):
                    messagebox.showerror("Export Error", "Incomplete HRV data. Please recalculate HRV.")
                    return
                
                # Enhanced export dengan metadata
                radar_rr_list = [float(x[0]) if isinstance(x, (list, tuple)) else float(x) 
                            for x in results['radar_rr']]
                
                # Simple radar-only export dengan enhanced info
                timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
                df = pd.DataFrame({
                    'Radar_RR_ms': radar_rr_list,
                    'Analysis_Method': ['Vibrant_Paper_Method'] * len(radar_rr_list),
                    'Export_Timestamp': [timestamp] * len(radar_rr_list)
                })
                
                df.to_csv(file_path, index=False)
                
                # Vibrant success feedback
                messagebox.showinfo("🎨 Vibrant Export Complete", 
                    f"✨ Raw HRV results exported successfully!\n\n"
                    f"📁 File: {os.path.basename(file_path)}\n"
                    f"📊 Radar RR intervals: {len(radar_rr_list)}\n"
                    f"🎯 Mean RR: {np.mean(radar_rr_list):.1f} ms\n"
                    f"📈 Ready for external processing!\n"
                    f"🕒 Export time: {timestamp}")

            except Exception as e:
                messagebox.showerror("Export Error", f"Failed to export raw HRV results: {str(e)}")

    def calculate_rr_rmse(self, ecg_rr_intervals, radar_rr_intervals):
        """
        Calculate simple RMSE between ECG and Radar RR intervals
        Following agrimetsoft.com formula: RMSE = √(Σ(observed - predicted)²/n)
        
        Args:
            ecg_rr_intervals: Array of ECG RR intervals in ms (observed/reference)
            radar_rr_intervals: Array of Radar RR intervals in ms (predicted/model)
        
        Returns:
            dict: Contains RMSE, correlation, MAE, and comparison info
        """
        try:
            # Convert to numpy arrays
            ecg_rr = np.array(ecg_rr_intervals)
            radar_rr = np.array(radar_rr_intervals)
            
            # Handle different lengths by taking minimum length
            min_length = min(len(ecg_rr), len(radar_rr))
            if min_length == 0:
                return {
                    'rmse': np.nan,
                    'correlation': np.nan,
                    'mae': np.nan,
                    'n_points': 0,
                    'error': 'No data points available'
                }
            
            # Truncate to same length
            ecg_rr = ecg_rr[:min_length]
            radar_rr = radar_rr[:min_length]
            
            # Remove any NaN or infinite values
            valid_mask = np.isfinite(ecg_rr) & np.isfinite(radar_rr)
            ecg_clean = ecg_rr[valid_mask]
            radar_clean = radar_rr[valid_mask]
            
            if len(ecg_clean) < 2:
                return {
                    'rmse': np.nan,
                    'correlation': np.nan,
                    'mae': np.nan,
                    'n_points': len(ecg_clean),
                    'error': 'Insufficient valid data points'
                }
            
            # Calculate differences (ECG as observed, Radar as predicted)
            differences = ecg_clean - radar_clean
            
            # Calculate RMSE: √(Σ(observed - predicted)²/n)
            squared_differences = differences ** 2
            mse = np.mean(squared_differences)
            rmse = np.sqrt(mse)
            
            # Calculate correlation coefficient
            correlation = np.corrcoef(ecg_clean, radar_clean)[0, 1]
            
            # Calculate Mean Absolute Error (MAE)
            mae = np.mean(np.abs(differences))
            
            # Additional statistics
            mean_bias = np.mean(differences)  # Systematic bias
            std_differences = np.std(differences)  # Random error
            
            return {
                'rmse': rmse,
                'correlation': correlation,
                'mae': mae,
                'mean_bias': mean_bias,
                'std_error': std_differences,
                'n_points': len(ecg_clean),
                'ecg_mean': np.mean(ecg_clean),
                'radar_mean': np.mean(radar_clean),
                'error': None
            }
            
        except Exception as e:
            return {
                'rmse': np.nan,
                'correlation': np.nan,
                'mae': np.nan,
                'n_points': 0,
                'error': f'Calculation error: {str(e)}'
            }
        
    def check_peak_time_alignment(self, radar_peak_times, ecg_peak_times, tolerance=0.1):
        # Find overlapping time range
        radar_start, radar_end = radar_peak_times[0], radar_peak_times[-1]
        ecg_start, ecg_end = ecg_peak_times[0], ecg_peak_times[-1]
        
        overlap_start = max(radar_start, ecg_start)
        overlap_end = min(radar_end, ecg_end)
        overlap_duration = overlap_end - overlap_start
        
        # Filter to overlapping range
        radar_mask = (radar_peak_times >= overlap_start) & (radar_peak_times <= overlap_end)
        ecg_mask = (ecg_peak_times >= overlap_start) & (ecg_peak_times <= overlap_end)
        
        radar_overlap = radar_peak_times[radar_mask]
        ecg_overlap = ecg_peak_times[ecg_mask]
        
        alignment_metrics = {
            'overlap_duration': overlap_duration,
            'radar_peaks_overlap': len(radar_overlap),
            'ecg_peaks_overlap': len(ecg_overlap),
            'radar_total_peaks': len(radar_peak_times),
            'ecg_total_peaks': len(ecg_peak_times),
            'overlap_start': overlap_start,
            'overlap_end': overlap_end,
            'radar_range': (radar_start, radar_end),
            'ecg_range': (ecg_start, ecg_end)
        }
        
        # Calculate peak rate difference
        if overlap_duration > 0:
            radar_rate = len(radar_overlap) / overlap_duration * 60  # peaks per minute
            ecg_rate = len(ecg_overlap) / overlap_duration * 60
            rate_difference = abs(radar_rate - ecg_rate)
            alignment_metrics['radar_rate_bpm'] = radar_rate
            alignment_metrics['ecg_rate_bpm'] = ecg_rate
            alignment_metrics['rate_difference_bpm'] = rate_difference
        else:
            alignment_metrics['radar_rate_bpm'] = 0
            alignment_metrics['ecg_rate_bpm'] = 0
            alignment_metrics['rate_difference_bpm'] = float('inf')
        
        # Calculate temporal alignment using cross-correlation
        alignment_quality = 'poor'
        time_offset_suggestion = 0.0
        
        if len(radar_overlap) > 5 and len(ecg_overlap) > 5:
            # Create time bins for cross-correlation
            time_bins = np.linspace(overlap_start, overlap_end, 200)
            
            radar_hist, _ = np.histogram(radar_overlap, bins=time_bins)
            ecg_hist, _ = np.histogram(ecg_overlap, bins=time_bins)
            
            # Cross-correlation
            correlation = np.correlate(radar_hist, ecg_hist, mode='full')
            correlation_normalized = correlation / (np.linalg.norm(radar_hist) * np.linalg.norm(ecg_hist))
            
            # Find best offset
            max_corr_idx = np.argmax(correlation_normalized)
            offset_bins = max_corr_idx - (len(ecg_hist) - 1)
            
            bin_width = (overlap_end - overlap_start) / (len(time_bins) - 1)
            time_offset_suggestion = offset_bins * bin_width
            
            max_correlation = correlation_normalized[max_corr_idx]
            alignment_metrics['max_correlation'] = max_correlation
            alignment_metrics['time_offset_suggestion'] = time_offset_suggestion
            
            # Determine alignment quality
            if max_correlation > 0.7 and rate_difference < 10:
                alignment_quality = 'good'
            elif max_correlation > 0.5 and rate_difference < 20:
                alignment_quality = 'fair'
            else:
                alignment_quality = 'poor'
        
        alignment_metrics['alignment_quality'] = alignment_quality
        
        return alignment_metrics, radar_overlap, ecg_overlap

    def trim_to_overlapping_range(self, radar_peak_times, ecg_peak_times, buffer_seconds=5.0):
        """
        Trim both radar and ECG peak times to overlapping range with optional buffer
        """
        radar_start, radar_end = radar_peak_times[0], radar_peak_times[-1]
        ecg_start, ecg_end = ecg_peak_times[0], ecg_peak_times[-1]
        
        # Find overlapping range with buffer
        overlap_start = max(radar_start, ecg_start) + buffer_seconds
        overlap_end = min(radar_end, ecg_end) - buffer_seconds
        
        if overlap_end <= overlap_start:
            # Fallback: no buffer if range too small
            overlap_start = max(radar_start, ecg_start)
            overlap_end = min(radar_end, ecg_end)
        
        # Trim to overlapping range
        radar_mask = (radar_peak_times >= overlap_start) & (radar_peak_times <= overlap_end)
        ecg_mask = (ecg_peak_times >= overlap_start) & (ecg_peak_times <= overlap_end)
        
        radar_trimmed = radar_peak_times[radar_mask]
        ecg_trimmed = ecg_peak_times[ecg_mask]
        
        return radar_trimmed, ecg_trimmed, overlap_start, overlap_end

    def calculate_synchronized_rr_intervals(self, radar_peaks_trimmed, ecg_peaks_trimmed, 
                                        time_offset=0.0, min_rr=300, max_rr=1500):
        """
        Calculate RR intervals from synchronized, trimmed peak times
        """
        # Apply time offset to radar peaks
        radar_peaks_sync = radar_peaks_trimmed - time_offset
        
        # Calculate RR intervals
        radar_rr = np.diff(radar_peaks_sync) * 1000  # Convert to ms
        radar_rr_times = radar_peaks_sync[:-1]
        
        ecg_rr = np.diff(ecg_peaks_trimmed) * 1000  # Convert to ms
        ecg_rr_times = ecg_peaks_trimmed[:-1]
        
        # Filter RR intervals
        radar_valid_mask = (radar_rr >= min_rr) & (radar_rr <= max_rr)
        ecg_valid_mask = (ecg_rr >= min_rr) & (ecg_rr <= max_rr)
        
        radar_rr_filtered = radar_rr[radar_valid_mask]
        radar_rr_times_filtered = radar_rr_times[radar_valid_mask]
        
        ecg_rr_filtered = ecg_rr[ecg_valid_mask]
        ecg_rr_times_filtered = ecg_rr_times[ecg_valid_mask]
        
        return (radar_rr_filtered, radar_rr_times_filtered, 
                ecg_rr_filtered, ecg_rr_times_filtered)

    def plot_peak_alignment_analysis(self, radar_peaks, ecg_peaks, radar_rr, radar_rr_times, 
                                    ecg_rr, ecg_rr_times, alignment_metrics):
        """
        Create detailed plots for peak alignment analysis
        """
        # Clear existing plots
        for ax in [self.ax_rr_radar, self.ax_rr_ecg, self.ax_rr_compare]:
            ax.clear()
        
        # Plot 1: Radar RR intervals
        self.ax_rr_radar.plot(radar_rr_times, radar_rr, 'b.-', markersize=4, linewidth=1, alpha=0.8)
        self.ax_rr_radar.set_title(f'Radar RR Intervals (Synchronized)\nRate: {alignment_metrics["radar_rate_bpm"]:.1f} BPM')
        self.ax_rr_radar.set_xlabel('Time (s)')
        self.ax_rr_radar.set_ylabel('RR Interval (ms)')
        self.ax_rr_radar.grid(True, alpha=0.3)
        
        # Add statistical info
        if len(radar_rr) > 0:
            radar_mean = np.mean(radar_rr)
            radar_std = np.std(radar_rr)
            self.ax_rr_radar.axhline(y=radar_mean, color='blue', linestyle='--', alpha=0.7, 
                                    label=f'Mean: {radar_mean:.0f}±{radar_std:.0f} ms')
            self.ax_rr_radar.legend()
        
        # Plot 2: ECG RR intervals
        self.ax_rr_ecg.plot(ecg_rr_times, ecg_rr, 'r.-', markersize=4, linewidth=1, alpha=0.8)
        self.ax_rr_ecg.set_title(f'ECG RR Intervals (Reference)\nRate: {alignment_metrics["ecg_rate_bpm"]:.1f} BPM')
        self.ax_rr_ecg.set_xlabel('Time (s)')
        self.ax_rr_ecg.set_ylabel('RR Interval (ms)')
        self.ax_rr_ecg.grid(True, alpha=0.3)
        
        # Add statistical info
        if len(ecg_rr) > 0:
            ecg_mean = np.mean(ecg_rr)
            ecg_std = np.std(ecg_rr)
            self.ax_rr_ecg.axhline(y=ecg_mean, color='red', linestyle='--', alpha=0.7,
                                label=f'Mean: {ecg_mean:.0f}±{ecg_std:.0f} ms')
            self.ax_rr_ecg.legend()
        
        # Plot 3: Overlaid comparison with alignment info
        self.ax_rr_compare.plot(radar_rr_times, radar_rr, 'b.-', markersize=3, linewidth=1.5, 
                            alpha=0.7, label=f'Radar ({len(radar_rr)} intervals)')
        self.ax_rr_compare.plot(ecg_rr_times, ecg_rr, 'r.-', markersize=3, linewidth=1.5, 
                            alpha=0.7, label=f'ECG ({len(ecg_rr)} intervals)')
        
        # Add alignment quality indicator
        quality_color = {'good': 'green', 'fair': 'orange', 'poor': 'red'}
        alignment_quality = alignment_metrics['alignment_quality']
        
        self.ax_rr_compare.set_title(f'RR Intervals Comparison\n'
                                    f'Alignment: {alignment_quality.upper()} '
                                    f'(Δ Rate: {alignment_metrics["rate_difference_bpm"]:.1f} BPM)')
        
        # Color the title based on alignment quality
        title_color = quality_color.get(alignment_quality, 'black')
        self.ax_rr_compare.title.set_color(title_color)
        
        self.ax_rr_compare.set_xlabel('Time (s)')
        self.ax_rr_compare.set_ylabel('RR Interval (ms)')
        self.ax_rr_compare.legend()
        self.ax_rr_compare.grid(True, alpha=0.3)
        
        # Add time range indicators
        if 'overlap_start' in alignment_metrics and 'overlap_end' in alignment_metrics:
            for ax in [self.ax_rr_radar, self.ax_rr_ecg, self.ax_rr_compare]:
                ax.axvline(x=alignment_metrics['overlap_start'], color='green', 
                        linestyle=':', alpha=0.5, label='Overlap Range')
                ax.axvline(x=alignment_metrics['overlap_end'], color='green', 
                        linestyle=':', alpha=0.5)

    def enhanced_calculate_hrv_analysis(self):
        """Enhanced HRV analysis with improved peak alignment and visualization"""
        # Check if we have the required data
        if not hasattr(self, 'paper_analysis_results') or self.paper_analysis_results is None:
            messagebox.showwarning("Warning", "No paper method results available. Please run the paper algorithm first.")
            return
        
        if not self.ecg_loaded:
            messagebox.showwarning("Warning", "No ECG data loaded. Please load ECG data first.")
            return
        
        # Clear results
        self.hrv_results_text.delete(1.0, tk.END)
        for ax in [self.ax_rr_radar, self.ax_rr_ecg, self.ax_rr_compare, 
                self.ax_hrv_radar, self.ax_hrv_ecg, self.ax_hrv_compare]:
            ax.clear()
        
        try:
            # Get parameters
            window_size = self.hrv_window_var.get()
            overlap_percent = self.hrv_overlap_var.get()
            min_rr = self.min_rr_var.get()
            max_rr = self.max_rr_var.get()
            
            # Get peak times
            radar_peaks = self.paper_analysis_results['final_peaks']
            radar_times = self.signals['timestamps']
            radar_peak_times = radar_times[radar_peaks]
            
            ecg_peaks = self.detect_r_peaks(self.ecg_data, self.ecg_fs)
            ecg_peak_times = self.ecg_time[ecg_peaks]
            
            self.hrv_results_text.insert(tk.END, "ENHANCED HRV ANALYSIS WITH PEAK ALIGNMENT\n")
            self.hrv_results_text.insert(tk.END, "=" * 55 + "\n")
            self.hrv_results_text.insert(tk.END, f"Window Size: {window_size} s\n")
            self.hrv_results_text.insert(tk.END, f"Overlap: {overlap_percent}%\n")
            self.hrv_results_text.insert(tk.END, f"RR Filter: {min_rr}-{max_rr} ms\n\n")
            
            # STEP 1: Check peak time alignment
            alignment_metrics, radar_overlap, ecg_overlap = self.check_peak_time_alignment(
                radar_peak_times, ecg_peak_times)
            
            self.hrv_results_text.insert(tk.END, f"PEAK ALIGNMENT ANALYSIS:\n")
            self.hrv_results_text.insert(tk.END, f"Overlap duration: {alignment_metrics['overlap_duration']:.1f} s\n")
            self.hrv_results_text.insert(tk.END, f"Radar peaks in overlap: {alignment_metrics['radar_peaks_overlap']}\n")
            self.hrv_results_text.insert(tk.END, f"ECG peaks in overlap: {alignment_metrics['ecg_peaks_overlap']}\n")
            self.hrv_results_text.insert(tk.END, f"Alignment quality: {alignment_metrics['alignment_quality'].upper()}\n")
            
            if 'max_correlation' in alignment_metrics:
                self.hrv_results_text.insert(tk.END, f"Cross-correlation: {alignment_metrics['max_correlation']:.3f}\n")
            if 'time_offset_suggestion' in alignment_metrics:
                self.hrv_results_text.insert(tk.END, f"Suggested time offset: {alignment_metrics['time_offset_suggestion']:.3f} s\n")
            
            # STEP 2: Trim to overlapping range
            radar_trimmed, ecg_trimmed, overlap_start, overlap_end = self.trim_to_overlapping_range(
                radar_peak_times, ecg_peak_times)
            
            self.hrv_results_text.insert(tk.END, f"\nTRIMMED TO OVERLAP RANGE:\n")
            self.hrv_results_text.insert(tk.END, f"Time range: {overlap_start:.1f} - {overlap_end:.1f} s\n")
            self.hrv_results_text.insert(tk.END, f"Radar peaks after trimming: {len(radar_trimmed)}\n")
            self.hrv_results_text.insert(tk.END, f"ECG peaks after trimming: {len(ecg_trimmed)}\n")
            
            # STEP 3: Apply synchronization if enabled
            time_offset = 0.0
            if hasattr(self, 'enable_sync_var') and self.enable_sync_var.get():
                if 'time_offset_suggestion' in alignment_metrics:
                    max_offset = self.max_offset_var.get()
                    suggested_offset = alignment_metrics['time_offset_suggestion']
                    
                    if abs(suggested_offset) <= max_offset:
                        time_offset = suggested_offset
                        self.hrv_results_text.insert(tk.END, f"Applied time offset: {time_offset:.3f} s\n")
                    else:
                        self.hrv_results_text.insert(tk.END, f"Offset {suggested_offset:.3f} s exceeds max {max_offset} s - not applied\n")
            
            # STEP 4: Calculate synchronized RR intervals
            radar_rr, radar_rr_times, ecg_rr, ecg_rr_times = self.calculate_synchronized_rr_intervals(
                radar_trimmed, ecg_trimmed, time_offset, min_rr, max_rr)
            
            self.hrv_results_text.insert(tk.END, f"\nSYNCHRONIZED RR INTERVALS:\n")
            self.hrv_results_text.insert(tk.END, f"Valid radar RR intervals: {len(radar_rr)}\n")
            self.hrv_results_text.insert(tk.END, f"Valid ECG RR intervals: {len(ecg_rr)}\n")
            
            if len(radar_rr) > 0 and len(ecg_rr) > 0:
                # Calculate basic statistics
                radar_rr_mean = np.mean(radar_rr)
                ecg_rr_mean = np.mean(ecg_rr)
                rr_bias = radar_rr_mean - ecg_rr_mean
                
                self.hrv_results_text.insert(tk.END, f"Radar RR mean: {radar_rr_mean:.1f} ms\n")
                self.hrv_results_text.insert(tk.END, f"ECG RR mean: {ecg_rr_mean:.1f} ms\n")
                self.hrv_results_text.insert(tk.END, f"RR bias (radar-ECG): {rr_bias:.1f} ms\n")
            
            # STEP 5: Create alignment visualization
            self.plot_peak_alignment_analysis(radar_peaks, ecg_peaks, radar_rr, radar_rr_times,
                                            ecg_rr, ecg_rr_times, alignment_metrics)
            
            # STEP 6: Continue with HRV parameter calculation (existing code)
            radar_stats, radar_hrv, _ = self.calculate_hrv_parameters(radar_rr, window_size, overlap_percent)
            ecg_stats, ecg_hrv, _ = self.calculate_hrv_parameters(ecg_rr, window_size, overlap_percent)
            
            # STEP 7: Plot HRV parameters (reuse existing plotting code)
            # ... existing HRV plotting code ...
            
            # STEP 8: Calculate comparison metrics
            if len(radar_rr) > 0 and len(ecg_rr) > 0:
                # Synchronize for direct comparison
                min_len = min(len(radar_hrv['SDNN']), len(ecg_hrv['SDNN']))
                if min_len > 0:
                    radar_sdnn_sync = radar_hrv['SDNN'][:min_len]
                    ecg_sdnn_sync = ecg_hrv['SDNN'][:min_len]
                    
                    # Calculate metrics
                    sdnn_rmse = np.sqrt(np.mean((radar_sdnn_sync - ecg_sdnn_sync)**2))
                    sdnn_nrmse = sdnn_rmse / np.mean(ecg_sdnn_sync) * 100
                    sdnn_correlation = np.corrcoef(radar_sdnn_sync, ecg_sdnn_sync)[0, 1]
                    
                    self.hrv_results_text.insert(tk.END, f"\nCOMPARISON METRICS:\n")
                    self.hrv_results_text.insert(tk.END, f"SDNN RMSE: {sdnn_rmse:.2f} ms\n")
                    self.hrv_results_text.insert(tk.END, f"SDNN NRMSE: {sdnn_nrmse:.1f}%\n")
                    self.hrv_results_text.insert(tk.END, f"SDNN Correlation: {sdnn_correlation:.3f}\n")
            
            # STEP 9: Provide alignment recommendations
            self.hrv_results_text.insert(tk.END, f"\nALIGNMENT RECOMMENDATIONS:\n")
            
            quality = alignment_metrics['alignment_quality']
            if quality == 'poor':
                self.hrv_results_text.insert(tk.END, f"⚠️ Poor alignment detected!\n")
                self.hrv_results_text.insert(tk.END, f"• Check signal quality and processing parameters\n")
                self.hrv_results_text.insert(tk.END, f"• Consider adjusting time window or filters\n")
                if 'time_offset_suggestion' in alignment_metrics:
                    self.hrv_results_text.insert(tk.END, f"• Try time offset: {alignment_metrics['time_offset_suggestion']:.3f} s\n")
            elif quality == 'fair':
                self.hrv_results_text.insert(tk.END, f"⚠️ Fair alignment - proceed with caution\n")
                self.hrv_results_text.insert(tk.END, f"• Results may have moderate reliability\n")
            else:
                self.hrv_results_text.insert(tk.END, f"✅ Good alignment achieved!\n")
                self.hrv_results_text.insert(tk.END, f"• Results should be reliable\n")
            
            # Store enhanced results
            self.hrv_analysis_results = {
                # ... existing results ...
                'alignment_metrics': alignment_metrics,
                'time_offset_applied': time_offset,
                'trimmed_radar_peaks': radar_trimmed,
                'trimmed_ecg_peaks': ecg_trimmed,
                'overlap_range': (overlap_start, overlap_end)
            }
            
            self.fig_hrv.tight_layout()
            self.canvas_hrv.draw()
            
        except Exception as e:
            messagebox.showerror("Error", f"Enhanced HRV analysis failed: {str(e)}")
            print(f"Enhanced HRV analysis error: {e}")
            traceback.print_exc()

    def save_hrv_results(self):
        """Save HRV analysis results"""
        if not hasattr(self, 'hrv_analysis_results'):
            messagebox.showwarning("Warning", "No HRV results to save. Please calculate HRV first.")
            return
        
        # Save figure
        fig_file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("All files", "*.*")],
            title="Save HRV Analysis Results"
        )
        if fig_file_path:
            self.fig_hrv.savefig(fig_file_path, dpi=300, bbox_inches='tight')
        
        # Save data
        data_file_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            title="Save HRV Data"
        )
        if data_file_path:
            results = self.hrv_analysis_results
            
            # Create comprehensive dataframe
            max_len = max(
                len(results['radar_rr']),
                len(results['ecg_rr']),
                len(results['radar_hrv']['SDNN']),
                len(results['ecg_hrv']['SDNN'])
            )
            
            # Initialize arrays with NaN
            data_dict = {}
            
            # RR intervals
            radar_rr_padded = np.full(max_len, np.nan)
            radar_rr_padded[:len(results['radar_rr'])] = results['radar_rr']
            data_dict['Radar_RR_ms'] = radar_rr_padded
            
            ecg_rr_padded = np.full(max_len, np.nan)
            ecg_rr_padded[:len(results['ecg_rr'])] = results['ecg_rr']
            data_dict['ECG_RR_ms'] = ecg_rr_padded
            
            # HRV parameters
            for param in ['SDNN', 'RMSSD', 'pNN50', 'HR_mean']:
                radar_param_padded = np.full(max_len, np.nan)
                radar_param_padded[:len(results['radar_hrv'][param])] = results['radar_hrv'][param]
                data_dict[f'Radar_{param}'] = radar_param_padded
                
                ecg_param_padded = np.full(max_len, np.nan)
                ecg_param_padded[:len(results['ecg_hrv'][param])] = results['ecg_hrv'][param]
                data_dict[f'ECG_{param}'] = ecg_param_padded
            
            df = pd.DataFrame(data_dict)
            df.to_csv(data_file_path, index=False)
            
            # Save summary statistics
            summary_file_path = data_file_path.replace('.csv', '_summary.txt')
            with open(summary_file_path, 'w') as f:
                f.write("HRV ANALYSIS SUMMARY\n")
                f.write("=" * 50 + "\n\n")
                
                f.write("RADAR HRV STATISTICS:\n")
                for key, value in results['radar_stats'].items():
                    f.write(f"{key}: {value:.3f}\n")
                
                f.write("\nECG HRV STATISTICS:\n")
                for key, value in results['ecg_stats'].items():
                    f.write(f"{key}: {value:.3f}\n")
                
                f.write("\nCOMPARISON METRICS:\n")
                for key, value in results['comparison_metrics'].items():
                    f.write(f"{key}: {value:.3f}\n")
            
            messagebox.showinfo("Save Complete", 
                f"HRV results saved to:\n{fig_file_path}\n{data_file_path}\n{summary_file_path}")

    def create_main_interface(self):
        self.root = tk.Tk()
        self.root.title("EEMD Radar Signal Analysis with Paper Method Peak Detection")
        self.root.geometry("1800x1000")
        
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill=tk.BOTH, expand=True)
        
        # Main Analysis Tab (Scrollable)
        main_tab = ttk.Frame(notebook)
        notebook.add(main_tab, text="Main Analysis")
        self.create_main_analysis_tab(main_tab)
        
        # Heart Rate EEMD Tab (Scrollable) 
        hr_eemd_tab = ttk.Frame(notebook)
        notebook.add(hr_eemd_tab, text="Heart Rate EEMD")
        self.create_hr_eemd_tab(hr_eemd_tab)
        
        # Paper Method Tab (NEW - Clean Implementation)
        paper_tab = ttk.Frame(notebook)
        notebook.add(paper_tab, text="Paper Method Steps 2-7")
        self.create_paper_method_tab(paper_tab)

        # HRV Analysis Tab (NEW)
        hrv_tab = ttk.Frame(notebook)
        notebook.add(hrv_tab, text="HRV Analysis")
        self.create_hrv_analysis_tab(hrv_tab)

        # HRV Feature Extraction Tab (NEW)
        hrv_features_tab = ttk.Frame(notebook)
        notebook.add(hrv_features_tab, text="HRV Features")
        self.create_hrv_features_tab(hrv_features_tab)
        
        self.root.mainloop()

    def apply_moving_average(self, signal, window_size):
        """Apply moving average filter to signal"""
        if len(signal) < window_size:
            return signal
        
        if window_size <= 0:
            return signal
        
        # Method: Centered moving average (better for preserving signal shape)
        filtered_signal = np.zeros_like(signal)
        half_window = window_size // 2
        
        for i in range(len(signal)):
            start_idx = max(0, i - half_window)
            end_idx = min(len(signal), i + half_window + 1)
            filtered_signal[i] = np.mean(signal[start_idx:end_idx])
        
        return filtered_signal
    
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

    def step4b_consolidate_waves(self, filtered_peaks, filtered_values, time_data, fs):
        if len(filtered_peaks) <= 1:
            return filtered_peaks, filtered_values
        consolidation_window = 0.6  # seconds
        window_samples = int(consolidation_window * fs)
        
        # Convert peaks to time domain for easier processing
        peak_times = time_data[filtered_peaks]
        
        consolidated_peaks = []
        consolidated_values = []
        
        # Group peaks that are within consolidation window
        i = 0
        while i < len(filtered_peaks):
            current_time = peak_times[i]
            current_group_indices = [i]  # Start group with current peak
            current_group_peaks = [filtered_peaks[i]]
            current_group_values = [filtered_values[i]]
            
            # Find all peaks within consolidation window
            j = i + 1
            while j < len(filtered_peaks) and (peak_times[j] - current_time) <= consolidation_window:
                current_group_indices.append(j)
                current_group_peaks.append(filtered_peaks[j])
                current_group_values.append(filtered_values[j])
                j += 1
            
            # Within this group, keep only the peak with highest absolute value
            group_values_abs = np.abs(current_group_values)
            highest_idx = np.argmax(group_values_abs)
            
            consolidated_peaks.append(current_group_peaks[highest_idx])
            consolidated_values.append(current_group_values[highest_idx])
            
            # Move to next group (skip all peaks in current group)
            i = j
        
        return np.array(consolidated_peaks), np.array(consolidated_values)
    # NEW FUNCTION - Add this to the EEMDPaperAnalyzer class
    def compute_adaptive_rr_constraints(self, peaks, time_data, window_size=30):
        """
        Compute adaptive RR interval constraints from most stable segment of peaks
        
        Parameters:
        - peaks: array of peak indices
        - time_data: time array
        - window_size: size of sliding window to find stable segment
        
        Returns:
        - rr_mean: mean RR interval from most stable segment
        - rr_min: minimum allowed RR interval (0.75 × mean)
        - rr_max: maximum allowed RR interval (1.25 × mean)
        - stable_segment_info: dict with stability analysis info
        """
        if len(peaks) < 3:
            # Not enough peaks for RR analysis, return default values
            default_rr = 0.8  # Default ~75 BPM
            return default_rr, default_rr * 0.75, default_rr * 1.25, {'status': 'insufficient_peaks'}
        
        # Compute all RR intervals
        peak_times = time_data[peaks]
        rr_intervals = np.diff(peak_times)
        
        if len(rr_intervals) < window_size:
            # Use all intervals if less than window size
            rr_mean = np.mean(rr_intervals)
            rr_std = np.std(rr_intervals)
            stable_segment_info = {
                'status': 'all_intervals_used',
                'segment_start': 0,
                'segment_end': len(rr_intervals),
                'segment_size': len(rr_intervals),
                'segment_std': rr_std,
                'all_rr_std': rr_std
            }
        else:
            # Find sliding window with lowest standard deviation
            min_std = float('inf')
            best_start = 0
            
            for start in range(len(rr_intervals) - window_size + 1):
                end = start + window_size
                segment_rr = rr_intervals[start:end]
                segment_std = np.std(segment_rr)
                
                if segment_std < min_std:
                    min_std = segment_std
                    best_start = start
            
            # Extract most stable segment
            best_end = best_start + window_size
            stable_rr_segment = rr_intervals[best_start:best_end]
            rr_mean = np.mean(stable_rr_segment)
            
            stable_segment_info = {
                'status': 'stable_segment_found',
                'segment_start': best_start,
                'segment_end': best_end,
                'segment_size': window_size,
                'segment_std': min_std,
                'all_rr_std': np.std(rr_intervals),
                'stability_improvement': (np.std(rr_intervals) - min_std) / np.std(rr_intervals) * 100
            }
        
        # Define adaptive thresholds
        rr_min = rr_mean * 0.75
        rr_max = rr_mean * 1.25
        
        stable_segment_info.update({
            'rr_mean': rr_mean,
            'rr_min': rr_min,
            'rr_max': rr_max,
            'mean_hr_bpm': 60.0 / rr_mean if rr_mean > 0 else 0
        })
        
        return rr_mean, rr_min, rr_max, stable_segment_info

    # FIXED FUNCTION - Replace the problematic step4c_adaptive_rr_validation_enhanced
    def step4c_adaptive_rr_validation_enhanced(self, consolidated_peaks, consolidated_values, 
                                            candidate_peaks, candidate_values, time_data, signal):
        """
        Enhanced Step 4C: Apply adaptive RR interval constraints with stricter close peak removal
        """
        if len(consolidated_peaks) < 3:
            return consolidated_peaks, consolidated_values, {'status': 'insufficient_peaks_for_rr'}
        
        try:
            # Step 1: Compute adaptive RR constraints from initial coarse detection
            rr_mean, rr_min, rr_max, stability_info = self.compute_adaptive_rr_constraints(
                consolidated_peaks, time_data, window_size=min(30, len(consolidated_peaks)//2))
            
            # ENHANCEMENT 1: More aggressive minimum RR threshold
            # Use stricter minimum based on physiological limits
            physiological_min_rr = 0.4  # 400ms = 150 BPM maximum
            adaptive_min_rr = max(physiological_min_rr, rr_mean * 0.6)  # Changed from 0.75 to 0.6
            
            # ENHANCEMENT 2: Two-pass validation for stubborn close peaks
            validated_peaks, validated_values, pass1_stats = self.rr_validation_pass(
                consolidated_peaks, consolidated_values, candidate_peaks, candidate_values, 
                time_data, rr_mean, adaptive_min_rr, rr_max, pass_name="Pass 1")
            
            # Check if we still have close peaks after first pass
            if len(validated_peaks) > 1:
                close_peaks_remain = self.check_for_close_peaks(validated_peaks, time_data, adaptive_min_rr)
                
                if close_peaks_remain['has_close_peaks']:
                    # ENHANCEMENT 3: Second pass with even stricter criteria
                    stricter_min_rr = max(physiological_min_rr, rr_mean * 0.7)  # Even stricter
                    
                    validated_peaks, validated_values, pass2_stats = self.rr_validation_pass(
                        validated_peaks, validated_values, candidate_peaks, candidate_values,
                        time_data, rr_mean, stricter_min_rr, rr_max, pass_name="Pass 2 (Stricter)")
                    
                    # ENHANCEMENT 4: Final close peak cleanup
                    validated_peaks, validated_values, cleanup_stats = self.final_close_peak_cleanup(
                        validated_peaks, validated_values, time_data, adaptive_min_rr)
                    
                    final_stats = {
                        'pass1_stats': pass1_stats,
                        'pass2_stats': pass2_stats,
                        'cleanup_stats': cleanup_stats,
                        'enhancement_applied': True
                    }
                else:
                    final_stats = {
                        'pass1_stats': pass1_stats,
                        'enhancement_applied': False,
                        'message': 'No close peaks detected after first pass'
                    }
            else:
                final_stats = {
                    'pass1_stats': pass1_stats,
                    'enhancement_applied': False,
                    'message': 'Insufficient peaks for second pass'
                }
            
            # Compile comprehensive statistics - FIXED
            rr_stats = {
                'status': 'completed_enhanced',
                'initial_peaks': len(consolidated_peaks),
                'validated_peaks': len(validated_peaks),
                'rejected_peaks': len(consolidated_peaks) - len(validated_peaks),  # FIXED: Use count instead of list
                'recovered_peaks': pass1_stats.get('recovered', 0),  # FIXED: Safe access
                'rr_constraints': {
                    'rr_mean': rr_mean,
                    'rr_min': adaptive_min_rr,
                    'rr_max': rr_max,
                    'mean_hr_bpm': 60.0 / rr_mean if rr_mean > 0 else 0
                },
                'stability_info': stability_info,
                'enhancement_details': final_stats
            }
            
            return np.array(validated_peaks), np.array(validated_values), rr_stats
            
        except Exception as e:
            print(f"Error in enhanced RR validation: {e}")
            # Fallback to simple validation
            return self.simple_close_peak_removal(consolidated_peaks, consolidated_values, time_data)
    
    #
    def step4c_comprehensive_rr_correction_adaptive(self, consolidated_peaks, consolidated_values, 
                                                candidate_peaks, candidate_values, time_data, signal):
        if len(consolidated_peaks) < 5:
            return consolidated_peaks, consolidated_values, {'status': 'insufficient_peaks'}
        
        # PHASE 1: Find robust RR baseline from most stable segment
        try:
            stable_rr_mean, baseline_stats = self.find_robust_rr_baseline(consolidated_peaks, time_data)
        except Exception as e:
            # Fallback: calculate simple RR mean if robust method fails
            peak_times = time_data[consolidated_peaks]
            rr_intervals = np.diff(peak_times)
            stable_rr_mean = np.median(rr_intervals)  # Use median for robustness
            baseline_stats = {'method': 'fallback_median', 'rr_mean': stable_rr_mean}
            self.debug_print(f"Fallback RR calculation: {stable_rr_mean:.3f}s")
        
        # Ensure stable_rr_mean is valid
        if stable_rr_mean <= 0 or stable_rr_mean > 3.0:
            # Ultimate fallback for invalid RR
            stable_rr_mean = 0.8  # Default ~75 BPM
            self.debug_print(f"Using default RR mean: {stable_rr_mean:.3f}s")
        
        # PHASE 2: Define fully adaptive thresholds based on rr_mean
        rr_min_adaptive = max(0.35, 0.65 * stable_rr_mean)      # Adaptive minimum RR
        rr_gap_threshold = 1.4 * stable_rr_mean                 # Gap detection threshold
        rr_max_search = 2.5 * stable_rr_mean                    # Maximum search range
        
        self.debug_print(f"Adaptive HR Correction - RR Mean: {stable_rr_mean:.3f}s")
        self.debug_print(f"Adaptive Thresholds - Min: {rr_min_adaptive:.3f}s, Gap: {rr_gap_threshold:.3f}s, Max: {rr_max_search:.3f}s")
        
        # PHASE 3: Multi-pass adaptive correction
        current_peaks = consolidated_peaks.copy()
        current_values = consolidated_values.copy()
        
        correction_stats = {
            'iterations': 0,
            'total_removed': 0,
            'total_added': 0,
            'rr_mean': stable_rr_mean,
            'adaptive_thresholds': {
                'rr_min_adaptive': rr_min_adaptive,
                'rr_gap_threshold': rr_gap_threshold,
                'rr_max_search': rr_max_search
            }
        }
        for iteration in range(7):
            correction_stats['iterations'] = iteration + 1
            
            # Step A: Adaptive close peak removal
            cleaned_peaks, cleaned_values, removed_count = self.remove_close_peaks_adaptive(
                current_peaks, current_values, time_data, signal, stable_rr_mean)
            
            # Step B: Adaptive missing peak restoration
            final_peaks, final_values, added_count = self.add_missing_peaks_adaptive(
                cleaned_peaks, cleaned_values, candidate_peaks, candidate_values, 
                time_data, signal, stable_rr_mean)
            correction_stats['total_removed'] += removed_count
            correction_stats['total_added'] += added_count
            
            self.debug_print(f"Adaptive Iteration {iteration+1}: Removed {removed_count}, Added {added_count}, Total peaks: {len(final_peaks)}")
            
            # Adaptive convergence check
            if iteration >= 2 and removed_count == 0 and added_count == 0:
                self.debug_print(f"Adaptive correction converged after {iteration+1} iterations")
                break
            elif iteration >= 5 and removed_count + added_count <= 1:
                self.debug_print(f"Adaptive correction near-convergence after {iteration+1} iterations")
                break
            
            # Update for next iteration
            current_peaks = final_peaks
            current_values = final_values
            
            # ADAPTIVE ENHANCEMENT: Update RR mean every few iterations for dynamic adaptation
            if iteration % 2 == 1 and len(final_peaks) >= 5:
                updated_rr_mean, _ = self.find_robust_rr_baseline(final_peaks, time_data)
                if abs(updated_rr_mean - stable_rr_mean) / stable_rr_mean < 0.3:  # Only update if change < 30%
                    stable_rr_mean = updated_rr_mean
                    self.debug_print(f"Updated RR mean to {stable_rr_mean:.3f}s")
        
        # PHASE 4: Final validation
        final_stats = self.validate_final_rr_intervals_adaptive(final_peaks, time_data, stable_rr_mean)
        correction_stats.update(final_stats)
        
        return final_peaks, final_values, correction_stats


    def add_missing_peaks_adaptive(self, peaks, values, candidate_peaks, candidate_values, 
                                time_data, signal, rr_mean):
        if len(peaks) < 2:
            return peaks, values, 0
        
        # Compute adaptive gap detection threshold
        rr_gap_threshold = 1.4 * rr_mean  # ✅ Adaptive gap threshold
        
        peak_times = time_data[peaks]
        augmented_peaks = list(peaks)
        augmented_values = list(values)
        added_count = 0
        
        i = 0
        while i < len(augmented_peaks) - 1:
            current_time = time_data[augmented_peaks[i]]
            next_time = time_data[augmented_peaks[i + 1]]
            gap_duration = next_time - current_time
            
            # ✅ Check if gap exceeds adaptive threshold
            if gap_duration > rr_gap_threshold:
                self.debug_print(f"Adaptive gap detected: {gap_duration:.3f}s (threshold: {rr_gap_threshold:.3f}s)")
                
                # ✅ Compute expected number of missing peaks
                expected_count = round((gap_duration / rr_mean)) - 1
                expected_count = max(0, min(3, expected_count))  # Limit to max 3 peaks per gap
                
                self.debug_print(f"Expected {expected_count} peaks in gap (gap/rr_mean: {gap_duration/rr_mean:.2f})")
                
                if expected_count > 0:
                    # ✅ Find candidate peaks in the gap
                    gap_mask = (candidate_peaks > augmented_peaks[i]) & (candidate_peaks < augmented_peaks[i + 1])
                    gap_candidates = candidate_peaks[gap_mask]
                    gap_values = candidate_values[gap_mask]
                    
                    restored_peaks = []
                    
                    if len(gap_candidates) > 0:
                        # ✅ STRATEGY 1: Select best candidates using adaptive criteria
                        restored_peaks = self.select_adaptive_gap_candidates(
                            current_time, next_time, gap_candidates, gap_values, time_data,
                            rr_mean, expected_count, signal)
                    
                    # ✅ STRATEGY 2: Signal-based fallback if insufficient candidates
                    if len(restored_peaks) < expected_count:
                        additional_peaks = self.signal_based_adaptive_detection(
                            current_time, next_time, rr_mean, time_data, signal, 
                            expected_count - len(restored_peaks))
                        
                        for peak_time, peak_val in additional_peaks:
                            # Convert time back to index
                            peak_idx = np.argmin(np.abs(time_data - peak_time))
                            restored_peaks.append((peak_idx, peak_val))
                    
                    # ✅ Insert all restored peaks in chronological order
                    restored_peaks.sort(key=lambda x: time_data[x[0]])  # Sort by time
                    
                    for restore_idx, (peak_idx, peak_val) in enumerate(restored_peaks):
                        insert_position = i + 1 + restore_idx
                        augmented_peaks.insert(insert_position, peak_idx)
                        augmented_values.insert(insert_position, peak_val)
                        added_count += 1
                        
                        self.debug_print(f"Restored peak at {time_data[peak_idx]:.3f}s (magnitude: {peak_val:.4f})")
            
            # ✅ Update i to account for inserted peaks
            current_gap_additions = 0
            while (i + 1 + current_gap_additions < len(augmented_peaks) and 
                time_data[augmented_peaks[i + 1 + current_gap_additions]] < next_time):
                current_gap_additions += 1
            
            i += 1 + current_gap_additions
        
        return np.array(augmented_peaks), np.array(augmented_values), added_count

    def evaluate_local_signal_quality(self, peak_idx, signal, time_data):
        """
        Evaluate the local signal characteristics around a peak
        """
        window_size = 10  # samples
        start_idx = max(0, peak_idx - window_size)
        end_idx = min(len(signal), peak_idx + window_size + 1)
        
        if end_idx - start_idx < 3:
            return 0.5  # Neutral score for edge cases
        
        local_signal = signal[start_idx:end_idx]
        peak_value = signal[peak_idx]
        
        # Check if it's actually a local maximum/minimum
        is_local_max = np.all(local_signal <= peak_value)
        is_local_min = np.all(local_signal >= peak_value)
        
        if is_local_max or is_local_min:
            prominence_score = 1.0
        else:
            # Calculate relative prominence
            local_max = np.max(local_signal)
            local_min = np.min(local_signal)
            if local_max != local_min:
                prominence_score = abs(peak_value - np.median(local_signal)) / (local_max - local_min)
            else:
                prominence_score = 0.5
        
        return min(1.0, prominence_score)

    def remove_close_peaks_adaptive(self, peaks, values, time_data, signal, rr_mean):
        if len(peaks) < 2:
            return peaks, values, 0
        
        # Compute adaptive minimum RR threshold
        rr_min_adaptive = max(0.35, 0.65 * rr_mean)  # Adaptive minimum as specified
        
        peak_times = time_data[peaks]
        cleaned_peaks = []
        cleaned_values = []
        removed_count = 0
        
        i = 0
        while i < len(peaks):
            current_peak = peaks[i]
            current_value = values[i]
            current_time = peak_times[i]
            
            # ✅ Group all peaks within adaptive minimum threshold
            close_group = [(current_peak, current_value, current_time)]
            j = i + 1
            
            while j < len(peaks) and (peak_times[j] - current_time) < rr_min_adaptive:
                close_group.append((peaks[j], values[j], peak_times[j]))
                j += 1
            
            if len(close_group) > 1:
                # ✅ Multiple close peaks detected - keep only the best one
                best_peak = self.select_best_from_close_group_adaptive(close_group, time_data, signal, rr_mean)
                cleaned_peaks.append(best_peak[0])
                cleaned_values.append(best_peak[1])
                removed_count += len(close_group) - 1
                
                self.debug_print(f"Adaptive: Close group of {len(close_group)} peaks, kept peak at {best_peak[2]:.3f}s (RR_min: {rr_min_adaptive:.3f}s)")
            else:
                # ✅ Single peak - keep it (no close neighbors)
                cleaned_peaks.append(current_peak)
                cleaned_values.append(current_value)
            
            i = j  # Move to next group
        
        return np.array(cleaned_peaks), np.array(cleaned_values), removed_count

    def evaluate_signal_smoothness(self, peak_idx, signal, window=5):
        """
        Evaluate signal smoothness around a peak
        """
        start_idx = max(0, peak_idx - window)
        end_idx = min(len(signal), peak_idx + window + 1)
        
        if end_idx - start_idx < 3:
            return 0.5
        
        local_signal = signal[start_idx:end_idx]
        
        # Calculate second derivative (measure of smoothness)
        if len(local_signal) >= 3:
            second_derivative = np.diff(local_signal, n=2)
            smoothness = 1.0 / (1.0 + np.std(second_derivative))
        else:
            smoothness = 0.5
        
        return min(1.0, smoothness)


    # NEW FUNCTION - Find the most stable RR baseline
    def find_robust_rr_baseline(self, peaks, time_data, min_segment_size=10):
        """
        Find the most stable RR interval baseline from the cleanest segment
        """
        if len(peaks) < min_segment_size:
            # Fallback to overall median
            peak_times = time_data[peaks]
            rr_intervals = np.diff(peak_times)
            return np.median(rr_intervals), {'method': 'fallback_median'}
        
        peak_times = time_data[peaks]
        rr_intervals = np.diff(peak_times)
        
        # Try multiple segment sizes to find the most stable one
        best_stability = float('inf')
        best_rr_mean = None
        best_segment_info = None
        
        for segment_size in [min_segment_size, min(20, len(rr_intervals)//2), min(30, len(rr_intervals)//3)]:
            if segment_size > len(rr_intervals):
                continue
                
            # Find segment with lowest coefficient of variation (std/mean)
            for start in range(len(rr_intervals) - segment_size + 1):
                end = start + segment_size
                segment = rr_intervals[start:end]
                
                # Filter out obvious outliers (> 2 std from median)
                segment_median = np.median(segment)
                segment_std = np.std(segment)
                outlier_mask = np.abs(segment - segment_median) <= 2 * segment_std
                clean_segment = segment[outlier_mask]
                
                if len(clean_segment) >= segment_size * 0.7:  # At least 70% of segment is clean
                    stability = np.std(clean_segment) / np.mean(clean_segment)  # Coefficient of variation
                    
                    if stability < best_stability:
                        best_stability = stability
                        best_rr_mean = np.mean(clean_segment)
                        best_segment_info = {
                            'start': start,
                            'end': end,
                            'size': len(clean_segment),
                            'stability': stability,
                            'mean': best_rr_mean,
                            'std': np.std(clean_segment)
                        }
        
        if best_rr_mean is None:
            # Ultimate fallback
            best_rr_mean = np.median(rr_intervals)
            best_segment_info = {'method': 'ultimate_fallback'}
        
        return best_rr_mean, best_segment_info

    # UTILITY FUNCTION - Debug printing
    def debug_print(self, message):
        """Print debug messages (can be disabled)"""
        if hasattr(self, 'enable_debug') and self.enable_debug:
            print(f"[DEBUG] {message}")
        # For GUI display
        if hasattr(self, 'paper_results_text'):
            self.paper_results_text.insert(tk.END, f"Debug: {message}\n")


    # INTEGRATION - Update run_paper_algorithm() to use comprehensive correction
    def update_run_paper_algorithm_for_comprehensive_correction(self):
        pass

    # UPDATE RESULTS DISPLAY
    def update_results_display_for_comprehensive_correction(self):
        """
        Update the results display in run_paper_algorithm():
        
        ADD this after the comprehensive correction:
        """
        example_display = '''
        self.paper_results_text.insert(tk.END, f"STEP 4C COMPREHENSIVE RR CORRECTION:\\n")
        self.paper_results_text.insert(tk.END, f"Baseline RR: {rr_correction_stats['rr_mean']:.3f}s\\n")
        self.paper_results_text.insert(tk.END, f"RR Range: {rr_correction_stats['thresholds']['min']:.3f}s - {rr_correction_stats['thresholds']['max']:.3f}s\\n")
        self.paper_results_text.insert(tk.END, f"Iterations: {rr_correction_stats['iterations']}\\n")
        self.paper_results_text.insert(tk.END, f"Peaks removed (too close): {rr_correction_stats['total_removed']}\\n")
        self.paper_results_text.insert(tk.END, f"Peaks added (gaps filled): {rr_correction_stats['total_added']}\\n")
        self.paper_results_text.insert(tk.END, f"Final peak count: {len(final_peaks)}\\n")
        
        if 'quality' in rr_correction_stats:
            self.paper_results_text.insert(tk.END, f"RR Quality: {rr_correction_stats['quality'].upper()}\\n")
        self.paper_results_text.insert(tk.END, "\\n")
        '''

    # SIMPLIFIED INTEGRATION EXAMPLE
    def simple_integration_example(self):
        """
        If you want to test this quickly, add this function and call it after Step 4B:
        """
        def quick_test_comprehensive_correction(self, peaks_before_step5, values_before_step5):
            """Quick test function"""
            corrected_peaks, corrected_values, stats = self.step4c_comprehensive_rr_correction_adaptive(
                peaks_before_step5, values_before_step5, 
                self.paper_analysis_results['step3_candidate_peaks'],
                self.paper_analysis_results['step3_candidate_values'],
                self.signals['timestamps'], 
                self.latest_combined_imf
            )
            
            print(f"Before: {len(peaks_before_step5)} peaks")
            print(f"After: {len(corrected_peaks)} peaks")
            print(f"Removed: {stats.get('total_removed', 0)}")
            print(f"Added: {stats.get('total_added', 0)}")
            
            return corrected_peaks, corrected_values


    # SIMPLIFIED BACKUP FUNCTION - Add this as a fallback
    def simple_close_peak_removal(self, peaks, values, time_data, min_interval=0.4):
        """
        Simple backup function to remove close peaks if enhanced version fails
        """
        if len(peaks) < 2:
            return peaks, values, {'status': 'simple_fallback', 'initial_peaks': len(peaks), 'validated_peaks': len(peaks)}
        
        peak_times = time_data[peaks]
        final_peaks = []
        final_values = []
        
        # Always keep first peak
        final_peaks.append(peaks[0])
        final_values.append(values[0])
        
        for i in range(1, len(peaks)):
            last_time = time_data[final_peaks[-1]]
            current_time = peak_times[i]
            
            if (current_time - last_time) >= min_interval:
                # Far enough from last peak - keep it
                final_peaks.append(peaks[i])
                final_values.append(values[i])
            else:
                # Too close - compare magnitudes and keep higher one
                if abs(values[i]) > abs(final_values[-1]):
                    # Current peak is higher - replace the last one
                    final_peaks[-1] = peaks[i]
                    final_values[-1] = values[i]
                # Otherwise keep the previous peak and skip current
        
        simple_stats = {
            'status': 'simple_fallback_completed',
            'initial_peaks': len(peaks),
            'validated_peaks': len(final_peaks),
            'rejected_peaks': len(peaks) - len(final_peaks),
            'recovered_peaks': 0,
            'method': 'simple_magnitude_based_removal'
        }
        
        return np.array(final_peaks), np.array(final_values), simple_stats


    # NEW FUNCTION - Add this to the EEMDPaperAnalyzer class
    def attempt_peak_recovery(self, last_peak, current_peak, candidate_peaks, candidate_values, 
                            time_data, rr_mean, rr_min, rr_max):
        # Find candidate peaks between last accepted and current
        between_mask = (candidate_peaks > last_peak) & (candidate_peaks < current_peak)
        between_candidates = candidate_peaks[between_mask]
        between_values = candidate_values[between_mask]
        
        if len(between_candidates) == 0:
            return {'recovered_peak': None, 'recovered_value': None, 'method': 'no_candidates'}
        
        # Calculate expected position based on mean RR
        last_time = time_data[last_peak]
        expected_time = last_time + rr_mean
        
        # Find candidate closest to expected time that creates valid RR intervals
        best_candidate = None
        best_value = None
        best_score = float('inf')
        
        for candidate, value in zip(between_candidates, between_values):
            candidate_time = time_data[candidate]
            
            # Check if this candidate creates valid RR intervals
            rr_to_candidate = candidate_time - last_time
            rr_from_candidate = time_data[current_peak] - candidate_time
            
            if rr_min <= rr_to_candidate <= rr_max and rr_min <= rr_from_candidate <= rr_max:
                # Valid RR intervals - score by timing accuracy and magnitude
                timing_error = abs(candidate_time - expected_time)
                magnitude_score = abs(value)  # Prefer higher magnitude
                
                # Combined score (lower is better for timing, higher is better for magnitude)
                score = timing_error / magnitude_score if magnitude_score > 0 else timing_error
                
                if score < best_score:
                    best_score = score
                    best_candidate = candidate
                    best_value = value
        
        return {
            'recovered_peak': best_candidate,
            'recovered_value': best_value,
            'method': 'rr_guided_recovery' if best_candidate is not None else 'recovery_failed',
            'candidates_evaluated': len(between_candidates),
            'expected_time': expected_time
        }
    
    def calculate_rr_mean_for_restoration(self, peaks, time_data, window_size=20):
        if len(peaks) < 3:
            # Not enough peaks, use default RR
            default_rr = 0.8  # ~75 BPM
            return default_rr, {'status': 'insufficient_peaks', 'method': 'default'}
        
        # Compute all RR intervals
        peak_times = time_data[peaks]
        rr_intervals = np.diff(peak_times)
        
        if len(rr_intervals) < window_size:
            # Use all intervals if less than window size
            rr_mean = np.median(rr_intervals)  # Use median for robustness
            return rr_mean, {
                'status': 'all_intervals_used',
                'method': 'median_all',
                'intervals_count': len(rr_intervals),
                'rr_std': np.std(rr_intervals)
            }
        
        # Find sliding window with lowest standard deviation (most stable)
        min_std = float('inf')
        best_start = 0
        
        for start in range(len(rr_intervals) - window_size + 1):
            end = start + window_size
            segment_rr = rr_intervals[start:end]
            segment_std = np.std(segment_rr)
            
            if segment_std < min_std:
                min_std = segment_std
                best_start = start
        
        # Extract most stable segment and compute mean
        best_end = best_start + window_size
        stable_segment = rr_intervals[best_start:best_end]
        rr_mean = np.mean(stable_segment)
        
        stability_stats = {
            'status': 'stable_segment_found',
            'method': 'sliding_window_min_std',
            'segment_start': best_start,
            'segment_end': best_end,
            'segment_size': window_size,
            'segment_std': min_std,
            'all_intervals_std': np.std(rr_intervals),
            'stability_improvement': (np.std(rr_intervals) - min_std) / np.std(rr_intervals) * 100
        }
        
        return rr_mean, stability_stats

    # NEW FUNCTION - Add this for adaptive missed peak restoration
    def restore_missed_peak_adaptive(self, peak1, peak2, candidate_peaks, candidate_values, 
                                    time_data, rr_mean, vth):
        # Find candidates between the two peaks
        between_mask = (candidate_peaks > peak1) & (candidate_peaks < peak2)
        between_candidates = candidate_peaks[between_mask]
        between_values = candidate_values[between_mask]
        
        if len(between_candidates) == 0:
            return {
                'peak_restored': False,
                'reason': 'no_candidates_between_peaks',
                'candidates_evaluated': 0
            }
        
        # Calculate expected time position for missed peak
        peak1_time = time_data[peak1]
        expected_time = peak1_time + rr_mean
        
        # CONSTRAINT 1: Amplitude check - candidates must pass minimal threshold
        # Use a lower threshold than original vth for missed peaks (they might be weaker)
        min_amplitude_threshold = vth * 0.3  # 30% of original threshold
        amplitude_mask = np.abs(between_values) >= min_amplitude_threshold
        
        if not np.any(amplitude_mask):
            return {
                'peak_restored': False,
                'reason': 'no_candidates_pass_amplitude_check',
                'candidates_evaluated': len(between_candidates),
                'min_amplitude_threshold': min_amplitude_threshold
            }
        
        # Filter candidates by amplitude
        valid_candidates = between_candidates[amplitude_mask]
        valid_values = between_values[amplitude_mask]
        valid_times = time_data[valid_candidates]
        
        # CONSTRAINT 2: Timing check - select candidate closest to expected time
        time_differences = np.abs(valid_times - expected_time)
        closest_idx = np.argmin(time_differences)
        
        best_candidate = valid_candidates[closest_idx]
        best_value = valid_values[closest_idx]
        best_time = valid_times[closest_idx]
        timing_error = time_differences[closest_idx]
        
        # CONSTRAINT 3: Validate that restored peak creates reasonable RR intervals
        rr_to_restored = best_time - peak1_time
        rr_from_restored = time_data[peak2] - best_time
        
        # Both intervals should be within reasonable range (0.3 to 2.0 seconds)
        min_rr = 0.3  # 200 BPM max
        max_rr = 2.0  # 30 BPM min
        
        if min_rr <= rr_to_restored <= max_rr and min_rr <= rr_from_restored <= max_rr:
            # Restoration successful
            return {
                'peak_restored': True,
                'restored_peak': best_candidate,
                'restored_value': best_value,
                'expected_time': expected_time,
                'actual_time': best_time,
                'timing_error': timing_error,
                'rr_to_restored': rr_to_restored,
                'rr_from_restored': rr_from_restored,
                'candidates_evaluated': len(between_candidates),
                'amplitude_passed': len(valid_candidates),
                'method': 'adaptive_rr_guided'
            }
        else:
            # Intervals not physiologically reasonable
            return {
                'peak_restored': False,
                'reason': 'unreasonable_rr_intervals',
                'rr_to_restored': rr_to_restored,
                'rr_from_restored': rr_from_restored,
                'candidates_evaluated': len(between_candidates)
            }
    #
    def select_best_from_close_group_adaptive(self, close_group, time_data, signal, rr_mean):
        if len(close_group) == 1:
            return close_group[0]
        
        best_peak = None
        best_score = -1
        
        for peak_idx, peak_value, peak_time in close_group:
            # ✅ CRITERION 1: Signal quality and prominence (40% weight)
            prominence_score = self.evaluate_local_signal_quality(peak_idx, signal, time_data)
            
            # ✅ CRITERION 2: Magnitude relative to group (30% weight)
            magnitude_score = abs(peak_value) / max(abs(pv) for _, pv, _ in close_group)
            
            # ✅ CRITERION 3: Adaptive timing preference (20% weight)
            # Prefer peaks that would create RR intervals closest to rr_mean
            timing_score = self.evaluate_adaptive_timing_fit(peak_idx, time_data, signal, rr_mean)
            
            # ✅ CRITERION 4: Signal smoothness and waveform quality (10% weight)
            smoothness_score = self.evaluate_signal_smoothness(peak_idx, signal)
            
            # ✅ Adaptive combined score
            combined_score = (0.4 * prominence_score + 
                            0.3 * magnitude_score + 
                            0.2 * timing_score + 
                            0.1 * smoothness_score)
            
            if combined_score > best_score:
                best_score = combined_score
                best_peak = (peak_idx, peak_value, peak_time)
        
        return best_peak

    def select_adaptive_gap_candidates(self, start_time, end_time, candidates, values, time_data, 
                                    rr_mean, max_peaks, signal):
        if len(candidates) == 0 or max_peaks == 0:
            return []
        
        gap_duration = end_time - start_time
        candidate_times = time_data[candidates]
        
        # ✅ Calculate expected peak positions based on adaptive RR mean
        expected_positions = []
        for j in range(max_peaks):
            expected_time = start_time + (j + 1) * (gap_duration / (max_peaks + 1))
            expected_positions.append(expected_time)
        
        # ✅ Score each candidate using adaptive criteria
        candidate_scores = []
        
        for i, (candidate, value) in enumerate(zip(candidates, values)):
            candidate_time = candidate_times[i]
            
            # ✅ ADAPTIVE CRITERION 1: Timing fit to expected RR pattern (40% weight)
            min_timing_error = min(abs(candidate_time - exp_pos) for exp_pos in expected_positions)
            timing_score = 1.0 / (1.0 + min_timing_error / rr_mean)
            
            # ✅ ADAPTIVE CRITERION 2: Signal prominence and quality (30% weight)
            prominence_score = self.evaluate_local_signal_quality(candidate, signal, time_data)
            
            # ✅ ADAPTIVE CRITERION 3: Magnitude relative to local context (20% weight)
            magnitude_score = abs(value) / np.max(np.abs(values))
            
            # ✅ ADAPTIVE CRITERION 4: RR interval consistency (10% weight)
            consistency_score = self.evaluate_rr_consistency_adaptive(
                candidate_time, start_time, end_time, rr_mean)
            
            # ✅ Adaptive combined score
            combined_score = (0.4 * timing_score + 
                            0.3 * prominence_score + 
                            0.2 * magnitude_score + 
                            0.1 * consistency_score)
            
            candidate_scores.append((combined_score, candidate, value, candidate_time))
        
        # ✅ Select best non-conflicting candidates
        candidate_scores.sort(reverse=True)  # Sort by score (best first)
        
        selected_peaks = []
        min_separation = max(0.25, 0.5 * rr_mean)  # Adaptive minimum separation
        
        for score, candidate, value, time in candidate_scores:
            if len(selected_peaks) >= max_peaks:
                break
            
            # ✅ Check for conflicts with already selected peaks
            conflict = False
            for _, _, _, selected_time in selected_peaks:
                if abs(time - selected_time) < min_separation:
                    conflict = True
                    break
            
            # ✅ Check boundary conflicts
            if (time - start_time) < min_separation or (end_time - time) < min_separation:
                conflict = True
            
            if not conflict:
                selected_peaks.append((score, candidate, value, time))
        
        # ✅ Sort by time and return
        selected_peaks.sort(key=lambda x: x[3])
        return [(candidate, value) for _, candidate, value, _ in selected_peaks]

    def signal_based_adaptive_detection(self, start_time, end_time, rr_mean, time_data, signal, max_peaks):
        if max_peaks <= 0:
            return []
        
        # Find time indices for the gap
        start_idx = np.argmin(np.abs(time_data - start_time))
        end_idx = np.argmin(np.abs(time_data - end_time))
        
        if end_idx - start_idx < 10:
            return []
        
        # Extract gap signal
        gap_signal = signal[start_idx:end_idx+1]
        gap_time = time_data[start_idx:end_idx+1]
        gap_duration = end_time - start_time
        
        # ✅ Adaptive parameters based on rr_mean
        min_distance = max(5, int(0.6 * rr_mean * len(gap_signal) / gap_duration))
        
        # ✅ Try adaptive prominence thresholds
        from scipy.signal import find_peaks
        
        for prominence_factor in [0.4, 0.3, 0.2, 0.1]:  # Progressively more sensitive
            prominence = prominence_factor * np.std(gap_signal)
            
            peaks_in_gap, properties = find_peaks(gap_signal, 
                                                distance=min_distance,
                                                prominence=prominence)
            
            if len(peaks_in_gap) >= max_peaks:
                break
        
        if len(peaks_in_gap) == 0:
            return []
        
        # ✅ Convert to time domain and score by adaptive criteria
        detected_candidates = []
        for peak_rel_idx in peaks_in_gap:
            peak_time = gap_time[peak_rel_idx]
            peak_value = gap_signal[peak_rel_idx]
            
            # ✅ Score by expected timing pattern
            expected_positions = [start_time + (j+1) * rr_mean for j in range(max_peaks)]
            timing_errors = [abs(peak_time - exp_pos) for exp_pos in expected_positions]
            timing_score = 1.0 / (1.0 + min(timing_errors) / rr_mean)
            
            detected_candidates.append((timing_score, peak_time, peak_value))
        
        # ✅ Select best candidates by timing score
        detected_candidates.sort(reverse=True)  # Best timing first
        return [(time, value) for _, time, value in detected_candidates[:max_peaks]]

    def evaluate_adaptive_timing_fit(self, peak_idx, time_data, signal, rr_mean):
        peak_time = time_data[peak_idx]
        
        # Find neighboring peaks in a window around this peak
        window_duration = 3 * rr_mean  # Look 3 RR intervals around
        window_mask = (np.abs(time_data - peak_time) <= window_duration)
        
        if np.sum(window_mask) < 10:  # Not enough context
            return 0.5
        
        # Simple timing regularity score
        # In a real implementation, you'd analyze local RR patterns
        # For now, return neutral score
        return 0.5

    def evaluate_rr_consistency_adaptive(self, candidate_time, start_time, end_time, rr_mean):
        # Check if adding this candidate creates intervals close to rr_mean
        rr_before = candidate_time - start_time
        rr_after = end_time - candidate_time
        
        # Score based on how close intervals are to expected rr_mean
        before_error = abs(rr_before - rr_mean) / rr_mean
        after_error = abs(rr_after - rr_mean) / rr_mean
        
        avg_error = (before_error + after_error) / 2
        consistency_score = 1.0 / (1.0 + avg_error)
        
        return consistency_score

    def validate_final_rr_intervals_adaptive(self, peaks, time_data, rr_mean):
        if len(peaks) < 2:
            return {'validation': 'insufficient_peaks'}
        
        peak_times = time_data[peaks]
        rr_intervals = np.diff(peak_times)
        
        # ✅ Adaptive quality thresholds based on rr_mean
        rr_min_acceptable = max(0.35, 0.6 * rr_mean)
        rr_max_acceptable = min(2.0, 1.6 * rr_mean)
        
        # Count intervals in adaptive ranges
        close_intervals = np.sum(rr_intervals < rr_min_acceptable)
        normal_intervals = np.sum((rr_intervals >= rr_min_acceptable) & (rr_intervals <= rr_max_acceptable))
        long_intervals = np.sum(rr_intervals > rr_max_acceptable)
        
        # ✅ Adaptive quality assessment
        total_intervals = len(rr_intervals)
        normal_ratio = normal_intervals / total_intervals
        
        if normal_ratio >= 0.9 and close_intervals == 0:
            quality = 'excellent'
        elif normal_ratio >= 0.8 and close_intervals <= 2:
            quality = 'good'
        elif normal_ratio >= 0.7:
            quality = 'fair'
        else:
            quality = 'poor'
        
        return {
            'validation': 'adaptive_completed',
            'quality': quality,
            'final_peak_count': len(peaks),
            'rr_statistics': {
                'mean': np.mean(rr_intervals),
                'std': np.std(rr_intervals),
                'target_rr_mean': rr_mean,
                'deviation_from_target': abs(np.mean(rr_intervals) - rr_mean) / rr_mean * 100
            },
            'adaptive_interval_distribution': {
                'close_intervals': close_intervals,
                'normal_intervals': normal_intervals,
                'long_intervals': long_intervals,
                'total_intervals': total_intervals,
                'normal_ratio': normal_ratio
            },
            'adaptive_thresholds': {
                'rr_min_acceptable': rr_min_acceptable,
                'rr_max_acceptable': rr_max_acceptable
            }
        }
    def step4c_comprehensive_rr_correction_ultra_adaptive(self, consolidated_peaks, consolidated_values, 
                                                      candidate_peaks, candidate_values, time_data, signal):
        """
        ULTRA-ADAPTIVE: Most aggressive missing peak detection for HRV accuracy
        """
        if len(consolidated_peaks) < 5:
            return consolidated_peaks, consolidated_values, {'status': 'insufficient_peaks'}
        
        # PHASE 1: Calculate RR baseline with better robustness
        stable_rr_mean = self.calculate_robust_rr_baseline_enhanced(consolidated_peaks, time_data)
        
        # PHASE 2: ULTRA-AGGRESSIVE thresholds for better HRV
        rr_min_strict = max(0.30, 0.60 * stable_rr_mean)      # Even stricter close peak removal
        rr_gap_aggressive = 1.25 * stable_rr_mean             # More aggressive gap detection (was 1.4)
        rr_max_search = 3.0 * stable_rr_mean                  # Wider search range
        
        self.debug_print(f"🎯 ULTRA-ADAPTIVE HR Correction - RR Mean: {stable_rr_mean:.3f}s ({60/stable_rr_mean:.1f} BPM)")
        self.debug_print(f"🎯 Ultra Thresholds - Min: {rr_min_strict:.3f}s, Gap: {rr_gap_aggressive:.3f}s")
        
        # PHASE 3: Enhanced multi-pass correction with more iterations
        current_peaks = consolidated_peaks.copy()
        current_values = consolidated_values.copy()
        
        correction_stats = {
            'iterations': 0,
            'total_removed': 0,
            'total_added': 0,
            'rr_mean': stable_rr_mean,
            'mode': 'ultra_adaptive'
        }
        
        for iteration in range(10):  # More iterations for thorough correction
            correction_stats['iterations'] = iteration + 1
            
            # Step A: Ultra-strict close peak removal
            cleaned_peaks, cleaned_values, removed_count = self.remove_close_peaks_ultra_strict(
                current_peaks, current_values, time_data, signal, stable_rr_mean)
            
            # Step B: Ultra-aggressive missing peak restoration
            final_peaks, final_values, added_count = self.add_missing_peaks_ultra_aggressive(
                cleaned_peaks, cleaned_values, candidate_peaks, candidate_values, 
                time_data, signal, stable_rr_mean, rr_gap_aggressive)
            
            correction_stats['total_removed'] += removed_count
            correction_stats['total_added'] += added_count
            
            self.debug_print(f"🔄 Ultra Iteration {iteration+1}: Removed {removed_count}, Added {added_count}, Total: {len(final_peaks)}")
            
            # Dynamic convergence - allow more corrections in early iterations
            if iteration >= 3 and removed_count == 0 and added_count == 0:
                break
            elif iteration >= 7 and removed_count + added_count <= 1:
                break
            
            current_peaks = final_peaks
            current_values = final_values
            
            # Update RR mean more frequently for ultra-adaptation
            if iteration % 2 == 0 and len(final_peaks) >= 5:
                updated_rr_mean = self.calculate_robust_rr_baseline_enhanced(final_peaks, time_data)
                if 0.7 <= updated_rr_mean / stable_rr_mean <= 1.3:  # Allow 30% change
                    stable_rr_mean = updated_rr_mean
        
        # PHASE 4: Final HRV-optimized validation
        final_stats = self.validate_for_hrv_quality(final_peaks, time_data, stable_rr_mean)
        correction_stats.update(final_stats)
        
        return final_peaks, final_values, correction_stats


    def calculate_robust_rr_baseline_enhanced(self, peaks, time_data):
        """
        Enhanced RR baseline calculation with better outlier handling
        """
        if len(peaks) < 3:
            return 0.8  # Default ~75 BPM
        
        peak_times = time_data[peaks]
        rr_intervals = np.diff(peak_times)
        
        # Multi-stage outlier removal for better baseline
        # Stage 1: Remove extreme outliers (>3 std from median)
        median_rr = np.median(rr_intervals)
        mad = np.median(np.abs(rr_intervals - median_rr))  # Median Absolute Deviation
        
        # Use MAD for robust outlier detection
        outlier_threshold = 3 * mad
        clean_intervals = rr_intervals[np.abs(rr_intervals - median_rr) <= outlier_threshold]
        
        if len(clean_intervals) < len(rr_intervals) * 0.5:  # If too many outliers, use all
            clean_intervals = rr_intervals
        
        # Stage 2: Find most stable segment if enough intervals
        if len(clean_intervals) >= 10:
            window_size = min(10, len(clean_intervals) // 2)
            min_cv = float('inf')  # Coefficient of variation
            best_mean = median_rr
            
            for i in range(len(clean_intervals) - window_size + 1):
                segment = clean_intervals[i:i + window_size]
                segment_mean = np.mean(segment)
                segment_cv = np.std(segment) / segment_mean if segment_mean > 0 else float('inf')
                
                if segment_cv < min_cv:
                    min_cv = segment_cv
                    best_mean = segment_mean
            
            baseline_rr = best_mean
        else:
            baseline_rr = np.mean(clean_intervals)
        
        # Validate and constrain to physiological range
        baseline_rr = max(0.3, min(2.0, baseline_rr))  # 30-200 BPM range
        
        return baseline_rr


    def remove_close_peaks_ultra_strict(self, peaks, values, time_data, signal, rr_mean):
        """
        Ultra-strict close peak removal with enhanced selection
        """
        if len(peaks) < 2:
            return peaks, values, 0
        
        rr_min_ultra = max(0.30, 0.60 * rr_mean)  # Even stricter than before
        
        peak_times = time_data[peaks]
        cleaned_peaks = []
        cleaned_values = []
        removed_count = 0
        
        i = 0
        while i < len(peaks):
            current_peak = peaks[i]
            current_value = values[i]
            current_time = peak_times[i]
            
            # Collect ultra-close group
            close_group = [(current_peak, current_value, current_time)]
            j = i + 1
            
            while j < len(peaks) and (peak_times[j] - current_time) < rr_min_ultra:
                close_group.append((peaks[j], values[j], peak_times[j]))
                j += 1
            
            if len(close_group) > 1:
                # Ultra-enhanced selection with more criteria
                best_peak = self.select_ultra_best_peak(close_group, time_data, signal, rr_mean)
                cleaned_peaks.append(best_peak[0])
                cleaned_values.append(best_peak[1])
                removed_count += len(close_group) - 1
                
                self.debug_print(f"🔥 Ultra-strict: {len(close_group)} → 1 peak at {best_peak[2]:.3f}s")
            else:
                cleaned_peaks.append(current_peak)
                cleaned_values.append(current_value)
            
            i = j
        
        return np.array(cleaned_peaks), np.array(cleaned_values), removed_count


    def add_missing_peaks_ultra_aggressive(self, peaks, values, candidate_peaks, candidate_values, 
                                        time_data, signal, rr_mean, gap_threshold):
        """
        Ultra-aggressive missing peak restoration for HRV optimization
        """
        if len(peaks) < 2:
            return peaks, values, 0
        
        peak_times = time_data[peaks]
        augmented_peaks = list(peaks)
        augmented_values = list(values)
        added_count = 0
        
        i = 0
        while i < len(augmented_peaks) - 1:
            current_time = time_data[augmented_peaks[i]]
            next_time = time_data[augmented_peaks[i + 1]]
            gap_duration = next_time - current_time
            
            # Ultra-aggressive gap detection (was 1.4, now 1.25)
            if gap_duration > gap_threshold:
                # More aggressive expected count calculation
                expected_count = max(1, round((gap_duration / rr_mean)) - 1)
                expected_count = min(4, expected_count)  # Allow up to 4 peaks per gap
                
                self.debug_print(f"🎯 Ultra gap: {gap_duration:.3f}s, expecting {expected_count} peaks")
                
                if expected_count > 0:
                    # STRATEGY 1: Enhanced candidate selection
                    gap_mask = (candidate_peaks > augmented_peaks[i]) & (candidate_peaks < augmented_peaks[i + 1])
                    gap_candidates = candidate_peaks[gap_mask]
                    gap_values = candidate_values[gap_mask]
                    
                    restored_peaks = []
                    
                    if len(gap_candidates) > 0:
                        restored_peaks = self.select_ultra_gap_candidates(
                            current_time, next_time, gap_candidates, gap_values, time_data,
                            rr_mean, expected_count, signal)
                    
                    # STRATEGY 2: More aggressive signal-based detection
                    if len(restored_peaks) < expected_count:
                        additional_peaks = self.ultra_signal_detection(
                            current_time, next_time, rr_mean, time_data, signal, 
                            expected_count - len(restored_peaks))
                        
                        for peak_time, peak_val in additional_peaks:
                            peak_idx = np.argmin(np.abs(time_data - peak_time))
                            restored_peaks.append((peak_idx, peak_val))
                    
                    # STRATEGY 3: Interpolation-based peak insertion if still insufficient
                    if len(restored_peaks) < expected_count:
                        interpolated_peaks = self.interpolate_missing_peaks(
                            current_time, next_time, rr_mean, time_data, signal,
                            expected_count - len(restored_peaks))
                        
                        for peak_time, peak_val in interpolated_peaks:
                            peak_idx = np.argmin(np.abs(time_data - peak_time))
                            restored_peaks.append((peak_idx, peak_val))
                    
                    # Insert all restored peaks
                    restored_peaks.sort(key=lambda x: time_data[x[0]])
                    
                    for restore_idx, (peak_idx, peak_val) in enumerate(restored_peaks):
                        insert_position = i + 1 + restore_idx
                        augmented_peaks.insert(insert_position, peak_idx)
                        augmented_values.insert(insert_position, peak_val)
                        added_count += 1
                        
                        self.debug_print(f"✅ Added peak at {time_data[peak_idx]:.3f}s")
            
            # Update index accounting for insertions
            current_gap_additions = 0
            while (i + 1 + current_gap_additions < len(augmented_peaks) and 
                time_data[augmented_peaks[i + 1 + current_gap_additions]] < next_time):
                current_gap_additions += 1
            
            i += 1 + current_gap_additions
        
        return np.array(augmented_peaks), np.array(augmented_values), added_count


    def select_ultra_best_peak(self, close_group, time_data, signal, rr_mean):
        """
        Ultra-enhanced peak selection with 5 criteria
        """
        if len(close_group) == 1:
            return close_group[0]
        
        best_peak = None
        best_score = -1
        
        for peak_idx, peak_value, peak_time in close_group:
            # Enhanced scoring with 5 criteria
            prominence = self.evaluate_local_signal_quality(peak_idx, signal, time_data)
            magnitude = abs(peak_value) / max(abs(pv) for _, pv, _ in close_group)
            smoothness = self.evaluate_signal_smoothness(peak_idx, signal)
            
            # NEW: Waveform shape analysis
            shape_score = self.evaluate_waveform_shape(peak_idx, signal)
            
            # NEW: RR consistency with expected timing
            timing_score = self.evaluate_timing_consistency(peak_time, close_group, rr_mean)
            
            # Ultra-weighted scoring
            combined_score = (0.3 * prominence + 
                            0.25 * magnitude + 
                            0.2 * shape_score +
                            0.15 * timing_score +
                            0.1 * smoothness)
            
            if combined_score > best_score:
                best_score = combined_score
                best_peak = (peak_idx, peak_value, peak_time)
        
        return best_peak


    def select_ultra_gap_candidates(self, start_time, end_time, candidates, values, time_data, 
                                    rr_mean, max_peaks, signal):
        """
        Ultra-sophisticated gap candidate selection
        """
        if len(candidates) == 0:
            return []
        
        gap_duration = end_time - start_time
        candidate_times = time_data[candidates]
        
        # Calculate optimal peak positions using golden ratio spacing
        optimal_positions = []
        for j in range(max_peaks):
            # Use uniform spacing as baseline
            pos = start_time + (j + 1) * gap_duration / (max_peaks + 1)
            optimal_positions.append(pos)
        
        # Score candidates with enhanced criteria
        candidate_scores = []
        
        for i, (candidate, value) in enumerate(zip(candidates, values)):
            candidate_time = candidate_times[i]
            
            # 5-factor scoring system
            timing_score = self.calculate_timing_score(candidate_time, optimal_positions, rr_mean)
            prominence_score = self.evaluate_local_signal_quality(candidate, signal, time_data)
            magnitude_score = abs(value) / np.max(np.abs(values))
            shape_score = self.evaluate_waveform_shape(candidate, signal)
            consistency_score = self.evaluate_rr_consistency_adaptive(candidate_time, start_time, end_time, rr_mean)
            
            # Ultra-weighted scoring for gap filling
            combined_score = (0.35 * timing_score + 
                            0.25 * prominence_score + 
                            0.2 * magnitude_score + 
                            0.15 * shape_score +
                            0.05 * consistency_score)
            
            candidate_scores.append((combined_score, candidate, value, candidate_time))
        
        # Select best non-conflicting candidates
        candidate_scores.sort(reverse=True)
        selected_peaks = []
        min_separation = max(0.2, 0.4 * rr_mean)  # More aggressive separation
        
        for score, candidate, value, time in candidate_scores:
            if len(selected_peaks) >= max_peaks:
                break
            
            # Check conflicts
            conflict = False
            for _, _, _, selected_time in selected_peaks:
                if abs(time - selected_time) < min_separation:
                    conflict = True
                    break
            
            if not conflict and (time - start_time) >= min_separation and (end_time - time) >= min_separation:
                selected_peaks.append((score, candidate, value, time))
        
        selected_peaks.sort(key=lambda x: x[3])  # Sort by time
        return [(candidate, value) for _, candidate, value, _ in selected_peaks]


    def ultra_signal_detection(self, start_time, end_time, rr_mean, time_data, signal, max_peaks):
        """
        Ultra-aggressive signal-based peak detection with multiple methods
        """
        if max_peaks <= 0:
            return []
        
        start_idx = np.argmin(np.abs(time_data - start_time))
        end_idx = np.argmin(np.abs(time_data - end_time))
        
        if end_idx - start_idx < 5:
            return []
        
        gap_signal = signal[start_idx:end_idx+1]
        gap_time = time_data[start_idx:end_idx+1]
        gap_duration = end_time - start_time
        
        from scipy.signal import find_peaks
        
        # Method 1: Adaptive prominence
        min_distance = max(3, int(0.4 * rr_mean * len(gap_signal) / gap_duration))
        
        detected_peaks = []
        
        # Try multiple detection strategies
        for method, params in [
            ('prominence', {'prominence': 0.2 * np.std(gap_signal)}),
            ('height', {'height': np.mean(gap_signal) + 0.1 * np.std(gap_signal)}),
            ('adaptive', {'prominence': 0.1 * np.std(gap_signal), 'width': 2})
        ]:
            
            peaks_found, _ = find_peaks(gap_signal, distance=min_distance, **params)
            
            if len(peaks_found) > 0:
                for peak_idx in peaks_found:
                    peak_time = gap_time[peak_idx]
                    peak_value = gap_signal[peak_idx]
                    
                    # Score by timing optimality
                    expected_times = [start_time + (j+1) * rr_mean for j in range(max_peaks)]
                    timing_errors = [abs(peak_time - exp_t) for exp_t in expected_times]
                    timing_score = 1.0 / (1.0 + min(timing_errors) / rr_mean)
                    
                    detected_peaks.append((timing_score, peak_time, peak_value, method))
            
            if len(detected_peaks) >= max_peaks:
                break
        
        # Return best peaks by timing score
        detected_peaks.sort(reverse=True)
        unique_peaks = []
        
        for score, peak_time, peak_value, method in detected_peaks:
            # Check for duplicates
            too_close = False
            for _, existing_time, _, _ in unique_peaks:
                if abs(peak_time - existing_time) < 0.3 * rr_mean:
                    too_close = True
                    break
            
            if not too_close:
                unique_peaks.append((score, peak_time, peak_value, method))
            
            if len(unique_peaks) >= max_peaks:
                break
        
        return [(time, value) for _, time, value, _ in unique_peaks]


    def interpolate_missing_peaks(self, start_time, end_time, rr_mean, time_data, signal, max_peaks):
        """
        Last resort: interpolate peaks at expected positions
        """
        if max_peaks <= 0:
            return []
        
        gap_duration = end_time - start_time
        interpolated_peaks = []
        
        for j in range(max_peaks):
            # Calculate expected time
            expected_time = start_time + (j + 1) * gap_duration / (max_peaks + 1)
            
            # Find closest signal index
            expected_idx = np.argmin(np.abs(time_data - expected_time))
            
            # Get signal value at expected position
            if 0 <= expected_idx < len(signal):
                expected_value = signal[expected_idx]
                interpolated_peaks.append((expected_time, expected_value))
            
            self.debug_print(f"🔮 Interpolated peak at {expected_time:.3f}s")
        
        return interpolated_peaks


    def evaluate_waveform_shape(self, peak_idx, signal, window=8):
        """
        Evaluate the quality of the waveform shape around a peak
        """
        start_idx = max(0, peak_idx - window)
        end_idx = min(len(signal), peak_idx + window + 1)
        
        if end_idx - start_idx < 5:
            return 0.5
        
        local_signal = signal[start_idx:end_idx]
        peak_pos = peak_idx - start_idx
        
        if peak_pos < 0 or peak_pos >= len(local_signal):
            return 0.5
        
        # Check for proper peak shape (rising then falling)
        left_rising = True
        right_falling = True
        
        if peak_pos > 0:
            left_rising = local_signal[peak_pos] > local_signal[peak_pos - 1]
        if peak_pos < len(local_signal) - 1:
            right_falling = local_signal[peak_pos] > local_signal[peak_pos + 1]
        
        shape_score = 0.5
        if left_rising and right_falling:
            shape_score = 1.0
        elif left_rising or right_falling:
            shape_score = 0.7
        
        return shape_score


    def evaluate_timing_consistency(self, peak_time, close_group, rr_mean):
        """
        Evaluate how well a peak fits expected timing in a close group
        """
        group_times = [pt for _, _, pt in close_group]
        group_center = np.mean(group_times)
        group_span = max(group_times) - min(group_times)
        
        if group_span == 0:
            return 1.0
        
        # Prefer peaks closer to group center
        center_distance = abs(peak_time - group_center)
        timing_score = 1.0 - (center_distance / (group_span + 1e-6))
        
        return max(0.0, min(1.0, timing_score))


    def calculate_timing_score(self, candidate_time, optimal_positions, rr_mean):
        """
        Calculate how well a candidate fits optimal timing positions
        """
        if not optimal_positions:
            return 0.5
        
        # Find closest optimal position
        timing_errors = [abs(candidate_time - pos) for pos in optimal_positions]
        min_error = min(timing_errors)
        
        # Convert to score (0-1, higher is better)
        timing_score = 1.0 / (1.0 + min_error / rr_mean)
        
        return timing_score


    def validate_for_hrv_quality(self, peaks, time_data, rr_mean):
        """
        Specialized validation for HRV analysis quality
        """
        if len(peaks) < 3:
            return {'validation': 'insufficient_for_hrv'}
        
        peak_times = time_data[peaks]
        rr_intervals = np.diff(peak_times)
        
        # HRV-specific quality metrics
        rr_cv = np.std(rr_intervals) / np.mean(rr_intervals)  # Coefficient of variation
        
        # Count problematic intervals
        very_short = np.sum(rr_intervals < 0.4)  # < 150 BPM
        very_long = np.sum(rr_intervals > 1.5)   # > 40 BPM
        
        # HRV quality assessment
        hrv_quality = 'excellent'
        if very_short > 0 or very_long > 2:
            hrv_quality = 'poor'
        elif very_long > 0 or rr_cv > 0.3:
            hrv_quality = 'fair'
        elif rr_cv > 0.2:
            hrv_quality = 'good'
        
        return {
            'validation': 'hrv_optimized',
            'hrv_quality': hrv_quality,
            'rr_cv': rr_cv,
            'problematic_intervals': very_short + very_long,
            'total_intervals': len(rr_intervals),
            'mean_rr': np.mean(rr_intervals),
            'target_rr': rr_mean
        }

    def step4c_enhanced_rr_correction_robust(self, consolidated_peaks, consolidated_values, 
                                            candidate_peaks, candidate_values, time_data, signal):
        if len(consolidated_peaks) < 5:
            return consolidated_peaks, consolidated_values, {'status': 'insufficient_peaks'}
        
        # Phase 1: Calculate baseline RR
        peak_times = time_data[consolidated_peaks]
        rr_intervals = np.diff(peak_times)
        baseline_rr = np.median(rr_intervals[rr_intervals < 1.5])  # Exclude very long intervals
        baseline_rr = max(0.5, min(1.2, baseline_rr))  # Physiological limits
        
        # Phase 2: Define thresholds
        rr_min_threshold = max(0.35, 0.6 * baseline_rr)    # 60% of baseline
        rr_gap_threshold = 1.3 * baseline_rr               # 130% triggers gap search
        
        print(f"Baseline RR: {baseline_rr:.3f}s, Min: {rr_min_threshold:.3f}s, Gap: {rr_gap_threshold:.3f}s")
        
        current_peaks = consolidated_peaks.copy()
        current_values = consolidated_values.copy()
        
        stats = {'baseline_rr': baseline_rr, 'total_removed': 0, 'total_added': 0}
        
        # Phase 3: Remove close peaks (2 iterations)
        for iteration in range(2):
            cleaned_peaks, cleaned_values, removed = self.remove_close_peaks_simple(
                current_peaks, current_values, time_data, rr_min_threshold)
            stats['total_removed'] += removed
            current_peaks, current_values = cleaned_peaks, cleaned_values
            if removed == 0:
                break
        
        # Phase 4: Fill gaps (3 iterations)
        for iteration in range(3):
            filled_peaks, filled_values, added = self.fill_gaps_simple(
                current_peaks, current_values, candidate_peaks, candidate_values, 
                time_data, signal, baseline_rr, rr_gap_threshold)
            stats['total_added'] += added
            current_peaks, current_values = filled_peaks, filled_values
            if added == 0:
                break
        
        return current_peaks, current_values, stats

    def remove_close_peaks_simple(self, peaks, values, time_data, min_threshold):
        """Simple close peak removal"""
        if len(peaks) < 2:
            return peaks, values, 0
        
        peak_times = time_data[peaks]
        cleaned_peaks = []
        cleaned_values = []
        removed_count = 0
        
        i = 0
        while i < len(peaks):
            current_peak = peaks[i]
            current_value = values[i]
            current_time = peak_times[i]
            
            # Find close peaks
            close_group = [(current_peak, current_value)]
            j = i + 1
            while j < len(peaks) and (peak_times[j] - current_time) < min_threshold:
                close_group.append((peaks[j], values[j]))
                j += 1
            
            if len(close_group) > 1:
                # Keep the one with highest absolute value
                best_peak = max(close_group, key=lambda x: abs(x[1]))
                cleaned_peaks.append(best_peak[0])
                cleaned_values.append(best_peak[1])
                removed_count += len(close_group) - 1
            else:
                cleaned_peaks.append(current_peak)
                cleaned_values.append(current_value)
            
            i = j
        
        return np.array(cleaned_peaks), np.array(cleaned_values), removed_count

    def fill_gaps_simple(self, peaks, values, candidate_peaks, candidate_values, 
                        time_data, signal, baseline_rr, gap_threshold):
        """Simple gap filling"""
        if len(peaks) < 2:
            return peaks, values, 0
        
        peak_times = time_data[peaks]
        filled_peaks = list(peaks)
        filled_values = list(values)
        added_count = 0
        
        i = 0
        while i < len(filled_peaks) - 1:
            current_time = time_data[filled_peaks[i]]
            next_time = time_data[filled_peaks[i + 1]]
            gap_duration = next_time - current_time
            
            if gap_duration > gap_threshold:
                # Look for candidate peaks in the gap
                gap_mask = (candidate_peaks > filled_peaks[i]) & (candidate_peaks < filled_peaks[i + 1])
                gap_candidates = candidate_peaks[gap_mask]
                gap_values = candidate_values[gap_mask]
                
                if len(gap_candidates) > 0:
                    # Select best candidate (highest magnitude closest to gap center)
                    gap_center_time = (current_time + next_time) / 2
                    gap_times = time_data[gap_candidates]
                    
                    # Score by magnitude and timing
                    scores = []
                    for idx, (cand, val) in enumerate(zip(gap_candidates, gap_values)):
                        timing_score = 1.0 / (1.0 + abs(gap_times[idx] - gap_center_time))
                        magnitude_score = abs(val) / np.max(np.abs(gap_values))
                        combined_score = 0.6 * magnitude_score + 0.4 * timing_score
                        scores.append((combined_score, cand, val))
                    
                    # Insert best candidate
                    best_score, best_peak, best_value = max(scores)
                    filled_peaks.insert(i + 1, best_peak)
                    filled_values.insert(i + 1, best_value)
                    added_count += 1
            
            i += 1
        
        return np.array(filled_peaks), np.array(filled_values), added_count
    
    def calculate_hrv_with_mav(self):
        """Calculate HRV with optional Moving Average filtering"""
        # Check if we have the required data
        if not hasattr(self, 'paper_analysis_results') or self.paper_analysis_results is None:
            messagebox.showwarning("Warning", "No paper method results available. Please run the paper algorithm first.")
            return
        
        if not self.ecg_loaded:
            messagebox.showwarning("Warning", "No ECG data loaded. Please load ECG data first.")
            return
        
        # Clear results
        self.hrv_results_text.delete(1.0, tk.END)
        for ax in [self.ax_rr_radar, self.ax_rr_ecg, self.ax_rr_compare, 
                   self.ax_hrv_radar, self.ax_hrv_ecg, self.ax_hrv_compare, self.ax_peak_compare]:
            ax.clear()
        
        try:
            # Get parameters
            window_size = self.hrv_window_var.get()
            overlap_percent = self.hrv_overlap_var.get()
            min_rr = self.min_rr_var.get()
            max_rr = self.max_rr_var.get()
            enable_mav = self.hrv_enable_mav_var.get()
            mav_window = self.hrv_mav_window_var.get()
            
            self.hrv_results_text.insert(tk.END, "HRV ANALYSIS WITH MAV OPTION\n")
            self.hrv_results_text.insert(tk.END, "=" * 50 + "\n")
            self.hrv_results_text.insert(tk.END, f"Window Size: {window_size} s\n")
            self.hrv_results_text.insert(tk.END, f"Overlap: {overlap_percent}%\n")
            self.hrv_results_text.insert(tk.END, f"RR Filter: {min_rr}-{max_rr} ms\n")
            self.hrv_results_text.insert(tk.END, f"MAV Applied: {enable_mav}\n")
            if enable_mav:
                self.hrv_results_text.insert(tk.END, f"MAV Window: {mav_window}\n")
            self.hrv_results_text.insert(tk.END, "\n")
            
            # Plot peak detection comparison
            self.plot_peak_detection_comparison()
            
            # Calculate RR intervals from radar peaks
            radar_peaks = self.paper_analysis_results['final_peaks']
            radar_times = self.signals['timestamps']
            radar_rr, radar_rr_times = self.calculate_rr_intervals(radar_peaks, radar_times)
            radar_rr_filtered, radar_rr_times_filtered = self.filter_rr_intervals(
                radar_rr, radar_rr_times, min_rr, max_rr)
            
            # Calculate RR intervals from ECG peaks
            ecg_peaks = self.detect_r_peaks(self.ecg_data, self.ecg_fs)
            ecg_rr, ecg_rr_times = self.calculate_rr_intervals(ecg_peaks, self.ecg_time)
            ecg_rr_filtered, ecg_rr_times_filtered = self.filter_rr_intervals(
                ecg_rr, ecg_rr_times, min_rr, max_rr)
            
            # Apply MAV if enabled
            if enable_mav:
                radar_rr_mav = self.apply_moving_average(radar_rr_filtered, mav_window)
                ecg_rr_mav = self.apply_moving_average(ecg_rr_filtered, mav_window)
            else:
                radar_rr_mav = radar_rr_filtered
                ecg_rr_mav = ecg_rr_filtered
            
            # Calculate HRV parameters for both raw and MAV data
            radar_stats_raw, radar_hrv_raw, _ = self.calculate_hrv_parameters(
                radar_rr_filtered, window_size, overlap_percent)
            ecg_stats_raw, ecg_hrv_raw, _ = self.calculate_hrv_parameters(
                ecg_rr_filtered, window_size, overlap_percent)
            
            radar_stats_mav, radar_hrv_mav, _ = self.calculate_hrv_parameters(
                radar_rr_mav, window_size, overlap_percent)
            ecg_stats_mav, ecg_hrv_mav, _ = self.calculate_hrv_parameters(
                ecg_rr_mav, window_size, overlap_percent)
            
            # Plot RR intervals
            self.plot_rr_intervals_with_mav(radar_rr_filtered, radar_rr_times_filtered, radar_rr_mav,
                                        ecg_rr_filtered, ecg_rr_times_filtered, ecg_rr_mav, enable_mav)
            
        
            # Store results for export
            self.hrv_results_raw = {
                'radar_rr': radar_rr_filtered,
                'radar_rr_times': radar_rr_times_filtered,
                'ecg_rr': ecg_rr_filtered,
                'ecg_rr_times': ecg_rr_times_filtered,
                'radar_hrv': radar_hrv_raw,
                'ecg_hrv': ecg_hrv_raw,
                'radar_stats': radar_stats_raw,
                'ecg_stats': ecg_stats_raw
            }
            
            self.hrv_results_mav = {
                'radar_rr': radar_rr_mav,
                'radar_rr_times': radar_rr_times_filtered[:len(radar_rr_mav)],
                'ecg_rr': ecg_rr_mav,
                'ecg_rr_times': ecg_rr_times_filtered[:len(ecg_rr_mav)],
                'radar_hrv': radar_hrv_mav,
                'ecg_hrv': ecg_hrv_mav,
                'radar_stats': radar_stats_mav,
                'ecg_stats': ecg_stats_mav,
                'mav_window': mav_window
            }

            # Plot HRV parameters
            self.plot_hrv_parameters_with_mav(radar_hrv_raw, radar_hrv_mav, ecg_hrv_raw, ecg_hrv_mav, enable_mav)
            
            # Display results
            self.display_hrv_results(radar_stats_raw, radar_stats_mav, ecg_stats_raw, ecg_stats_mav, enable_mav)
            
            self.fig_hrv.tight_layout()
            self.canvas_hrv.draw()

            # Enable editing after successful HRV calculation
            if hasattr(self, 'edit_radar_btn'):
                self.edit_radar_btn.config(state='normal')
                self.hrv_edit_info.config(text="Click 'Edit Radar Peaks' to start editing")
                print("DEBUG: Edit button enabled")  # DEBUG
            else:
                print("DEBUG: edit_radar_btn not found")  # DEBUG

            # Store original peaks for editing
            if hasattr(self, 'paper_analysis_results') and self.paper_analysis_results:
                self.original_radar_peaks = self.paper_analysis_results['final_peaks'].copy()
                self.original_radar_values = self.paper_analysis_results['final_values'].copy()
            
        except Exception as e:
            messagebox.showerror("Error", f"HRV analysis failed: {str(e)}")
            print(f"HRV analysis error: {e}")
            traceback.print_exc()

    def plot_rr_intervals_with_mav(self, radar_rr_raw, radar_rr_times, radar_rr_mav,
                                ecg_rr_raw, ecg_rr_times, ecg_rr_mav, enable_mav):
        """Plot RR intervals with MAV comparison"""
        # Plot radar RR intervals
        if enable_mav:
            self.ax_rr_radar.plot(radar_rr_times, radar_rr_raw, 'b.-', markersize=2, linewidth=1, alpha=0.6, label='Raw')
            self.ax_rr_radar.plot(radar_rr_times[:len(radar_rr_mav)], radar_rr_mav, 'r-', linewidth=2, label='MAV')
            self.ax_rr_radar.set_title('Radar RR Intervals (Raw vs MAV)')
        else:
            self.ax_rr_radar.plot(radar_rr_times, radar_rr_raw, 'b.-', markersize=3, linewidth=1, label='Raw')
            self.ax_rr_radar.set_title('Radar RR Intervals (Raw)')
        
        self.ax_rr_radar.set_xlabel('Time (s)')
        self.ax_rr_radar.set_ylabel('RR Interval (ms)')
        self.ax_rr_radar.legend()
        self.ax_rr_radar.grid(True)
        
        # Plot ECG RR intervals
        if enable_mav:
            self.ax_rr_ecg.plot(ecg_rr_times, ecg_rr_raw, 'r.-', markersize=2, linewidth=1, alpha=0.6, label='Raw')
            self.ax_rr_ecg.plot(ecg_rr_times[:len(ecg_rr_mav)], ecg_rr_mav, 'g-', linewidth=2, label='MAV')
            self.ax_rr_ecg.set_title('ECG RR Intervals (Raw vs MAV)')
        else:
            self.ax_rr_ecg.plot(ecg_rr_times, ecg_rr_raw, 'r.-', markersize=3, linewidth=1, label='Raw')
            self.ax_rr_ecg.set_title('ECG RR Intervals (Raw)')
        
        self.ax_rr_ecg.set_xlabel('Time (s)')
        self.ax_rr_ecg.set_ylabel('RR Interval (ms)')
        self.ax_rr_ecg.legend()
        self.ax_rr_ecg.grid(True)
        
        # Compare RR intervals
        if enable_mav:
            min_len = min(len(radar_rr_mav), len(ecg_rr_mav))
            self.ax_rr_compare.plot(radar_rr_times[:min_len], radar_rr_mav[:min_len], 
                                'b-', linewidth=2, alpha=0.7, label='Radar MAV')
            self.ax_rr_compare.plot(ecg_rr_times[:min_len], ecg_rr_mav[:min_len], 
                                'r-', linewidth=2, alpha=0.7, label='ECG MAV')
            self.ax_rr_compare.set_title('RR Intervals Comparison (MAV)')
        else:
            min_len = min(len(radar_rr_raw), len(ecg_rr_raw))
            self.ax_rr_compare.plot(radar_rr_times[:min_len], radar_rr_raw[:min_len], 
                                'b.-', markersize=2, linewidth=1, alpha=0.7, label='Radar Raw')
            self.ax_rr_compare.plot(ecg_rr_times[:min_len], ecg_rr_raw[:min_len], 
                                'r.-', markersize=2, linewidth=1, alpha=0.7, label='ECG Raw')
            self.ax_rr_compare.set_title('RR Intervals Comparison (Raw)')
        
        self.ax_rr_compare.set_xlabel('Time (s)')
        self.ax_rr_compare.set_ylabel('RR Interval (ms)')
        self.ax_rr_compare.legend()
        self.ax_rr_compare.grid(True)

    def edit_radar_peaks(self):
        print("DEBUG: edit_radar_peaks() called")  # DEBUG
    
        if not hasattr(self, 'paper_analysis_results') or self.paper_analysis_results is None:
            print("DEBUG: No paper_analysis_results")  # DEBUG
            messagebox.showwarning("Warning", "No radar peaks available. Run paper method first.")
            return
        
        if not hasattr(self, 'signals') or self.signals is None:
            print("DEBUG: No signals")  # DEBUG
            messagebox.showwarning("Warning", "No radar signal available.")
            return
        
        print("DEBUG: Starting editing mode")  # DEBUG
        self.hrv_editing_mode = True
        
        # Initialize manual peaks from current results
        if hasattr(self, 'edited_radar_peaks') and self.edited_radar_peaks is not None:
            self.manual_radar_peaks = list(self.edited_radar_peaks.copy())
            print(f"DEBUG: Using edited peaks: {len(self.manual_radar_peaks)}")  # DEBUG
        else:
            self.manual_radar_peaks = list(self.paper_analysis_results['final_peaks'].copy())
            print(f"DEBUG: Using original peaks: {len(self.manual_radar_peaks)}")  # DEBUG
            
            # Setup editing
            self.hrv_edit_segment_start = 0
            self.exit_edit_btn.config(state='normal')
            self.hrv_prev_btn.config(state='normal')
            self.hrv_next_btn.config(state='normal')
            self.update_hrv_btn.config(state='normal')
            
            # Connect click events to peak comparison plot
            self.ax_peak_compare.figure.canvas.mpl_connect('button_press_event', self.on_radar_click)
            self.plot_radar_edit_segment()  # Ini akan override plot comparison dengan editing view
            self.hrv_edit_info.config(text="EDITING: Left-click=ADD, Right-click=REMOVE radar peaks")

    def prev_segment_hrv(self):
        """Navigate to previous 30s segment"""
        self.hrv_edit_segment_start = max(0, self.hrv_edit_segment_start - self.hrv_edit_segment_duration)
        self.plot_radar_edit_segment()

    def next_segment_hrv(self):
        """Navigate to next 30s segment"""
        if hasattr(self, 'signals') and self.signals:
            total_duration = self.signals['timestamps'][-1] - self.signals['timestamps'][0]
            max_start = total_duration - self.hrv_edit_segment_duration
            self.hrv_edit_segment_start = min(self.hrv_edit_segment_start + self.hrv_edit_segment_duration, max_start)
            self.plot_radar_edit_segment()   

    def on_radar_click(self, event):
        """Handle mouse clicks for manual radar peak editing"""
        if not self.hrv_editing_mode or event.inaxes != self.ax_peak_compare:  
            return
        
        if event.xdata is None:
            return
        
        time_clicked = event.xdata
        
        # Find closest sample index in radar signal
        if hasattr(self, 'signals') and self.signals:
            radar_times = self.signals['timestamps']
            sample_clicked = np.argmin(np.abs(radar_times - time_clicked))
            
            # Window threshold for detection (in samples)
            window = int(0.05 * self.signals['fs'])  # 50ms window
            
            if event.button == 1:  # Left click - Add peak
                # Check if peak already exists nearby
                if all(abs(sample_clicked - peak) > window for peak in self.manual_radar_peaks):
                    self.manual_radar_peaks.append(sample_clicked)
                    self.manual_radar_peaks.sort()
                    
            elif event.button == 3:  # Right click - Remove nearest peak
                for peak in self.manual_radar_peaks[:]:  # Use slice copy for safe iteration
                    if abs(sample_clicked - peak) <= window:
                        self.manual_radar_peaks.remove(peak)
                        break
            
            self.plot_radar_edit_segment()

    def plot_radar_edit_segment(self):
        """Plot 30s segment for editing radar peaks"""
        print(f"DEBUG: plot_radar_edit_segment called, segment_start: {self.hrv_edit_segment_start}")  # DEBUG
        
        if not hasattr(self, 'signals') or not hasattr(self, 'manual_radar_peaks'):
            print("DEBUG: Missing signals or manual_radar_peaks")  # DEBUG
            return
        
        try:
            self.ax_peak_compare.clear()
            
            # Get segment indices
            radar_times = self.signals['timestamps']
            start_idx = np.argmin(np.abs(radar_times - self.hrv_edit_segment_start))
            end_time = self.hrv_edit_segment_start + self.hrv_edit_segment_duration
            end_idx = np.argmin(np.abs(radar_times - end_time))
            
            print(f"DEBUG: Segment indices: {start_idx} to {end_idx}")  # DEBUG
            
            # Get radar signal for segment
            if hasattr(self, 'latest_combined_imf') and self.latest_combined_imf is not None:
                radar_signal = self.latest_combined_imf
                print(f"DEBUG: Using latest_combined_imf, length: {len(radar_signal)}")  # DEBUG
            else:
                print("DEBUG: No latest_combined_imf available")  # DEBUG
                self.ax_peak_compare.text(0.5, 0.5, 'No radar signal available for editing\nCombine IMFs first in Heart Rate EEMD tab', 
                                        ha='center', va='center', transform=self.ax_peak_compare.transAxes)
                self.canvas_hrv.draw()
                return
            
            segment_time = radar_times[start_idx:end_idx]
            segment_signal = radar_signal[start_idx:end_idx]
            
            # Normalize for display
            if len(segment_signal) > 0:
                segment_signal_norm = (segment_signal - np.mean(segment_signal)) / np.std(segment_signal)
                
                # Plot radar signal
                self.ax_peak_compare.plot(segment_time, segment_signal_norm, 'b-', linewidth=1, 
                                        alpha=0.7, label='Radar Signal')
                
                # Plot manual radar peaks in this segment
                peaks_in_segment = [p for p in self.manual_radar_peaks if start_idx <= p < end_idx]
                print(f"DEBUG: Peaks in segment: {len(peaks_in_segment)}")  # DEBUG
                
                if peaks_in_segment:
                    peak_times = radar_times[peaks_in_segment]
                    # Ensure indices are within bounds
                    valid_peak_indices = [p - start_idx for p in peaks_in_segment if 0 <= p - start_idx < len(segment_signal_norm)]
                    if valid_peak_indices:
                        peak_values = segment_signal_norm[valid_peak_indices]
                        self.ax_peak_compare.plot(peak_times[:len(valid_peak_indices)], peak_values, 'ro', markersize=8, 
                                                label=f'Radar Peaks ({len(valid_peak_indices)})')
                
                # Add ECG reference if available (static, not editable)
                if hasattr(self, 'ecg_loaded') and self.ecg_loaded:
                    try:
                        ecg_start_idx = np.argmin(np.abs(self.ecg_time - self.hrv_edit_segment_start))
                        ecg_end_idx = np.argmin(np.abs(self.ecg_time - end_time))
                        
                        if ecg_end_idx > ecg_start_idx:
                            ecg_segment_time = self.ecg_time[ecg_start_idx:ecg_end_idx]
                            ecg_segment = self.ecg_data[ecg_start_idx:ecg_end_idx]
                            
                            if len(ecg_segment) > 0:
                                ecg_segment_norm = (ecg_segment - np.mean(ecg_segment)) / np.std(ecg_segment)
                                ecg_offset = 3.0
                                
                                self.ax_peak_compare.plot(ecg_segment_time, ecg_segment_norm + ecg_offset, 
                                                        'r-', linewidth=1, alpha=0.5, label='ECG Reference')
                                
                                # ECG peaks in segment
                                ecg_peaks = self.detect_r_peaks(self.ecg_data, self.ecg_fs)
                                ecg_peaks_in_segment = [p for p in ecg_peaks if ecg_start_idx <= p < ecg_end_idx]
                                if ecg_peaks_in_segment:
                                    ecg_peak_times = self.ecg_time[ecg_peaks_in_segment]
                                    ecg_peak_values = ((self.ecg_data[ecg_peaks_in_segment] - np.mean(ecg_segment)) / 
                                                    np.std(ecg_segment) + ecg_offset)
                                    self.ax_peak_compare.plot(ecg_peak_times, ecg_peak_values, 'rs', markersize=6, 
                                                            alpha=0.7, label=f'ECG R-Peaks ({len(ecg_peaks_in_segment)})')
                    except Exception as e:
                        print(f"DEBUG: ECG plotting error: {e}")
                
                self.ax_peak_compare.set_xlim(segment_time[0], segment_time[-1])
                self.ax_peak_compare.set_title(f'Edit Radar Peaks - Segment {self.hrv_edit_segment_start:.0f}s to {end_time:.0f}s')
                self.ax_peak_compare.set_xlabel('Time (s)')
                self.ax_peak_compare.set_ylabel('Normalized Amplitude')
                self.ax_peak_compare.legend()
                self.ax_peak_compare.grid(True, alpha=0.3)
                
                # Update info
                total_peaks = len(self.manual_radar_peaks)
                self.hrv_edit_info.config(text=f"Editing: {total_peaks} total radar peaks. Left=ADD, Right=REMOVE")
                
                print("DEBUG: Plot completed successfully")  # DEBUG
            
            self.canvas_hrv.draw()
            
        except Exception as e:
            print(f"DEBUG: Error in plot_radar_edit_segment: {e}")  # DEBUG
            import traceback
            traceback.print_exc()
            
            self.ax_peak_compare.clear()
            self.ax_peak_compare.text(0.5, 0.5, f'Error plotting edit segment:\n{str(e)}', 
                                    ha='center', va='center', transform=self.ax_peak_compare.transAxes)
            self.canvas_hrv.draw()

    def update_rr_comparison_plot_with_edited(self):
        """Update RR interval comparison plot with latest edited peaks"""
        try:
            if not hasattr(self, 'latest_edited_rr_intervals') or self.latest_edited_rr_intervals is None:
                return
                
            if not hasattr(self, 'hrv_results_raw') or not self.hrv_results_raw:
                return
            
            # Get the comparison plot axes (assuming it's ax_rr_comparison or similar)
            # You'll need to adjust this based on your actual plot structure
            if hasattr(self, 'ax_rr_comparison'):
                ax = self.ax_rr_comparison
            elif hasattr(self, 'ax_hrv_rr'):
                ax = self.ax_hrv_rr
            else:
                # Find the RR comparison plot
                for attr_name in dir(self):
                    if 'ax_' in attr_name and 'rr' in attr_name.lower():
                        ax = getattr(self, attr_name)
                        break
                else:
                    print("Could not find RR comparison plot axis")
                    return
            
            # Clear and replot
            ax.clear()
            
            # Plot ECG RR intervals (if available)
            if 'ecg_rr' in self.hrv_results_raw:
                ecg_rr = self.hrv_results_raw['ecg_rr']
                ecg_time_cumsum = np.cumsum(ecg_rr) / 1000  # Convert to seconds
                ax.plot(ecg_time_cumsum, ecg_rr, 'r-', linewidth=1, alpha=0.7, label='ECG RR')
            
            # Plot edited radar RR intervals
            edited_rr = self.latest_edited_rr_intervals
            radar_time_cumsum = np.cumsum(edited_rr) / 1000  # Convert to seconds
            ax.plot(radar_time_cumsum, edited_rr, 'b-', linewidth=1.5, alpha=0.8, label='Radar RR (Edited)')
            
            ax.set_title('RR Intervals Comparison - Updated with Edited Peaks')
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('RR Interval (ms)')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Refresh the canvas
            if hasattr(self, 'canvas_hrv'):
                self.canvas_hrv.draw()
            elif hasattr(self, 'canvas_features'):
                self.canvas_features.draw()
                
            print("DEBUG: RR comparison plot updated with edited peaks")
            
        except Exception as e:
            print(f"Error updating RR comparison plot: {e}")

    def update_hrv_with_edited_peaks_safe(self):
        """Enhanced update function with robust error handling for comparison metrics"""
        if not hasattr(self, 'manual_radar_peaks') or not self.manual_radar_peaks:
            messagebox.showwarning("Warning", "No edited peaks available")
            return
        
        try:
            import time
            
            # Store edited peaks with timestamp
            self.edited_radar_peaks = np.array(self.manual_radar_peaks)
            radar_times = self.signals['timestamps']
            self.edited_radar_values = self.latest_combined_imf[self.edited_radar_peaks]
            self.edited_peaks_timestamp = time.time()
            
            # Calculate RR intervals from edited peaks
            edited_rr, edited_rr_times = self.calculate_rr_intervals(self.edited_radar_peaks, radar_times)
            
            # Apply same filtering as before
            min_rr = self.min_rr_var.get()
            max_rr = self.max_rr_var.get()
            edited_rr_filtered, edited_rr_times_filtered = self.filter_rr_intervals(
                edited_rr, edited_rr_times, min_rr, max_rr)
            
            # Store latest edited RR intervals for HRV Features tab
            self.latest_edited_rr_intervals = edited_rr_filtered.copy()
            self.latest_edited_rr_times = edited_rr_times_filtered.copy()
            
            print(f"DEBUG: Updated edited RR intervals - Count: {len(self.latest_edited_rr_intervals)}, Mean: {np.mean(self.latest_edited_rr_intervals):.1f}ms")
            
            # Apply MAV if enabled
            enable_mav = self.hrv_enable_mav_var.get()
            if enable_mav:
                mav_window = self.hrv_mav_window_var.get()
                edited_rr_mav = self.apply_moving_average(edited_rr_filtered, mav_window)
            else:
                edited_rr_mav = edited_rr_filtered
            
            # Calculate HRV parameters for edited data
            window_size = self.hrv_window_var.get()
            overlap_percent = self.hrv_overlap_var.get()
            
            edited_stats_raw, edited_hrv_raw, _ = self.calculate_hrv_parameters(
                edited_rr_filtered, window_size, overlap_percent)
            edited_stats_mav, edited_hrv_mav, _ = self.calculate_hrv_parameters(
                edited_rr_mav, window_size, overlap_percent)
            
            # Calculate simple RR intervals RMSE with ECG (Primary RMSE)
            rr_rmse_results = {'raw': None, 'mav': None}
            
            if hasattr(self, 'hrv_results_raw') and self.hrv_results_raw and 'ecg_rr' in self.hrv_results_raw:
                # Calculate simple RR-to-RR RMSE
                ecg_rr_raw = self.hrv_results_raw['ecg_rr']
                rr_rmse_results['raw'] = self.calculate_rr_rmse(ecg_rr_raw, edited_rr_filtered)
                
                if enable_mav and hasattr(self, 'hrv_results_mav') and self.hrv_results_mav and 'ecg_rr' in self.hrv_results_mav:
                    ecg_rr_mav = self.hrv_results_mav['ecg_rr']
                    min_len_mav = min(len(ecg_rr_mav), len(edited_rr_mav))
                    rr_rmse_results['mav'] = self.calculate_rr_rmse(ecg_rr_mav[:min_len_mav], edited_rr_mav[:min_len_mav])
            
            # Store edited results
            self.hrv_results_edited = {
                'radar_rr': edited_rr_filtered,
                'radar_rr_times': edited_rr_times_filtered,
                'radar_rr_mav': edited_rr_mav,
                'radar_hrv_raw': edited_hrv_raw,
                'radar_hrv_mav': edited_hrv_mav,
                'radar_stats_raw': edited_stats_raw,
                'radar_stats_mav': edited_stats_mav,
                'total_edited_peaks': len(self.edited_radar_peaks),
                'update_timestamp': self.edited_peaks_timestamp,
                'rr_rmse': rr_rmse_results  # Store RR RMSE results
            }
            
            # SAFE: Calculate HRV comparison metrics with error handling
            if hasattr(self, 'hrv_results_raw') and self.hrv_results_raw:
                try:
                    print("DEBUG: Attempting RAW comparison metrics calculation...")
                    edited_comparison_raw = self.calculate_hrv_comparison_metrics(
                        edited_hrv_raw, self.hrv_results_raw['ecg_hrv'])
                    
                    if edited_comparison_raw.get('error'):
                        print(f"DEBUG: RAW comparison error: {edited_comparison_raw['error']}")
                        self.hrv_results_edited['comparison_raw'] = None
                        self.hrv_results_edited['comparison_raw_error'] = edited_comparison_raw['error']
                    else:
                        self.hrv_results_edited['comparison_raw'] = edited_comparison_raw
                        print("DEBUG: RAW comparison metrics calculated successfully")
                    
                except Exception as raw_error:
                    print(f"DEBUG: RAW comparison metrics failed: {raw_error}")
                    self.hrv_results_edited['comparison_raw'] = None
                    self.hrv_results_edited['comparison_raw_error'] = str(raw_error)
                
                try:
                    print("DEBUG: Attempting MAV comparison metrics calculation...")
                    if hasattr(self, 'hrv_results_mav') and self.hrv_results_mav:
                        edited_comparison_mav = self.calculate_hrv_comparison_metrics(
                            edited_hrv_mav, self.hrv_results_mav['ecg_hrv'])
                        
                        if edited_comparison_mav.get('error'):
                            print(f"DEBUG: MAV comparison error: {edited_comparison_mav['error']}")
                            self.hrv_results_edited['comparison_mav'] = None
                            self.hrv_results_edited['comparison_mav_error'] = edited_comparison_mav['error']
                        else:
                            self.hrv_results_edited['comparison_mav'] = edited_comparison_mav
                            print("DEBUG: MAV comparison metrics calculated successfully")
                    else:
                        print("DEBUG: No MAV data available for comparison")
                        
                except Exception as mav_error:
                    print(f"DEBUG: MAV comparison metrics failed: {mav_error}")
                    self.hrv_results_edited['comparison_mav'] = None
                    self.hrv_results_edited['comparison_mav_error'] = str(mav_error)
            
            # Update display
            self.display_edited_hrv_results_safe()
            
            # Update plot
            self.update_rr_comparison_plot_with_edited()
            
            # Update info with timestamp and RMSE summary (focus on RR RMSE)
            current_time_str = time.strftime('%H:%M:%S')
            
            # Add RR RMSE summary to info text (primary metric)
            rmse_summary = ""
            if rr_rmse_results['raw'] and np.isfinite(rr_rmse_results['raw']['rmse']):
                rmse_val = rr_rmse_results['raw']['rmse']
                corr_val = rr_rmse_results['raw']['correlation']
                rmse_summary = f" | RR RMSE: {rmse_val:.1f}ms, r={corr_val:.3f}"
            
            self.hrv_edit_info.config(text=f"✅ HRV updated with {len(self.edited_radar_peaks)} edited peaks (Latest: {current_time_str}){rmse_summary}")
            
            print(f"DEBUG: HRV update completed at {current_time_str}")
            if rr_rmse_results['raw']:
                print(f"DEBUG: RR RMSE = {rr_rmse_results['raw']['rmse']:.2f}ms, Correlation = {rr_rmse_results['raw']['correlation']:.3f}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to update HRV with edited peaks: {str(e)}")
            print(f"Edit HRV update error: {e}")
            import traceback
            traceback.print_exc()

    def display_edited_hrv_results_safe(self):
        """Safe display function that handles comparison errors gracefully"""
        if not hasattr(self, 'hrv_results_edited'):
            return
        results = self.hrv_results_edited
        
        self.hrv_results_text.insert(tk.END, f"\n" + "="*70 + "\n")
        self.hrv_results_text.insert(tk.END, f"EDITED RADAR PEAKS HRV RESULTS:\n")
        self.hrv_results_text.insert(tk.END, f"="*70 + "\n")
        self.hrv_results_text.insert(tk.END, f"Total edited peaks: {results['total_edited_peaks']}\n")
        self.hrv_results_text.insert(tk.END, f"Update timestamp: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(results['update_timestamp']))}\n")
        
        # PRIMARY METRIC: Display RR Intervals RMSE Results (most important)
        if 'rr_rmse' in results and results['rr_rmse']['raw']:
            rmse_raw = results['rr_rmse']['raw']
            
            self.hrv_results_text.insert(tk.END, f"\n🎯 RR INTERVALS ACCURACY (Primary Metric):\n")
            self.hrv_results_text.insert(tk.END, f"-" * 50 + "\n")
            
            if rmse_raw['error']:
                self.hrv_results_text.insert(tk.END, f"Error: {rmse_raw['error']}\n")
            else:
                self.hrv_results_text.insert(tk.END, f"Data points compared: {rmse_raw['n_points']}\n")
                self.hrv_results_text.insert(tk.END, f"ECG RR mean: {rmse_raw['ecg_mean']:.1f} ms\n")
                self.hrv_results_text.insert(tk.END, f"Radar RR mean: {rmse_raw['radar_mean']:.1f} ms\n")
                self.hrv_results_text.insert(tk.END, f"\n📊 Accuracy Metrics:\n")
                self.hrv_results_text.insert(tk.END, f"  RMSE: {rmse_raw['rmse']:.2f} ms ⭐\n")
                self.hrv_results_text.insert(tk.END, f"  Correlation (r): {rmse_raw['correlation']:.4f}\n")
                self.hrv_results_text.insert(tk.END, f"  Mean Absolute Error: {rmse_raw['mae']:.2f} ms\n")
                self.hrv_results_text.insert(tk.END, f"  Mean Bias: {rmse_raw['mean_bias']:.2f} ms\n")
                
                # Interpretation
                self.hrv_results_text.insert(tk.END, f"\n📋 Interpretation:\n")
                if rmse_raw['rmse'] < 10:
                    accuracy = "✅ Excellent"
                elif rmse_raw['rmse'] < 20:
                    accuracy = "✅ Good"
                elif rmse_raw['rmse'] < 50:
                    accuracy = "⚠️ Moderate"
                else:
                    accuracy = "❌ Poor"
                
                self.hrv_results_text.insert(tk.END, f"  Accuracy: {accuracy} (RMSE = {rmse_raw['rmse']:.1f}ms)\n")
                
                if rmse_raw['correlation'] > 0.9:
                    trend = "✅ Excellent trend matching"
                elif rmse_raw['correlation'] > 0.8:
                    trend = "✅ Good trend matching"
                elif rmse_raw['correlation'] > 0.6:
                    trend = "⚠️ Moderate trend matching"
                else:
                    trend = "❌ Poor trend matching"
                
                self.hrv_results_text.insert(tk.END, f"  Trend: {trend} (r = {rmse_raw['correlation']:.3f})\n")
        
        # SECONDARY: HRV Statistics
        self.hrv_results_text.insert(tk.END, f"\n📈 EDITED RADAR HRV STATISTICS:\n")
        self.hrv_results_text.insert(tk.END, f"-" * 50 + "\n")
        self.hrv_results_text.insert(tk.END, f"  SDNN: {results['radar_stats_raw']['SDNN_mean']:.2f} ± {results['radar_stats_raw']['SDNN_std']:.2f} ms\n")
        self.hrv_results_text.insert(tk.END, f"  RMSSD: {results['radar_stats_raw']['RMSSD_mean']:.2f} ± {results['radar_stats_raw']['RMSSD_std']:.2f} ms\n")
        self.hrv_results_text.insert(tk.END, f"  pNN50: {results['radar_stats_raw']['pNN50_mean']:.2f}%\n")
        
        # OPTIONAL: HRV Comparison Metrics (if available)
        if 'comparison_raw' in results:
            if results['comparison_raw'] is None:
                # Show error if comparison failed
                error_msg = results.get('comparison_raw_error', 'Unknown error')
                self.hrv_results_text.insert(tk.END, f"\n❌ HRV COMPARISON METRICS:\n")
                self.hrv_results_text.insert(tk.END, f"Calculation failed: {error_msg}\n")
            else:
                # Show successful comparison
                comp = results['comparison_raw']
                self.hrv_results_text.insert(tk.END, f"\n📊 HRV COMPARISON METRICS (Secondary):\n")
                self.hrv_results_text.insert(tk.END, f"-" * 50 + "\n")
                
                if 'sdnn_rmse' in comp and np.isfinite(comp['sdnn_rmse']):
                    self.hrv_results_text.insert(tk.END, f"  SDNN RMSE: {comp['sdnn_rmse']:.2f} ms\n")
                if 'sdnn_correlation' in comp and np.isfinite(comp['sdnn_correlation']):
                    self.hrv_results_text.insert(tk.END, f"  SDNN Correlation: {comp['sdnn_correlation']:.3f}\n")
                if 'rmssd_rmse' in comp and np.isfinite(comp['rmssd_rmse']):
                    self.hrv_results_text.insert(tk.END, f"  RMSSD RMSE: {comp['rmssd_rmse']:.2f} ms\n")
                if 'rmssd_correlation' in comp and np.isfinite(comp['rmssd_correlation']):
                    self.hrv_results_text.insert(tk.END, f"  RMSSD Correlation: {comp['rmssd_correlation']:.3f}\n")
        
        self.hrv_results_text.insert(tk.END, f"\n" + "="*70 + "\n")

    
    def export_edited_hrv(self):
        """Export edited HRV results"""
        if not hasattr(self, 'hrv_results_edited'):
            messagebox.showwarning("Warning", "No edited HRV results to export.")
            return
        
        file_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            title="Save Edited HRV Results",
            initialfile="hrv_edited_results.csv"
        )
        
        if file_path:
            try:
                results = self.hrv_results_edited
                
                # Create dataframe with edited results
                data_dict = {
                    'Edited_Radar_RR_ms': results['radar_rr'],
                    'ECG_RR_ms': self.hrv_results_raw['ecg_rr'][:len(results['radar_rr'])]
                }
                
                df = pd.DataFrame(data_dict)
                df.to_csv(file_path, index=False)
                
                messagebox.showinfo("Export Complete", f"Edited HRV results exported to:\n{file_path}")
            except Exception as e:
                messagebox.showerror("Export Error", f"Failed to export: {str(e)}")

    def plot_hrv_parameters_with_mav(self, radar_hrv_raw, radar_hrv_mav, ecg_hrv_raw, ecg_hrv_mav, enable_mav):
        """Plot HRV parameters with MAV comparison"""
        # Plot radar HRV
        if enable_mav and len(radar_hrv_mav['times']) > 0:
            self.ax_hrv_radar.plot(radar_hrv_raw['times'], radar_hrv_raw['SDNN'], 'b-', alpha=0.6, label='SDNN Raw')
            self.ax_hrv_radar.plot(radar_hrv_mav['times'], radar_hrv_mav['SDNN'], 'r-', linewidth=2, label='SDNN MAV')
            self.ax_hrv_radar.set_title('Radar HRV Parameters (Raw vs MAV)')
        else:
            self.ax_hrv_radar.plot(radar_hrv_raw['times'], radar_hrv_raw['SDNN'], 'b-', label='SDNN Raw')
            self.ax_hrv_radar.set_title('Radar HRV Parameters (Raw)')
        
        self.ax_hrv_radar.set_xlabel('Window Index')
        self.ax_hrv_radar.set_ylabel('SDNN (ms)')
        self.ax_hrv_radar.legend()
        self.ax_hrv_radar.grid(True)
        
        # Plot ECG HRV
        if enable_mav and len(ecg_hrv_mav['times']) > 0:
            self.ax_hrv_ecg.plot(ecg_hrv_raw['times'], ecg_hrv_raw['SDNN'], 'r-', alpha=0.6, label='SDNN Raw')
            self.ax_hrv_ecg.plot(ecg_hrv_mav['times'], ecg_hrv_mav['SDNN'], 'g-', linewidth=2, label='SDNN MAV')
            self.ax_hrv_ecg.set_title('ECG HRV Parameters (Raw vs MAV)')
        else:
            self.ax_hrv_ecg.plot(ecg_hrv_raw['times'], ecg_hrv_raw['SDNN'], 'r-', label='SDNN Raw')
            self.ax_hrv_ecg.set_title('ECG HRV Parameters (Raw)')
        
        self.ax_hrv_ecg.set_xlabel('Window Index')
        self.ax_hrv_ecg.set_xlabel('Window Index')
        self.ax_hrv_ecg.set_ylabel('SDNN (ms)')
        self.ax_hrv_ecg.legend()
        self.ax_hrv_ecg.grid(True)
        
        # Compare HRV parameters
        if enable_mav:
            min_len = min(len(radar_hrv_mav['SDNN']), len(ecg_hrv_mav['SDNN']))
            if min_len > 0:
                self.ax_hrv_compare.plot(range(min_len), radar_hrv_mav['SDNN'][:min_len], 
                                    'b-', linewidth=2, label='Radar SDNN MAV')
                self.ax_hrv_compare.plot(range(min_len), ecg_hrv_mav['SDNN'][:min_len], 
                                    'r-', linewidth=2, label='ECG SDNN MAV')
                self.ax_hrv_compare.set_title('HRV Comparison (MAV)')
        else:
            min_len = min(len(radar_hrv_raw['SDNN']), len(ecg_hrv_raw['SDNN']))
            if min_len > 0:
                self.ax_hrv_compare.plot(range(min_len), radar_hrv_raw['SDNN'][:min_len], 
                                    'b-', label='Radar SDNN Raw')
                self.ax_hrv_compare.plot(range(min_len), ecg_hrv_raw['SDNN'][:min_len], 
                                    'r-', label='ECG SDNN Raw')
                self.ax_hrv_compare.set_title('HRV Comparison (Raw)')
        
        self.ax_hrv_compare.set_xlabel('Window Index')
        self.ax_hrv_compare.set_ylabel('SDNN (ms)')
        self.ax_hrv_compare.legend()
        self.ax_hrv_compare.grid(True)

    def display_hrv_results(self, radar_stats_raw, radar_stats_mav, ecg_stats_raw, ecg_stats_mav, enable_mav):
        """Display HRV results with safe error handling"""
        self.hrv_results_text.insert(tk.END, f"RAW HRV RESULTS:\n")
        self.hrv_results_text.insert(tk.END, f"=" * 30 + "\n")
        
        self.hrv_results_text.insert(tk.END, f"RADAR RAW:\n")
        self.hrv_results_text.insert(tk.END, f"  SDNN: {radar_stats_raw['SDNN_mean']:.2f} ± {radar_stats_raw['SDNN_std']:.2f} ms\n")
        self.hrv_results_text.insert(tk.END, f"  RMSSD: {radar_stats_raw['RMSSD_mean']:.2f} ± {radar_stats_raw['RMSSD_std']:.2f} ms\n")
        self.hrv_results_text.insert(tk.END, f"  pNN50: {radar_stats_raw['pNN50_mean']:.2f}%\n")
        self.hrv_results_text.insert(tk.END, f"  HR: {radar_stats_raw['HR_mean']:.1f} ± {radar_stats_raw['HR_std']:.1f} BPM\n\n")
        
        self.hrv_results_text.insert(tk.END, f"ECG RAW:\n")
        self.hrv_results_text.insert(tk.END, f"  SDNN: {ecg_stats_raw['SDNN_mean']:.2f} ± {ecg_stats_raw['SDNN_std']:.2f} ms\n")
        self.hrv_results_text.insert(tk.END, f"  RMSSD: {ecg_stats_raw['RMSSD_mean']:.2f} ± {ecg_stats_raw['RMSSD_std']:.2f} ms\n")
        self.hrv_results_text.insert(tk.END, f"  pNN50: {ecg_stats_raw['pNN50_mean']:.2f}%\n")
        self.hrv_results_text.insert(tk.END, f"  HR: {ecg_stats_raw['HR_mean']:.1f} ± {ecg_stats_raw['HR_std']:.1f} BPM\n\n")
        
        # Safe calculation of RAW comparison metrics
        if hasattr(self, 'hrv_results_raw') and self.hrv_results_raw:
            try:
                raw_comparison = self.calculate_hrv_comparison_metrics(
                    self.hrv_results_raw['radar_hrv'], self.hrv_results_raw['ecg_hrv'])
                
                self.hrv_results_text.insert(tk.END, f"RAW COMPARISON METRICS:\n")
                self.hrv_results_text.insert(tk.END, f"  SDNN RMSE: {raw_comparison['sdnn_rmse']:.2f} ms\n")
                self.hrv_results_text.insert(tk.END, f"  SDNN NRMSE: {raw_comparison['sdnn_nrmse']:.1f}%\n")
                self.hrv_results_text.insert(tk.END, f"  SDNN Correlation: {raw_comparison['sdnn_correlation']:.3f}\n")
                self.hrv_results_text.insert(tk.END, f"  RMSSD RMSE: {raw_comparison['rmssd_rmse']:.2f} ms\n")
                self.hrv_results_text.insert(tk.END, f"  RMSSD NRMSE: {raw_comparison['rmssd_nrmse']:.1f}%\n")
                self.hrv_results_text.insert(tk.END, f"  RMSSD Correlation: {raw_comparison['rmssd_correlation']:.3f}\n\n")
            except Exception as e:
                self.hrv_results_text.insert(tk.END, f"RAW COMPARISON METRICS: Calculation failed\n\n")
                print(f"Raw comparison metrics error: {e}")
        else:
            self.hrv_results_text.insert(tk.END, f"RAW COMPARISON METRICS: Data not available\n\n")
        
        if enable_mav:
            self.hrv_results_text.insert(tk.END, f"MAV HRV RESULTS:\n")
            self.hrv_results_text.insert(tk.END, f"=" * 30 + "\n")
            
            self.hrv_results_text.insert(tk.END, f"RADAR MAV:\n")
            self.hrv_results_text.insert(tk.END, f"  SDNN: {radar_stats_mav['SDNN_mean']:.2f} ± {radar_stats_mav['SDNN_std']:.2f} ms\n")
            self.hrv_results_text.insert(tk.END, f"  RMSSD: {radar_stats_mav['RMSSD_mean']:.2f} ± {radar_stats_mav['RMSSD_std']:.2f} ms\n")
            self.hrv_results_text.insert(tk.END, f"  pNN50: {radar_stats_mav['pNN50_mean']:.2f}%\n")
            self.hrv_results_text.insert(tk.END, f"  HR: {radar_stats_mav['HR_mean']:.1f} ± {radar_stats_mav['HR_std']:.1f} BPM\n\n")
            
            self.hrv_results_text.insert(tk.END, f"ECG MAV:\n")
            self.hrv_results_text.insert(tk.END, f"  SDNN: {ecg_stats_mav['SDNN_mean']:.2f} ± {ecg_stats_mav['SDNN_std']:.2f} ms\n")
            self.hrv_results_text.insert(tk.END, f"  RMSSD: {ecg_stats_mav['RMSSD_mean']:.2f} ± {ecg_stats_mav['RMSSD_std']:.2f} ms\n")
            self.hrv_results_text.insert(tk.END, f"  pNN50: {ecg_stats_mav['pNN50_mean']:.2f}%\n")
            self.hrv_results_text.insert(tk.END, f"  HR: {ecg_stats_mav['HR_mean']:.1f} ± {ecg_stats_mav['HR_std']:.1f} BPM\n\n")
            
            # Safe calculation of MAV comparison metrics
            if hasattr(self, 'hrv_results_mav') and self.hrv_results_mav:
                try:
                    mav_comparison = self.calculate_hrv_comparison_metrics(
                        self.hrv_results_mav['radar_hrv'], self.hrv_results_mav['ecg_hrv'])
                    
                    self.hrv_results_text.insert(tk.END, f"MAV COMPARISON METRICS:\n")
                    self.hrv_results_text.insert(tk.END, f"  SDNN RMSE: {mav_comparison['sdnn_rmse']:.2f} ms\n")
                    self.hrv_results_text.insert(tk.END, f"  SDNN NRMSE: {mav_comparison['sdnn_nrmse']:.1f}%\n")
                    self.hrv_results_text.insert(tk.END, f"  SDNN Correlation: {mav_comparison['sdnn_correlation']:.3f}\n")
                    self.hrv_results_text.insert(tk.END, f"  RMSSD RMSE: {mav_comparison['rmssd_rmse']:.2f} ms\n")
                    self.hrv_results_text.insert(tk.END, f"  RMSSD NRMSE: {mav_comparison['rmssd_nrmse']:.1f}%\n")
                    self.hrv_results_text.insert(tk.END, f"  RMSSD Correlation: {mav_comparison['rmssd_correlation']:.3f}\n\n")
                    
                    # Safe improvement analysis
                    if hasattr(self, 'hrv_results_raw') and self.hrv_results_raw:
                        try:
                            raw_comparison = self.calculate_hrv_comparison_metrics(
                                self.hrv_results_raw['radar_hrv'], self.hrv_results_raw['ecg_hrv'])
                            
                            # Calculate improvement metrics
                            sdnn_rmse_improvement = raw_comparison['sdnn_rmse'] - mav_comparison['sdnn_rmse']
                            rmssd_rmse_improvement = raw_comparison['rmssd_rmse'] - mav_comparison['rmssd_rmse']
                            sdnn_corr_improvement = mav_comparison['sdnn_correlation'] - raw_comparison['sdnn_correlation']
                            rmssd_corr_improvement = mav_comparison['rmssd_correlation'] - raw_comparison['rmssd_correlation']
                            
                            self.hrv_results_text.insert(tk.END, f"MAV IMPROVEMENT ANALYSIS:\n")
                            self.hrv_results_text.insert(tk.END, f"  SDNN RMSE Change: {sdnn_rmse_improvement:+.2f} ms\n")
                            self.hrv_results_text.insert(tk.END, f"  RMSSD RMSE Change: {rmssd_rmse_improvement:+.2f} ms\n")
                            self.hrv_results_text.insert(tk.END, f"  SDNN Correlation Change: {sdnn_corr_improvement:+.3f}\n")
                            self.hrv_results_text.insert(tk.END, f"  RMSSD Correlation Change: {rmssd_corr_improvement:+.3f}\n")
                            
                            # Overall assessment
                            improvements = [sdnn_rmse_improvement > 0, rmssd_rmse_improvement > 0, 
                                        sdnn_corr_improvement > 0, rmssd_corr_improvement > 0]
                            improvement_count = sum(improvements)
                            
                            if improvement_count >= 3:
                                assessment = "SIGNIFICANT IMPROVEMENT"
                            elif improvement_count >= 2:
                                assessment = "MODERATE IMPROVEMENT"
                            elif improvement_count >= 1:
                                assessment = "SLIGHT IMPROVEMENT"
                            else:
                                assessment = "NO IMPROVEMENT"
                            
                            self.hrv_results_text.insert(tk.END, f"  Overall MAV Effect: {assessment}\n\n")
                        except Exception as e:
                            self.hrv_results_text.insert(tk.END, f"  MAV Improvement Analysis: Calculation failed\n\n")
                            print(f"MAV improvement analysis error: {e}")
                    
                except Exception as e:
                    self.hrv_results_text.insert(tk.END, f"MAV COMPARISON METRICS: Calculation failed\n\n")
                    print(f"MAV comparison metrics error: {e}")
            else:
                self.hrv_results_text.insert(tk.END, f"MAV COMPARISON METRICS: Data not available\n\n")
    def update_hrv_comparison_plot_with_edited(self):
        """Update HRV comparison plot to include edited results"""
        if not hasattr(self, 'hrv_results_edited'):
            return
        
        # Clear and replot HRV comparison with edited data
        self.ax_hrv_compare.clear()
        
        try:
            # Get minimum length for synchronized plotting
            min_len = len(self.hrv_results_edited['radar_hrv_raw']['SDNN'])
            
            if hasattr(self, 'hrv_results_raw') and self.hrv_results_raw:
                min_len = min(min_len, len(self.hrv_results_raw['radar_hrv']['SDNN']))
                min_len = min(min_len, len(self.hrv_results_raw['ecg_hrv']['SDNN']))
            
            if min_len > 0:
                indices = range(min_len)
                
                # Plot ECG reference
                if hasattr(self, 'hrv_results_raw'):
                    self.ax_hrv_compare.plot(indices, self.hrv_results_raw['ecg_hrv']['SDNN'][:min_len], 
                                        'r-', alpha=0.8, linewidth=1.5, label='ECG Reference')
                
                # Plot original radar
                if hasattr(self, 'hrv_results_raw'):
                    self.ax_hrv_compare.plot(indices, self.hrv_results_raw['radar_hrv']['SDNN'][:min_len], 
                                        'b-', alpha=0.6, linewidth=1, label='Radar Original')
                
                # Plot edited radar
                self.ax_hrv_compare.plot(indices, self.hrv_results_edited['radar_hrv_raw']['SDNN'][:min_len], 
                                    'g-', linewidth=2, label='Radar Edited')
                
                self.ax_hrv_compare.set_title('HRV SDNN Comparison: Original vs Edited vs ECG')
            else:
                self.ax_hrv_compare.text(0.5, 0.5, 'Insufficient data for comparison', 
                                    ha='center', va='center', transform=self.ax_hrv_compare.transAxes)
        
        except Exception as e:
            self.ax_hrv_compare.text(0.5, 0.5, f'Error updating comparison: {str(e)}', 
                                ha='center', va='center', transform=self.ax_hrv_compare.transAxes)
            print(f"HRV comparison update error: {e}")
        
        self.ax_hrv_compare.set_xlabel('Window Index')
        self.ax_hrv_compare.set_ylabel('SDNN (ms)')
        self.ax_hrv_compare.legend()
        self.ax_hrv_compare.grid(True)
        
        self.canvas_hrv.draw()

    def export_raw_hrv(self):
        print("[DEBUG] Raw HRV export button clicked")
        if not hasattr(self, 'hrv_results_raw') or not self.hrv_results_raw:
            messagebox.showwarning("Warning", "No raw HRV results to export. Calculate HRV first.")
            return
        
        file_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            title="Save Raw HRV Results",
            initialfile="hrv_raw_results.csv"
        )
        
        if file_path:
            try:
                results = self.hrv_results_raw

                if not all(key in results for key in ['ecg_rr', 'radar_rr', 'ecg_hrv', 'radar_hrv']):
                    messagebox.showerror("Export Error", "Incomplete HRV data. Please recalculate HRV.")
                    return
                
                # Convert to flat list of floats
                ecg_rr_list = [float(x[0]) if isinstance(x, (list, tuple)) else float(x) for x in results['ecg_rr']]
                radar_rr_list = [float(x[0]) if isinstance(x, (list, tuple)) else float(x) for x in results['radar_rr']]

                max_len = max(len(ecg_rr_list), len(radar_rr_list))

                ecg_rr_padded = np.full(max_len, np.nan)
                ecg_rr_padded[:len(ecg_rr_list)] = ecg_rr_list

                radar_rr_padded = np.full(max_len, np.nan)
                radar_rr_padded[:len(radar_rr_list)] = radar_rr_list

                data_dict = {
                    'Radar_RR_ms': radar_rr_padded
                }

                df = pd.DataFrame(data_dict)
                df.to_csv(file_path, index=False)


                messagebox.showinfo("Export Complete", f"Raw HRV results exported to:\n{file_path}")

            except Exception as e:
                messagebox.showerror("Export Error", f"Failed to export raw HRV results: {str(e)}")

    def export_mav_hrv(self):
        if not hasattr(self, 'hrv_results_mav') or not self.hrv_results_mav:
            messagebox.showwarning("Warning", "No MAV HRV results to export. Calculate HRV with MAV enabled first.")
            return
        
        file_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            title="Save MAV HRV Results",
            initialfile="hrv_mav_results.csv"
        )
        
        if file_path:
            try:
                results = self.hrv_results_mav

                if not all(key in results for key in ['ecg_rr', 'radar_rr', 'ecg_hrv', 'radar_hrv']):
                    messagebox.showerror("Export Error", "Incomplete MAV HRV data. Please recalculate HRV.")
                    return
                
                # Convert to flat list of floats
                ecg_rr_list = [float(x[0]) if isinstance(x, (list, tuple)) else float(x) for x in results['ecg_rr']]
                radar_rr_list = [float(x[0]) if isinstance(x, (list, tuple)) else float(x) for x in results['radar_rr']]

                max_len = max(len(ecg_rr_list), len(radar_rr_list))

                ecg_rr_padded = np.full(max_len, np.nan)
                ecg_rr_padded[:len(ecg_rr_list)] = ecg_rr_list

                radar_rr_padded = np.full(max_len, np.nan)
                radar_rr_padded[:len(radar_rr_list)] = radar_rr_list

                data_dict = {
                    'ECG_RR_MAV_ms': ecg_rr_padded,
                    'Radar_RR_MAV_ms': radar_rr_padded
                }

                df = pd.DataFrame(data_dict)
                df.to_csv(file_path, index=False)


                messagebox.showinfo("Export Complete", f"MAV HRV results exported to:\n{file_path}")

            except Exception as e:
                messagebox.showerror("Export Error", f"Failed to export MAV HRV results: {str(e)}")

    def calculate_hrv_comparison_metrics(self, radar_hrv, ecg_hrv):
        """
        Robust calculation of comparison metrics between radar and ECG HRV parameters
        Handles various data structure issues and missing data gracefully
        """
        try:
            # Initialize default metrics
            default_metrics = {
                'sdnn_rmse': np.nan,
                'sdnn_nrmse': np.nan,
                'sdnn_correlation': np.nan,
                'rmssd_rmse': np.nan,
                'rmssd_nrmse': np.nan,
                'rmssd_correlation': np.nan,
                'pnn50_rmse': np.nan,
                'pnn50_correlation': np.nan,
                'hr_rmse': np.nan,
                'hr_correlation': np.nan,
                'error': None
            }
            
            # Check if inputs are valid
            if not radar_hrv or not ecg_hrv:
                print("DEBUG: Empty radar_hrv or ecg_hrv data")
                default_metrics['error'] = "Empty HRV data"
                return default_metrics
            
            # Debug: Print data structure
            print(f"DEBUG: radar_hrv keys: {list(radar_hrv.keys()) if isinstance(radar_hrv, dict) else 'Not a dict'}")
            print(f"DEBUG: ecg_hrv keys: {list(ecg_hrv.keys()) if isinstance(ecg_hrv, dict) else 'Not a dict'}")
            
            comparison_metrics = {}
            
            # List of HRV parameters to compare
            hrv_parameters = ['SDNN', 'RMSSD', 'pNN50', 'HR_mean']
            
            for param in hrv_parameters:
                try:
                    # Check if parameter exists in both datasets
                    if param not in radar_hrv or param not in ecg_hrv:
                        print(f"DEBUG: Parameter {param} missing in data")
                        comparison_metrics[f'{param.lower()}_rmse'] = np.nan
                        comparison_metrics[f'{param.lower()}_correlation'] = np.nan
                        if param != 'HR_mean':  # HR_mean doesn't have NRMSE
                            comparison_metrics[f'{param.lower()}_nrmse'] = np.nan
                        continue
                    
                    # Get data arrays
                    radar_data = radar_hrv[param]
                    ecg_data = ecg_hrv[param]
                    
                    # Convert to numpy arrays and handle different data types
                    if isinstance(radar_data, (list, tuple)):
                        radar_array = np.array(radar_data)
                    elif isinstance(radar_data, np.ndarray):
                        radar_array = radar_data
                    else:
                        # Single value
                        radar_array = np.array([radar_data])
                    
                    if isinstance(ecg_data, (list, tuple)):
                        ecg_array = np.array(ecg_data)
                    elif isinstance(ecg_data, np.ndarray):
                        ecg_array = ecg_data
                    else:
                        # Single value
                        ecg_array = np.array([ecg_data])
                    
                    # Debug: Print array info
                    print(f"DEBUG: {param} - radar_array shape: {radar_array.shape}, ecg_array shape: {ecg_array.shape}")
                    
                    # Handle empty arrays
                    if len(radar_array) == 0 or len(ecg_array) == 0:
                        print(f"DEBUG: Empty arrays for {param}")
                        comparison_metrics[f'{param.lower()}_rmse'] = np.nan
                        comparison_metrics[f'{param.lower()}_correlation'] = np.nan
                        if param != 'HR_mean':
                            comparison_metrics[f'{param.lower()}_nrmse'] = np.nan
                        continue
                    
                    # Synchronize array lengths (take minimum)
                    min_length = min(len(radar_array), len(ecg_array))
                    radar_sync = radar_array[:min_length]
                    ecg_sync = ecg_array[:min_length]
                    
                    # Remove NaN and infinite values
                    valid_mask = np.isfinite(radar_sync) & np.isfinite(ecg_sync)
                    radar_clean = radar_sync[valid_mask]
                    ecg_clean = ecg_sync[valid_mask]
                    
                    print(f"DEBUG: {param} - clean data length: {len(radar_clean)}")
                    
                    if len(radar_clean) < 2:
                        print(f"DEBUG: Insufficient clean data for {param}")
                        comparison_metrics[f'{param.lower()}_rmse'] = np.nan
                        comparison_metrics[f'{param.lower()}_correlation'] = np.nan
                        if param != 'HR_mean':
                            comparison_metrics[f'{param.lower()}_nrmse'] = np.nan
                        continue
                    
                    # Calculate RMSE
                    rmse = np.sqrt(np.mean((radar_clean - ecg_clean)**2))
                    comparison_metrics[f'{param.lower()}_rmse'] = rmse
                    
                    # Calculate NRMSE (normalized by mean of ECG) - except for HR_mean
                    if param != 'HR_mean':
                        ecg_mean = np.mean(ecg_clean)
                        if ecg_mean != 0:
                            nrmse = (rmse / abs(ecg_mean)) * 100
                            comparison_metrics[f'{param.lower()}_nrmse'] = nrmse
                        else:
                            comparison_metrics[f'{param.lower()}_nrmse'] = np.nan
                    
                    # Calculate correlation
                    if len(radar_clean) > 1 and np.std(radar_clean) > 0 and np.std(ecg_clean) > 0:
                        correlation = np.corrcoef(radar_clean, ecg_clean)[0, 1]
                        comparison_metrics[f'{param.lower()}_correlation'] = correlation
                    else:
                        comparison_metrics[f'{param.lower()}_correlation'] = np.nan
                    
                    print(f"DEBUG: {param} - RMSE: {rmse:.3f}, Correlation: {comparison_metrics[f'{param.lower()}_correlation']:.3f}")
                    
                except Exception as param_error:
                    print(f"DEBUG: Error calculating metrics for {param}: {param_error}")
                    comparison_metrics[f'{param.lower()}_rmse'] = np.nan
                    comparison_metrics[f'{param.lower()}_correlation'] = np.nan
                    if param != 'HR_mean':
                        comparison_metrics[f'{param.lower()}_nrmse'] = np.nan
            
            # Handle legacy key names for backward compatibility
            if 'sdnn_rmse' in comparison_metrics:
                comparison_metrics['sdnn_rmse'] = comparison_metrics['sdnn_rmse']
            if 'rmssd_rmse' in comparison_metrics:
                comparison_metrics['rmssd_rmse'] = comparison_metrics['rmssd_rmse']
            
            print(f"DEBUG: Final comparison_metrics keys: {list(comparison_metrics.keys())}")
            return comparison_metrics
            
        except Exception as e:
            print(f"DEBUG: Critical error in calculate_hrv_comparison_metrics: {e}")
            import traceback
            traceback.print_exc()
            
            default_metrics['error'] = str(e)
            return default_metrics

    def calculate_initial_rr_rmse(self, radar_rr, ecg_rr):
        """Calculate initial RR RMSE when first running HRV analysis"""
        try:
            # Calculate RR RMSE for initial analysis
            initial_rmse = self.calculate_rr_rmse(ecg_rr, radar_rr)
            
            # Store in hrv_results_raw for reference
            if not hasattr(self, 'hrv_results_raw'):
                self.hrv_results_raw = {}
            
            self.hrv_results_raw['rr_rmse'] = initial_rmse
            
            # Print initial RMSE
            if initial_rmse and not initial_rmse['error']:
                print(f"Initial RR RMSE: {initial_rmse['rmse']:.2f}ms, Correlation: {initial_rmse['correlation']:.3f}")
                
            return initial_rmse
            
        except Exception as e:
            print(f"Error calculating initial RR RMSE: {e}")
            return None
        
    def switch_to_comparison_mode(self):
        """Switch plot to show peak comparison"""
        self.hrv_editing_mode = False
        self.plot_peak_detection_comparison()

    def switch_to_editing_mode(self):
        """Switch plot to show editing interface"""
        self.hrv_editing_mode = True
        self.plot_radar_edit_segment()
    
    def exit_editing_mode(self):
        """Exit editing mode and return to comparison view"""
        print("DEBUG: Exiting editing mode")  # DEBUG
        self.hrv_editing_mode = False
        self.hrv_prev_btn.config(state='disabled')
        self.hrv_next_btn.config(state='disabled')
        self.exit_edit_btn.config(state='disabled')
        
        # Disconnect specific click events
        try:
            if hasattr(self, 'click_connection'):
                self.ax_peak_compare.figure.canvas.mpl_disconnect(self.click_connection)
                print("DEBUG: Disconnected click events")  # DEBUG
        except:
            print("DEBUG: Could not disconnect click events")  # DEBUG
        
        # Return to comparison plot
        self.plot_peak_detection_comparison()
        self.hrv_edit_info.config(text="Editing exited. Click 'Edit Radar Peaks' to edit again.")
        print("DEBUG: Returned to comparison mode")  # DEBUG
    
    def create_hrv_features_tab(self, parent):
        """Create HRV Feature Extraction tab with MAV HRV option"""
        def create_features_content(scrollable_frame):
            main_frame = ttk.Frame(scrollable_frame)
            main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            left_panel = ttk.LabelFrame(main_frame, text="HRV Feature Extraction Controls", width=350)
            left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10, expand=False)
            left_panel.pack_propagate(False)
            
            # Information frame
            info_frame = ttk.LabelFrame(left_panel, text="Feature Information")
            info_frame.pack(fill=tk.X, expand=False, padx=5, pady=5)
            
            info_label = ttk.Label(info_frame, 
                                text="Extract 12 PAF prediction features:\n"
                                    "• 2 Spectral (LF, HF Power)\n"
                                    "• 6 Bispectral (P1, P2, H1-H4)\n"
                                    "• 4 Non-linear (SD1, SD2, SD1/SD2, SamEn)\n"
                                    "\nBased on Mohebbi & Ghassemian (2012)", 
                                justify=tk.LEFT, foreground="darkblue")
            info_label.pack(padx=5, pady=5)
            
            # Data source selection
            source_frame = ttk.LabelFrame(left_panel, text="RR Intervals Source")
            source_frame.pack(fill=tk.X, expand=False, padx=5, pady=5)

            self.rr_source_var = tk.StringVar(value="paper_method")

            # Existing options
            ttk.Radiobutton(source_frame, text="From Paper Method Results (Radar)", 
                        variable=self.rr_source_var, value="paper_method").pack(anchor=tk.W, padx=5, pady=2)
            ttk.Radiobutton(source_frame, text="From Edited Radar Peaks", 
                        variable=self.rr_source_var, value="edited_peaks").pack(anchor=tk.W, padx=5, pady=2)
            
            # NEW: MAV HRV options
            self.mav_radar_radio = ttk.Radiobutton(source_frame, text="From MAV Filtered Radar RR", 
                        variable=self.rr_source_var, value="mav_radar")
            self.mav_radar_radio.pack(anchor=tk.W, padx=5, pady=2)
            
            self.mav_ecg_radio = ttk.Radiobutton(source_frame, text="From MAV Filtered ECG RR", 
                        variable=self.rr_source_var, value="mav_ecg")
            self.mav_ecg_radio.pack(anchor=tk.W, padx=5, pady=2)
            
            # Existing options
            ttk.Radiobutton(source_frame, text="From ECG R-Peaks", 
                        variable=self.rr_source_var, value="ecg_peaks").pack(anchor=tk.W, padx=5, pady=2)
            ttk.Radiobutton(source_frame, text="Load from CSV/NPY File", 
                        variable=self.rr_source_var, value="load_file").pack(anchor=tk.W, padx=5, pady=2)
            
            # NEW: MAV info panel
            mav_info_frame = ttk.Frame(source_frame)
            mav_info_frame.pack(fill=tk.X, padx=5, pady=2)
            
            self.mav_info_label = ttk.Label(mav_info_frame, text="MAV data not available", 
                                        font=("Arial", 8), foreground="gray")
            self.mav_info_label.pack(padx=5, pady=2)
            
            # Update MAV info on startup
            self.update_mav_info_display()
            
            # Comparison mode
            comparison_frame = ttk.LabelFrame(left_panel, text="Comparison Mode")
            comparison_frame.pack(fill=tk.X, expand=False, padx=5, pady=5)

            self.comparison_mode_var = tk.BooleanVar(value=False)
            ttk.Checkbutton(comparison_frame, text="Extract Both Radar & ECG Features", 
                        variable=self.comparison_mode_var).pack(anchor=tk.W, padx=5, pady=2)

            comparison_info = ttk.Label(comparison_frame, 
                                    text="When checked: Extract features from both\nradar (paper method/edited/MAV) and ECG for comparison", 
                                    font=("Arial", 8), foreground="gray")
            comparison_info.pack(padx=5, pady=2)
            
            # File loading
            file_frame = ttk.Frame(source_frame)
            file_frame.pack(fill=tk.X, padx=5, pady=2)
            
            self.load_rr_features_btn = ttk.Button(file_frame, text="Load RR File", 
                                                command=self.load_rr_for_features)
            self.load_rr_features_btn.pack(side=tk.LEFT, padx=2)
            
            self.rr_file_label = ttk.Label(file_frame, text="No file loaded", font=("Arial", 8))
            self.rr_file_label.pack(side=tk.LEFT, padx=5)
            
            # Processing parameters
            params_frame = ttk.LabelFrame(left_panel, text="Processing Parameters")
            params_frame.pack(fill=tk.X, expand=False, padx=5, pady=5)
            
            # RR interval filtering
            filter_frame = ttk.Frame(params_frame)
            filter_frame.pack(fill=tk.X, padx=5, pady=2)
            ttk.Label(filter_frame, text="Min RR (ms):").pack(side=tk.LEFT)
            self.features_min_rr_var = tk.DoubleVar(value=300.0)
            ttk.Entry(filter_frame, textvariable=self.features_min_rr_var, width=8).pack(side=tk.RIGHT)
            
            filter_frame2 = ttk.Frame(params_frame)
            filter_frame2.pack(fill=tk.X, padx=5, pady=2)
            ttk.Label(filter_frame2, text="Max RR (ms):").pack(side=tk.LEFT)
            self.features_max_rr_var = tk.DoubleVar(value=1500.0)
            ttk.Entry(filter_frame2, textvariable=self.features_max_rr_var, width=8).pack(side=tk.RIGHT)
            
            # Minimum RR intervals required
            min_rr_frame = ttk.Frame(params_frame)
            min_rr_frame.pack(fill=tk.X, padx=5, pady=2)
            ttk.Label(min_rr_frame, text="Min RR Count:").pack(side=tk.LEFT)
            self.min_rr_count_var = tk.IntVar(value=50)
            ttk.Entry(min_rr_frame, textvariable=self.min_rr_count_var, width=8).pack(side=tk.RIGHT)
            
            # Processing buttons
            button_frame = ttk.Frame(left_panel)
            button_frame.pack(fill=tk.X, expand=False, padx=5, pady=10)

            self.extract_features_main_btn = ttk.Button(button_frame, text="Extract HRV Features", 
                                                    command=self.extract_hrv_features_main)
            self.extract_features_main_btn.pack(fill=tk.X, padx=5, pady=5)

            # Save buttons frame
            save_frame = ttk.LabelFrame(button_frame, text="Save Features")
            save_frame.pack(fill=tk.X, padx=5, pady=5)

            self.save_radar_csv_btn = ttk.Button(save_frame, text="Save Radar Features", 
                                                command=self.save_radar_features_csv)
            self.save_radar_csv_btn.pack(fill=tk.X, padx=5, pady=2)
            self.save_radar_csv_btn.config(state='disabled')

            self.save_ecg_csv_btn = ttk.Button(save_frame, text="Save ECG Features", 
                                            command=self.save_ecg_features_csv)
            self.save_ecg_csv_btn.pack(fill=tk.X, padx=5, pady=2)
            self.save_ecg_csv_btn.config(state='disabled')

            self.save_both_csv_btn = ttk.Button(save_frame, text="Save Both (Radar + ECG)", 
                                            command=self.save_both_features_csv)
            self.save_both_csv_btn.pack(fill=tk.X, padx=5, pady=2)
            self.save_both_csv_btn.config(state='disabled')

            self.print_features_console_btn = ttk.Button(button_frame, text="Print to Console", 
                                                        command=self.print_hrv_features_console)
            self.print_features_console_btn.pack(fill=tk.X, padx=5, pady=5)
            self.print_features_console_btn.config(state='disabled')
            
            # Results display
            results_frame = ttk.LabelFrame(left_panel, text="Feature Results")
            results_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
            
            self.features_results_text = tk.Text(results_frame, width=50, height=20)
            features_scrollbar = ttk.Scrollbar(results_frame, orient=tk.VERTICAL, 
                                            command=self.features_results_text.yview)
            self.features_results_text.configure(yscrollcommand=features_scrollbar.set)
            
            self.features_results_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
            features_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            
            # Right panel for plots
            right_panel = ttk.Frame(main_frame)
            right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            # Create figure for feature visualization
            self.fig_features = Figure(figsize=(9, 10))
            gs_features = self.fig_features.add_gridspec(3, 2, height_ratios=[1, 1, 1], 
                                                        width_ratios=[1, 1], hspace=0.3, wspace=0.9)
            
            self.ax_rr_features = self.fig_features.add_subplot(gs_features[0, :])  # RR intervals plot
            self.ax_psd_features = self.fig_features.add_subplot(gs_features[1, 0])  # PSD plot
            self.ax_poincare_features = self.fig_features.add_subplot(gs_features[1, 1])  # Poincaré plot
            self.ax_bispectrum_features = self.fig_features.add_subplot(gs_features[2, :])  # Bispectrum plot
            
            # Initialize plots
            for ax in [self.ax_rr_features, self.ax_psd_features, self.ax_poincare_features, self.ax_bispectrum_features]:
                ax.text(0.5, 0.5, 'Extract HRV features to see visualization', 
                    ha='center', va='center', transform=ax.transAxes)
            
            self.canvas_features = FigureCanvasTkAgg(self.fig_features, master=right_panel)
            
            # Add toolbar
            features_toolbar_frame = ttk.Frame(right_panel)
            features_toolbar_frame.pack(side=tk.TOP, fill=tk.X)
            
            features_toolbar = NavigationToolbar2Tk(self.canvas_features, features_toolbar_frame)
            features_toolbar.update()
            
            self.canvas_features.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # CORRECTED: Use the existing create_scrollable_tab function
        self.create_scrollable_tab(parent, create_features_content)


    def update_mav_info_display(self):
        """Update MAV information display in HRV Features tab"""
        try:
            if not hasattr(self, 'mav_info_label'):
                return
            
            # Check if MAV data is available
            mav_available = False
            mav_info_text = "MAV data not available"
            
            if hasattr(self, 'hrv_results_mav') and self.hrv_results_mav:
                radar_count = len(self.hrv_results_mav.get('radar_rr', []))
                ecg_count = len(self.hrv_results_mav.get('ecg_rr', []))
                mav_window = self.hrv_results_mav.get('mav_window', 'Unknown')
                
                if radar_count > 0 or ecg_count > 0:
                    mav_available = True
                    mav_info_text = f"MAV available: Radar({radar_count}), ECG({ecg_count}), Window: {mav_window}"
            
            # Update label text and color
            self.mav_info_label.config(text=mav_info_text)
            
            if mav_available:
                self.mav_info_label.config(foreground="darkgreen")
                # Enable MAV radio buttons
                if hasattr(self, 'mav_radar_radio'):
                    self.mav_radar_radio.config(state='normal')
                if hasattr(self, 'mav_ecg_radio'):
                    self.mav_ecg_radio.config(state='normal')
            else:
                self.mav_info_label.config(foreground="gray")
                # Disable MAV radio buttons
                if hasattr(self, 'mav_radar_radio'):
                    self.mav_radar_radio.config(state='disabled')
                if hasattr(self, 'mav_ecg_radio'):
                    self.mav_ecg_radio.config(state='disabled')
                    
        except Exception as e:
            print(f"Error updating MAV info display: {e}")


    def load_rr_for_features(self):
        """Load RR intervals from file for feature extraction"""
        file_path = filedialog.askopenfilename(
            title="Load RR Intervals for Feature Extraction",
            filetypes=[("NumPy files", "*.npy"), ("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                if file_path.endswith('.csv'):
                    df = pd.read_csv(file_path)
                    # Try to find RR column
                    rr_columns = [col for col in df.columns if 'rr' in col.lower() or 'interval' in col.lower()]
                    if rr_columns:
                        self.hrv_rr_intervals = df[rr_columns[0]].dropna().values
                    else:
                        self.hrv_rr_intervals = df.iloc[:, 0].dropna().values
                else:
                    self.hrv_rr_intervals = np.load(file_path)
                
                self.rr_file_label.config(text=f"Loaded: {os.path.basename(file_path)} ({len(self.hrv_rr_intervals)} intervals)")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load RR intervals: {str(e)}")
                self.rr_file_label.config(text="Load failed")
                
    def extract_hrv_features_main(self):
        """Extract HRV features based on selected source (enhanced with MAV)"""
        try:
            # Update MAV info before extraction
            self.update_mav_info_display()
            
            # Check if comparison mode is enabled
            comparison_mode = self.comparison_mode_var.get()
            
            if comparison_mode:
                # Extract both radar and ECG features (use existing function)
                self.extract_comparison_hrv_features_with_rr_rmse()
            else:
                # Extract single source features (enhanced version)
                self.extract_single_source_features()
                
        except Exception as e:
            messagebox.showerror("Error", f"Feature extraction failed: {str(e)}")
            print(f"Feature extraction error: {e}")
            import traceback
            traceback.print_exc()

    def extract_single_source_features(self):
        """Extract features from single source including MAV options"""
        source = self.rr_source_var.get()
        
        if source == "paper_method":
            if not hasattr(self, 'paper_analysis_results') or self.paper_analysis_results is None:
                messagebox.showwarning("Warning", "No paper method results available. Run paper algorithm first.")
                return
            
            # Calculate RR from paper method peaks
            radar_peaks = self.paper_analysis_results['final_peaks']
            radar_times = self.signals['timestamps']
            rr_intervals, _ = self.calculate_rr_intervals(radar_peaks, radar_times)
            source_name = "Radar (Paper Method)"
            
        elif source == "edited_peaks":
            # Use latest edited RR intervals that are stored and tracked
            if not hasattr(self, 'latest_edited_rr_intervals') or self.latest_edited_rr_intervals is None:
                messagebox.showwarning("Warning", "No edited peaks available. Edit peaks first in HRV Analysis tab and click 'Update HRV (Edited)'.")
                return
            
            rr_intervals = self.latest_edited_rr_intervals.copy()
            source_name = f"Radar (Edited Peaks - {len(rr_intervals)} intervals)"
            print(f"DEBUG: Using stored edited RR intervals - Count: {len(rr_intervals)}, Mean: {np.mean(rr_intervals):.1f}ms")
            
        # NEW: MAV Radar option
        elif source == "mav_radar":
            if not hasattr(self, 'hrv_results_mav') or self.hrv_results_mav is None or 'radar_rr' not in self.hrv_results_mav:
                messagebox.showwarning("Warning", "No MAV radar data available. Run HRV Analysis with MAV enabled first.")
                return
            
            rr_intervals = self.hrv_results_mav['radar_rr'].copy()
            mav_window = self.hrv_results_mav.get('mav_window', 'Unknown')
            source_name = f"Radar MAV (Window: {mav_window})"
            print(f"DEBUG: Using MAV radar RR intervals - Count: {len(rr_intervals)}, Mean: {np.mean(rr_intervals):.1f}ms, Window: {mav_window}")
            
        # NEW: MAV ECG option
        elif source == "mav_ecg":
            if not hasattr(self, 'hrv_results_mav') or self.hrv_results_mav is None or 'ecg_rr' not in self.hrv_results_mav:
                messagebox.showwarning("Warning", "No MAV ECG data available. Run HRV Analysis with MAV enabled first.")
                return
            
            rr_intervals = self.hrv_results_mav['ecg_rr'].copy()
            mav_window = self.hrv_results_mav.get('mav_window', 'Unknown')
            source_name = f"ECG MAV (Window: {mav_window})"
            print(f"DEBUG: Using MAV ECG RR intervals - Count: {len(rr_intervals)}, Mean: {np.mean(rr_intervals):.1f}ms, Window: {mav_window}")
            
        elif source == "ecg_peaks":
            if not hasattr(self, 'ecg_loaded') or not self.ecg_loaded:
                messagebox.showwarning("Warning", "No ECG data loaded. Load ECG data first in HRV Analysis tab.")
                return
            
            # Calculate RR from ECG peaks
            ecg_peaks = self.detect_r_peaks(self.ecg_data, self.ecg_fs)
            rr_intervals, _ = self.calculate_rr_intervals(ecg_peaks, self.ecg_time)
            source_name = "ECG R-Peaks"
            
        elif source == "load_file":
            if not hasattr(self, 'hrv_rr_intervals') or self.hrv_rr_intervals is None:
                messagebox.showwarning("Warning", "No RR intervals loaded. Load file first.")
                return
            
            rr_intervals = self.hrv_rr_intervals
            source_name = "Loaded File"
        
        else:
            messagebox.showerror("Error", "Invalid source selection")
            return
        
        # For edited_peaks and MAV sources, skip additional filtering since they're already processed
        if source in ["edited_peaks", "mav_radar", "mav_ecg"]:
            rr_filtered = rr_intervals  # Already filtered/processed
        else:
            # Filter RR intervals for other sources
            min_rr = self.features_min_rr_var.get()
            max_rr = self.features_max_rr_var.get()
            min_count = self.min_rr_count_var.get()
            
            valid_mask = (rr_intervals >= min_rr) & (rr_intervals <= max_rr)
            rr_filtered = rr_intervals[valid_mask]
            
            if len(rr_filtered) < min_count:
                messagebox.showwarning("Warning", 
                    f"Insufficient RR intervals after filtering: {len(rr_filtered)} < {min_count} required")
                return
        
        # Store filtered RR intervals
        self.hrv_rr_intervals = rr_filtered
        
        # Extract features
        self.extract_all_hrv_features(rr_filtered)
        
        # Store source info
        self.current_feature_source = source_name
        self.comparison_results = None  # Clear comparison results
        
        # Create visualizations
        self.create_hrv_feature_plots(rr_filtered)
        
        # Display results
        self.display_hrv_feature_results()
        
        # Enable correct save buttons
        if hasattr(self, 'save_radar_csv_btn'):
            self.save_radar_csv_btn.config(state='normal')
        if hasattr(self, 'save_ecg_csv_btn'):
            self.save_ecg_csv_btn.config(state='disabled')
        if hasattr(self, 'save_both_csv_btn'):
            self.save_both_csv_btn.config(state='disabled')
        if hasattr(self, 'print_features_console_btn'):
            self.print_features_console_btn.config(state='normal')
        
        self.features_results_text.insert(tk.END, f"\n✅ Feature extraction completed for {source_name}!")

    def extract_comparison_hrv_features_with_mav(self):
        """Enhanced comparison feature extraction with MAV options"""
        # Get radar RR intervals
        radar_source = self.rr_source_var.get()
        
        if radar_source == "paper_method":
            if not hasattr(self, 'paper_analysis_results') or self.paper_analysis_results is None:
                messagebox.showwarning("Warning", "No paper method results available. Run paper algorithm first.")
                return
            radar_peaks = self.paper_analysis_results['final_peaks']
            radar_times = self.signals['timestamps']
            radar_rr, _ = self.calculate_rr_intervals(radar_peaks, radar_times)
            radar_name = "Radar (Paper Method)"
            
        elif radar_source == "edited_peaks":
            if not hasattr(self, 'latest_edited_rr_intervals') or self.latest_edited_rr_intervals is None:
                messagebox.showwarning("Warning", "No edited peaks available. Edit peaks first in HRV Analysis tab and click 'Update HRV (Edited)'.")
                return
            
            radar_rr = self.latest_edited_rr_intervals.copy()
            if hasattr(self, 'edited_peaks_timestamp'):
                timestamp_str = time.strftime('%H:%M:%S', time.localtime(self.edited_peaks_timestamp))
                radar_name = f"Radar (Edited Peaks - Updated: {timestamp_str})"
            else:
                radar_name = "Radar (Edited Peaks)"
                
            print(f"DEBUG: Using edited RR for comparison - Count: {len(radar_rr)}, Mean: {np.mean(radar_rr):.1f}ms")
            
        # NEW: MAV Radar option for comparison
        elif radar_source == "mav_radar":
            if not hasattr(self, 'hrv_results_mav') or self.hrv_results_mav is None or 'radar_rr' not in self.hrv_results_mav:
                messagebox.showwarning("Warning", "No MAV radar data available. Run HRV Analysis with MAV enabled first.")
                return
            
            radar_rr = self.hrv_results_mav['radar_rr'].copy()
            mav_window = self.hrv_results_mav.get('mav_window', 'Unknown')
            radar_name = f"Radar MAV (Window: {mav_window})"
            print(f"DEBUG: Using MAV radar RR for comparison - Count: {len(radar_rr)}, Mean: {np.mean(radar_rr):.1f}ms")
            
        # NEW: MAV ECG option for comparison (special case - compare MAV ECG with raw ECG)
        elif radar_source == "mav_ecg":
            if not hasattr(self, 'hrv_results_mav') or self.hrv_results_mav is None or 'ecg_rr' not in self.hrv_results_mav:
                messagebox.showwarning("Warning", "No MAV ECG data available. Run HRV Analysis with MAV enabled first.")
                return
            
            radar_rr = self.hrv_results_mav['ecg_rr'].copy()  # Using ECG MAV as "radar" for comparison
            mav_window = self.hrv_results_mav.get('mav_window', 'Unknown')
            radar_name = f"ECG MAV (Window: {mav_window})"
            print(f"DEBUG: Using MAV ECG RR as primary for comparison - Count: {len(radar_rr)}, Mean: {np.mean(radar_rr):.1f}ms")
            
        else:
            messagebox.showwarning("Warning", "For comparison mode, please select a valid radar source (Paper Method, Edited Peaks, or MAV)")
            return
        
        # Get ECG RR intervals (for comparison)
        if not hasattr(self, 'ecg_loaded') or not self.ecg_loaded:
            messagebox.showwarning("Warning", "No ECG data loaded. Load ECG data first in HRV Analysis tab.")
            return
        
        # For MAV ECG comparison, use raw ECG as reference
        if radar_source == "mav_ecg":
            # Compare MAV ECG vs Raw ECG
            if hasattr(self, 'hrv_results_raw') and 'ecg_rr' in self.hrv_results_raw:
                ecg_rr = self.hrv_results_raw['ecg_rr'].copy()
                ecg_name = "ECG Raw (Reference)"
            else:
                ecg_peaks = self.detect_r_peaks(self.ecg_data, self.ecg_fs)
                ecg_rr, _ = self.calculate_rr_intervals(ecg_peaks, self.ecg_time)
                ecg_name = "ECG R-Peaks"
        else:
            # Normal comparison with raw ECG
            ecg_peaks = self.detect_r_peaks(self.ecg_data, self.ecg_fs)
            ecg_rr, _ = self.calculate_rr_intervals(ecg_peaks, self.ecg_time)
            ecg_name = "ECG R-Peaks"
        
        # Filter both RR intervals with same criteria (unless using MAV data)
        min_rr = self.features_min_rr_var.get()
        max_rr = self.features_max_rr_var.get()
        min_count = self.min_rr_count_var.get()
        
        # For MAV data, skip filtering as it's already processed
        if radar_source in ["edited_peaks", "mav_radar", "mav_ecg"]:
            radar_rr_filtered = radar_rr  # Already filtered/processed
        else:
            radar_valid_mask = (radar_rr >= min_rr) & (radar_rr <= max_rr)
            radar_rr_filtered = radar_rr[radar_valid_mask]
        
        # Always filter ECG data unless it's from stored results
        if radar_source == "mav_ecg" and hasattr(self, 'hrv_results_raw'):
            ecg_rr_filtered = ecg_rr  # Already processed
        else:
            ecg_valid_mask = (ecg_rr >= min_rr) & (ecg_rr <= max_rr)
            ecg_rr_filtered = ecg_rr[ecg_valid_mask]
        
        # Check minimum count
        if len(radar_rr_filtered) < min_count or len(ecg_rr_filtered) < min_count:
            messagebox.showwarning("Warning", f"Insufficient RR intervals after filtering")
            return
        
        # Calculate RR RMSE for comparison
        rr_rmse_comparison = self.calculate_rr_rmse(ecg_rr_filtered, radar_rr_filtered)
        
        # Extract HRV features
        self.extract_all_hrv_features(radar_rr_filtered)
        radar_features = self.hrv_features_extracted.copy()
        
        self.extract_all_hrv_features(ecg_rr_filtered)
        ecg_features = self.hrv_features_extracted.copy()
        
        # Store comparison results with RR RMSE
        self.comparison_results = {
            'radar_features': radar_features,
            'ecg_features': ecg_features,
            'radar_rr': radar_rr_filtered,
            'ecg_rr': ecg_rr_filtered,
            'radar_name': radar_name,
            'ecg_name': ecg_name,
            'rr_rmse': rr_rmse_comparison
        }
        
        # For plotting, use radar as primary
        self.hrv_features_extracted = radar_features
        self.hrv_rr_intervals = radar_rr_filtered
        self.current_feature_source = f"{radar_name} vs {ecg_name} Comparison"
        
        # Create comparison visualizations
        self.create_comparison_feature_plots()
        
        # Display comparison results with RR RMSE
        self.display_comparison_feature_results_with_rr_rmse()
        
        # Enable save buttons
        if hasattr(self, 'save_radar_csv_btn'):
            self.save_radar_csv_btn.config(state='normal')
        if hasattr(self, 'save_ecg_csv_btn'):
            self.save_ecg_csv_btn.config(state='normal')
        if hasattr(self, 'save_both_csv_btn'):
            self.save_both_csv_btn.config(state='normal')
        if hasattr(self, 'print_features_console_btn'):
            self.print_features_console_btn.config(state='normal')
        
        # Show RMSE in success message
        rmse_msg = ""
        if rr_rmse_comparison and not rr_rmse_comparison['error']:
            rmse_msg = f" | RR RMSE: {rr_rmse_comparison['rmse']:.1f}ms"
        
        self.features_results_text.insert(tk.END, f"\n✅ Comparison feature extraction completed!{rmse_msg}")


        
    def sample_entropy(self, signal, m=2, r_factor=0.2):
        """Compute sample entropy of signal - copied from Preprocessing.py"""
        import math
        r = r_factor * np.std(signal)
        n = len(signal)
        def _phi(m):
            x = np.array([signal[i:i+m] for i in range(n - m + 1)])
            C = np.sum(np.max(np.abs(x[:, None] - x[None, :]), axis=2) <= r, axis=0) - 1
            return np.sum(C) / (n - m + 1)
        try:
            return -np.log(_phi(m+1) / _phi(m))
        except:
            return 0

    def ar_psd_burg(self, rr_intervals, fs=4):
        """AR-based PSD computation - adapted from Preprocessing.py"""
        try:
            # Note: Install spectrum package if not available: pip install spectrum
            from spectrum import arburg
            
            # Resample
            rr_interp = np.interp(np.linspace(0, len(rr_intervals), len(rr_intervals)*fs), 
                                np.arange(len(rr_intervals)), rr_intervals)
            rr_interp = rr_interp - np.mean(rr_interp)

            # AR modeling
            order = 16
            ar_coeffs, e, _ = arburg(rr_interp, order)
            ar_coeffs = np.array(ar_coeffs)

            nfft = 512
            freqs = np.linspace(0, fs/2, nfft)
            psd = np.zeros_like(freqs)
            for i, f in enumerate(freqs):
                omega = np.exp(-2j * np.pi * f / fs)
                psd[i] = e / np.abs(np.polyval(ar_coeffs[::-1], omega))**2
            psd = np.real(psd)
            psd = psd / 100.0

            # Power features
            lf_band = (freqs >= 0.04) & (freqs < 0.15)
            hf_band = (freqs >= 0.15) & (freqs <= 0.4)
            lf_power = np.trapz(psd[lf_band], freqs[lf_band]) if np.any(lf_band) else 0
            hf_power = np.trapz(psd[hf_band], freqs[hf_band]) if np.any(hf_band) else 0

            peak_lf = freqs[lf_band][np.argmax(psd[lf_band])] if np.any(lf_band) else 0
            peak_hf = freqs[hf_band][np.argmax(psd[hf_band])] if np.any(hf_band) else 0

            return freqs, psd, lf_power, hf_power, peak_lf, peak_hf
        
        except ImportError:
            print("Warning: spectrum package not available. Using scipy.signal.welch instead.")
            from scipy.signal import welch
            
            # Fallback to Welch method
            freqs, psd = welch(rr_intervals, fs=4, nperseg=min(256, len(rr_intervals)//4))
            
            lf_band = (freqs >= 0.04) & (freqs < 0.15)
            hf_band = (freqs >= 0.15) & (freqs <= 0.4)
            lf_power = np.trapz(psd[lf_band], freqs[lf_band]) if np.any(lf_band) else 0
            hf_power = np.trapz(psd[hf_band], freqs[hf_band]) if np.any(hf_band) else 0
            
            peak_lf = freqs[lf_band][np.argmax(psd[lf_band])] if np.any(lf_band) else 0
            peak_hf = freqs[hf_band][np.argmax(psd[hf_band])] if np.any(hf_band) else 0
            
            return freqs, psd, lf_power, hf_power, peak_lf, peak_hf

    def compute_bispectrum(self, rr_intervals):
        """Compute bispectrum - copied from Preprocessing.py"""
        x = rr_intervals - np.mean(rr_intervals)
        N = len(x)
        max_lag = min(50, N // 2)

        C = np.zeros((max_lag, max_lag))
        for m in range(max_lag):
            for n in range(max_lag):
                if m + n < N:
                    C[m, n] = np.mean(x[:N - m - n] * x[m:N - n] * x[m + n:])

        from scipy.fft import fft2
        B = np.abs(fft2(C))
        f1 = np.linspace(0, 1, B.shape[0])
        f2 = np.linspace(0, 1, B.shape[1])

        self.last_bispectrum = B
        self.last_f1 = f1
        self.last_f2 = f2

    def bispectrum_features(self, rr_intervals):
        """Extract bispectral features - copied from Preprocessing.py"""
        try:
            self.compute_bispectrum(rr_intervals)

            B = self.last_bispectrum
            F1, F2 = np.meshgrid(self.last_f1, self.last_f2)

            omega_mask = (F1 > F2) & (F1 + F2 < 1.0)
            B_omega = B.copy()
            B_omega[~omega_mask] = 0

            omega_values = B_omega[omega_mask]
            omega_values = omega_values[omega_values > 1e-12]

            if len(omega_values) == 0:
                return {'P1': 0, 'P2': 0, 'H1': 0, 'H2': 0, 'H3': 0, 'H4': 0}

            # Entropy features
            p_n = omega_values / np.sum(omega_values)
            P1 = -np.sum(p_n * np.log(p_n + 1e-12))

            omega_squared = omega_values ** 2
            q_n = omega_squared / np.sum(omega_squared)
            P2 = -np.sum(q_n * np.log(q_n + 1e-12))

            # H1: log amplitude sum in Ω
            H1 = np.sum(np.log(omega_values + 1e-12))

            # H2–H4: only diagonal values in Ω
            f_diag = self.last_f1
            valid_diag_idx = np.where(f_diag < 0.5)[0]

            diagonal_values = B[valid_diag_idx, valid_diag_idx]
            diag_valid = diagonal_values[diagonal_values > 1e-12]

            if len(diag_valid) == 0:
                return {'P1': P1, 'P2': P2, 'H1': H1, 'H2': 0, 'H3': 0, 'H4': 0}

            log_diag = np.log(diag_valid)
            H2 = np.sum(log_diag)

            N_diag = len(diag_valid)
            k = np.arange(1, N_diag + 1)
            H3 = np.sum(k * log_diag) / N_diag
            H4 = np.sum((k - H3) ** 2 * log_diag) / N_diag

            return {'P1': P1, 'P2': P2, 'H1': H1, 'H2': H2, 'H3': H3, 'H4': H4}

        except Exception as e:
            print(f"Error in bispectrum_features: {e}")
            return {'P1': 0, 'P2': 0, 'H1': 0, 'H2': 0, 'H3': 0, 'H4': 0}

    def calculate_poincare_features(self, rr_intervals):
        """Calculate Poincaré plot features - ensure this exists"""
        if len(rr_intervals) < 2:
            return {'SD1': 0, 'SD2': 0, 'SD1_SD2_ratio': 0}
        
        # Create Poincaré plot data
        rr_n = rr_intervals[:-1]
        rr_n1 = rr_intervals[1:]
        
        # Calculate distances from lines y=x and y=-x+2*RR_mean
        # SD1: standard deviation perpendicular to line y=x
        diff = rr_n1 - rr_n
        SD1 = np.std(diff) / np.sqrt(2)
        
        # SD2: standard deviation along line y=x  
        sum_vals = rr_n1 + rr_n
        SD2 = np.std(sum_vals) / np.sqrt(2)
        
        # SD1/SD2 ratio
        SD1_SD2_ratio = SD1 / SD2 if SD2 != 0 else 0
        
        return {'SD1': SD1, 'SD2': SD2, 'SD1_SD2_ratio': SD1_SD2_ratio}

    def extract_all_hrv_features(self, rr_intervals):
        """Extract all 12 HRV features"""
        try:
            # 1. Spectral features (2)
            freqs, psd, lf_power, hf_power, peak_lf, peak_hf = self.ar_psd_burg(rr_intervals)
            
            # 2. Bispectral features (6)
            bispectral_features = self.bispectrum_features(rr_intervals)
            
            # 3. Non-linear features (4)
            poincare_features = self.calculate_poincare_features(rr_intervals)
            sample_entropy = self.sample_entropy(rr_intervals)
            
            # Combine all features
            self.hrv_features_extracted = {
                # Spectral (2)
                'LF_power': lf_power,
                'HF_power': hf_power,
                'peak_LF': peak_lf,
                'peak_HF': peak_hf,
                
                # Bispectral (6)
                'P1': bispectral_features['P1'],
                'P2': bispectral_features['P2'],
                'H1': bispectral_features['H1'],
                'H2': bispectral_features['H2'],
                'H3': bispectral_features['H3'],
                'H4': bispectral_features['H4'],
                
                # Non-linear (4)
                'SD1': poincare_features['SD1'],
                'SD2': poincare_features['SD2'],
                'SD1_SD2_ratio': poincare_features['SD1_SD2_ratio'],
                'SamEn': sample_entropy,
                
                # For plotting
                'frequencies': freqs,
                'psd': psd
            }
            
        except Exception as e:
            print(f"Error extracting features: {e}")
            self.hrv_features_extracted = None
    
    def create_hrv_feature_plots(self, rr_intervals):
        """Create visualization plots for HRV features"""
        try:
            # Clear all plots
            for ax in [self.ax_rr_features, self.ax_psd_features, self.ax_poincare_features, self.ax_bispectrum_features]:
                ax.clear()
            
            # 1. RR intervals time series
            rr_times = np.cumsum(rr_intervals) / 1000  # Convert to seconds
            self.ax_rr_features.plot(rr_times, rr_intervals, 'b-', linewidth=1, alpha=0.8)
            self.ax_rr_features.set_title('RR Intervals Time Series')
            self.ax_rr_features.set_xlabel('Time (s)')
            self.ax_rr_features.set_ylabel('RR Interval (ms)')
            self.ax_rr_features.grid(True, alpha=0.3)
            
            # 2. Power Spectral Density
            if 'frequencies' in self.hrv_features_extracted:
                freqs = self.hrv_features_extracted['frequencies']
                psd = self.hrv_features_extracted['psd']
                
                self.ax_psd_features.plot(freqs, psd, 'b-', linewidth=1.5)
                
                # Highlight LF and HF bands
                lf_mask = (freqs >= 0.04) & (freqs <= 0.15)
                hf_mask = (freqs >= 0.15) & (freqs <= 0.4)
                
                if np.any(lf_mask):
                    self.ax_psd_features.fill_between(freqs[lf_mask], psd[lf_mask], 
                                                    alpha=0.5, color='red', label='LF')
                if np.any(hf_mask):
                    self.ax_psd_features.fill_between(freqs[hf_mask], psd[hf_mask], 
                                                    alpha=0.5, color='green', label='HF')
                
                self.ax_psd_features.set_xlim(0, 0.5)
                self.ax_psd_features.legend()
            
            self.ax_psd_features.set_title('Power Spectral Density')
            self.ax_psd_features.set_xlabel('Frequency (Hz)')
            self.ax_psd_features.set_ylabel('PSD')
            self.ax_psd_features.grid(True, alpha=0.3)
            
            # 3. Poincaré plot
            if len(rr_intervals) > 1:
                rr_n = rr_intervals[:-1]
                rr_n1 = rr_intervals[1:]
                
                self.ax_poincare_features.scatter(rr_n, rr_n1, alpha=0.6, s=20, color='blue')
                
                # Add identity line
                min_rr = min(min(rr_n), min(rr_n1))
                max_rr = max(max(rr_n), max(rr_n1))
                self.ax_poincare_features.plot([min_rr, max_rr], [min_rr, max_rr], 'k--', alpha=0.5)
                
                self.ax_poincare_features.set_xlabel('RR(n) [ms]')
                self.ax_poincare_features.set_ylabel('RR(n+1) [ms]')
                self.ax_poincare_features.axis('equal')
            
            self.ax_poincare_features.set_title('Poincaré Plot')
            self.ax_poincare_features.grid(True, alpha=0.3)
            
            # 4. Bispectrum contour
            if hasattr(self, 'last_bispectrum'):
                B = self.last_bispectrum
                F1, F2 = np.meshgrid(self.last_f1, self.last_f2)
                
                if B.size > 0:
                    contour = self.ax_bispectrum_features.contour(F1, F2, B, levels=20, cmap='jet')
                    self.ax_bispectrum_features.set_xlabel('f1')
                    self.ax_bispectrum_features.set_ylabel('f2')
            
            self.ax_bispectrum_features.set_title('Bispectrum |B(f1,f2)|')
            self.ax_bispectrum_features.grid(True, alpha=0.3)
            
            self.fig_features.tight_layout()
            self.canvas_features.draw()
            
        except Exception as e:
            print(f"Error creating plots: {e}")
    
    def create_comparison_feature_plots(self):
        """Create comparison visualization plots for radar vs ECG HRV features"""
        try:
            if not hasattr(self, 'comparison_results') or self.comparison_results is None:
                return
            
            results = self.comparison_results
            radar_rr = results['radar_rr']
            ecg_rr = results['ecg_rr']
            radar_features = results['radar_features']
            ecg_features = results['ecg_features']
            
            # Clear all plots
            for ax in [self.ax_rr_features, self.ax_psd_features, self.ax_poincare_features, self.ax_bispectrum_features]:
                ax.clear()
            
            # 1. RR intervals comparison
            radar_times = np.cumsum(radar_rr) / 1000
            ecg_times = np.cumsum(ecg_rr) / 1000
            
            self.ax_rr_features.plot(radar_times, radar_rr, 'b-', linewidth=1, alpha=0.8, 
                                    label=f'{results["radar_name"]} ({len(radar_rr)} intervals)')
            self.ax_rr_features.plot(ecg_times, ecg_rr, 'r-', linewidth=1, alpha=0.8, 
                                    label=f'{results["ecg_name"]} ({len(ecg_rr)} intervals)')
            self.ax_rr_features.set_title('RR Intervals Comparison')
            self.ax_rr_features.set_xlabel('Time (s)')
            self.ax_rr_features.set_ylabel('RR Interval (ms)')
            self.ax_rr_features.legend()
            self.ax_rr_features.grid(True, alpha=0.3)
            
            # 2. PSD comparison
            if 'frequencies' in radar_features and 'frequencies' in ecg_features:
                radar_freqs = radar_features['frequencies']
                radar_psd = radar_features['psd']
                ecg_freqs = ecg_features['frequencies']
                ecg_psd = ecg_features['psd']
                
                self.ax_psd_features.plot(radar_freqs, radar_psd, 'b-', linewidth=1.5, alpha=0.8, 
                                        label=f'{results["radar_name"]}')
                self.ax_psd_features.plot(ecg_freqs, ecg_psd, 'r-', linewidth=1.5, alpha=0.8, 
                                        label=f'{results["ecg_name"]}')
                
                # Highlight frequency bands for both
                lf_mask_r = (radar_freqs >= 0.04) & (radar_freqs <= 0.15)
                hf_mask_r = (radar_freqs >= 0.15) & (radar_freqs <= 0.4)
                
                if np.any(lf_mask_r):
                    self.ax_psd_features.fill_between(radar_freqs[lf_mask_r], radar_psd[lf_mask_r], 
                                                    alpha=0.3, color='blue', label='LF Band')
                if np.any(hf_mask_r):
                    self.ax_psd_features.fill_between(radar_freqs[hf_mask_r], radar_psd[hf_mask_r], 
                                                    alpha=0.3, color='green', label='HF Band')
                
                self.ax_psd_features.set_xlim(0, 0.5)
                self.ax_psd_features.legend()
            
            self.ax_psd_features.set_title('Power Spectral Density Comparison')
            self.ax_psd_features.set_xlabel('Frequency (Hz)')
            self.ax_psd_features.set_ylabel('PSD')
            self.ax_psd_features.grid(True, alpha=0.3)
            
            # 3. Poincaré plots comparison
            if len(radar_rr) > 1 and len(ecg_rr) > 1:
                # Radar Poincaré
                radar_rr_n = radar_rr[:-1]
                radar_rr_n1 = radar_rr[1:]
                
                # ECG Poincaré
                ecg_rr_n = ecg_rr[:-1]
                ecg_rr_n1 = ecg_rr[1:]
                
                self.ax_poincare_features.scatter(radar_rr_n, radar_rr_n1, alpha=0.6, s=20, 
                                                color='blue', label=f'{results["radar_name"]}')
                self.ax_poincare_features.scatter(ecg_rr_n, ecg_rr_n1, alpha=0.6, s=20, 
                                                color='red', label=f'{results["ecg_name"]}')
                
                # Add identity line
                all_rr = np.concatenate([radar_rr_n, radar_rr_n1, ecg_rr_n, ecg_rr_n1])
                min_rr = np.min(all_rr)
                max_rr = np.max(all_rr)
                self.ax_poincare_features.plot([min_rr, max_rr], [min_rr, max_rr], 'k--', alpha=0.5)
                
                self.ax_poincare_features.set_xlabel('RR(n) [ms]')
                self.ax_poincare_features.set_ylabel('RR(n+1) [ms]')
                self.ax_poincare_features.legend()
                self.ax_poincare_features.axis('equal')
            
            self.ax_poincare_features.set_title('Poincaré Plot Comparison')
            self.ax_poincare_features.grid(True, alpha=0.3)
            
            # 4. Feature comparison bar chart
            feature_names = ['LF_power', 'HF_power', 'P1', 'P2', 'H1', 'H2', 'H3', 'H4', 
                            'SD1', 'SD2', 'SD1_SD2_ratio', 'SamEn']
            
            radar_values = [radar_features[name] for name in feature_names]
            ecg_values = [ecg_features[name] for name in feature_names]
            
            x = np.arange(len(feature_names))
            width = 0.35
            
            self.ax_bispectrum_features.bar(x - width/2, radar_values, width, 
                                        label=f'{results["radar_name"]}', alpha=0.8, color='blue')
            self.ax_bispectrum_features.bar(x + width/2, ecg_values, width, 
                                        label=f'{results["ecg_name"]}', alpha=0.8, color='red')
            
            self.ax_bispectrum_features.set_xlabel('Features')
            self.ax_bispectrum_features.set_ylabel('Feature Values')
            self.ax_bispectrum_features.set_title('12 HRV Features Comparison')
            self.ax_bispectrum_features.set_xticks(x)
            self.ax_bispectrum_features.set_xticklabels(feature_names, rotation=45, ha='right')
            self.ax_bispectrum_features.legend()
            self.ax_bispectrum_features.grid(True, alpha=0.3)
            
            self.fig_features.tight_layout()
            self.canvas_features.draw()
            
        except Exception as e:
            print(f"Error creating comparison plots: {e}")
    
    def display_comparison_feature_results_with_rr_rmse(self):
        """Enhanced comparison display with RR RMSE"""
        if not hasattr(self, 'comparison_results'):
            return
        
        self.features_results_text.delete(1.0, tk.END)
        
        results = self.comparison_results
        
        text = "HRV FEATURE COMPARISON RESULTS (with RR RMSE)\n"
        text += "=" * 60 + "\n\n"
        
        text += f"RADAR SOURCE: {results['radar_name']}\n"
        text += f"ECG SOURCE: {results['ecg_name']}\n\n"
        
        # NEW: Display RR RMSE first (most important metric)
        if 'rr_rmse' in results and results['rr_rmse'] and not results['rr_rmse']['error']:
            rmse = results['rr_rmse']
            text += "RR INTERVALS ACCURACY:\n"
            text += "-" * 30 + "\n"
            text += f"RMSE: {rmse['rmse']:.2f} ms\n"
            text += f"Correlation: {rmse['correlation']:.4f}\n"
            text += f"Mean Absolute Error: {rmse['mae']:.2f} ms\n"
            text += f"Data points: {rmse['n_points']}\n\n"
        
        # Rest of the feature comparison...
        radar_features = results['radar_features']
        ecg_features = results['ecg_features']
        
        feature_names = ['LF_power', 'HF_power', 'P1', 'P2', 'H1', 'H2', 'H3', 'H4', 
                        'SD1', 'SD2', 'SD1_SD2_ratio', 'SamEn']
        
        radar_vector = [radar_features[name] for name in feature_names]
        ecg_vector = [ecg_features[name] for name in feature_names]
        
        feature_correlation = np.corrcoef(radar_vector, ecg_vector)[0, 1]
        feature_rmse = np.sqrt(np.mean((np.array(radar_vector) - np.array(ecg_vector))**2))
        
        text += "HRV FEATURES COMPARISON:\n"
        text += "-" * 30 + "\n"
        text += f"Feature Vector Correlation: {feature_correlation:.4f}\n"
        text += f"Feature Vector RMSE: {feature_rmse:.6f}\n\n"
        
        # Feature vectors for copying
        text += "RADAR FEATURE VECTOR:\n"
        text += ",".join([f"{v:.10f}" for v in radar_vector]) + "\n\n"
        
        text += "ECG FEATURE VECTOR:\n"
        text += ",".join([f"{v:.10f}" for v in ecg_vector]) + "\n\n"
        
        self.features_results_text.insert(tk.END, text)


    def display_hrv_feature_results(self):
        """Display extracted features in text box"""
        if self.hrv_features_extracted is None:
            return
        
        self.features_results_text.delete(1.0, tk.END)
        
        features = self.hrv_features_extracted
        
        text = "HRV FEATURE EXTRACTION RESULTS\n"
        text += "=" * 50 + "\n\n"
        
        text += f"RR INTERVALS INFO:\n"
        text += f"Total intervals: {len(self.hrv_rr_intervals)}\n"
        text += f"Mean RR: {np.mean(self.hrv_rr_intervals):.1f} ms\n"
        text += f"Std RR: {np.std(self.hrv_rr_intervals):.1f} ms\n\n"
        
        text += "SPECTRAL FEATURES (2):\n"
        text += "-" * 25 + "\n"
        text += f"LF Power: {features['LF_power']:.6f}\n"
        text += f"HF Power: {features['HF_power']:.6f}\n"
        text += f"Peak LF: {features['peak_LF']:.4f} Hz\n"
        text += f"Peak HF: {features['peak_HF']:.4f} Hz\n\n"
        
        text += "BISPECTRAL FEATURES (6):\n"
        text += "-" * 25 + "\n"
        text += f"P1 (Norm. Bisp. Entropy): {features['P1']:.4f}\n"
        text += f"P2 (Norm. Bisp. Sq. Entropy): {features['P2']:.4f}\n"
        text += f"H1 (Sum Log Amplitudes): {features['H1']:.4f}\n"
        text += f"H2 (Sum Log Diag): {features['H2']:.4f}\n"
        text += f"H3 (1st Order Moment): {features['H3']:.4f}\n"
        text += f"H4 (2nd Order Moment): {features['H4']:.4f}\n\n"
        
        text += "NON-LINEAR FEATURES (4):\n"
        text += "-" * 25 + "\n"
        text += f"SD1: {features['SD1']:.2f} ms\n"
        text += f"SD2: {features['SD2']:.2f} ms\n"
        text += f"SD1/SD2 Ratio: {features['SD1_SD2_ratio']:.4f}\n"
        text += f"Sample Entropy: {features['SamEn']:.4f}\n\n"
        
        text += "TOTAL FEATURES: 12\n"
        text += "(As per Mohebbi & Ghassemian, 2012)\n\n"
        
        # Feature vector for easy copying
        text += "FEATURE VECTOR (copy-paste ready):\n"
        text += "-" * 35 + "\n"
        feature_vector = [
            features['LF_power'], features['HF_power'],
            features['P1'], features['P2'], features['H1'], features['H2'], features['H3'], features['H4'],
            features['SD1'], features['SD2'], features['SD1_SD2_ratio'], features['SamEn']
        ]
        text += ",".join([f"{v:.10f}" for v in feature_vector]) + "\n"
        
        self.features_results_text.insert(tk.END, text)

    def print_hrv_features_console(self):
        """Print HRV features to console for easy copying"""
        try:
            # Check if comparison mode
            if hasattr(self, 'comparison_results') and self.comparison_results is not None:
                self.print_comparison_features_console()
            else:
                self.print_single_features_console()
                
        except Exception as e:
            print(f"Error printing features: {e}")

    def print_single_features_console(self):
        """Print single source features to console"""
        if self.hrv_features_extracted is None:
            print("No features extracted.")
            return
        
        features = self.hrv_features_extracted
        source_name = getattr(self, 'current_feature_source', 'Unknown')
        
        # Print individual features
        print("\n" + "="*60)
        print(f"HRV FEATURES EXTRACTED - {source_name}")
        print("="*60)
        print(f"LF_power: {features['LF_power']:.10f}")
        print(f"HF_power: {features['HF_power']:.10f}")
        print(f"P1: {features['P1']:.10f}")
        print(f"P2: {features['P2']:.10f}")
        print(f"H1: {features['H1']:.10f}")
        print(f"H2: {features['H2']:.10f}")
        print(f"H3: {features['H3']:.10f}")
        print(f"H4: {features['H4']:.10f}")
        print(f"SD1: {features['SD1']:.10f}")
        print(f"SD2: {features['SD2']:.10f}")
        print(f"SD1_SD2_ratio: {features['SD1_SD2_ratio']:.10f}")
        print(f"SamEn: {features['SamEn']:.10f}")
        
        # Print as CSV row
        feature_vector = [
            features['LF_power'], features['HF_power'],
            features['P1'], features['P2'], features['H1'], features['H2'], features['H3'], features['H4'],
            features['SD1'], features['SD2'], features['SD1_SD2_ratio'], features['SamEn']
        ]
        
        print("\n" + "="*60)
        print("CSV ROW (copy this line):")
        print("="*60)
        print(",".join([f"{v:.10f}" for v in feature_vector]))
        print("="*60 + "\n")
        
        # Update GUI
        self.features_results_text.insert(tk.END, f"\n📋 Features printed to console - check terminal!")

    def print_comparison_features_console(self):
        """Print comparison features to console"""
        results = self.comparison_results
        radar_features = results['radar_features']
        ecg_features = results['ecg_features']
        
        print("\n" + "="*80)
        print("COMPARISON HRV FEATURES EXTRACTED")
        print("="*80)
        
        # Print radar features
        print(f"\nRADAR FEATURES - {results['radar_name']}:")
        print("-" * 50)
        radar_vector = []
        for name in ['LF_power', 'HF_power', 'P1', 'P2', 'H1', 'H2', 'H3', 'H4', 'SD1', 'SD2', 'SD1_SD2_ratio', 'SamEn']:
            print(f"{name}: {radar_features[name]:.10f}")
            radar_vector.append(radar_features[name])
        
        # Print ECG features  
        print(f"\nECG FEATURES - {results['ecg_name']}:")
        print("-" * 50)
        ecg_vector = []
        for name in ['LF_power', 'HF_power', 'P1', 'P2', 'H1', 'H2', 'H3', 'H4', 'SD1', 'SD2', 'SD1_SD2_ratio', 'SamEn']:
            print(f"{name}: {ecg_features[name]:.10f}")
            ecg_vector.append(ecg_features[name])
        
        print("\n" + "="*80)
        print("CSV ROWS (copy these lines):")
        print("="*80)
        print("RADAR:", ",".join([f"{v:.10f}" for v in radar_vector]))
        print("ECG  :", ",".join([f"{v:.10f}" for v in ecg_vector]))
        print("="*80 + "\n")
        
        # Update GUI
        self.features_results_text.insert(tk.END, f"\n📋 Comparison features printed to console - check terminal!")

    def save_radar_features_csv(self):
        """Save radar features to fixed CSV file (append mode like Preprocessing.py)"""
        try:
            # Determine which radar features to use
            if hasattr(self, 'comparison_results') and self.comparison_results is not None:
                # Use radar features from comparison
                features = self.comparison_results['radar_features']
                source_name = self.comparison_results['radar_name']
            else:
                # Use single source features (should be radar)
                if not hasattr(self, 'hrv_features_extracted') or self.hrv_features_extracted is None:
                    messagebox.showwarning("Warning", "No radar features available.")
                    return
                features = self.hrv_features_extracted
                source_name = getattr(self, 'current_feature_source', 'Unknown')
            
            # Create feature row (without label, will be added manually)
            feature_row = {
                'LF_power': features['LF_power'],
                'HF_power': features['HF_power'],
                'P1': features['P1'],
                'P2': features['P2'],
                'H1': features['H1'],
                'H2': features['H2'],
                'H3': features['H3'],
                'H4': features['H4'],
                'SD1': features['SD1'],
                'SD2': features['SD2'],
                'SD1_SD2_ratio': features['SD1_SD2_ratio'],
                'SamEn': features['SamEn'],
                'Label': ''  # Empty for manual entry
            }
            
            # Fixed CSV path (like in Preprocessing.py)
            csv_path = os.path.join(os.getcwd(), "radar_hrv_features.csv")
            file_exists = os.path.exists(csv_path)
            
            # Create DataFrame and append
            df = pd.DataFrame([feature_row])
            df.to_csv(csv_path, mode='a', index=False, header=not file_exists)
            
            self.features_results_text.insert(tk.END, f"\n✅ Radar features appended to radar_hrv_features.csv")
            messagebox.showinfo("Save Complete", f"Radar features saved to:\n{csv_path}\n\nSource: {source_name}")
        
        except Exception as e:
            messagebox.showerror("Save Error", f"Failed to save radar features: {str(e)}")

    def save_radar_features_csv(self):
        """Save radar features to fixed CSV file (append mode like Preprocessing.py)"""
        try:
            # Determine which radar features to use
            if hasattr(self, 'comparison_results') and self.comparison_results is not None:
                # Use radar features from comparison
                features = self.comparison_results['radar_features']
                source_name = self.comparison_results['radar_name']
            else:
                # Use single source features (should be radar)
                if not hasattr(self, 'hrv_features_extracted') or self.hrv_features_extracted is None:
                    messagebox.showwarning("Warning", "No radar features available.")
                    return
                features = self.hrv_features_extracted
                source_name = getattr(self, 'current_feature_source', 'Unknown')
            
            # Create feature row (without label, will be added manually)
            feature_row = {
                'LF_power': features['LF_power'],
                'HF_power': features['HF_power'],
                'P1': features['P1'],
                'P2': features['P2'],
                'H1': features['H1'],
                'H2': features['H2'],
                'H3': features['H3'],
                'H4': features['H4'],
                'SD1': features['SD1'],
                'SD2': features['SD2'],
                'SD1_SD2_ratio': features['SD1_SD2_ratio'],
                'SamEn': features['SamEn'],
                'Label': ''  # Empty for manual entry
            }
            
            # Fixed CSV path (like in Preprocessing.py)
            csv_path = os.path.join(os.getcwd(), "radar_hrv_features.csv")
            file_exists = os.path.exists(csv_path)
            
            # Create DataFrame and append
            df = pd.DataFrame([feature_row])
            df.to_csv(csv_path, mode='a', index=False, header=not file_exists)
            
            self.features_results_text.insert(tk.END, f"\n✅ Radar features appended to radar_hrv_features.csv")
            messagebox.showinfo("Save Complete", f"Radar features saved to:\n{csv_path}\n\nSource: {source_name}")
            
        except Exception as e:
            messagebox.showerror("Save Error", f"Failed to save radar features: {str(e)}")

    def save_ecg_features_csv(self):
        """Save ECG features to fixed CSV file (append mode like Preprocessing.py)"""
        try:
            # Determine which ECG features to use
            if hasattr(self, 'comparison_results') and self.comparison_results is not None:
                # Use ECG features from comparison
                features = self.comparison_results['ecg_features']
                source_name = self.comparison_results['ecg_name']
            else:
                # Single source mode - check if it's ECG
                if not hasattr(self, 'hrv_features_extracted') or self.hrv_features_extracted is None:
                    messagebox.showwarning("Warning", "No ECG features available.")
                    return
                
                source = self.rr_source_var.get()
                if source != "ecg_peaks":
                    messagebox.showwarning("Warning", "Current features are not from ECG source.")
                    return
                    
                features = self.hrv_features_extracted
                source_name = "ECG R-Peaks"
            
            # Create feature row (without label, will be added manually)
            feature_row = {
                'LF_power': features['LF_power'],
                'HF_power': features['HF_power'],
                'P1': features['P1'],
                'P2': features['P2'],
                'H1': features['H1'],
                'H2': features['H2'],
                'H3': features['H3'],
                'H4': features['H4'],
                'SD1': features['SD1'],
                'SD2': features['SD2'],
                'SD1_SD2_ratio': features['SD1_SD2_ratio'],
                'SamEn': features['SamEn'],
                'Label': ''  # Empty for manual entry
            }
            
            # Fixed CSV path (separate file for ECG)
            csv_path = os.path.join(os.getcwd(), "ecg_hrv_features.csv")
            file_exists = os.path.exists(csv_path)
            
            # Create DataFrame and append
            df = pd.DataFrame([feature_row])
            df.to_csv(csv_path, mode='a', index=False, header=not file_exists)
            
            self.features_results_text.insert(tk.END, f"\n✅ ECG features appended to ecg_hrv_features.csv")
            messagebox.showinfo("Save Complete", f"ECG features saved to:\n{csv_path}\n\nSource: {source_name}")
            
        except Exception as e:
            messagebox.showerror("Save Error", f"Failed to save ECG features: {str(e)}")

    def save_both_features_csv(self):
        """Save both radar and ECG features to their respective CSV files"""
        try:
            if not hasattr(self, 'comparison_results') or self.comparison_results is None:
                messagebox.showwarning("Warning", "No comparison results available. Use comparison mode first.")
                return
            
            # Save radar features
            self.save_radar_features_csv()
            
            # Save ECG features  
            self.save_ecg_features_csv()
            
            self.features_results_text.insert(tk.END, f"\n✅ Both radar and ECG features saved to separate CSV files")
            
        except Exception as e:
            messagebox.showerror("Save Error", f"Failed to save both features: {str(e)}")

    # TAMBAHKAN FUNGSI-FUNGSI INI KE kim_findpeaks_method.py

    # 1. BISPECTRUM COMPUTATION (dari Preprocessing.py)
    def compute_bispectrum_vibrant(self, rr_intervals):
        """
        Compute bispectrum |B(f1,f2)| from third-order cumulant of RR intervals.
        Style dari Preprocessing.py untuk hasil yang vibrant dan jelas.
        """
        x = rr_intervals - np.mean(rr_intervals)
        N = len(x)
        max_lag = min(50, N // 2)

        C = np.zeros((max_lag, max_lag))
        for m in range(max_lag):
            for n in range(max_lag):
                if m + n < N:
                    C[m, n] = np.mean(x[:N - m - n] * x[m:N - n] * x[m + n:])

        B = np.abs(fft2(C))
        f1 = np.linspace(0, 1, B.shape[0])
        f2 = np.linspace(0, 1, B.shape[1])

        self.last_bispectrum = B
        self.last_f1 = f1
        self.last_f2 = f2
        
        return B, f1, f2

    # 2. BISPECTRUM FEATURES (dari Preprocessing.py)
    def bispectrum_features_vibrant(self, rr_intervals):
        """
        Extract bispectral features P1, P2, H1–H4 menggunakan style Preprocessing.py
        """
        try:
            # Compute fresh bispectrum
            B, f1, f2 = self.compute_bispectrum_vibrant(rr_intervals)
            
            F1, F2 = np.meshgrid(f1, f2)
            omega_mask = (F1 > F2) & (F1 + F2 < 1.0)
            B_omega = B.copy()
            B_omega[~omega_mask] = 0

            omega_values = B_omega[omega_mask]
            omega_values = omega_values[omega_values > 1e-12]

            if len(omega_values) == 0:
                return {'P1': 0, 'P2': 0, 'H1': 0, 'H2': 0, 'H3': 0, 'H4': 0}

            # Entropy features
            p_n = omega_values / np.sum(omega_values)
            P1 = -np.sum(p_n * np.log(p_n + 1e-12))

            omega_squared = omega_values ** 2
            q_n = omega_squared / np.sum(omega_squared)
            P2 = -np.sum(q_n * np.log(q_n + 1e-12))

            # H1: log amplitude sum in Ω
            H1 = np.sum(np.log(omega_values + 1e-12))

            # H2–H4: diagonal values in Ω
            f_diag = f1
            valid_diag_idx = np.where(f_diag < 0.5)[0]
            diagonal_values = B[valid_diag_idx, valid_diag_idx]
            diag_valid = diagonal_values[diagonal_values > 1e-12]

            if len(diag_valid) == 0:
                return {'P1': P1, 'P2': P2, 'H1': H1, 'H2': 0, 'H3': 0, 'H4': 0}

            log_diag = np.log(diag_valid)
            H2 = np.sum(log_diag)

            N_diag = len(diag_valid)
            k = np.arange(1, N_diag + 1)
            H3 = np.sum(k * log_diag) / N_diag
            H4 = np.sum((k - H3) ** 2 * log_diag) / N_diag

            return {'P1': P1, 'P2': P2, 'H1': H1, 'H2': H2, 'H3': H3, 'H4': H4}

        except Exception as e:
            print(f"Error in bispectrum_features: {e}")
            return {'P1': 0, 'P2': 0, 'H1': 0, 'H2': 0, 'H3': 0, 'H4': 0}

    # 3. VIBRANT BISPECTRUM PLOTTING
    def plot_bispectrum_contour_vibrant(self, ax):
        """
        Plot bispectrum sebagai contour yang vibrant dan jelas.
        Adopted dari Preprocessing.py
        """
        try:
            if hasattr(self, 'last_bispectrum') and hasattr(self, 'last_f1') and hasattr(self, 'last_f2'):
                B = self.last_bispectrum
                f1_axis = self.last_f1
                f2_axis = self.last_f2

                if B.size > 0:
                    F1, F2 = np.meshgrid(f1_axis, f2_axis)

                    # Level kontur yang vibrant
                    vmin = np.percentile(B, 5)
                    vmax = np.percentile(B, 95)
                    levels = np.linspace(vmin, vmax, 20)

                    # Plot kontur dengan colormap vibrant
                    contour = ax.contour(F1, F2, B, levels=levels, cmap='jet', linewidths=0.8)
                    
                    # Tambahkan filled contour untuk lebih vibrant
                    filled_contour = ax.contourf(F1, F2, B, levels=levels, cmap='jet', alpha=0.6)

                    # Colorbar
                    try:
                        cbar = self.fig_features.colorbar(filled_contour, ax=ax, fraction=0.046, pad=0.04)
                        cbar.set_label('|B(f1,f2)|', fontsize=10)
                    except:
                        pass

                    # Grid yang rapi
                    ax.grid(True, linestyle='--', alpha=0.4, color='white', linewidth=1)
                else:
                    ax.text(0.5, 0.5, 'No bispectrum data', ha='center', va='center', 
                        transform=ax.transAxes, fontsize=12, 
                        bbox=dict(boxstyle='round', alpha=0.8, facecolor='lightblue'))
            else:
                ax.text(0.5, 0.5, 'No bispectrum computed', ha='center', va='center', 
                    transform=ax.transAxes, fontsize=12,
                    bbox=dict(boxstyle='round', alpha=0.8, facecolor='lightcoral'))

            # Styling yang rapi
            ax.set_xlabel('f1 (Normalized Frequency)', fontsize=11, fontweight='bold')
            ax.set_ylabel('f2 (Normalized Frequency)', fontsize=11, fontweight='bold')
            ax.set_title('|B(f1,f2)| Contour Plot', fontsize=12, fontweight='bold', pad=20)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_aspect('equal')  # Membuat bentuk persegi

        except Exception as e:
            print(f"Error plotting bispectrum: {e}")
            ax.text(0.5, 0.5, f'Error: {str(e)}', ha='center', va='center', 
                transform=ax.transAxes, color='red', fontweight='bold')

    # 4. VIBRANT POINCARE PLOTTING
    def plot_poincare_with_arrows_vibrant(self, ax, rr_intervals):
        """
        Plot Poincaré yang vibrant dengan SD1/SD2 arrows.
        Adopted dari Preprocessing.py
        """
        try:
            if len(rr_intervals) > 1:
                rr_n = rr_intervals[:-1]
                rr_n1 = rr_intervals[1:]
                
                # Scatter plot dengan warna vibrant
                scatter = ax.scatter(rr_n, rr_n1, alpha=0.7, s=25, c=np.arange(len(rr_n)), 
                                cmap='viridis', edgecolors='black', linewidth=0.5)
                
                # Calculate RRm (mean)
                RRm = np.mean(rr_intervals)
                
                # Reference lines
                min_rr = min(min(rr_n), min(rr_n1))
                max_rr = max(max(rr_n), max(rr_n1))
                
                # Identity line (y = x)
                ax.plot([min_rr, max_rr], [min_rr, max_rr], 'k--', alpha=0.8, 
                    linewidth=3, label='y = x', zorder=3)
                
                # Second line (y = -x + 2*RRm)
                x_line = np.linspace(min_rr, max_rr, 100)
                y_line = -x_line + 2 * RRm
                ax.plot(x_line, y_line, 'r--', alpha=0.8, linewidth=3, 
                    label=f'y = -x + 2RRm', zorder=3)
                
                # Calculate SD1 and SD2
                distances_id = np.abs(rr_n1 - rr_n) / np.sqrt(2)
                SD1 = np.std(distances_id)
                
                distances_2 = np.abs(rr_n1 + rr_n - 2*RRm) / np.sqrt(2)
                SD2 = np.std(distances_2)
                
                # Center point
                center_x, center_y = RRm, RRm
                
                # SD1 vector (vibrant red)
                sd1_length = SD1 * 2
                sd1_dx = -sd1_length / np.sqrt(2)
                sd1_dy = sd1_length / np.sqrt(2)
                
                ax.arrow(center_x, center_y, sd1_dx, sd1_dy, 
                        head_width=max_rr*0.015, head_length=max_rr*0.015, 
                        fc='red', ec='darkred', linewidth=3, alpha=0.9,
                        label=f'SD1={SD1:.1f}ms', zorder=4)
                
                # SD2 vector (vibrant green)
                sd2_length = SD2 * 2
                sd2_dx = sd2_length / np.sqrt(2)
                sd2_dy = sd2_length / np.sqrt(2)
                
                ax.arrow(center_x, center_y, sd2_dx, sd2_dy, 
                        head_width=max_rr*0.015, head_length=max_rr*0.015, 
                        fc='green', ec='darkgreen', linewidth=3, alpha=0.9,
                        label=f'SD2={SD2:.1f}ms', zorder=4)
                
                # Center point yang prominent
                ax.plot(center_x, center_y, 'ko', markersize=12, markeredgecolor='white', 
                    markeredgewidth=2, label=f'Center (RRm={RRm:.0f}ms)', zorder=5)
                
                # Legend yang rapi
                ax.legend(fontsize=10, loc='upper right', framealpha=0.9, 
                        fancybox=True, shadow=True)
                
                # Colorbar untuk scatter
                try:
                    cbar = self.fig_features.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
                    cbar.set_label('Point Index', fontsize=10)
                except:
                    pass
                    
            else:
                ax.text(0.5, 0.5, 'Insufficient RR intervals', 
                    ha='center', va='center', transform=ax.transAxes,
                    fontsize=12, bbox=dict(boxstyle='round', alpha=0.8, facecolor='orange'))
            
            # Styling yang vibrant
            ax.set_title('Poincaré Plot - SD1/SD2 Analysis', fontsize=14, fontweight='bold', pad=20)
            ax.set_xlabel('RR(n) [ms]', fontsize=12, fontweight='bold')
            ax.set_ylabel('RR(n+1) [ms]', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3, linestyle='-', linewidth=1)
            ax.set_aspect('equal')
            
            # Background gradient (optional)
            ax.set_facecolor('#f8f9fa')
            
        except Exception as e:
            print(f"Error in Poincaré plot: {e}")
            ax.text(0.5, 0.5, f'Error: {str(e)}', ha='center', va='center', 
                transform=ax.transAxes, color='red', fontweight='bold')

    # 5. VIBRANT PSD PLOTTING  
    def plot_power_spectral_density_vibrant(self, ax, frequencies, psd, lf_power, hf_power):
        """
        Plot PSD yang vibrant dengan area fill berwarna.
        Adopted dari Preprocessing.py
        """
        try:
            if len(frequencies) > 0 and len(psd) > 0:
                # Main PSD line
                ax.plot(frequencies, psd, 'b-', linewidth=2.5, alpha=0.8, label='PSD')
                ax.fill_between(frequencies, psd, alpha=0.2, color='blue')

                # LF dan HF bands dengan warna vibrant
                lf_mask = (frequencies >= 0.04) & (frequencies <= 0.15)
                hf_mask = (frequencies >= 0.15) & (frequencies <= 0.4)

                if np.any(lf_mask):
                    ax.fill_between(frequencies[lf_mask], psd[lf_mask], alpha=0.7, 
                                color='red', label=f'LF: {lf_power:.4f}')
                if np.any(hf_mask):
                    ax.fill_between(frequencies[hf_mask], psd[hf_mask], alpha=0.7, 
                                color='green', label=f'HF: {hf_power:.4f}')

                # Dynamic scaling
                ax.set_xlim(0, 0.5)
                psd_max = np.max(psd)
                ax.set_ylim(0, psd_max * 1.2)
                
                # Scientific notation yang rapi
                from matplotlib.ticker import ScalarFormatter
                ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
                ax.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
                
                # Legend yang rapi
                ax.legend(fontsize=10, loc='upper right', framealpha=0.9, 
                        fancybox=True, shadow=True)

            else:
                ax.text(0.5, 0.5, 'No spectral data available', ha='center', va='center', 
                    transform=ax.transAxes, fontsize=12,
                    bbox=dict(boxstyle='round', alpha=0.8, facecolor='lightgray'))

            # Styling vibrant
            ax.set_title('Power Spectral Density (AR Model)', fontsize=14, fontweight='bold', pad=20)
            ax.set_xlabel('Frequency (Hz)', fontsize=12, fontweight='bold')
            ax.set_ylabel('Power Spectral Density', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3, linestyle='-', linewidth=1)
            ax.set_facecolor('#f8f9fa')

        except Exception as e:
            print(f"Error in PSD plot: {e}")
            ax.text(0.5, 0.5, f'Error: {str(e)}', ha='center', va='center', 
                transform=ax.transAxes, color='red', fontweight='bold')

    # 6. MAIN VISUALIZATION FUNCTION
    def create_vibrant_hrv_visualization(self, results):
        """
        Create visualization menggunakan style vibrant dari Preprocessing.py
        """
        try:
            # Clear previous plots
            for ax in [self.ax_rr_features, self.ax_psd_features, 
                    self.ax_poincare_features, self.ax_bispectrum_features]:
                ax.clear()

            # Extract data
            radar_rr = results['radar_rr']
            radar_features = results['radar_features']
            ecg_features = results['ecg_features']
            
            # 1. RR Intervals Plot (top span)
            rr_times = np.cumsum(radar_rr) / 1000  # Convert to seconds
            self.ax_rr_features.plot(rr_times, radar_rr, 'b.-', markersize=3, 
                                    linewidth=1.5, alpha=0.8, label='Radar RR')
            
            mean_rr = np.mean(radar_rr)
            std_rr = np.std(radar_rr)
            self.ax_rr_features.axhline(mean_rr, color='red', linestyle='-', 
                                    alpha=0.8, linewidth=2, label=f'Mean: {mean_rr:.1f}ms')
            self.ax_rr_features.fill_between(rr_times, mean_rr - std_rr, mean_rr + std_rr, 
                                            alpha=0.2, color='orange', label=f'±1 STD')
            
            self.ax_rr_features.set_title('RR Intervals Over Time', fontsize=14, fontweight='bold')
            self.ax_rr_features.set_xlabel('Time (s)', fontweight='bold')
            self.ax_rr_features.set_ylabel('RR Interval (ms)', fontweight='bold')
            self.ax_rr_features.legend()
            self.ax_rr_features.grid(True, alpha=0.3)
            self.ax_rr_features.set_facecolor('#f8f9fa')

            # 2. PSD Plot (bottom left)
            if 'frequencies' in radar_features:
                self.plot_power_spectral_density_vibrant(
                    self.ax_psd_features, 
                    radar_features['frequencies'],
                    radar_features['psd'],
                    radar_features['LF_power'],
                    radar_features['HF_power']
                )

            # 3. Poincaré Plot (bottom right)
            self.plot_poincare_with_arrows_vibrant(self.ax_poincare_features, radar_rr)

            # 4. Bispectrum Plot (full bottom)
            # First compute bispectrum
            self.bispectrum_features_vibrant(radar_rr)
            self.plot_bispectrum_contour_vibrant(self.ax_bispectrum_features)

            # Tight layout
            self.fig_features.tight_layout(pad=3.0)
            self.canvas_features.draw()

        except Exception as e:
            print(f"Error creating vibrant visualization: {e}")
            messagebox.showerror("Visualization Error", f"Failed to create visualization: {str(e)}")

def main():
    try:
        default_file = 'AFsub_rs2_take2_20250603_155416_downsampled.h5'
        
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