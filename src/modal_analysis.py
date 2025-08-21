"""
模态参数估计模块
从分离的模态信号中估计频率、阻尼比和振型
"""

import numpy as np
from scipy import signal
from scipy.signal import hilbert, find_peaks
import matplotlib.pyplot as plt


class ModalParameterEstimator:
    """模态参数估计器"""
    
    def __init__(self, fs=240):
        """
        初始化估计器
        
        参数:
            fs: 采样频率 (Hz)
        """
        self.fs = fs
        self.frequencies = []
        self.damping_ratios = []
        self.mode_shapes = []
        
    def estimate_frequency(self, modal_signal):
        """
        估计模态频率（使用FFT和峰值检测）
        
        参数:
            modal_signal: 模态信号时间序列
            
        返回:
            frequency: 估计的频率 (Hz)
        """
        # FFT
        n = len(modal_signal)
        fft_vals = np.fft.fft(modal_signal)
        fft_freqs = np.fft.fftfreq(n, 1/self.fs)
        
        # 只取正频率部分
        positive_freq_idx = fft_freqs > 0
        fft_freqs = fft_freqs[positive_freq_idx]
        fft_magnitude = np.abs(fft_vals[positive_freq_idx])
        
        # 找到最大峰值
        peak_idx = np.argmax(fft_magnitude)
        frequency = fft_freqs[peak_idx]
        
        return frequency
    
    def estimate_frequency_hilbert(self, modal_signal):
        """
        使用Hilbert变换估计瞬时频率
        
        参数:
            modal_signal: 模态信号时间序列
            
        返回:
            frequency: 平均频率 (Hz)
            inst_freq: 瞬时频率序列
        """
        # Hilbert变换
        analytic_signal = hilbert(modal_signal)
        inst_phase = np.unwrap(np.angle(analytic_signal))
        
        # 瞬时频率
        inst_freq = np.diff(inst_phase) / (2.0 * np.pi) * self.fs
        
        # 去除边缘效应，取中间80%的数据
        n = len(inst_freq)
        start = int(0.1 * n)
        end = int(0.9 * n)
        
        # 平均频率
        frequency = np.mean(inst_freq[start:end])
        
        return frequency, inst_freq
    
    def estimate_damping_ratio(self, modal_signal, frequency):
        """
        估计阻尼比（使用对数衰减法）
        
        参数:
            modal_signal: 模态信号时间序列
            frequency: 模态频率 (Hz)
            
        返回:
            damping_ratio: 阻尼比 (%)
        """
        # 获取信号包络（使用Hilbert变换）
        analytic_signal = hilbert(modal_signal)
        envelope = np.abs(analytic_signal)
        
        # 找到峰值
        peaks, _ = find_peaks(envelope, height=np.max(envelope)*0.1)
        
        if len(peaks) < 2:
            return 0.0
        
        # 选择合适的峰值进行计算
        # 使用前面几个较大的峰值
        n_peaks_to_use = min(10, len(peaks))
        peaks = peaks[:n_peaks_to_use]
        peak_values = envelope[peaks]
        
        # 对数衰减拟合
        # ln(A) = ln(A0) - ζωn*t
        t = peaks / self.fs
        log_peaks = np.log(peak_values)
        
        # 线性拟合
        coeffs = np.polyfit(t, log_peaks, 1)
        decay_rate = -coeffs[0]
        
        # 计算阻尼比
        # decay_rate = ζ * ωn = ζ * 2π * fn
        omega_n = 2 * np.pi * frequency
        zeta = decay_rate / omega_n
        
        # 转换为百分比
        damping_ratio = zeta * 100
        
        return damping_ratio
    
    def estimate_damping_half_power(self, modal_signal, frequency):
        """
        使用半功率带宽法估计阻尼比
        
        参数:
            modal_signal: 模态信号
            frequency: 模态频率
            
        返回:
            damping_ratio: 阻尼比 (%)
        """
        # FFT
        fft_vals = np.fft.fft(modal_signal)
        fft_freqs = np.fft.fftfreq(len(modal_signal), 1/self.fs)
        
        # 只取正频率部分
        positive_freq_idx = fft_freqs > 0
        fft_freqs = fft_freqs[positive_freq_idx]
        fft_magnitude = np.abs(fft_vals[positive_freq_idx])
        
        # 找到峰值
        peak_idx = np.argmax(fft_magnitude)
        peak_magnitude = fft_magnitude[peak_idx]
        
        # 半功率点（-3dB）
        half_power = peak_magnitude / np.sqrt(2)
        
        # 找到半功率带宽
        # 向左搜索
        left_idx = peak_idx
        while left_idx > 0 and fft_magnitude[left_idx] > half_power:
            left_idx -= 1
            
        # 向右搜索
        right_idx = peak_idx
        while right_idx < len(fft_magnitude)-1 and fft_magnitude[right_idx] > half_power:
            right_idx += 1
            
        # 带宽
        if right_idx > left_idx:
            bandwidth = fft_freqs[right_idx] - fft_freqs[left_idx]
            # 阻尼比 ζ ≈ Δf / (2*fn)
            damping_ratio = (bandwidth / (2 * frequency)) * 100
        else:
            damping_ratio = 0.0
            
        return damping_ratio
    
    def extract_mode_shapes(self, U_r, mixing_matrix):
        """
        提取模态振型
        
        参数:
            U_r: PCA降维矩阵 (N_pixels, r)
            mixing_matrix: CP算法的混合矩阵 (r, r)
            
        返回:
            mode_shapes: 模态振型矩阵 (N_pixels, n_modes)
        """
        # 根据论文公式: φ_i = U_r * γ_i
        # 其中γ_i是混合矩阵的列
        mode_shapes = U_r @ mixing_matrix
        
        # 归一化每个振型
        for i in range(mode_shapes.shape[1]):
            mode_shape = mode_shapes[:, i]
            # 使用最大值归一化
            mode_shapes[:, i] = mode_shape / np.max(np.abs(mode_shape))
            
        return mode_shapes
    
    def estimate_all_parameters(self, separated_signals, U_r=None, mixing_matrix=None):
        """
        估计所有模态参数
        
        参数:
            separated_signals: 分离的模态信号 (n_modes, n_samples)
            U_r: PCA降维矩阵（可选）
            mixing_matrix: 混合矩阵（可选）
            
        返回:
            results: 包含所有估计参数的字典
        """
        n_modes = separated_signals.shape[0]
        
        self.frequencies = []
        self.damping_ratios = []
        
        print("\n估计模态参数...")
        for i in range(n_modes):
            modal_signal = separated_signals[i, :]
            
            # 估计频率
            freq = self.estimate_frequency(modal_signal)
            self.frequencies.append(freq)
            
            # 估计阻尼比
            damping = self.estimate_damping_ratio(modal_signal, freq)
            self.damping_ratios.append(damping)
            
            print(f"模态 {i+1}: 频率 = {freq:.2f} Hz, 阻尼比 = {damping:.2f}%")
        
        # 提取振型（如果提供了必要的矩阵）
        if U_r is not None and mixing_matrix is not None:
            self.mode_shapes = self.extract_mode_shapes(U_r, mixing_matrix)
            print(f"提取了 {self.mode_shapes.shape[1]} 个模态振型")
        
        results = {
            'frequencies': np.array(self.frequencies),
            'damping_ratios': np.array(self.damping_ratios),
            'mode_shapes': self.mode_shapes if len(self.mode_shapes) > 0 else None
        }
        
        return results
    
    def plot_results(self, separated_signals, save_path=None):
        """
        绘制估计结果
        
        参数:
            separated_signals: 分离的模态信号
            save_path: 保存路径（可选）
        """
        n_modes = separated_signals.shape[0]
        n_show = min(n_modes, 3)
        
        fig, axes = plt.subplots(n_show, 3, figsize=(15, 4*n_show))
        if n_show == 1:
            axes = axes.reshape(1, -1)
        
        for i in range(n_show):
            modal_signal = separated_signals[i, :]
            t = np.arange(len(modal_signal)) / self.fs
            
            # 时域信号
            axes[i, 0].plot(t, modal_signal, 'b-', linewidth=0.5)
            axes[i, 0].set_ylabel(f'模态 {i+1}')
            axes[i, 0].set_xlabel('时间 (s)')
            axes[i, 0].grid(True, alpha=0.3)
            axes[i, 0].set_title(f'时域信号')
            
            # 包络和阻尼
            analytic_signal = hilbert(modal_signal)
            envelope = np.abs(analytic_signal)
            axes[i, 1].plot(t, envelope, 'r-', linewidth=1, label='包络')
            axes[i, 1].plot(t, modal_signal, 'b-', linewidth=0.5, alpha=0.5, label='信号')
            axes[i, 1].set_xlabel('时间 (s)')
            axes[i, 1].grid(True, alpha=0.3)
            axes[i, 1].legend()
            axes[i, 1].set_title(f'阻尼比 = {self.damping_ratios[i]:.2f}%')
            
            # 频谱
            freqs, psd = signal.periodogram(modal_signal, fs=self.fs)
            axes[i, 2].semilogy(freqs, psd, 'g-', linewidth=1)
            axes[i, 2].axvline(x=self.frequencies[i], color='r', linestyle='--', 
                             label=f'f = {self.frequencies[i]:.2f} Hz')
            axes[i, 2].set_xlim([0, 50])
            axes[i, 2].set_xlabel('频率 (Hz)')
            axes[i, 2].grid(True, alpha=0.3)
            axes[i, 2].legend()
            axes[i, 2].set_title('功率谱密度')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()