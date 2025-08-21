"""
相位提取模块
从Complex Steerable Pyramid的响应中提取局部相位信息
"""

import numpy as np
from tqdm import tqdm


class PhaseExtractor:
    """相位提取器"""
    
    def __init__(self):
        """初始化相位提取器"""
        pass
    
    def extract_phases_from_video(self, video_frames, pyramid_decomposer, scale=1):
        """
        从视频帧序列中提取相位信息
        
        参数:
            video_frames: 视频帧序列 (T, H, W) 或 (T, H, W, C)
            pyramid_decomposer: Complex Steerable Pyramid分解器
            scale: 使用的尺度（默认为1）
            
        返回:
            phases: 相位矩阵 (N_pixels, T)
            amplitudes: 振幅矩阵 (N_pixels, T)
        """
        n_frames = len(video_frames)
        
        # 存储所有帧的相位和振幅
        all_phases = []
        all_amplitudes = []
        
        print("提取视频帧的相位信息...")
        for frame_idx in tqdm(range(n_frames)):
            frame = video_frames[frame_idx]
            
            # 如果是彩色图像，转换为灰度
            if len(frame.shape) == 3:
                frame = np.mean(frame, axis=2)
            
            # Complex Steerable Pyramid分解
            pyramid = pyramid_decomposer.decompose(frame)
            
            # 提取指定尺度的响应（使用第一个方向）
            if scale <= len(pyramid['band_pass']):
                response = pyramid['band_pass'][scale-1][0]  # 使用第一个方向
            else:
                # 如果尺度超出范围，使用低通分量
                response = pyramid['low_pass']
            
            # 提取相位和振幅
            phase = np.angle(response)
            amplitude = np.abs(response)
            
            # 展平为向量
            phase_vector = phase.flatten()
            amplitude_vector = amplitude.flatten()
            
            all_phases.append(phase_vector)
            all_amplitudes.append(amplitude_vector)
        
        # 转换为矩阵形式 (N_pixels, T)
        phases = np.array(all_phases).T
        amplitudes = np.array(all_amplitudes).T
        
        return phases, amplitudes
    
    def remove_temporal_mean(self, phases):
        """
        移除时间平均值（去中心化）
        
        参数:
            phases: 相位矩阵 (N_pixels, T)
            
        返回:
            centered_phases: 去中心化的相位矩阵
        """
        # 计算每个像素的时间平均值
        temporal_mean = np.mean(phases, axis=1, keepdims=True)
        
        # 去中心化
        centered_phases = phases - temporal_mean
        
        return centered_phases
    
    def unwrap_phases(self, phases):
        """
        相位展开（处理相位跳变）
        
        参数:
            phases: 相位矩阵 (N_pixels, T)
            
        返回:
            unwrapped: 展开后的相位
        """
        unwrapped = np.zeros_like(phases)
        
        # 对每个像素的时间序列进行相位展开
        for i in range(phases.shape[0]):
            unwrapped[i, :] = np.unwrap(phases[i, :])
            
        return unwrapped
    
    def compute_phase_differences(self, phases):
        """
        计算相位差（用于运动估计）
        
        参数:
            phases: 相位矩阵 (N_pixels, T)
            
        返回:
            phase_diffs: 相位差矩阵 (N_pixels, T-1)
        """
        # 计算相邻帧之间的相位差
        phase_diffs = np.diff(phases, axis=1)
        
        # 处理相位跳变（将差值限制在-π到π之间）
        phase_diffs = np.mod(phase_diffs + np.pi, 2*np.pi) - np.pi
        
        return phase_diffs
    
    def filter_phases(self, phases, low_freq=0.5, high_freq=30, fps=240):
        """
        带通滤波相位信号
        
        参数:
            phases: 相位矩阵 (N_pixels, T)
            low_freq: 低频截止频率 (Hz)
            high_freq: 高频截止频率 (Hz)
            fps: 视频帧率
            
        返回:
            filtered: 滤波后的相位
        """
        from scipy import signal
        
        # 设计带通滤波器
        nyquist = fps / 2
        low = low_freq / nyquist
        high = high_freq / nyquist
        
        # 防止频率超出范围
        low = max(0.001, min(low, 0.999))
        high = max(low + 0.001, min(high, 0.999))
        
        # Butterworth带通滤波器
        b, a = signal.butter(4, [low, high], btype='band')
        
        # 对每个像素的时间序列进行滤波
        filtered = np.zeros_like(phases)
        for i in range(phases.shape[0]):
            # 使用filtfilt进行零相位滤波
            filtered[i, :] = signal.filtfilt(b, a, phases[i, :])
            
        return filtered