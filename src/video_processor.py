"""
视频处理主模块
整合所有算法实现完整的视频振动模态分析
"""

import numpy as np
import os

# 尝试导入可选库
try:
    import cv2
except ImportError:
    cv2 = None
    print("警告: OpenCV未安装，视频加载功能将不可用")

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("警告: Matplotlib未安装，可视化功能将不可用")

try:
    from tqdm import tqdm
except ImportError:
    # 简单的tqdm替代
    def tqdm(iterable, desc=None):
        if desc:
            print(desc)
        return iterable

from .steerable_pyramid import ComplexSteerablePyramid
from .phase_extraction import PhaseExtractor
from .pca_analysis import PCAAnalyzer
from .complexity_pursuit import ComplexityPursuit
from .modal_analysis import ModalParameterEstimator


class VideoModalAnalysis:
    """视频模态分析主类"""
    
    def __init__(self, n_scales=5, n_orientations=4, fps=240):
        """
        初始化视频模态分析器
        
        参数:
            n_scales: 金字塔尺度数
            n_orientations: 方向数
            fps: 视频帧率
        """
        self.n_scales = n_scales
        self.n_orientations = n_orientations
        self.fps = fps
        
        # 初始化各个组件
        self.pyramid = ComplexSteerablePyramid(n_scales, n_orientations)
        self.phase_extractor = PhaseExtractor()
        self.pca_analyzer = PCAAnalyzer()
        self.cp_separator = ComplexityPursuit()
        self.modal_estimator = ModalParameterEstimator(fps)
        
        # 存储结果
        self.video_frames = None
        self.phases = None
        self.principal_components = None
        self.separated_signals = None
        self.modal_parameters = None
        
    def load_video(self, video_path, max_frames=None, downsample_factor=1):
        """
        加载视频文件
        
        参数:
            video_path: 视频文件路径
            max_frames: 最大帧数（None表示全部）
            downsample_factor: 空间下采样因子
            
        返回:
            frames: 视频帧数组
        """
        print(f"加载视频: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        
        # 获取视频信息
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"原始视频: {width}x{height}, {total_frames}帧, {original_fps:.1f} fps")
        
        # 确定要读取的帧数
        if max_frames is not None:
            n_frames = min(max_frames, total_frames)
        else:
            n_frames = total_frames
            
        # 读取视频帧
        frames = []
        for i in tqdm(range(n_frames), desc="读取视频帧"):
            ret, frame = cap.read()
            if not ret:
                break
                
            # 转换为灰度图
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # 空间下采样
            if downsample_factor > 1:
                gray = cv2.resize(gray, 
                                 (width//downsample_factor, height//downsample_factor),
                                 interpolation=cv2.INTER_AREA)
            
            # 归一化到[0, 1]
            gray = gray.astype(np.float32) / 255.0
            
            frames.append(gray)
        
        cap.release()
        
        self.video_frames = np.array(frames)
        print(f"加载了 {len(self.video_frames)} 帧, 尺寸: {self.video_frames[0].shape}")
        
        return self.video_frames
    
    def load_frames(self, frames):
        """
        直接加载帧数组（用于已经预处理的数据）
        
        参数:
            frames: 帧数组 (n_frames, height, width)
        """
        self.video_frames = np.array(frames)
        if self.video_frames.ndim == 4:  # 如果是彩色图像
            # 转换为灰度
            self.video_frames = np.mean(self.video_frames, axis=3)
        
        # 归一化
        if self.video_frames.max() > 1:
            self.video_frames = self.video_frames.astype(np.float32) / 255.0
            
        print(f"加载了 {len(self.video_frames)} 帧, 尺寸: {self.video_frames[0].shape}")
        
    def analyze(self, scale=1, filter_phases=True, low_freq=0.5, high_freq=30):
        """
        执行完整的模态分析流程
        
        参数:
            scale: 使用的金字塔尺度
            filter_phases: 是否对相位进行滤波
            low_freq: 滤波低频截止
            high_freq: 滤波高频截止
            
        返回:
            results: 分析结果字典
        """
        if self.video_frames is None:
            raise ValueError("请先加载视频数据")
        
        print("\n=== 开始视频模态分析 ===\n")
        
        # 步骤1: 提取相位
        print("步骤1: 提取局部相位...")
        self.phases, amplitudes = self.phase_extractor.extract_phases_from_video(
            self.video_frames, self.pyramid, scale=scale
        )
        
        # 步骤2: 去中心化
        print("步骤2: 去除时间平均值...")
        self.phases = self.phase_extractor.remove_temporal_mean(self.phases)
        
        # 步骤3: 滤波（可选）
        if filter_phases:
            print(f"步骤3: 带通滤波 ({low_freq}-{high_freq} Hz)...")
            self.phases = self.phase_extractor.filter_phases(
                self.phases, low_freq, high_freq, self.fps
            )
        
        # 步骤4: PCA降维
        print("步骤4: PCA降维...")
        self.principal_components, U_r = self.pca_analyzer.fit_transform(self.phases)
        
        # 步骤5: 盲源分离
        print("步骤5: Complexity Pursuit盲源分离...")
        self.separated_signals = self.cp_separator.fit_transform(self.principal_components)
        
        # 步骤6: 估计模态参数
        print("步骤6: 估计模态参数...")
        mixing_matrix = self.cp_separator.get_mixing_matrix()
        self.modal_parameters = self.modal_estimator.estimate_all_parameters(
            self.separated_signals, U_r, mixing_matrix
        )
        
        print("\n=== 分析完成 ===\n")
        
        # 整理结果
        results = {
            'frequencies': self.modal_parameters['frequencies'],
            'damping_ratios': self.modal_parameters['damping_ratios'],
            'mode_shapes': self.modal_parameters['mode_shapes'],
            'separated_signals': self.separated_signals,
            'principal_components': self.principal_components,
            'U_r': U_r,
            'mixing_matrix': mixing_matrix
        }
        
        return results
    
    def visualize_results(self, results=None, save_dir=None):
        """
        可视化分析结果
        
        参数:
            results: 分析结果（如果为None，使用内部存储的结果）
            save_dir: 保存图像的目录
        """
        if results is None:
            results = self.modal_parameters
            
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
        
        # 绘制特征值谱
        if self.phases is not None:
            save_path = os.path.join(save_dir, 'eigenvalue_spectrum.png') if save_dir else None
            self.pca_analyzer.plot_eigenvalue_spectrum(self.phases, save_path)
        
        # 绘制主成分
        if self.principal_components is not None:
            save_path = os.path.join(save_dir, 'principal_components.png') if save_dir else None
            self.pca_analyzer.plot_principal_components(self.principal_components, save_path)
        
        # 绘制分离的信号
        if self.separated_signals is not None:
            save_path = os.path.join(save_dir, 'separated_signals.png') if save_dir else None
            self.cp_separator.plot_separated_signals(self.separated_signals, self.fps, save_path)
        
        # 绘制模态参数估计结果
        if self.separated_signals is not None:
            save_path = os.path.join(save_dir, 'modal_parameters.png') if save_dir else None
            self.modal_estimator.plot_results(self.separated_signals, save_path)
    
    def save_results(self, save_path):
        """
        保存分析结果到文件
        
        参数:
            save_path: 保存路径（.npz格式）
        """
        np.savez(save_path,
                frequencies=self.modal_parameters['frequencies'],
                damping_ratios=self.modal_parameters['damping_ratios'],
                mode_shapes=self.modal_parameters['mode_shapes'],
                separated_signals=self.separated_signals,
                principal_components=self.principal_components)
        
        print(f"结果已保存到: {save_path}")
    
    def load_results(self, load_path):
        """
        从文件加载分析结果
        
        参数:
            load_path: 加载路径
        """
        data = np.load(load_path)
        
        self.modal_parameters = {
            'frequencies': data['frequencies'],
            'damping_ratios': data['damping_ratios'],
            'mode_shapes': data['mode_shapes']
        }
        self.separated_signals = data['separated_signals']
        self.principal_components = data['principal_components']
        
        print(f"结果已从 {load_path} 加载")
        
    def print_summary(self):
        """打印分析结果摘要"""
        if self.modal_parameters is None:
            print("尚未进行分析")
            return
            
        print("\n" + "="*50)
        print(" 模态分析结果摘要 ")
        print("="*50)
        
        n_modes = len(self.modal_parameters['frequencies'])
        
        print(f"\n检测到 {n_modes} 个模态:\n")
        print(f"{'模态':<6} {'频率 (Hz)':<12} {'阻尼比 (%)':<12}")
        print("-"*30)
        
        for i in range(n_modes):
            freq = self.modal_parameters['frequencies'][i]
            damp = self.modal_parameters['damping_ratios'][i]
            print(f"{i+1:<6} {freq:<12.2f} {damp:<12.2f}")
        
        print("\n" + "="*50)