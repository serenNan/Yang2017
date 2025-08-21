"""
Complex Steerable Pyramid 实现
基于论文中的多尺度分解方法
"""

import numpy as np
from scipy import signal
from scipy.ndimage import convolve


class ComplexSteerablePyramid:
    """Complex Steerable Pyramid 滤波器实现"""
    
    def __init__(self, n_scales=5, n_orientations=4):
        """
        初始化Complex Steerable Pyramid
        
        参数:
            n_scales: 尺度数量（默认5）
            n_orientations: 方向数量（默认4）
        """
        self.n_scales = n_scales
        self.n_orientations = n_orientations
        
    def _create_filters(self, image_shape):
        """创建滤波器组"""
        rows, cols = image_shape
        
        # 创建频率网格
        u = np.arange(-cols/2, cols/2) / cols
        v = np.arange(-rows/2, rows/2) / rows
        u_grid, v_grid = np.meshgrid(u, v)
        
        # 极坐标
        rho = np.sqrt(u_grid**2 + v_grid**2)
        theta = np.arctan2(v_grid, u_grid)
        
        # 防止除零
        rho[rho == 0] = 1e-10
        
        filters = {}
        
        # 低通滤波器
        filters['low_pass'] = self._low_pass_filter(rho)
        
        # 高通滤波器
        filters['high_pass'] = self._high_pass_filter(rho)
        
        # 带通滤波器（每个尺度和方向）
        filters['band_pass'] = []
        for scale in range(self.n_scales):
            scale_filters = []
            for ori in range(self.n_orientations):
                angle = ori * np.pi / self.n_orientations
                bp_filter = self._band_pass_filter(rho, theta, scale, angle)
                scale_filters.append(bp_filter)
            filters['band_pass'].append(scale_filters)
            
        return filters
    
    def _low_pass_filter(self, rho):
        """创建低通滤波器"""
        cutoff = 0.5
        transition = 0.1
        
        # 使用平滑过渡
        lp = np.zeros_like(rho)
        mask = rho <= cutoff - transition
        lp[mask] = 1.0
        
        transition_mask = (rho > cutoff - transition) & (rho < cutoff + transition)
        lp[transition_mask] = 0.5 * (1 + np.cos(np.pi * (rho[transition_mask] - cutoff + transition) / (2 * transition)))
        
        return lp
    
    def _high_pass_filter(self, rho):
        """创建高通滤波器"""
        cutoff = 0.5
        transition = 0.1
        
        # 使用平滑过渡
        hp = np.zeros_like(rho)
        mask = rho >= cutoff + transition
        hp[mask] = 1.0
        
        transition_mask = (rho > cutoff - transition) & (rho < cutoff + transition)
        hp[transition_mask] = 0.5 * (1 - np.cos(np.pi * (rho[transition_mask] - cutoff + transition) / (2 * transition)))
        
        return hp
    
    def _band_pass_filter(self, rho, theta, scale, orientation):
        """创建带通滤波器（Gabor类型）"""
        # 径向部分
        center_freq = 0.5 * (2 ** (-scale))
        bandwidth = 0.5
        
        radial = np.exp(-(np.log(rho / center_freq) ** 2) / (2 * (bandwidth ** 2)))
        radial[rho == 0] = 0
        
        # 角度部分
        angular = np.exp(1j * orientation) * np.exp(-(theta - orientation) ** 2 / (2 * (np.pi / self.n_orientations) ** 2))
        
        return radial * angular
    
    def decompose(self, image):
        """
        对图像进行Complex Steerable Pyramid分解
        
        参数:
            image: 输入图像（灰度）
            
        返回:
            pyramid: 金字塔分解结果字典
        """
        # 确保输入是灰度图像
        if len(image.shape) == 3:
            image = np.mean(image, axis=2)
            
        # FFT变换到频域
        image_fft = np.fft.fft2(image)
        image_fft = np.fft.fftshift(image_fft)
        
        # 创建滤波器
        filters = self._create_filters(image.shape)
        
        # 金字塔分解
        pyramid = {
            'low_pass': None,
            'high_pass': None,
            'band_pass': []
        }
        
        # 低通分量
        lp_fft = image_fft * filters['low_pass']
        pyramid['low_pass'] = np.fft.ifft2(np.fft.ifftshift(lp_fft))
        
        # 高通分量
        hp_fft = image_fft * filters['high_pass']
        pyramid['high_pass'] = np.fft.ifft2(np.fft.ifftshift(hp_fft))
        
        # 带通分量（每个尺度和方向）
        for scale in range(self.n_scales):
            scale_responses = []
            for ori in range(self.n_orientations):
                bp_fft = image_fft * filters['band_pass'][scale][ori]
                response = np.fft.ifft2(np.fft.ifftshift(bp_fft))
                scale_responses.append(response)
            pyramid['band_pass'].append(scale_responses)
            
        return pyramid
    
    def reconstruct(self, pyramid):
        """
        从金字塔重建图像
        
        参数:
            pyramid: 金字塔分解结果
            
        返回:
            reconstructed: 重建的图像
        """
        # 初始化重建图像
        shape = pyramid['low_pass'].shape
        reconstructed_fft = np.zeros(shape, dtype=complex)
        
        # 添加低通分量
        if pyramid['low_pass'] is not None:
            lp_fft = np.fft.fftshift(np.fft.fft2(pyramid['low_pass']))
            reconstructed_fft += lp_fft
        
        # 添加高通分量
        if pyramid['high_pass'] is not None:
            hp_fft = np.fft.fftshift(np.fft.fft2(pyramid['high_pass']))
            reconstructed_fft += hp_fft
        
        # 添加带通分量
        for scale_responses in pyramid['band_pass']:
            for response in scale_responses:
                bp_fft = np.fft.fftshift(np.fft.fft2(response))
                reconstructed_fft += bp_fft
                
        # 逆FFT重建
        reconstructed = np.fft.ifft2(np.fft.ifftshift(reconstructed_fft))
        
        return np.real(reconstructed)