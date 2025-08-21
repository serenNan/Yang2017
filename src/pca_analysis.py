"""
PCA降维模块
用于处理高维像素数据的降维
"""

import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


class PCAAnalyzer:
    """PCA分析器"""
    
    def __init__(self, energy_threshold=0.99, eigenvalue_threshold=0.01):
        """
        初始化PCA分析器
        
        参数:
            energy_threshold: 能量保留阈值（默认0.99，保留99%的能量）
            eigenvalue_threshold: 特征值阈值（相对于最大特征值的比例）
        """
        self.energy_threshold = energy_threshold
        self.eigenvalue_threshold = eigenvalue_threshold
        self.pca = None
        self.n_components = None
        self.U_r = None  # 降维矩阵
        
    def fit_transform(self, phase_matrix):
        """
        对相位矩阵进行PCA降维
        
        参数:
            phase_matrix: 相位矩阵 (N_pixels, T_frames)
            
        返回:
            principal_components: 主成分 (r, T_frames)
            U_r: 前r个主成分向量 (N_pixels, r)
        """
        print(f"输入矩阵形状: {phase_matrix.shape}")
        
        # 计算协方差矩阵的特征值（使用SVD更高效）
        # phase_matrix = U * S * V^T
        U, S, Vt = np.linalg.svd(phase_matrix, full_matrices=False)
        
        # 特征值（奇异值的平方）
        eigenvalues = S**2
        
        # 归一化特征值
        eigenvalues_normalized = eigenvalues / eigenvalues[0]
        
        # 计算累积能量
        cumulative_energy = np.cumsum(eigenvalues) / np.sum(eigenvalues)
        
        # 确定主成分数量
        # 方法1：基于能量阈值
        n_energy = np.argmax(cumulative_energy >= self.energy_threshold) + 1
        
        # 方法2：基于特征值衰减
        n_eigenvalue = np.sum(eigenvalues_normalized > self.eigenvalue_threshold)
        
        # 方法3：检测特征值的急剧下降
        eigenvalue_ratios = eigenvalues[1:] / eigenvalues[:-1]
        n_drop = 1
        for i, ratio in enumerate(eigenvalue_ratios):
            if ratio < 0.5:  # 如果下降超过50%
                n_drop = i + 1
                break
        
        # 取三种方法的最小值（更保守的选择）
        self.n_components = min(n_energy, n_eigenvalue, n_drop)
        self.n_components = max(self.n_components, 3)  # 至少保留3个分量
        
        print(f"能量阈值建议: {n_energy} 个主成分")
        print(f"特征值阈值建议: {n_eigenvalue} 个主成分")
        print(f"特征值衰减建议: {n_drop} 个主成分")
        print(f"最终选择: {self.n_components} 个主成分")
        print(f"保留能量: {cumulative_energy[self.n_components-1]:.2%}")
        
        # 提取前r个主成分
        self.U_r = U[:, :self.n_components]  # (N_pixels, r)
        
        # 计算主成分（降维后的表示）
        # η = U_r^T * δ'
        principal_components = self.U_r.T @ phase_matrix  # (r, T)
        
        return principal_components, self.U_r
    
    def inverse_transform(self, principal_components):
        """
        从主成分重建原始数据
        
        参数:
            principal_components: 主成分 (r, T)
            
        返回:
            reconstructed: 重建的数据 (N_pixels, T)
        """
        if self.U_r is None:
            raise ValueError("需要先调用fit_transform")
            
        # δ' ≈ U_r * η
        reconstructed = self.U_r @ principal_components
        
        return reconstructed
    
    def plot_eigenvalue_spectrum(self, phase_matrix, save_path=None):
        """
        绘制特征值谱
        
        参数:
            phase_matrix: 相位矩阵
            save_path: 保存路径（可选）
        """
        # 计算特征值
        _, S, _ = np.linalg.svd(phase_matrix, full_matrices=False)
        eigenvalues = S**2
        
        # 归一化
        eigenvalues = eigenvalues / eigenvalues[0]
        
        # 绘图
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # 线性尺度
        ax1.plot(eigenvalues[:50], 'b-', linewidth=2)
        ax1.axhline(y=self.eigenvalue_threshold, color='r', linestyle='--', label=f'阈值={self.eigenvalue_threshold}')
        ax1.set_xlabel('索引')
        ax1.set_ylabel('归一化特征值')
        ax1.set_title('特征值谱（线性尺度）')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # 对数尺度
        ax2.semilogy(eigenvalues[:50], 'b-', linewidth=2)
        ax2.axhline(y=self.eigenvalue_threshold, color='r', linestyle='--', label=f'阈值={self.eigenvalue_threshold}')
        ax2.set_xlabel('索引')
        ax2.set_ylabel('归一化特征值（对数）')
        ax2.set_title('特征值谱（对数尺度）')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
        
    def plot_principal_components(self, principal_components, save_path=None):
        """
        绘制主成分时间序列
        
        参数:
            principal_components: 主成分矩阵 (r, T)
            save_path: 保存路径（可选）
        """
        n_components = principal_components.shape[0]
        n_show = min(n_components, 6)  # 最多显示6个
        
        fig, axes = plt.subplots(n_show, 1, figsize=(12, 2*n_show))
        if n_show == 1:
            axes = [axes]
            
        for i in range(n_show):
            axes[i].plot(principal_components[i, :], 'b-', linewidth=0.5)
            axes[i].set_ylabel(f'PC {i+1}')
            axes[i].grid(True, alpha=0.3)
            
        axes[-1].set_xlabel('时间帧')
        axes[0].set_title('主成分时间序列')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()