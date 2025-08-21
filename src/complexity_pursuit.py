"""
Complexity Pursuit (CP) 盲源分离算法
用于从混合信号中分离独立的模态成分
"""

import numpy as np
from scipy import signal as scipy_signal
from scipy.linalg import eigh
import matplotlib.pyplot as plt


class ComplexityPursuit:
    """Complexity Pursuit 算法实现"""
    
    def __init__(self, max_iter=100, tol=1e-6):
        """
        初始化CP算法
        
        参数:
            max_iter: 最大迭代次数
            tol: 收敛容差
        """
        self.max_iter = max_iter
        self.tol = tol
        self.W = None  # 解混矩阵
        self.mixing_matrix = None  # 混合矩阵
        
    def fit(self, X):
        """
        使用CP算法进行盲源分离
        
        参数:
            X: 输入信号矩阵 (n_channels, n_samples)
            
        返回:
            S: 分离的源信号 (n_sources, n_samples)
        """
        n_channels, n_samples = X.shape
        
        # 预白化
        X_white, whitening_matrix = self._whiten(X)
        
        # 初始化解混矩阵
        W = np.eye(n_channels)
        
        print("执行Complexity Pursuit算法...")
        for iteration in range(self.max_iter):
            W_old = W.copy()
            
            # 对每个分量进行优化
            for i in range(n_channels):
                # 提取第i个分量
                w_i = W[i, :]
                
                # 计算当前估计的源信号
                s_i = w_i @ X_white
                
                # 计算复杂度的梯度（使用时间可预测性作为目标）
                gradient = self._compute_gradient(s_i, X_white)
                
                # 更新权重
                w_i_new = gradient
                
                # 去相关（与其他已提取的分量正交化）
                for j in range(i):
                    w_i_new -= (w_i_new @ W[j, :].T) * W[j, :]
                
                # 归一化
                w_i_new = w_i_new / np.linalg.norm(w_i_new)
                
                W[i, :] = w_i_new
            
            # 检查收敛
            change = np.max(np.abs(W - W_old))
            if change < self.tol:
                print(f"算法在第 {iteration+1} 次迭代后收敛")
                break
        
        # 保存解混矩阵
        self.W = W @ whitening_matrix
        
        # 计算混合矩阵（W的逆）
        self.mixing_matrix = np.linalg.pinv(self.W)
        
        # 计算分离的源信号
        S = self.W @ X
        
        return S
    
    def _whiten(self, X):
        """
        对信号进行白化处理
        
        参数:
            X: 输入信号 (n_channels, n_samples)
            
        返回:
            X_white: 白化后的信号
            whitening_matrix: 白化矩阵
        """
        # 中心化
        X_centered = X - np.mean(X, axis=1, keepdims=True)
        
        # 计算协方差矩阵
        cov = (X_centered @ X_centered.T) / X.shape[1]
        
        # 特征值分解
        eigenvalues, eigenvectors = eigh(cov)
        
        # 按特征值降序排序
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # 构造白化矩阵
        # 添加小的正则化项避免数值问题
        D = np.diag(1.0 / np.sqrt(eigenvalues + 1e-10))
        whitening_matrix = D @ eigenvectors.T
        
        # 白化信号
        X_white = whitening_matrix @ X_centered
        
        return X_white, whitening_matrix
    
    def _compute_gradient(self, s, X):
        """
        计算复杂度目标函数的梯度
        使用时间可预测性作为复杂度度量
        
        参数:
            s: 当前估计的源信号
            X: 白化后的混合信号
            
        返回:
            gradient: 梯度向量
        """
        n_samples = len(s)
        
        # 计算时间延迟版本
        s_delayed = np.concatenate([[0], s[:-1]])
        
        # 时间可预测性：E[s(t) * s(t-1)]
        predictability = s * s_delayed
        
        # 非线性函数（用于增强稀疏性）
        g = np.tanh(s)
        g_prime = 1 - np.tanh(s)**2
        
        # 计算梯度
        # 结合时间结构和稀疏性
        gradient = (X @ predictability.T) / n_samples + \
                  (X @ g.T) / n_samples
        
        return gradient
    
    def transform(self, X):
        """
        使用学习到的解混矩阵分离信号
        
        参数:
            X: 混合信号 (n_channels, n_samples)
            
        返回:
            S: 分离的信号 (n_sources, n_samples)
        """
        if self.W is None:
            raise ValueError("需要先调用fit方法")
            
        return self.W @ X
    
    def fit_transform(self, X):
        """
        拟合并转换
        
        参数:
            X: 混合信号 (n_channels, n_samples)
            
        返回:
            S: 分离的信号 (n_sources, n_samples)
        """
        return self.fit(X)
    
    def plot_separated_signals(self, S, fs=240, save_path=None):
        """
        绘制分离的信号及其频谱
        
        参数:
            S: 分离的信号 (n_sources, n_samples)
            fs: 采样频率
            save_path: 保存路径（可选）
        """
        n_sources = S.shape[0]
        n_show = min(n_sources, 4)  # 最多显示4个
        
        fig, axes = plt.subplots(n_show, 2, figsize=(12, 3*n_show))
        if n_show == 1:
            axes = axes.reshape(1, -1)
        
        for i in range(n_show):
            # 时域信号
            t = np.arange(S.shape[1]) / fs
            axes[i, 0].plot(t, S[i, :], 'b-', linewidth=0.5)
            axes[i, 0].set_ylabel(f'模态 {i+1}')
            axes[i, 0].grid(True, alpha=0.3)
            if i == n_show - 1:
                axes[i, 0].set_xlabel('时间 (s)')
            
            # 频谱
            freqs, psd = scipy_signal.periodogram(S[i, :], fs=fs)
            axes[i, 1].semilogy(freqs, psd, 'r-', linewidth=1)
            axes[i, 1].set_xlim([0, 50])  # 显示0-50Hz
            axes[i, 1].grid(True, alpha=0.3)
            if i == n_show - 1:
                axes[i, 1].set_xlabel('频率 (Hz)')
            axes[i, 1].set_ylabel('功率谱密度')
        
        axes[0, 0].set_title('分离的模态信号')
        axes[0, 1].set_title('功率谱密度')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
    
    def get_mixing_matrix(self):
        """
        获取混合矩阵（W的逆）
        
        返回:
            mixing_matrix: 混合矩阵
        """
        if self.mixing_matrix is None:
            if self.W is not None:
                self.mixing_matrix = np.linalg.pinv(self.W)
            else:
                raise ValueError("需要先调用fit方法")
                
        return self.mixing_matrix