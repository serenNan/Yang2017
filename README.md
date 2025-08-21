# Yang2017 论文复现 - 基于相位的视频振动模态识别

本项目复现了论文 "Blind identification of full-field vibration modes from video measurements with phase-based video motion magnification" (Yang et al., 2017) 的核心方法。

## 项目概述

该方法通过视频测量，无需在结构表面放置标记点，即可自动识别结构的振动模态参数（频率、阻尼比和振型）。

## 主要特点

- 使用Complex Steerable Pyramid进行多尺度分解
- 基于相位的运动提取
- PCA降维处理高维像素数据
- Complexity Pursuit盲源分离算法分离模态
- 自动估计模态参数

## 安装依赖

```bash
pip install -r requirements.txt
```

## 项目结构

```
Yang2017/
├── requirements.txt          # 项目依赖
├── README.md                 # 项目说明
├── src/
│   ├── __init__.py
│   ├── steerable_pyramid.py # Complex Steerable Pyramid实现
│   ├── phase_extraction.py  # 相位提取模块
│   ├── pca_analysis.py      # PCA降维
│   ├── complexity_pursuit.py # CP盲源分离算法
│   ├── modal_analysis.py    # 模态参数估计
│   └── video_processor.py   # 视频处理主模块
├── example.py               # 示例脚本
└── data/                    # 数据文件夹（用户提供）
```

## 使用方法

```python
from src.video_processor import VideoModalAnalysis

# 初始化分析器
analyzer = VideoModalAnalysis()

# 加载视频
analyzer.load_video('your_video.mp4')

# 执行模态分析
frequencies, damping_ratios, mode_shapes = analyzer.analyze()
```

## 算法流程

1. **视频预处理**: 读取视频，降采样
2. **多尺度分解**: 使用Complex Steerable Pyramid分解每帧
3. **相位提取**: 提取局部相位信息
4. **PCA降维**: 处理高维像素数据
5. **盲源分离**: 使用Complexity Pursuit算法分离模态
6. **参数估计**: 估计频率和阻尼比

## 参考文献

Yang, Y., Dorn, C., Mancini, T., Talken, Z., Kenyon, G., Farrar, C., & Mascareñas, D. (2017). 
Blind identification of full-field vibration modes from video measurements with phase-based video motion magnification. 
Mechanical Systems and Signal Processing, 85, 567-590.