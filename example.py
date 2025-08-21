"""
示例脚本 - 展示如何使用Yang2017方法进行视频振动模态分析
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from src.video_processor import VideoModalAnalysis
import os


def generate_synthetic_video(n_frames=400, height=100, width=100, fps=240):
    """
    生成合成测试视频（包含多个振动模态）
    
    参数:
        n_frames: 帧数
        height: 高度
        width: 宽度
        fps: 帧率
        
    返回:
        frames: 视频帧数组
    """
    print("生成合成测试视频...")
    
    # 时间向量
    t = np.arange(n_frames) / fps
    
    # 定义模态参数
    frequencies = [6.0, 18.0, 25.0]  # Hz
    damping_ratios = [0.003, 0.002, 0.001]  # 阻尼比
    
    # 创建空间模式（模拟不同的振型）
    x = np.linspace(0, 1, width)
    y = np.linspace(0, 1, height)
    X, Y = np.meshgrid(x, y)
    
    # 模态振型
    mode_shapes = []
    mode_shapes.append(np.sin(np.pi * X) * np.sin(np.pi * Y))  # 第一阶模态
    mode_shapes.append(np.sin(2 * np.pi * X) * np.sin(np.pi * Y))  # 第二阶模态
    mode_shapes.append(np.sin(np.pi * X) * np.sin(2 * np.pi * Y))  # 第三阶模态
    
    # 生成视频帧
    frames = []
    for i in range(n_frames):
        frame = np.ones((height, width)) * 0.5  # 背景
        
        # 添加每个模态的贡献
        for j, (freq, damping, shape) in enumerate(zip(frequencies, damping_ratios, mode_shapes)):
            # 模态响应（包含阻尼）
            omega = 2 * np.pi * freq
            amplitude = np.exp(-damping * omega * t[i]) * np.cos(omega * t[i])
            
            # 添加到帧
            frame += 0.1 * amplitude * shape
        
        # 添加少量噪声
        noise = np.random.normal(0, 0.01, (height, width))
        frame += noise
        
        # 限制范围在[0, 1]
        frame = np.clip(frame, 0, 1)
        
        frames.append(frame)
    
    return np.array(frames)


def example_with_synthetic_data():
    """使用合成数据的示例"""
    
    print("\n" + "="*60)
    print(" Yang2017 视频振动模态分析 - 合成数据示例 ")
    print("="*60 + "\n")
    
    # 生成合成视频
    frames = generate_synthetic_video(n_frames=400, height=100, width=100, fps=240)
    print(f"生成的视频: {frames.shape[0]} 帧, 尺寸 {frames.shape[1]}x{frames.shape[2]}")
    
    # 创建分析器
    analyzer = VideoModalAnalysis(n_scales=5, n_orientations=4, fps=240)
    
    # 加载帧数据
    analyzer.load_frames(frames)
    
    # 执行分析
    results = analyzer.analyze(
        scale=1,  # 使用第一个尺度
        filter_phases=True,  # 进行带通滤波
        low_freq=1.0,  # 低频截止 1 Hz
        high_freq=50.0  # 高频截止 50 Hz
    )
    
    # 打印结果摘要
    analyzer.print_summary()
    
    # 可视化结果
    print("\n生成可视化图表...")
    analyzer.visualize_results(save_dir='results')
    
    # 保存结果
    analyzer.save_results('results/analysis_results.npz')
    
    return analyzer, results


def example_with_video_file(video_path):
    """使用实际视频文件的示例"""
    
    print("\n" + "="*60)
    print(" Yang2017 视频振动模态分析 - 视频文件示例 ")
    print("="*60 + "\n")
    
    # 检查文件是否存在
    if not os.path.exists(video_path):
        print(f"错误: 找不到视频文件 {video_path}")
        print("请提供有效的视频文件路径")
        return None, None
    
    # 创建分析器
    analyzer = VideoModalAnalysis(n_scales=5, n_orientations=4, fps=240)
    
    # 加载视频
    analyzer.load_video(
        video_path,
        max_frames=400,  # 限制帧数以加快处理
        downsample_factor=2  # 空间下采样因子
    )
    
    # 执行分析
    results = analyzer.analyze(
        scale=1,
        filter_phases=True,
        low_freq=1.0,
        high_freq=50.0
    )
    
    # 打印结果摘要
    analyzer.print_summary()
    
    # 可视化结果
    print("\n生成可视化图表...")
    analyzer.visualize_results(save_dir='results')
    
    # 保存结果
    analyzer.save_results('results/video_analysis_results.npz')
    
    return analyzer, results


def example_with_custom_data():
    """使用自定义数据的示例"""
    
    print("\n" + "="*60)
    print(" Yang2017 视频振动模态分析 - 自定义数据示例 ")
    print("="*60 + "\n")
    
    # 假设你有自己采集的数据
    # 这里我们创建一个简单的示例
    
    # 创建包含单个振动模式的简单数据
    n_frames = 300
    height, width = 50, 50
    fps = 100
    
    t = np.arange(n_frames) / fps
    x = np.linspace(0, 1, width)
    y = np.linspace(0, 1, height)
    X, Y = np.meshgrid(x, y)
    
    # 单一频率振动
    frequency = 10.0  # Hz
    mode_shape = np.sin(np.pi * X) * np.sin(np.pi * Y)
    
    frames = []
    for i in range(n_frames):
        # 时变振幅
        amplitude = np.sin(2 * np.pi * frequency * t[i])
        # 生成帧
        frame = 0.5 + 0.2 * amplitude * mode_shape
        frames.append(frame)
    
    frames = np.array(frames)
    
    # 创建分析器
    analyzer = VideoModalAnalysis(fps=fps)
    
    # 加载数据
    analyzer.load_frames(frames)
    
    # 执行分析
    results = analyzer.analyze(scale=1)
    
    # 打印结果
    analyzer.print_summary()
    
    return analyzer, results


if __name__ == "__main__":
    # 创建结果目录
    os.makedirs('results', exist_ok=True)
    
    # 运行示例
    print("请选择示例:")
    print("1. 使用合成数据")
    print("2. 使用视频文件")
    print("3. 使用自定义数据")
    
    choice = input("\n请输入选项 (1/2/3): ").strip()
    
    if choice == '1':
        analyzer, results = example_with_synthetic_data()
    elif choice == '2':
        video_path = input("请输入视频文件路径: ").strip()
        if not video_path:
            print("使用默认测试视频...")
            # 这里可以设置一个默认的测试视频路径
            video_path = "test_video.mp4"
        analyzer, results = example_with_video_file(video_path)
    elif choice == '3':
        analyzer, results = example_with_custom_data()
    else:
        print("无效选项")
        analyzer, results = None, None
    
    if analyzer is not None:
        print("\n分析完成！")
        print("结果已保存在 'results' 目录中")
        
        # 显示一些统计信息
        if results is not None:
            print(f"\n检测到的模态数: {len(results['frequencies'])}")
            print(f"频率范围: {results['frequencies'].min():.2f} - {results['frequencies'].max():.2f} Hz")
            print(f"平均阻尼比: {results['damping_ratios'].mean():.2f}%")