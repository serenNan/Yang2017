"""
快速测试脚本 - 验证系统是否正常工作
"""

import numpy as np
import sys
import os

# 添加src目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.video_processor import VideoModalAnalysis


def quick_test():
    """快速测试基本功能"""
    
    print("\n" + "="*50)
    print(" 快速功能测试 ")
    print("="*50 + "\n")
    
    # 创建简单的测试数据
    print("1. 生成测试数据...")
    n_frames = 100
    height, width = 32, 32
    fps = 100
    
    # 生成包含两个频率成分的信号
    t = np.arange(n_frames) / fps
    frames = []
    
    for i in range(n_frames):
        # 创建一个简单的振动模式
        frame = np.zeros((height, width))
        
        # 添加两个不同频率的振动
        x = np.linspace(0, 1, width)
        y = np.linspace(0, 1, height)
        X, Y = np.meshgrid(x, y)
        
        # 5 Hz 振动
        frame += 0.1 * np.sin(2 * np.pi * 5 * t[i]) * np.sin(np.pi * X) * np.sin(np.pi * Y)
        
        # 10 Hz 振动
        frame += 0.05 * np.sin(2 * np.pi * 10 * t[i]) * np.sin(2 * np.pi * X) * np.sin(np.pi * Y)
        
        # 添加背景
        frame += 0.5
        
        frames.append(frame)
    
    frames = np.array(frames)
    print(f"   生成的数据形状: {frames.shape}")
    
    # 创建分析器
    print("\n2. 初始化分析器...")
    analyzer = VideoModalAnalysis(n_scales=3, n_orientations=4, fps=fps)
    print("   分析器初始化成功")
    
    # 加载数据
    print("\n3. 加载数据...")
    analyzer.load_frames(frames)
    print("   数据加载成功")
    
    # 执行分析
    print("\n4. 执行模态分析...")
    try:
        results = analyzer.analyze(
            scale=1,
            filter_phases=True,
            low_freq=1.0,
            high_freq=20.0
        )
        print("   分析完成!")
        
        # 显示结果
        print("\n5. 分析结果:")
        print(f"   检测到 {len(results['frequencies'])} 个模态")
        
        for i, (freq, damp) in enumerate(zip(results['frequencies'], results['damping_ratios'])):
            print(f"   模态 {i+1}: 频率 = {freq:.2f} Hz, 阻尼比 = {damp:.2f}%")
        
        print("\n✓ 测试通过！所有功能正常工作。")
        
    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        return False
    
    return True


if __name__ == "__main__":
    success = quick_test()
    
    if success:
        print("\n" + "="*50)
        print(" 系统就绪 ")
        print("="*50)
        print("\n你可以运行以下命令开始使用:")
        print("  python example.py")
        print("\n或者在Python中:")
        print("  from src.video_processor import VideoModalAnalysis")
        print("  analyzer = VideoModalAnalysis()")
        print("  # 加载你的视频数据...")
    else:
        print("\n请检查错误信息并修复问题。")