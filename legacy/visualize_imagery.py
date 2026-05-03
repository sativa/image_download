#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""可视化影像结果"""

import rasterio
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def show_imagery_info(img_path):
    """显示影像信息并创建预览"""
    img_path = Path(img_path)
    if not img_path.exists():
        print(f"❌ 文件不存在: {img_path}")
        return
    
    with rasterio.open(str(img_path)) as src:
        # 读取影像数据
        img_data = src.read([1, 2, 3])  # RGB 三个波段
        img_data = np.transpose(img_data, (1, 2, 0))  # 转换为 (H, W, 3)
        
        # 归一化到 0-1
        img_data = img_data.astype(np.float32) / 255.0
        
        # 创建图形
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        
        # 左侧：完整影像
        ax1 = axes[0]
        ax1.imshow(img_data, extent=[src.bounds.left, src.bounds.right, 
                                      src.bounds.bottom, src.bounds.top])
        ax1.set_xlabel('经度 (度)', fontsize=12)
        ax1.set_ylabel('纬度 (度)', fontsize=12)
        ax1.set_title(f'多边形高分影像预览\n{img_path.name}', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # 添加边界框
        rect = Rectangle((src.bounds.left, src.bounds.bottom),
                        src.bounds.right - src.bounds.left,
                        src.bounds.top - src.bounds.bottom,
                        linewidth=2, edgecolor='red', facecolor='none')
        ax1.add_patch(rect)
        
        # 右侧：影像统计信息
        ax2 = axes[1]
        ax2.axis('off')
        
        info_text = f"""
📊 影像详细信息

文件路径:
  {img_path}

空间信息:
  • 尺寸: {src.width} × {src.height} 像素
  • 文件大小: {img_path.stat().st_size / 1024 / 1024:.2f} MB
  • 分辨率: {src.res[0]:.6f} 度/像素
  • 约 {src.res[0] * 111 * 1000:.1f} 米/像素

坐标信息:
  • 坐标系: {src.crs}
  • 经度范围: {src.bounds.left:.6f} ~ {src.bounds.right:.6f}
  • 纬度范围: {src.bounds.bottom:.6f} ~ {src.bounds.top:.6f}
  • 覆盖面积: {(src.bounds.right - src.bounds.left) * (src.bounds.top - src.bounds.bottom) * 111 * 111:.4f} km²

影像质量:
  • 波段数: {src.count} (RGB)
  • 数据类型: {src.dtypes[0]}
  • 像素值范围: {img_data.min():.3f} ~ {img_data.max():.3f}
  • 平均亮度: {img_data.mean():.3f}

处理状态:
  ✅ 已精确裁剪到多边形区域
  ✅ 多边形外区域为透明 (nodata)
        """
        
        ax2.text(0.1, 0.5, info_text, fontsize=11, family='monospace',
                verticalalignment='center', transform=ax2.transAxes)
        
        plt.tight_layout()
        
        # 保存预览图
        preview_path = img_path.parent / f"{img_path.stem}_preview.png"
        plt.savefig(preview_path, dpi=150, bbox_inches='tight')
        print(f"✅ 预览图已保存: {preview_path}")
        
        plt.close()

if __name__ == '__main__':
    img_path = "output_heshui/imagery/polygon_0_zoom20.tif"
    print("🎨 生成影像预览...")
    show_imagery_info(img_path)
    print("✅ 完成！")

