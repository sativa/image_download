#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""验证多边形与影像的关联"""

import geopandas as gpd
import rasterio
from pathlib import Path

def verify_polygon_imagery(gpkg_path, img_dir=None):
    """验证多边形与影像的关联"""
    gdf = gpd.read_file(gpkg_path)
    
    print("=" * 70)
    print("📊 多边形与影像关联验证")
    print("=" * 70)
    
    with_image = gdf[gdf['image_path'].notna()]
    print(f"\n总多边形数: {len(gdf)}")
    print(f"关联影像的多边形: {len(with_image)}")
    
    for idx, row in with_image.iterrows():
        print(f"\n多边形 {idx}:")
        print(f"  面积: {row.get('Area_km2', 'N/A'):.4f} km²" if 'Area_km2' in row else "  面积: N/A")
        print(f"  Change_Type: {row.get('Change_Type', 'N/A')}")
        print(f"  影像路径: {row['image_path']}")
        
        img_path_str = row['image_path']
        if Path(img_path_str).is_absolute():
            img_path = Path(img_path_str)
        else:
            # 相对路径，尝试多个可能的位置
            if img_dir:
                img_path = Path(img_dir) / Path(img_path_str).name
            else:
                img_path = Path(img_path_str)
            
            # 如果还是不存在，尝试相对于GPKG文件的位置
            if not img_path.exists():
                gpkg_dir = Path(gpkg_path).parent
                img_path = gpkg_dir / Path(img_path_str).name
        
        if img_path.exists():
            with rasterio.open(str(img_path)) as src:
                img_data = src.read(1)
                valid_pixels = (img_data != 0).sum()
                total_pixels = img_data.size
                
                # 计算有效区域面积
                # 使用更精确的方法计算每个像素的面积
                import math
                center_lat = (src.bounds.top + src.bounds.bottom) / 2.0
                pixel_res_x_deg = abs(src.bounds.right - src.bounds.left) / src.width
                pixel_res_y_deg = abs(src.bounds.top - src.bounds.bottom) / src.height
                
                # 转换为米（考虑纬度）
                meters_per_deg_lat = 111320.0
                meters_per_deg_lon = 111320.0 * math.cos(math.radians(center_lat))
                
                pixel_area_m2 = (pixel_res_x_deg * meters_per_deg_lon) * (pixel_res_y_deg * meters_per_deg_lat)
                valid_area_km2 = valid_pixels * pixel_area_m2 / 1000000.0
                
                print(f"  影像尺寸: {src.width} x {src.height} 像素")
                print(f"  有效像素: {valid_pixels}/{total_pixels} ({valid_pixels/total_pixels*100:.1f}%)")
                print(f"  有效区域: {valid_area_km2:.4f} km²")
                
                if 'Area_km2' in row:
                    area_ratio = valid_area_km2 / row['Area_km2']
                    print(f"  面积比: {area_ratio:.2f} (影像/多边形)")
                    if 0.8 < area_ratio < 1.5:
                        print(f"  ✅ 影像区域与多边形匹配")
                    else:
                        print(f"  ⚠️  面积差异较大")
        else:
            print(f"  ❌ 影像文件不存在: {img_path}")

if __name__ == '__main__':
    import sys
    gpkg_path = sys.argv[1] if len(sys.argv) > 1 else "output_heshui/imagery/soil_changes_only_类_with_imagery.gpkg"
    verify_polygon_imagery(gpkg_path, img_dir="output_heshui/imagery")

