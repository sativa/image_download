#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
下载高分卫星影像并与多边形关联

功能：
1. 读取 GPKG 文件中的多边形
2. 自动选择最快的影像源（ESRI、Microsoft Bing、Google）
3. 为每个多边形下载高分卫星影像（22级及以上）
4. 将影像保存为 GeoTIFF 并与多边形关联
"""

import os
import sys
import math
import time
import json
import signal
import atexit
from pathlib import Path
from typing import Tuple, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock, Event
import requests
from io import BytesIO
from PIL import Image
import numpy as np
import geopandas as gpd
from shapely.geometry import box
import rasterio
from rasterio.transform import from_bounds
from rasterio.crs import CRS
from rasterio.warp import calculate_default_transform, reproject, Resampling


def deg2num(lat_deg: float, lon_deg: float, zoom: int) -> Tuple[int, int]:
    """
    将经纬度转换为瓦片坐标
    
    Args:
        lat_deg: 纬度（度）
        lon_deg: 经度（度）
        zoom: 缩放级别
        
    Returns:
        (x, y): 瓦片坐标
    """
    lat_rad = math.radians(lat_deg)
    n = 2.0 ** zoom
    xtile = int((lon_deg + 180.0) / 360.0 * n)
    ytile = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return (xtile, ytile)


def num2deg(xtile: int, ytile: int, zoom: int) -> Tuple[float, float]:
    """
    将瓦片坐标转换为左上角经纬度
    
    Args:
        xtile: 瓦片 X 坐标
        ytile: 瓦片 Y 坐标
        zoom: 缩放级别
        
    Returns:
        (lon, lat): 左上角经纬度
    """
    n = 2.0 ** zoom
    lon_deg = xtile / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * ytile / n)))
    lat_deg = math.degrees(lat_rad)
    return (lon_deg, lat_deg)


def get_tile_bounds(xtile: int, ytile: int, zoom: int) -> Tuple[float, float, float, float]:
    """
    获取瓦片的边界（左、下、右、上）
    
    Args:
        xtile: 瓦片 X 坐标
        ytile: 瓦片 Y 坐标
        zoom: 缩放级别
        
    Returns:
        (left, bottom, right, top): 边界坐标
    """
    left, top = num2deg(xtile, ytile, zoom)
    right, bottom = num2deg(xtile + 1, ytile + 1, zoom)
    return (left, bottom, right, top)


# 影像源配置
IMAGERY_SOURCES = {
    'esri': {
        'name': 'ESRI World Imagery',
        'url_template': 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        'headers': {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        },
        'priority': 1  # 最高优先级（通常最快）
    },
    'bing': {
        'name': 'Microsoft Bing Maps',
        'url_template': 'https://ecn.t{server}.tiles.virtualearth.net/tiles/a{quadkey}.jpeg?g=1',
        'headers': {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        },
        'priority': 2,
        'quadkey_func': True  # 需要特殊处理
    },
    'google': {
        'name': 'Google Maps',
        'url_template': 'https://mt{server}.google.com/vt/lyrs=s&x={x}&y={y}&z={z}',
        'headers': {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Referer': 'https://www.google.com/maps/'
        },
        'priority': 3,
        'servers': ['0', '1', '2', '3']
    }
}

# 全局变量：当前使用的影像源
_current_source = None

# 全局Session对象（连接池）
_session_pool = {}
_session_lock = Lock()

# 全局停止事件（用于优雅退出）
_stop_event = Event()
_shutdown_in_progress = False


def quadkey_from_tile(xtile: int, ytile: int, zoom: int) -> str:
    """
    将瓦片坐标转换为 Bing Maps 的 QuadKey
    
    Args:
        xtile: 瓦片 X 坐标
        ytile: 瓦片 Y 坐标
        zoom: 缩放级别
        
    Returns:
        QuadKey 字符串
    """
    quadkey = ''
    for i in range(zoom, 0, -1):
        digit = 0
        mask = 1 << (i - 1)
        if (xtile & mask) != 0:
            digit += 1
        if (ytile & mask) != 0:
            digit += 2
        quadkey += str(digit)
    return quadkey


def get_tile_url(xtile: int, ytile: int, zoom: int, source_name: str) -> str:
    """
    获取指定影像源的瓦片 URL
    
    Args:
        xtile: 瓦片 X 坐标
        ytile: 瓦片 Y 坐标
        zoom: 缩放级别
        source_name: 影像源名称
        
    Returns:
        URL 字符串
    """
    source = IMAGERY_SOURCES[source_name]
    template = source['url_template']
    
    if source_name == 'esri':
        # ESRI 使用 XYZ 格式 (z/y/x)，需要翻转Y坐标
        # ESRI的Y坐标是从上往下的，而我们的ytile是从下往上的
        max_y = (2 ** zoom) - 1
        y_flipped = max_y - ytile
        return template.format(z=zoom, y=y_flipped, x=xtile)
    elif source_name == 'bing':
        # Bing 使用 QuadKey
        quadkey = quadkey_from_tile(xtile, ytile, zoom)
        server = (xtile + ytile) % 4
        return template.format(server=server, quadkey=quadkey)
    elif source_name == 'google':
        # Google 使用多个服务器
        server = (xtile + ytile) % len(source.get('servers', ['0']))
        return template.format(server=server, x=xtile, y=ytile, z=zoom)
    else:
        raise ValueError(f"未知的影像源: {source_name}")


def calculate_resolution_meters(zoom: int, latitude: float) -> float:
    """
    计算指定缩放级别在给定纬度处的分辨率（米/像素）
    
    Args:
        zoom: 缩放级别
        latitude: 纬度（度）
        
    Returns:
        分辨率（米/像素）
    """
    # 地球周长（米）
    earth_circumference = 40075017.0
    # 每个缩放级别的瓦片数量
    num_tiles = 2 ** zoom
    # 每个瓦片的像素数
    pixels_per_tile = 256
    # 每个缩放级别的总像素数
    total_pixels = num_tiles * pixels_per_tile
    # 在赤道的分辨率
    resolution_equator = earth_circumference / total_pixels
    # 在指定纬度的分辨率（考虑纬度因子）
    resolution = resolution_equator * math.cos(math.radians(latitude))
    return resolution


def get_zoom_for_resolution(target_resolution_m: float, latitude: float, max_zoom: int = 18) -> int:
    """
    根据目标分辨率计算合适的缩放级别
    
    Args:
        target_resolution_m: 目标分辨率（米/像素），例如 5.0 表示优于5米
        latitude: 纬度（度）
        max_zoom: 最大缩放级别
        
    Returns:
        合适的缩放级别
    """
    for zoom in range(max_zoom, 10, -1):  # 从高到低查找
        resolution = calculate_resolution_meters(zoom, latitude)
        if resolution <= target_resolution_m:
            return zoom
    return 10  # 如果都不满足，返回最小级别


def is_valid_imagery_tile(img: Image.Image, min_std: float = 1.0) -> bool:
    """
    检测瓦片是否是有效的影像（不是空白或错误页面）
    
    Args:
        img: PIL Image 对象
        min_std: 最小标准差阈值（用于判断是否有足够的颜色变化）
        
    Returns:
        True 如果是有效影像，False 如果是空白或无效
    """
    if img is None:
        return False
    
    # 转换为numpy数组
    arr = np.array(img)
    
    # 检查尺寸
    if arr.shape[0] != 256 or arr.shape[1] != 256:
        return False
    
    # 如果是RGB格式
    if len(arr.shape) == 3 and arr.shape[2] == 3:
        # 计算每个波段的标准差
        r_std = arr[:, :, 0].std()
        g_std = arr[:, :, 1].std()
        b_std = arr[:, :, 2].std()
        
        # 只过滤完全没有变化的瓦片（标准差<1.0）
        # 黄土高原地区可能颜色比较单一，但仍然是有效影像
        if r_std < min_std and g_std < min_std and b_std < min_std:
            return False
        
        # 检查是否是纯黑或纯白（完全没有变化）
        r_mean = arr[:, :, 0].mean()
        g_mean = arr[:, :, 1].mean()
        b_mean = arr[:, :, 2].mean()
        
        # 只过滤完全纯色的瓦片
        if (r_mean < 1 and g_mean < 1 and b_mean < 1) or \
           (r_mean > 254 and g_mean > 254 and b_mean > 254):
            if r_std < 0.5 and g_std < 0.5 and b_std < 0.5:
                return False
        
        return True
    else:
        # 单波段或其他格式
        std = arr.std()
        mean = arr.mean()
        if std < min_std:
            return False
        if (mean < 1 or mean > 254) and std < 0.5:
            return False
        return True


def get_session(source_name: str):
    """获取或创建Session对象（连接池）"""
    global _session_pool, _session_lock
    
    with _session_lock:
        if source_name not in _session_pool:
            session = requests.Session()
            # 配置连接池
            adapter = requests.adapters.HTTPAdapter(
                pool_connections=10,  # 连接池大小
                pool_maxsize=20,     # 最大连接数
                max_retries=0        # 禁用自动重试（我们自己处理）
            )
            session.mount('http://', adapter)
            session.mount('https://', adapter)
            _session_pool[source_name] = session
        return _session_pool[source_name]


def cleanup_sessions():
    """清理所有Session连接池（优雅退出）"""
    global _session_pool, _session_lock, _shutdown_in_progress
    
    if _shutdown_in_progress:
        return  # 避免重复清理
    
    _shutdown_in_progress = True
    
    with _session_lock:
        print("\n🔄 正在关闭所有HTTP连接...")
        for source_name, session in _session_pool.items():
            try:
                session.close()
                print(f"  ✅ 已关闭 {source_name} 的连接")
            except Exception as e:
                print(f"  ⚠️  关闭 {source_name} 连接时出错: {e}")
        _session_pool.clear()
        print("✅ 所有连接已关闭")


def signal_handler(signum, frame):
    """处理中断信号（Ctrl+C）"""
    global _stop_event
    
    print("\n\n⚠️  收到中断信号，正在停止下载...")
    _stop_event.set()  # 设置停止标志
    cleanup_sessions()  # 清理连接
    print("👋 程序已退出")
    sys.exit(0)


def download_tile(xtile: int, ytile: int, zoom: int, 
                 source_name: str = None, retry: int = 2, delay: float = 0.0) -> Optional[Image.Image]:
    """
    下载单个卫星影像瓦片（使用连接池优化）
    
    Args:
        xtile: 瓦片 X 坐标
        ytile: 瓦片 Y 坐标
        zoom: 缩放级别
        source_name: 影像源名称（如果为 None，使用全局当前源）
        retry: 重试次数
        delay: 请求延迟（秒）
        
    Returns:
        PIL Image 对象或 None
    """
    global _current_source, _stop_event
    
    # 检查是否收到停止信号
    if _stop_event.is_set():
        return None
    
    if source_name is None:
        source_name = _current_source or 'esri'
    
    source = IMAGERY_SOURCES[source_name]
    url = get_tile_url(xtile, ytile, zoom, source_name)
    
    # 使用Session连接池
    session = get_session(source_name)
    
    for attempt in range(retry):
        # 再次检查停止信号
        if _stop_event.is_set():
            return None
            
        try:
            if delay > 0:
                time.sleep(delay)
            # 使用Session而不是requests.get，复用连接
            response = session.get(url, headers=source['headers'], timeout=5)  # 减少超时时间
            if response.status_code == 200:
                # 检查是否是有效的图片
                content = response.content
                if len(content) > 100:  # 确保不是错误页面
                    try:
                        img = Image.open(BytesIO(content))
                        # 验证图片尺寸和格式
                        if img.size[0] == 256 and img.size[1] == 256:
                            # 转换为RGB（某些图片可能是RGBA或其他格式）
                            if img.mode != 'RGB':
                                img = img.convert('RGB')
                            # 直接返回，不做质量检测
                            return img
                    except Exception as e:
                        # 如果图片无法打开，可能是错误响应
                        pass
        except Exception:
            if attempt < retry - 1:
                time.sleep(delay * (attempt + 1))
                continue
    
    return None


def test_source_speed(xtile: int, ytile: int, zoom: int, source_name: str, num_tests: int = 3) -> Optional[float]:
    """
    测试影像源的下载速度
    
    Args:
        xtile: 测试瓦片 X 坐标
        ytile: 测试瓦片 Y 坐标
        zoom: 缩放级别
        source_name: 影像源名称
        num_tests: 测试次数
        
    Returns:
        平均响应时间（秒），如果失败返回 None
    """
    times = []
    for _ in range(num_tests):
        try:
            start = time.time()
            img = download_tile(xtile, ytile, zoom, source_name, retry=1, delay=0)
            if img:
                elapsed = time.time() - start
                times.append(elapsed)
        except:
            pass
    
    if times:
        return sum(times) / len(times)
    return None


def select_fastest_source(xtile: int, ytile: int, zoom: int) -> str:
    """
    自动选择最快的可用影像源
    
    Args:
        xtile: 测试瓦片 X 坐标
        ytile: 测试瓦片 Y 坐标
        zoom: 缩放级别
        
    Returns:
        最快的影像源名称
    """
    print("  🔍 测试各影像源速度...")
    results = {}
    
    # 按优先级顺序测试
    sources_by_priority = sorted(IMAGERY_SOURCES.items(), key=lambda x: x[1]['priority'])
    
    for source_name, source_info in sources_by_priority:
        print(f"    测试 {source_info['name']}...", end=' ')
        avg_time = test_source_speed(xtile, ytile, zoom, source_name)
        if avg_time:
            results[source_name] = avg_time
            print(f"✅ {avg_time:.2f}秒")
        else:
            print("❌ 不可用")
    
    if not results:
        print("  ⚠️  所有影像源都不可用，使用 ESRI 作为默认源")
        return 'esri'
    
    # 选择最快的
    fastest = min(results.items(), key=lambda x: x[1])
    print(f"  ✅ 选择最快的源: {IMAGERY_SOURCES[fastest[0]]['name']} ({fastest[1]:.2f}秒)")
    return fastest[0]


def get_tiles_for_bounds(bounds: Tuple[float, float, float, float], 
                         zoom: int) -> List[Tuple[int, int]]:
    """
    获取覆盖指定边界的所有瓦片坐标
    
    Args:
        bounds: (minx, miny, maxx, maxy) 边界坐标
        zoom: 缩放级别
        
    Returns:
        瓦片坐标列表 [(x, y), ...]
    """
    minx, miny, maxx, maxy = bounds
    
    # 计算覆盖边界的瓦片范围
    # 注意：在瓦片坐标系中，y 坐标从北到南递增（纬度越高，y 越小）
    x_min, y_top = deg2num(maxy, minx, zoom)  # 左上角
    x_max, y_bottom = deg2num(miny, maxx, zoom)  # 右下角
    
    # 确保 y 坐标范围正确（y_top < y_bottom）
    y_min = min(y_top, y_bottom)
    y_max = max(y_top, y_bottom)
    
    tiles = []
    for x in range(x_min, x_max + 1):
        for y in range(y_min, y_max + 1):
            tiles.append((x, y))
    
    return tiles


def download_and_merge_tiles(bounds: Tuple[float, float, float, float],
                              zoom: int,
                              output_path: Optional[str] = None,
                              polygon_geometry=None,
                              min_resolution_m: float = 5.0,
                              auto_adjust_zoom: bool = True,
                              max_workers: int = 50) -> Optional[np.ndarray]:
    """
    下载并合并覆盖指定边界的所有瓦片，并裁剪到多边形区域
    
    Args:
        bounds: (minx, miny, maxx, maxy) 边界坐标
        zoom: 缩放级别
        output_path: 输出 GeoTIFF 路径（可选）
        polygon_geometry: 多边形几何对象（用于裁剪，可选）
        min_resolution_m: 最小分辨率要求（米/像素），默认5.0
        auto_adjust_zoom: 是否自动调整缩放级别以找到有效影像
        
    Returns:
        合并后的影像数组或 None
    """
    minx, miny, maxx, maxy = bounds
    center_lat = (miny + maxy) / 2.0
    
    # 自动调整缩放级别以确保分辨率优于要求
    if auto_adjust_zoom:
        optimal_zoom = get_zoom_for_resolution(min_resolution_m, center_lat, max_zoom=18)
        if optimal_zoom != zoom:
            print(f"  🔍 自动调整缩放级别: {zoom} -> {optimal_zoom} (目标分辨率: <{min_resolution_m}米/像素)")
            zoom = optimal_zoom
    
    # 获取所有需要的瓦片
    tiles = get_tiles_for_bounds(bounds, zoom)
    
    # 计算下载区域大小
    area_deg = (maxx - minx) * (maxy - miny)
    area_km2 = area_deg * 111 * 111
    
    print(f"  需要下载 {len(tiles)} 个瓦片 (zoom={zoom})")
    print(f"  下载区域大小: {area_km2:.4f} km²")
    
    # 如果区域太大，给出警告
    if area_km2 > 1.0:
        print(f"  ⚠️  警告: 下载区域较大 ({area_km2:.2f} km²)，可能需要较长时间")
    elif area_km2 > 0.1:
        print(f"  ℹ️  下载区域: {area_km2:.4f} km²")
    else:
        print(f"  ✅ 下载区域: {area_km2:.4f} km² (小区域)")
    
    if not tiles:
        return None
    
    # 计算瓦片范围
    x_coords = [t[0] for t in tiles]
    y_coords = [t[1] for t in tiles]
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)
    
    # 计算输出图像尺寸（每个瓦片 256x256）
    width = (x_max - x_min + 1) * 256
    height = (y_max - y_min + 1) * 256
    
    # 创建输出图像
    merged_image = Image.new('RGB', (width, height))
    
    # 自动选择最快的影像源（使用第一个瓦片进行测试）
    global _current_source
    if _current_source is None and len(tiles) > 0:
        test_tile = tiles[0]
        _current_source = select_fastest_source(test_tile[0], test_tile[1], zoom)
    
    source_name = _current_source or 'esri'
    print(f"  📡 使用影像源: {IMAGERY_SOURCES[source_name]['name']}")
    
    # 下载并合并瓦片
    downloaded = 0
    failed = 0
    blank_tiles = 0  # 空白瓦片计数
    
    # 使用多线程并发下载
    actual_workers = min(max_workers, len(tiles))  # 不超过瓦片数量
    print(f"  📥 并发下载瓦片 (线程数: {actual_workers})...")
    
    # 线程锁用于保护共享变量
    lock = Lock()
    tile_results = {}  # 存储下载结果: (xtile, ytile) -> Image
    
    def download_single_tile(tile_info):
        """下载单个瓦片的辅助函数"""
        xtile, ytile = tile_info
        img = download_tile(xtile, ytile, zoom, source_name)
        return (xtile, ytile, img)
    
    # 使用线程池并发下载
    executor = ThreadPoolExecutor(max_workers=actual_workers)
    try:
        # 提交所有下载任务
        future_to_tile = {executor.submit(download_single_tile, tile): tile for tile in tiles}
        
        # 处理完成的任务
        completed = 0
        for future in as_completed(future_to_tile):
            # 检查停止信号
            if _stop_event.is_set():
                print("\n⚠️  检测到停止信号，正在取消剩余任务...")
                executor.shutdown(wait=False, cancel_futures=True)
                break
                
            completed += 1
            try:
                xtile, ytile, img = future.result()
                if img:
                    tile_results[(xtile, ytile)] = img
                    with lock:
                        downloaded += 1
                else:
                    with lock:
                        failed += 1
            except Exception as e:
                with lock:
                    failed += 1
            
            # 显示进度
            if completed % 5 == 0 or completed == len(tiles):
                with lock:
                    print(f"    进度: {completed}/{len(tiles)} ({downloaded} 成功, {failed} 失败)")
    finally:
        # 确保线程池被正确关闭，等待所有线程完成
        executor.shutdown(wait=True)
        print("  ✅ 所有下载线程已完成")
    
    # 将下载的瓦片合并到图像中
    for (xtile, ytile), img in tile_results.items():
        x_offset = (xtile - x_min) * 256
        y_offset = (ytile - y_min) * 256
        merged_image.paste(img, (x_offset, y_offset))
    
    resolution = calculate_resolution_meters(zoom, center_lat)
    print(f"  完成: {downloaded}/{len(tiles)} 个瓦片下载成功 ({downloaded/len(tiles)*100:.1f}%)")
    print(f"  实际分辨率: {resolution:.2f} 米/像素 (缩放级别: {zoom})")
    
    if blank_tiles > 0:
        print(f"  ⚠️  警告: {blank_tiles} 个瓦片是空白或无效影像")
    
    if downloaded == 0:
        return None
    
    # 转换为 numpy 数组
    img_array = np.array(merged_image)
    
    # 计算完整瓦片区域的边界（用于创建临时 GeoTIFF）
    # 获取左上角瓦片的左上角坐标
    left_tile, top_tile = deg2num(maxy, minx, zoom)
    right_tile, bottom_tile = deg2num(miny, maxx, zoom)
    
    # 计算完整瓦片区域的边界
    tile_left, tile_top = num2deg(x_min, y_min, zoom)
    tile_right, tile_bottom = num2deg(x_max + 1, y_max + 1, zoom)
    
    # 裁剪到边界框（先裁剪到边界框，后续再用多边形裁剪）
    left_pixel = (left_tile - x_min) * 256
    top_pixel = (top_tile - y_min) * 256
    right_pixel = ((right_tile - x_min) + 1) * 256
    bottom_pixel = ((bottom_tile - y_min) + 1) * 256
    
    # 裁剪到边界框
    img_array = img_array[top_pixel:bottom_pixel, left_pixel:right_pixel]
    
    # 计算裁剪后影像的边界
    pixel_width = (maxx - minx) / img_array.shape[1]
    pixel_height = (maxy - miny) / img_array.shape[0]
    
    # 如果指定了输出路径，保存为 GeoTIFF
    if output_path:
        # 计算仿射变换（从边界框开始）
        transform = from_bounds(minx, maxy, maxx, miny, 
                               img_array.shape[1], img_array.shape[0])
        
        # 如果有多边形几何，使用 rasterio mask 进行裁剪
        if polygon_geometry is not None:
            from shapely.geometry import mapping
            from rasterio import mask as rio_mask
            
            # 创建临时内存文件
            from rasterio.io import MemoryFile
            
            # 将影像转换为 (bands, height, width) 格式
            img_bands = np.transpose(img_array, (2, 0, 1))  # (3, H, W)
            
            # 创建临时 GeoTIFF 进行裁剪
            with MemoryFile() as memfile:
                with memfile.open(
                    driver='GTiff',
                    height=img_array.shape[0],
                    width=img_array.shape[1],
                    count=3,
                    dtype=img_array.dtype,
                    crs=CRS.from_epsg(4326),
                    transform=transform
                ) as temp_dst:
                    temp_dst.write(img_bands)
                    
                    # 使用多边形进行裁剪
                    try:
                        # 确保多边形是有效的几何对象
                        if hasattr(polygon_geometry, '__geo_interface__'):
                            geom = [mapping(polygon_geometry)]
                        else:
                            geom = [polygon_geometry]
                        
                        # 执行裁剪
                        out_image, out_transform = rio_mask.mask(
                            temp_dst, geom, crop=True, nodata=0
                        )
                        
                        # 转换回 (H, W, 3) 格式
                        out_image = np.transpose(out_image, (1, 2, 0))
                        
                        # 更新变换和尺寸
                        final_height, final_width = out_image.shape[:2]
                        
                        # 保存裁剪后的影像
                        with rasterio.open(
                            output_path,
                            'w',
                            driver='GTiff',
                            height=final_height,
                            width=final_width,
                            count=3,
                            dtype=out_image.dtype,
                            crs=CRS.from_epsg(4326),
                            transform=out_transform,
                            compress='lzw',
                            nodata=0
                        ) as dst:
                            # RGB 三个波段
                            for i in range(3):
                                dst.write(out_image[:, :, i], i + 1)
                        
                        print(f"  ✅ 影像已保存（已裁剪到多边形区域）: {output_path}")
                        return out_image
                        
                    except Exception as e:
                        print(f"  ⚠️  多边形裁剪失败，保存完整边界框影像: {e}")
                        # 如果裁剪失败，保存完整边界框影像
                        img_bands = np.transpose(img_array, (2, 0, 1))
                        with rasterio.open(
                            output_path,
                            'w',
                            driver='GTiff',
                            height=img_array.shape[0],
                            width=img_array.shape[1],
                            count=3,
                            dtype=img_array.dtype,
                            crs=CRS.from_epsg(4326),
                            transform=transform,
                            compress='lzw'
                        ) as dst:
                            for i in range(3):
                                dst.write(img_bands[i], i + 1)
                        print(f"  ✅ 影像已保存（完整边界框）: {output_path}")
                        return img_array
        else:
            # 没有多边形，直接保存边界框影像
            img_bands = np.transpose(img_array, (2, 0, 1))
            with rasterio.open(
                output_path,
                'w',
                driver='GTiff',
                height=img_array.shape[0],
                width=img_array.shape[1],
                count=3,
                dtype=img_array.dtype,
                crs=CRS.from_epsg(4326),
                transform=transform,
                compress='lzw'
            ) as dst:
                for i in range(3):
                    dst.write(img_bands[i], i + 1)
            
            print(f"  ✅ 影像已保存: {output_path}")
            return img_array
    
    # 如果没有指定输出路径，返回原始数组
    return img_array


def process_polygon_with_imagery(gdf: gpd.GeoDataFrame,
                                 idx: int,
                                 polygon: gpd.GeoSeries,
                                 zoom_level: int,
                                 output_dir: Path,
                                 buffer: float = 0.0001,
                                 min_resolution_m: float = 5.0,
                                 auto_adjust_zoom: bool = True,
                                 max_workers: int = 50) -> dict:
    """
    处理单个多边形，下载其对应的 Google 影像
    
    Args:
        gdf: GeoDataFrame
        idx: 多边形索引
        polygon: 多边形几何
        zoom_level: 缩放级别（22级及以上）
        output_dir: 输出目录
        buffer: 边界缓冲区（度）
        
    Returns:
        包含影像路径等信息的字典
    """
    # 获取多边形边界
    bounds = polygon.bounds
    minx, miny, maxx, maxy = bounds
    
    # 计算多边形面积（用于判断是否需要调整缓冲区）
    polygon_area_km2 = polygon.area * 111 * 111  # 粗略转换为km²
    
    # 根据多边形大小动态调整缓冲区
    # 小多边形使用更小的缓冲区，大多边形使用稍大的缓冲区
    if polygon_area_km2 < 0.01:  # 小于0.01 km²
        actual_buffer = buffer * 0.5  # 使用更小的缓冲区
    elif polygon_area_km2 < 0.1:  # 0.01-0.1 km²
        actual_buffer = buffer
    else:  # 大于0.1 km²
        actual_buffer = buffer * 1.5  # 稍大的缓冲区
    
    # 添加缓冲区以确保完整覆盖
    minx -= actual_buffer
    miny -= actual_buffer
    maxx += actual_buffer
    maxy += actual_buffer
    
    # 计算下载区域大小（用于提示）
    area_deg = (maxx - minx) * (maxy - miny)
    area_km2 = area_deg * 111 * 111  # 粗略转换为km²
    
    # 生成输出文件名
    # 使用多边形的属性来命名（如果有的话）
    filename_base = f"polygon_{idx}"
    if 'fid' in gdf.columns:
        filename_base = f"polygon_{gdf.loc[idx, 'fid']}"
    elif 'id' in gdf.columns:
        filename_base = f"polygon_{gdf.loc[idx, 'id']}"
    
    output_path = output_dir / f"{filename_base}_zoom{zoom_level}.tif"
    
    print(f"  边界: ({minx:.6f}, {miny:.6f}, {maxx:.6f}, {maxy:.6f})")
    print(f"  多边形面积: {polygon_area_km2:.4f} km²")
    print(f"  下载区域: {area_km2:.4f} km² (包含缓冲区)")
    print(f"  缩放级别: {zoom_level}")
    
    # 检查文件是否已存在
    if output_path.exists():
        try:
            # 验证文件是否有效（可以打开）
            import rasterio
            with rasterio.open(str(output_path)) as src:
                if src.width > 0 and src.height > 0:
                    print(f"  ✅ 文件已存在，跳过下载: {output_path}")
                    return {
                        'polygon_idx': idx,
                        'image_path': str(output_path),
                        'zoom_level': zoom_level,
                        'bounds': bounds,
                        'status': 'success',
                        'skipped': True
                    }
        except Exception as e:
            print(f"  ⚠️  已存在文件无效，将重新下载: {e}")
    
    # 下载并保存影像（传入多边形几何用于裁剪）
    try:
        result_img = download_and_merge_tiles(
            (minx, miny, maxx, maxy), 
            zoom_level, 
            str(output_path),
            polygon_geometry=polygon,  # 传入多边形几何用于裁剪
            min_resolution_m=min_resolution_m,
            auto_adjust_zoom=auto_adjust_zoom,
            max_workers=max_workers
        )
        
        if result_img is not None:
            return {
                'polygon_idx': idx,
                'image_path': str(output_path),
                'zoom_level': zoom_level,
                'bounds': bounds,
                'status': 'success'
            }
        else:
            return {
                'polygon_idx': idx,
                'image_path': None,
                'zoom_level': zoom_level,
                'bounds': bounds,
                'status': 'failed',
                'error': '下载失败：无法获取影像数据'
            }
    except Exception as e:
        print(f"  ❌ 处理失败: {e}")
        return {
            'polygon_idx': idx,
            'image_path': None,
            'zoom_level': zoom_level,
            'bounds': bounds,
            'status': 'failed',
            'error': str(e)
        }


def update_summary_file(summary_path: Path, new_result: dict):
    """
    增量更新摘要文件（线程安全）
    
    Args:
        summary_path: 摘要文件路径
        new_result: 新的结果记录
    """
    # 使用文件锁来避免并发写入冲突
    lock_file = summary_path.parent / f"{summary_path.name}.lock"
    max_retries = 10
    retry_delay = 0.1
    max_lock_age = 60  # 锁文件最大年龄（秒），超过此时间认为锁已失效
    
    for attempt in range(max_retries):
        try:
            # 检查锁文件是否存在，如果存在且太旧，删除它
            if lock_file.exists():
                lock_age = time.time() - lock_file.stat().st_mtime
                if lock_age > max_lock_age:
                    # 锁文件太旧，认为已失效，删除它
                    try:
                        lock_file.unlink()
                    except:
                        pass
            
            # 尝试创建锁文件（原子操作）
            if not lock_file.exists():
                lock_file.touch()
                time.sleep(0.01)  # 短暂延迟确保文件系统同步
                
                # 再次检查锁文件是否仍然存在（避免竞态条件）
                if lock_file.exists():
                    try:
                        # 读取已有摘要
                        existing_summary = []
                        if summary_path.exists():
                            try:
                                with open(summary_path, 'r', encoding='utf-8') as f:
                                    existing_summary = json.load(f)
                            except Exception:
                                pass
                        
                        # 创建索引并更新
                        existing_dict = {r['polygon_idx']: r for r in existing_summary}
                        existing_dict[new_result['polygon_idx']] = new_result
                        
                        # 保存更新后的摘要
                        merged_summary = list(existing_dict.values())
                        merged_summary.sort(key=lambda x: x['polygon_idx'])
                        
                        # 原子写入：先写入临时文件，再重命名
                        temp_path = summary_path.with_suffix('.tmp')
                        with open(temp_path, 'w', encoding='utf-8') as f:
                            json.dump(merged_summary, f, indent=2, ensure_ascii=False)
                        
                        # 原子替换
                        temp_path.replace(summary_path)
                        
                        # 删除锁文件
                        if lock_file.exists():
                            lock_file.unlink()
                        
                        return
                    except Exception as e:
                        # 清理锁文件
                        if lock_file.exists():
                            try:
                                lock_file.unlink()
                            except:
                                pass
                        if attempt < max_retries - 1:
                            time.sleep(retry_delay * (attempt + 1))
                            continue
                        raise
            else:
                # 锁文件存在，等待后重试
                time.sleep(retry_delay * (attempt + 1))
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(retry_delay * (attempt + 1))
                continue
            # 如果所有重试都失败，至少尝试直接写入（不保证线程安全）
            try:
                existing_summary = []
                if summary_path.exists():
                    try:
                        with open(summary_path, 'r', encoding='utf-8') as f:
                            existing_summary = json.load(f)
                    except Exception:
                        pass
                
                existing_dict = {r['polygon_idx']: r for r in existing_summary}
                existing_dict[new_result['polygon_idx']] = new_result
                merged_summary = list(existing_dict.values())
                merged_summary.sort(key=lambda x: x['polygon_idx'])
                
                with open(summary_path, 'w', encoding='utf-8') as f:
                    json.dump(merged_summary, f, indent=2, ensure_ascii=False)
            except Exception:
                pass  # 如果还是失败，至少不崩溃


def main():
    """主函数"""
    import argparse
    
    # 注册信号处理器（Ctrl+C）
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # 注册退出清理函数
    atexit.register(cleanup_sessions)
    
    parser = argparse.ArgumentParser(description='下载高分卫星影像并与多边形关联（支持 ESRI、Microsoft Bing、Google）')
    parser.add_argument('input_gpkg', type=str, help='输入的 GPKG 文件路径')
    parser.add_argument('--zoom', type=int, default=22, help='缩放级别（默认22，建议22-23）')
    parser.add_argument('--output-dir', type=str, default=None, help='输出目录（默认：输入文件同目录下的 imagery 文件夹）')
    parser.add_argument('--max-polygons', type=int, default=None, help='最大处理多边形数量（用于测试）')
    parser.add_argument('--buffer', type=float, default=0.0001, help='边界缓冲区（度，默认0.0001，会根据多边形大小自动调整）')
    parser.add_argument('--min-area', type=float, default=None, help='最小下载区域（公顷，ha），小于此值会跳过。例如：12.5 表示仅下载12.5公顷以上的图斑')
    parser.add_argument('--max-area', type=float, default=None, help='最大下载区域（km²），超过此值会跳过或警告')
    parser.add_argument('--source', type=str, default=None, 
                       choices=['esri', 'bing', 'google', 'auto'],
                       help='指定影像源（esri/bing/google/auto），默认 auto 自动选择最快的')
    parser.add_argument('--change-type', type=str, default=None,
                       help='只处理指定 Change_Type 的多边形（例如：灰褐土→新积土）')
    parser.add_argument('--change-types', type=str, nargs='+', default=None,
                       help='处理多个 Change_Type（例如：--change-types "灰褐土→新积土" "黄绵土→新积土"）')
    parser.add_argument('--min-resolution', type=float, default=5.0,
                       help='最小分辨率要求（米/像素），默认5.0，会自动调整缩放级别以满足要求')
    parser.add_argument('--no-auto-zoom', action='store_true',
                       help='禁用自动调整缩放级别（使用指定的--zoom参数）')
    parser.add_argument('--max-workers', type=int, default=50,
                       help='并发下载线程数（默认50，建议30-100，网络快可设置更高）')
    
    args = parser.parse_args()
    
    # 检查输入文件
    input_path = Path(args.input_gpkg)
    if not input_path.exists():
        print(f"❌ 错误: 文件不存在: {input_path}")
        sys.exit(1)
    
    # 设置输出目录
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = input_path.parent / 'imagery'
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📂 输入文件: {input_path}")
    print(f"📂 输出目录: {output_dir}")
    print(f"🔍 缩放级别: {args.zoom}")
    print(f"📏 最小分辨率要求: {args.min_resolution} 米/像素")
    if not args.no_auto_zoom:
        print(f"🔄 自动调整缩放级别: 启用")
    
    # 设置影像源
    global _current_source
    if args.source and args.source != 'auto':
        _current_source = args.source
        print(f"📡 指定影像源: {IMAGERY_SOURCES[args.source]['name']}")
    else:
        print(f"📡 影像源: 自动选择最快的可用源")
    
    # 读取 GPKG 文件
    print(f"\n读取 GPKG 文件...")
    try:
        gdf = gpd.read_file(str(input_path))
        print(f"✅ 成功读取 {len(gdf)} 个多边形")
        print(f"   列: {list(gdf.columns)}")
        print(f"   CRS: {gdf.crs}")
    except Exception as e:
        print(f"❌ 读取文件失败: {e}")
        sys.exit(1)
    
    # 确保使用 WGS84 坐标系
    if gdf.crs is None or gdf.crs.to_epsg() != 4326:
        print(f"⚠️  坐标系不是 WGS84，正在转换...")
        gdf = gdf.to_crs(epsg=4326)
        print(f"✅ 已转换为 WGS84")
    
    # 保存原始GeoDataFrame的副本（用于最终保存时包含所有变化类）
    gdf_original = gdf.copy()
    
    # 根据 Change_Type 筛选多边形（仅用于处理，不影响最终保存）
    if args.change_type:
        original_count = len(gdf)
        gdf = gdf[gdf['Change_Type'] == args.change_type]
        print(f"📋 筛选 Change_Type='{args.change_type}': {original_count} -> {len(gdf)} 个多边形（将处理这些）")
    elif args.change_types:
        original_count = len(gdf)
        gdf = gdf[gdf['Change_Type'].isin(args.change_types)]
        print(f"📋 筛选 Change_Type 在 {args.change_types}: {original_count} -> {len(gdf)} 个多边形（将处理这些）")
    else:
        print(f"📋 未指定筛选条件，将处理所有 {len(gdf)} 个多边形")
    
    if len(gdf) == 0:
        print("❌ 错误: 筛选后没有多边形需要处理")
        sys.exit(1)
    
    # 限制处理数量（用于测试）
    if args.max_polygons:
        gdf = gdf.head(args.max_polygons)
        print(f"⚠️  限制处理数量为: {args.max_polygons}")
    
    # 处理每个多边形
    results = []
    total = len(gdf)
    
    # 摘要文件路径
    summary_path = output_dir / 'download_summary.json'
    
    # 预先检查有多少文件已经存在，以及有多少会被面积过滤
    existing_count = 0
    area_filtered_count = 0
    
    for idx in gdf.index:
        polygon = gdf.loc[idx, 'geometry']
        
        # 检查面积过滤
        polygon_area_km2 = polygon.area * 111 * 111
        polygon_area_ha = polygon_area_km2 * 100
        
        # 最小面积过滤
        if args.min_area is not None and polygon_area_ha < args.min_area:
            area_filtered_count += 1
            continue
        
        # 最大面积过滤
        if args.max_area and polygon_area_km2 > args.max_area:
            area_filtered_count += 1
            continue
        
        # 检查文件是否已存在
        filename_base = f"polygon_{idx}"
        if 'fid' in gdf.columns:
            filename_base = f"polygon_{gdf.loc[idx, 'fid']}"
        elif 'id' in gdf.columns:
            filename_base = f"polygon_{gdf.loc[idx, 'id']}"
        output_path = output_dir / f"{filename_base}_zoom{args.zoom}.tif"
        if output_path.exists():
            try:
                with rasterio.open(str(output_path)) as src:
                    if src.width > 0 and src.height > 0:
                        existing_count += 1
            except:
                pass
    
    print(f"\n开始处理 {total} 个多边形...")
    if area_filtered_count > 0:
        print(f"   🔍 面积过滤: {area_filtered_count} 个 (不符合面积要求)")
    if existing_count > 0:
        print(f"   📦 已存在: {existing_count} 个 (将跳过)")
    need_download = total - existing_count - area_filtered_count
    if need_download > 0:
        print(f"   📥 需下载: {need_download} 个")
    print("=" * 60)
    
    processed_count = 0
    for idx in gdf.index:
        processed_count += 1
        polygon = gdf.loc[idx, 'geometry']
        
        # 显示进度
        progress_msg = f"[{processed_count}/{total}]"
        
        # 计算多边形面积（km²和公顷）
        polygon_area_km2 = polygon.area * 111 * 111
        polygon_area_ha = polygon_area_km2 * 100  # 1 km² = 100 公顷
        
        # 检查多边形面积（如果设置了最小面积限制）
        if args.min_area is not None:
            if polygon_area_ha < args.min_area:
                print(f"\n{progress_msg} ⚠️  跳过多边形 {idx}: 面积 {polygon_area_ha:.2f} ha < 最小限制 {args.min_area} ha")
                result = {
                    'polygon_idx': idx,
                    'image_path': None,
                    'zoom_level': None,
                    'bounds': polygon.bounds,
                    'status': 'skipped',
                    'error': f'面积小于限制: {polygon_area_ha:.2f} ha < {args.min_area} ha'
                }
                results.append(result)
                # 立即更新摘要文件
                update_summary_file(summary_path, result)
                continue
        
        # 检查多边形面积（如果设置了最大区域限制）
        if args.max_area:
            if polygon_area_km2 > args.max_area:
                print(f"\n{progress_msg} ⚠️  跳过多边形 {idx}: 面积 {polygon_area_km2:.4f} km² > 最大限制 {args.max_area} km²")
                result = {
                    'polygon_idx': idx,
                    'image_path': None,
                    'zoom_level': None,
                    'bounds': polygon.bounds,
                    'status': 'skipped',
                    'error': f'面积超过限制: {polygon_area_km2:.4f} km² > {args.max_area} km²'
                }
                results.append(result)
                # 立即更新摘要文件
                update_summary_file(summary_path, result)
                continue
        
        print(f"\n{progress_msg} 处理多边形 {idx}...")
        result = process_polygon_with_imagery(
            gdf, idx, polygon, args.zoom, output_dir, args.buffer,
            min_resolution_m=args.min_resolution,
            auto_adjust_zoom=not args.no_auto_zoom,
            max_workers=args.max_workers
        )
        results.append(result)
        
        # 立即更新摘要文件（增量保存）
        update_summary_file(summary_path, result)
    
    # 创建结果汇总
    print("\n" + "=" * 60)
    print("处理完成！")
    success_count = sum(1 for r in results if r['status'] == 'success')
    skipped_count = sum(1 for r in results if r.get('skipped', False))
    failed_count = sum(1 for r in results if r['status'] == 'failed')
    
    print(f"总计: {len(results)} 个多边形")
    print(f"  ✅ 新下载: {success_count - skipped_count}")
    print(f"  ⏭️  已存在(跳过): {skipped_count}")
    print(f"  ❌ 失败: {failed_count}")
    
    # 将影像路径添加到原始GeoDataFrame（包含所有变化类）
    # 首先确保原始GeoDataFrame有这些列
    if 'image_path' not in gdf_original.columns:
        gdf_original['image_path'] = None
    if 'zoom_level' not in gdf_original.columns:
        gdf_original['zoom_level'] = None
    
    # 将处理结果合并到原始GeoDataFrame
    for result in results:
        idx = result['polygon_idx']
        # 确保索引存在于原始GeoDataFrame中
        if idx in gdf_original.index:
            if result['status'] == 'success':
                # 包括新下载和跳过的（已存在的）文件
                gdf_original.loc[idx, 'image_path'] = result['image_path']
                gdf_original.loc[idx, 'zoom_level'] = result['zoom_level']
    
    # 统计所有变化类的数量
    if 'Change_Type' in gdf_original.columns:
        change_types = gdf_original['Change_Type'].unique()
        print(f"\n📊 原始数据包含 {len(change_types)} 种变化类: {list(change_types)}")
        for ct in change_types:
            count = len(gdf_original[gdf_original['Change_Type'] == ct])
            with_image = sum(1 for idx in gdf_original[gdf_original['Change_Type'] == ct].index 
                           if idx in gdf_original.index and gdf_original.loc[idx, 'image_path'] is not None)
            print(f"   - {ct}: {count} 个多边形 ({with_image} 个已关联影像)")
    
    # 先保存结果摘要（JSON文件较小，先保存可以避免内存问题）
    summary_path = output_dir / 'download_summary.json'
    import json
    
    print(f"\n💾 保存结果摘要...")
    try:
        # 读取已有摘要（如果存在）
        existing_summary = []
        if summary_path.exists():
            try:
                with open(summary_path, 'r', encoding='utf-8') as f:
                    existing_summary = json.load(f)
                print(f"   📋 读取已有摘要: {len(existing_summary)} 条记录")
            except Exception as e:
                print(f"   ⚠️  读取已有摘要失败: {e}")
        
        # 创建索引：polygon_idx -> 记录
        existing_dict = {r['polygon_idx']: r for r in existing_summary}
        
        # 合并结果：用新结果覆盖已有记录
        for result in results:
            existing_dict[result['polygon_idx']] = result
        
        # 保存合并后的摘要
        merged_summary = list(existing_dict.values())
        # 按 polygon_idx 排序
        merged_summary.sort(key=lambda x: x['polygon_idx'])
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(merged_summary, f, indent=2, ensure_ascii=False)
        print(f"   ✅ 已保存结果摘要: {summary_path} (共 {len(merged_summary)} 条记录)")
    except Exception as e:
        print(f"   ❌ 保存摘要文件失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 保存更新后的 GPKG（包含所有变化类和影像路径）
    output_gpkg = output_dir / f"{input_path.stem}_with_imagery.gpkg"
    print(f"\n💾 保存更新后的 GPKG...")
    print(f"   总多边形数: {len(gdf_original)}")
    print(f"   已关联影像: {sum(1 for p in gdf_original['image_path'] if p is not None)} 个")
    
    try:
        gdf_original.to_file(str(output_gpkg), driver='GPKG')
        print(f"   ✅ 已保存更新后的 GPKG: {output_gpkg}")
    except MemoryError:
        print(f"   ⚠️  内存不足，无法保存完整 GPKG 文件")
        print(f"   💡 建议: 减少并发线程数或分批处理")
        # 即使 GPKG 保存失败，摘要文件已保存，仍然可以继续
    except Exception as e:
        print(f"   ❌ 保存 GPKG 文件失败: {e}")
        import traceback
        traceback.print_exc()
        # 即使 GPKG 保存失败，摘要文件已保存，仍然可以继续
    
    # 清理所有连接
    cleanup_sessions()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  用户中断，正在退出...")
        cleanup_sessions()
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ 发生错误: {e}")
        cleanup_sessions()
        sys.exit(1)

