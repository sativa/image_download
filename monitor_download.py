#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
监控下载进度
"""
import time
import json
from pathlib import Path

def monitor_download(interval=30, max_checks=20):
    """监控下载进度"""
    summary_path = Path('output_heshui/imagery/download_summary.json')
    imagery_dir = Path('output_heshui/imagery')
    
    print("\n" + "="*60)
    print("📊 下载进度监控")
    print("="*60 + "\n")
    print(f"检查间隔: {interval}秒")
    print(f"最大检查次数: {max_checks}\n")
    
    last_count = 0
    last_file_count = 0
    
    for i in range(max_checks):
        # 统计文件数量
        if imagery_dir.exists():
            tif_files = list(imagery_dir.glob('polygon_*_zoom17.tif'))
            file_count = len(tif_files)
        else:
            file_count = 0
        
        # 读取摘要
        if summary_path.exists():
            with open(summary_path, 'r', encoding='utf-8') as f:
                summary = json.load(f)
            
            total = len(summary)
            success = sum(1 for r in summary if r['status'] == 'success')
            
            # 计算进度
            target = 73  # 目标总数
            progress = (success / target) * 100 if target > 0 else 0
            remaining = target - success
            
            # 检查是否有新进展
            new_records = total - last_count
            new_files = file_count - last_file_count
            
            print(f"[{i+1}/{max_checks}] {time.strftime('%H:%M:%S')}")
            print(f"  文件数: {file_count} ({'+' + str(new_files) if new_files > 0 else ''})")
            print(f"  摘要记录: {total} ({'+' + str(new_records) if new_records > 0 else ''})")
            print(f"  成功: {success}/{target} ({progress:.1f}%)")
            print(f"  剩余: {remaining} 个")
            
            if success >= target:
                print(f"\n✅ 下载完成！所有 {target} 个多边形已处理完成。")
                break
            
            if new_records > 0 or new_files > 0:
                print(f"  ✨ 有进展！新增 {new_records} 条记录，{new_files} 个文件")
            
            last_count = total
            last_file_count = file_count
        else:
            print(f"[{i+1}/{max_checks}] 摘要文件不存在，等待中...")
        
        if i < max_checks - 1:
            print()
            time.sleep(interval)
    
    print("\n" + "="*60 + "\n")

if __name__ == '__main__':
    monitor_download(interval=30, max_checks=20)

