import re
from collections import defaultdict
import sys

def parse_performance_log(file_path):
    """解析性能日志文件"""
    data = {}
    category_stats = defaultdict(lambda: {
        'total_time': 0,
        'count': 0,
        'samples': []
    })
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except Exception as e:
        print(f"错误: 无法读取文件 {file_path}")
        print(f"详细错误: {e}")
        return None
    
    # 跳过表头和分隔线
    start_parsing = False
    for line in lines:
        line = line.strip()
        
        # 检查是否为数据行
        if '|' in line and not line.startswith('==='):
            # 确保是数据行而非表头
            if 'Category' in line and 'Sample_Name' in line:
                continue
            
            # 解析数据行
            parts = [part.strip() for part in line.split('|')]
            
            if len(parts) >= 10:  # 确保有足够的部分
                try:
                    category = parts[0]
                    sample_name = parts[1]
                    
                    # 尝试解析性能时间，处理可能的异常
                    performance_str = parts[9]
                    if performance_str == '0':
                        performance = 0.0
                    else:
                        performance = float(performance_str)
                    
                    # 解析其他字段（可选）
                    m = int(parts[2])
                    k = int(parts[3])
                    n = int(parts[4])
                    nnz = int(parts[5])
                    
                    # 存储样本数据
                    sample_data = {
                        'name': sample_name,
                        'M': m,
                        'K': k,
                        'N': n,
                        'NNZ': nnz,
                        'performance': performance
                    }
                    
                    # 更新类别统计
                    category_stats[category]['total_time'] += performance
                    category_stats[category]['count'] += 1
                    category_stats[category]['samples'].append(sample_data)
                    
                    # 存储原始数据
                    key = f"{category}_{sample_name}"
                    data[key] = sample_data
                    
                except (ValueError, IndexError) as e:
                    print(f"警告: 无法解析行: {line}")
                    print(f"错误: {e}")
                    continue
    
    return data, category_stats

def print_statistics(category_stats):
    """打印统计信息"""
    if not category_stats:
        print("没有找到有效的数据")
        return
    
    print("\n" + "="*80)
    print("性能统计报告")
    print("="*80)
    
    # 计算总体统计
    total_all = 0
    count_all = 0
    
    for category, stats in category_stats.items():
        total_all += stats['total_time']
        count_all += stats['count']
    
    # 打印每个类别的统计
    print("\n按类别统计:")
    print("-"*80)
    print(f"{'类别':<15} {'样本数':<10} {'总时间(us)':<15} {'平均时间(us)':<15} {'占比(%)':<10}")
    print("-"*80)
    
    for category, stats in sorted(category_stats.items()):
        total_time = stats['total_time']
        count = stats['count']
        avg_time = total_time / count if count > 0 else 0
        percentage = (total_time / total_all * 100) if total_all > 0 else 0
        
        print(f"{category:<15} {count:<10} {total_time:<15.2f} {avg_time:<15.2f} {percentage:<10.2f}")
    
    # 打印总体统计
    print("-"*80)
    overall_avg = total_all / count_all if count_all > 0 else 0
    print(f"{'总计':<15} {count_all:<10} {total_all:<15.2f} {overall_avg:<15.2f} {'100.00':<10}")
    
    # 打印详细信息（可选）
    print("\n" + "="*80)
    print("各样本详细信息:")
    print("="*80)
    
    for category, stats in sorted(category_stats.items()):
        print(f"\n类别: {category}")
        print("-"*60)
        print(f"{'样本名称':<15} {'M':<6} {'K':<6} {'N':<6} {'NNZ':<10} {'性能(us)':<12}")
        print("-"*60)
        
        for sample in sorted(stats['samples'], key=lambda x: x['name']):
            print(f"{sample['name']:<15} {sample['M']:<6} {sample['K']:<6} {sample['N']:<6} "
                  f"{sample['NNZ']:<10} {sample['performance']:<12.2f}")

def main():
    # 检查命令行参数
    if len(sys.argv) < 2:
        print("使用方法: python stats_performance.py <日志文件路径>")
        print("示例: python stats_performance.py performance_log.txt")
        sys.exit(1)
    
    file_path = sys.argv[1]
    
    print(f"正在分析文件: {file_path}")
    print("="*80)
    
    # 解析日志文件
    result = parse_performance_log(file_path)
    
    if result is None:
        sys.exit(1)
    
    data, category_stats = result
    
    # 打印统计信息
    print_statistics(category_stats)
    
    # 保存结果到文件
    save_to_file(category_stats, "performance_summary.txt")

def save_to_file(category_stats, output_file):
    """将统计结果保存到文件"""
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("性能统计报告\n")
        f.write("="*80 + "\n\n")
        
        f.write("按类别统计:\n")
        f.write("-"*80 + "\n")
        f.write(f"{'类别':<15} {'样本数':<10} {'总时间(us)':<15} {'平均时间(us)':<15} {'占比(%)':<10}\n")
        f.write("-"*80 + "\n")
        
        total_all = sum(stats['total_time'] for stats in category_stats.values())
        count_all = sum(stats['count'] for stats in category_stats.values())
        
        for category, stats in sorted(category_stats.items()):
            total_time = stats['total_time']
            count = stats['count']
            avg_time = total_time / count if count > 0 else 0
            percentage = (total_time / total_all * 100) if total_all > 0 else 0
            
            f.write(f"{category:<15} {count:<10} {total_time:<15.2f} {avg_time:<15.2f} {percentage:<10.2f}\n")
        
        f.write("-"*80 + "\n")
        overall_avg = total_all / count_all if count_all > 0 else 0
        f.write(f"{'总计':<15} {count_all:<10} {total_all:<15.2f} {overall_avg:<15.2f} {'100.00':<10}\n")
    
    print(f"\n统计结果已保存到: {output_file}")

if __name__ == "__main__":
    main()