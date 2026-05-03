import re
import os

def calculate_average_time_for_item(file_path, item_name):
    total_time = 0.0
    sample_count = 0
    
    time_pattern = re.compile(r"Run 1:\s+([\d.]+)\s+ms")

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                if item_name in line and i + 1 < len(lines):
                    match = time_pattern.search(lines[i + 1])
                    if match:
                        total_time += float(match.group(1))
                        sample_count += 1
    except Exception as e:
        print(f"处理文件 {file_path} 时发生错误：{e}")
        return 0, 0.0, 0.0

    average_time = total_time / sample_count if sample_count > 0 else 0.0
    return sample_count, total_time, average_time


if __name__ == "__main__":
    # ===== 设置目录路径 =====
    directory_path = "../output"
    if not directory_path:
        directory_path = "."  # 使用当前目录
    elif not os.path.isdir(directory_path):
        print(f"错误：目录 '{directory_path}' 不存在，使用当前目录")
        directory_path = "."
    
    print(f"使用目录: {os.path.abspath(directory_path)}\n")
    
    # files_to_analyze = [
    #     'Bai.txt',
    #     'Gset.txt',   
    #     'HB.txt',
    #     'JGD_Homology.txt',
    #     'Pajek.txt',
    #     'VDOL.txt'
    # ]
    # files_to_analyze = [
    #     'Bai_default.txt',
    #     'Gset_default.txt',
    #     'HB_default.txt',
    #     'JGD_Homology_default.txt',
    #     'Pajek_default.txt',
    #     'VDOL_default.txt'
    # ]
    files_to_analyze = [
        'Bai_reorder.txt',
        'Gset_reorder.txt',
        'HB_reorder.txt',
        'JGD_Homology_reorder.txt',
        'Pajek_reorder.txt',
        'VDOL_reorder.txt'
    ]
    items_to_calculate = [
        "opRunner.RunOp",
        "aclnnBcsrSpmmCustom"
    ]

    # ===== 新增：用于汇总 =====
    files = []
    bcsr_op_runner_times = []
    bcsr_mmad_times = []

    for file_to_analyze in files_to_analyze:
        # 构建完整的文件路径
        full_file_path = os.path.join(directory_path, file_to_analyze)
        
        # 检查文件是否存在
        if not os.path.isfile(full_file_path):
            print(f"警告：文件 '{full_file_path}' 不存在，跳过处理")
            continue
            
        dataset_name = file_to_analyze.split('_')[0]
        files.append(dataset_name)

        print(f"'{file_to_analyze}'\n")

        for item in items_to_calculate:
            count, total, average = calculate_average_time_for_item(
                full_file_path, item
            )

            if count > 0:
                print(f"项目 '{item}' 的分析结果：")
                print(f"  共找到 {count} 个样本。")
                print(f"  总耗时: {total:.4f} ms")
                print(f"  平均耗时: {average:.4f} ms\n")

                # ===== 核心：存到 list 里 =====
                if item == "opRunner.RunOp":
                    bcsr_op_runner_times.append(average)
                elif item == "aclnnBcsrSpmmCustom":
                    bcsr_mmad_times.append(average)
            else:
                print(f"未在文件中找到 '{item}' 的有效样本数据。\n")

    # ===== 最终：直接可复制使用的列表 =====
    print("\n================= COPY BELOW =================\n")
    print(f"files = {files}")
    print(f"aclnnBcsrSpmmCustom = {[round(x, 4) for x in bcsr_op_runner_times]}")
    print(f"bcsr_mmad_times = {[round(x, 4) for x in bcsr_mmad_times]}")
    print("\n=============================================\n")