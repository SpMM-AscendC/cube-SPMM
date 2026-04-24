#!/bin/bash
CURRENT_DIR=$(
    cd $(dirname ${BASH_SOURCE:-$0})
    pwd
)

if [ -n "$ASCEND_INSTALL_PATH" ]; then
    _ASCEND_INSTALL_PATH=$ASCEND_INSTALL_PATH
elif [ -n "$ASCEND_HOME_PATH" ]; then
    _ASCEND_INSTALL_PATH=$ASCEND_HOME_PATH
else
    if [ -d "$HOME/Ascend/ascend-toolkit/latest" ]; then
        _ASCEND_INSTALL_PATH=$HOME/Ascend/ascend-toolkit/latest
    else
        _ASCEND_INSTALL_PATH=/usr/local/Ascend/ascend-toolkit/latest
    fi
fi

# 先source环境
source $_ASCEND_INSTALL_PATH/bin/setenv.bash


export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:$PATH
hash -r

export DDK_PATH=$_ASCEND_INSTALL_PATH
export NPU_HOST_LIB=$_ASCEND_INSTALL_PATH/$(arch)-$(uname -s | tr '[:upper:]' '[:lower:]')/devlib

function main {
    # 声明关联数组用于性能统计
    declare -A perf_sum perf_count

    # 1. 编译acl可执行文件
    cd $CURRENT_DIR
    rm -rf build
    mkdir -p build
    cd build
    cmake ../src -DCMAKE_SKIP_RPATH=TRUE
    if [ $? -ne 0 ]; then
        echo "[ERROR]: Cmake failed!"
        return 1
    fi
    echo "[INFO]: Cmake success!"
    make
    if [ $? -ne 0 ]; then
        echo "[ERROR]: Make failed!"
        return 1
    fi
    echo "[INFO]: Make success!"
    cd $CURRENT_DIR

    # 定义输入输出目录
    # INPUTS_DIR="../input_test"
    INPUTS_DIR="../inputs"
    # INPUTS_DIR="/root/autodl-tmp/datasets-coldense/DataSets"
    # INPUTS_DIR="../inputs_all"
    OUTPUT_DIR="../output_balance_split_2"
    MODE="default"

    # 创建输出目录
    mkdir -p ../output_all
    rm -rf $OUTPUT_DIR
    mkdir -p $OUTPUT_DIR

    # 创建失败样本日志
    FAILURE_LOG="$OUTPUT_DIR/failed_samples.log"
    echo "[$(date +%Y-%m-%d\ %H:%M:%S)] Failed Samples Log" > $FAILURE_LOG
    echo "================================================================" >> $FAILURE_LOG

    # 创建单个样本性能记录文件
    INDIVIDUAL_PERF_LOG="./result_info/individual_performance_coldense_balance_split_2.txt"
    echo "[$(date +%Y-%m-%d\ %H:%M:%S)] ColDense Individual Sample Performance" > $INDIVIDUAL_PERF_LOG
    echo "================================================================" >> $INDIVIDUAL_PERF_LOG
    echo "Format: Category | Sample_Name | M | K | N | NNZ | WindowNum | BlockNum | FillRate | Performance(us)" >> $INDIVIDUAL_PERF_LOG
    echo "================================================================" >> $INDIVIDUAL_PERF_LOG

    # 2. 查找所有测试用例并运行
    for mtx_file in $(find $INPUTS_DIR -name "*.mtx"); do
        sample_name=$(basename $mtx_file .mtx)
        category_dir=$(dirname $mtx_file)
        sample_dir="$category_dir/${sample_name}_colcondense"
        category=$(basename $category_dir)
        
        # if [[ "$category" != "Pajek" && "$category" != "JGD_Homology" && "$category" != "VDOL" ]]; then
        # if [[ "$category" != "Pajek" ]]; then
        #     continue
        # fi
        echo "==================== Running test for $sample_name ===================="

        # 3. 解析矩阵维度
        dims=$(python3 scripts/parse_matrix.py $mtx_file --reuse)
        if [ $? -ne 0 ]; then
            echo "[ERROR]: Failed to parse matrix dimensions for $mtx_file"
            echo "$sample_name | Parse dimension failed" >> $FAILURE_LOG
            continue
        fi
        read -r m k n nnz window_num block_num fill_rate <<< "$dims"
        echo "[INFO]: Matrix dimensions (M, K, N, NNZ): $m, $k, $n, $nnz"
        echo "[INFO]: Block info (WindowNum, BlockNum, Fill_rate): $window_num, $block_num, $fill_rate"

        # 4. 定义输入输出文件路径
        input_row_ptr="$sample_dir/rw_ptr.bin"
        input_col="$sample_dir/TC_col_ref.bin"
        input_values="$sample_dir/values.bin"
        input_b="$sample_dir/x2_gm.bin"
        # input_core_row_start="$sample_dir/core_row_start.bin"
        input_core_info="$sample_dir/core_info.bin"
        output_c="$OUTPUT_DIR/${sample_name}_output_c.bin"

        # 5. 运行可执行文件并使用msprof计时
        export LD_LIBRARY_PATH=$_ASCEND_INSTALL_PATH/opp/vendors/customize/op_api/lib:$LD_LIBRARY_PATH
        export LD_LIBRARY_PATH=${_ASCEND_INSTALL_PATH}/tools/simulator/Ascend910B2/lib:$LD_LIBRARY_PATH 
        echo "[INFO]: Execute op for $sample_name!"
        category=$(basename $category_dir)
        
        # 使用msprof进行性能测试
        msprof op --output=./msprof_out_coldense ./output/execute_spmm_op $m $k $n $window_num $block_num $input_row_ptr $input_col $input_values $input_b $input_core_info $output_c $category $sample_name $MODE
        # msprof op simulator --output=./msprof_out_coldense ./output/execute_spmm_op $m $k $n $window_num $block_num $input_row_ptr $input_col $input_values $input_b $input_core_info $output_c $category $sample_name $MODE
        exec_status=$?
        
        if [ $exec_status -ne 0 ]; then
            echo "[ERROR]: Acl executable run failed for sample $sample_name!"
            echo "$sample_name | Execution failed (exit code: $exec_status) | M=$m K=$k N=$n NNZ=$nnz" >> $FAILURE_LOG
            # # 清理可能存在的msprof输出
            # if [ -d "./msprof_out_coldense" ]; then
            #     rm -rf ./msprof_out_coldense
            # fi
            continue
        fi

        # 提取性能数据从CSV文件
        perf_value="0"
        op_basic_info_file=$(find ./msprof_out_coldense -name OpBasicInfo.csv 2>/dev/null | head -n 1)
        if [ -n "$op_basic_info_file" ] && [ -f "$op_basic_info_file" ]; then
            # 提取第二行、逗号分隔的第三个字段（清理非数字字符）
            perf_value=$(sed -n '2p' "$op_basic_info_file" | cut -d',' -f3 | sed 's/[^0-9.]//g')
            echo "[INFO]: Extracted performance value for $sample_name: $perf_value us"
            # 强制删除msprof_out文件夹
            if [ -d "./msprof_out_coldense" ]; then
                rm -rf ./msprof_out_coldense && echo "[INFO]: Deleted ./msprof_out_coldense folder"
            fi
        else
            echo "[WARN]: OpBasicInfo.csv not found for sample $sample_name, use default value 0"
            echo "$sample_name | OpBasicInfo.csv not found | M=$m K=$k N=$n NNZ=$nnz" >> $FAILURE_LOG
        fi

        # 记录单个样本的性能数据到文件（包含所有维度信息）
        echo "$category | $sample_name | $m | $k | $n | $nnz | $window_num | $block_num | $fill_rate | $perf_value" >> $INDIVIDUAL_PERF_LOG
        echo "" >> $INDIVIDUAL_PERF_LOG

        # 统计分类性能数据
        if [[ "$perf_value" =~ ^[0-9.]+$ ]] && [ "$perf_value" != "0" ]; then
            # 浮点累加：用awk计算当前总和+新值，保留8位小数
            current_sum=${perf_sum["$category_dir"]:-0.0}
            new_sum=$(echo "$current_sum $perf_value" | awk '{printf "%.8f", $1 + $2}')
            perf_sum["$category_dir"]=$new_sum
            # 样本数整数累加
            perf_count["$category_dir"]=$(( ${perf_count["$category_dir"]:-0} + 1 ))
        else
            echo "[WARN]: Invalid performance value '$perf_value' for $sample_name, skip statistics"
        fi

        # 6. 比较真值文件（只在输出文件存在时执行）
        if [ -f "$output_c" ]; then
            golden_bin="$sample_dir/golden.bin"
            if [ -f "$golden_bin" ]; then
                python3 scripts/verify_result.py $output_c $golden_bin $m $n $OUTPUT_DIR $sample_name > "$OUTPUT_DIR/${sample_name}_wrong_indices"
                if [ $? -ne 0 ]; then
                    echo "[ERROR]: Verify result failed for sample $sample_name!"
                    echo "[$sample_name] | Verification failed | M=$m K=$k N=$n NNZ=$nnz" >> $FAILURE_LOG
                    echo "[INFO]: Wrong indices saved to $OUTPUT_DIR/${sample_name}_wrong_indices"
                else
                    echo "[INFO]: Verify result success for sample $sample_name!"
                    # 验证成功则删除wrong_indices文件
                    rm -f "$OUTPUT_DIR/${sample_name}_wrong_indices"
                fi
            else
                echo "[WARN]: golden.bin not found for sample $sample_name. Skipping verification."
            fi

            # 7. 删除输出文件以节省空间（可选）
            rm -f $output_c
            # echo "[INFO]: Removed output file $output_c"
        else
            echo "[WARN]: Output file not generated for sample $sample_name, skipping verification"
            echo "$sample_name | Output file not generated | M=$m K=$k N=$n NNZ=$nnz" >> $FAILURE_LOG
        fi

        echo "[INFO]: Removed temp files"
        echo "==================== Finished test for $sample_name ===================="
        echo ""
        
        # break  # 取消注释可以只测试一个样本
        #删除中间文件
        #rm -rf $output_c $sample_dir
    done

    # 汇总统计并写入result.txt
    echo -e "\n==================== ColDense Performance Statistics ===================="
    # 追加时间戳，便于追溯
    echo -e "\n[$(date +%Y-%m-%d\ %H:%M:%S)] ColDense Performance Statistics" >> ./result_info/result_coldense.txt
    
    # 遍历所有分类目录的统计数据
    for cat_dir in "${!perf_sum[@]}"; do
        sum=${perf_sum[$cat_dir]}
        count=${perf_count[$cat_dir]}
        # awk计算平均值，避免除0，保留6位小数
        if [ "$count" -eq 0 ]; then
            avg="0.000000"
        else
            avg=$(echo "$sum $count" | awk '{printf "%.6f", $1 / $2}')
        fi
        cat_name=$(basename "$cat_dir")
        # 控制台输出
        echo "[INFO]: Category: $cat_name | Total Samples: $count | Total Performance: $sum us | Average: $avg us"
        # 追加写入result_coldense.txt
        echo "Category: $cat_name | Directory: $cat_dir | Total Samples: $count | Total Performance: $sum us | Average: $avg us" >> ./result_info/result_coldense.txt
        echo "" >> ./result_info/result_coldense.txt
    done
    
    echo "[INFO]: Performance statistics have been written to ./result_coldense.txt"
    echo "[INFO]: Individual sample performance has been written to $INDIVIDUAL_PERF_LOG"
    echo "[INFO]: Failed samples log has been written to $FAILURE_LOG"
}

main
