#!/bin/bash
#!!!!!!!!!!!!!!!!!!!!!当前配置文件为列浓缩+重排+负载均衡!!!!!!!!!!!!!!!!!!!!!!!!!!!
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
source $_ASCEND_INSTALL_PATH/bin/setenv.bash
export DDK_PATH=$_ASCEND_INSTALL_PATH
export NPU_HOST_LIB=$_ASCEND_INSTALL_PATH/$(arch)-$(uname -s | tr '[:upper:]' '[:lower:]')/devlib

function main {
    # 初始化性能统计关联数组（总和/样本数，无bc依赖，用awk计算）
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
    INPUTS_DIR="../inputs"
    OUTPUT_DIR="../output"
    MODE="reorder"

    # 清理并创建输出目录
    rm -rf $OUTPUT_DIR
    mkdir -p $OUTPUT_DIR
    if [ ! -d "../profile_result" ]; then
        mkdir -p "../profile_result"
    fi

    profile_result_DIR="../profile_result"
    INDIVIDUAL_FILE="$profile_result_DIR/individual_result_re_coldense_balance.txt"
    CATEGORY_RESULT_FILE="$profile_result_DIR/result_re_coldense_balance.txt"
    mkdir -p $profile_result_DIR

    FAILURE_LOG="$OUTPUT_DIR/failed_samples.log"
    rm -f $FAILURE_LOG

    # 2. 查找所有测试用例并运行
    for mtx_file in $(find $INPUTS_DIR -name "*.mtx"); do
        sample_name=$(basename $mtx_file .mtx)
        category_dir=$(dirname $mtx_file)
        sample_dir="$category_dir/${sample_name}_re_colcondense"
        
        echo "==================== Running test for $sample_name ===================="

        # 3. 解析矩阵维度
        dims=$(python3 scripts/parse_matrix_copy_reorder.py $mtx_file)
        if [ $? -ne 0 ]; then
            echo "[ERROR]: Failed to parse matrix dimensions for $mtx_file"
            continue
        fi
        read -r m k n nnz window_num block_num fill_rate<<< "$dims"
        echo "[INFO]: Matrix dimensions (M, K, N, NNZ): $m, $k, $n, $nnz"
        echo "[INFO]: Block info (WindowNum, BlockNum, Fill_rate): $window_num, $block_num, $fill_rate"

        # 4. 定义输入输出文件路径
        input_row_ptr="$sample_dir/rw_ptr.bin"
        input_col="$sample_dir/TC_col_ref.bin"
        input_values="$sample_dir/values.bin"
        input_ref="$sample_dir/reorder_ref.bin"
        input_core_info="$sample_dir/core_info.bin"
        input_b="$sample_dir/x2_gm.bin"
        output_c="$OUTPUT_DIR/${sample_name}_output_c.bin"

        # 5. 运行可执行文件并计时
        export LD_LIBRARY_PATH=$_ASCEND_INSTALL_PATH/opp/vendors/customize/op_api/lib:$LD_LIBRARY_PATH
        export LD_LIBRARY_PATH=${_ASCEND_INSTALL_PATH}/tools/simulator/Ascend910B2/lib:$LD_LIBRARY_PATH 
        category=$(basename $category_dir)
        #msprof op --output=./msprof_out_re_colcondense  
         msprof op --output=./msprof_out_re_colcondense  ./output/execute_spmm_op $m $k $n $window_num $block_num $input_row_ptr $input_col $input_values $input_b $input_core_info $output_c $category $sample_name $MODE $input_ref
        if [ $? -ne 0 ]; then
            echo "[ERROR]: Acl executable run failed for sample $sample_name!"
            continue
        fi

        # 提取性能数据
        perf_value="0"
        op_basic_info_file=$(find ./msprof_out_re_colcondense  -name OpBasicInfo.csv 2>/dev/null | head -n 1)
        if [ -n "$op_basic_info_file" ] && [ -f "$op_basic_info_file" ]; then
            perf_value=$(sed -n '2p' "$op_basic_info_file" | cut -d',' -f3 | sed 's/[^0-9.]//g')
            echo "[INFO]: Extracted performance value for $sample_name: $perf_value"
            #强制删除msprof_out文件夹
            if [ -d "./msprof_out_re_colcondense" ]; then
                rm -rf ./msprof_out_re_colcondense  && echo "[INFO]: Deleted ./msprof_out_re_colcondense folder"
            fi
        else
            echo "[WARN]: OpBasicInfo.csv not found for sample $sample_name, use default value 0"
        fi

        # 统计分类性能数据
        if [[ "$perf_value" =~ ^[0-9.]+$ ]]; then
            current_sum=${perf_sum["$category_dir"]:-0.0}
            new_sum=$(echo "$current_sum $perf_value" | awk '{printf "%.8f", $1 + $2}')
            perf_sum["$category_dir"]=$new_sum
            perf_count["$category_dir"]=$(( ${perf_count["$category_dir"]:-0} + 1 ))
        else
            echo "[WARN]: Invalid performance value '$perf_value' for $sample_name, skip statistics"
        fi

        # 6. 比较真值文件
        golden_bin="$sample_dir/golden.bin"
        verify_status="UNKNOWN"  # 初始化验证状态
        if [ -f "$golden_bin" ]; then
            python3 scripts/verify_result.py $output_c $golden_bin $m $n $OUTPUT_DIR $sample_name > "$OUTPUT_DIR/${sample_name}_wrong_indices"
            if [ $? -ne 0 ]; then
                echo "[ERROR]: Verify result failed for sample $sample_name!"
                echo "[$sample_name] (M, K, N, NNZ): $m, $k, $n, $nnz" >> $FAILURE_LOG
                verify_status="FAILED"
            else
                echo "[INFO]: Verify result success for $sample_name!"
                verify_status="SUCCESS"
            fi
        else
            echo "[WARN]: golden.bin not found for $sample_name. Skipping verification."
            verify_status="SKIPPED (no golden file)"
        fi

        echo "[$(date +%Y-%m-%d\ %H:%M:%S)] | Category: $category | Sample: $sample_name | Dims(M,K,N,NNZ): $m,$k,$n,$nnz | Perf: $perf_value | Verify: $verify_status" >> $INDIVIDUAL_FILE


        echo "==================== Finished test for $sample_name ===================="
        echo ""
        #删除中间文件
        rm -rf $OUTPUT_DIR $sample_dir
    done

    # 汇总统计并写入 分类结果文件
    echo -e "\n==================== Performance Statistics ===================="
    # 写入最新结果
    echo -e "[$(date +%Y-%m-%d\ %H:%M:%S)] Category Performance Statistics" >> $CATEGORY_RESULT_FILE
    # 遍历所有分类目录的统计数据
    for cat_dir in "${!perf_sum[@]}"; do
        sum=${perf_sum[$cat_dir]}
        count=${perf_count[$cat_dir]}
        if [ "$count" -eq 0 ]; then
            avg="0.000000"
        else
            avg=$(echo "$sum $count" | awk '{printf "%.6f", $1 / $2}')
        fi
        cat_name=$(basename "$cat_dir")
        # 控制台输出
        echo "[INFO]: Category: $cat_name | Total Samples: $count | Total Perf: $sum | Avg Perf: $avg"
        echo "Category: $cat_name | Dir: $cat_dir | Samples: $count | Total Perf: $sum | Avg Perf: $avg" >> $CATEGORY_RESULT_FILE
        echo "" >> $CATEGORY_RESULT_FILE
    done
    echo "" >> $INDIVIDUAL_FILE
    echo "[INFO]: Individual results: $INDIVIDUAL_FILE"
    echo "[INFO]: Category statistics: $CATEGORY_RESULT_FILE"
}

main
