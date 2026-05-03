
function main {
    declare -A perf_sum perf_count
    # 定义输入输出目录
    INPUTS_DIR="../inputs"
    OUTPUT_DIR="../reorder_bin"
    MODE="reorder"

    cate=""
    cate_refile=""

  while read -r incatedir; do
      while read -r sample_mtx; do
    IFS='/' read -ra dirlist <<< "$sample_mtx"
    cate=${dirlist[-2]}
    catedir="${OUTPUT_DIR}/${cate}"
    sample_name=${dirlist[-1]%.mtx}
    echo "==================== Gene_reorder for $sample_name ===================="
    #out_sample_dir="${catedir}/${dirlist[-1]}"
    mkdir -p "$catedir"
    python3 /root/autodl-tmp/zsx/ascendC_bcsr/AclNNInvocation/scripts/parse_matrix_copy_reorder.py $sample_mtx
    if [ $? -ne 0 ]; then
        echo "[ERROR]: Failed to parse matrix dimensions for $mtx_file"
        continue
    fi
    echo "==================== Gene_finish ===================="
    echo ""

  done   < <(find "$incatedir" -maxdepth 1 -type f -name "*.mtx")
done  < <(find "$INPUTS_DIR"  -maxdepth 1 -mindepth 1 -type d -name "*") 
}

main
