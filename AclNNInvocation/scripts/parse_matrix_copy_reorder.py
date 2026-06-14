import sys
import os
import numpy as np
import heapq
from datasketch import MinHash,MinHashLSH
import scipy.sparse as ssp
import scipy.io as sio
import queue
import time
def _ceil_to(x: int, base: int) -> int:
    return ((x + base - 1) // base) * base


def optimal_partition_split(rw_ptr, num_cores=20):
    """
    按 TC_block 粒度均分到 num_cores 个核，达到 minimax 理论下界 ceil(total/P)。
    每个核可跨行窗口边界，kernel 通过 blk_start/blk_end 裁剪首末行窗口。

    返回 core_info: int32 数组, shape [num_cores, 4]
    每行: [rw_start, rw_end, blk_start, blk_end]
    """
    n = len(rw_ptr) - 1
    total = int(rw_ptr[n]) if n > 0 else 0

    if total == 0 or n == 0:
        return np.array([[n, n, 0, 0]] * num_cores, dtype=np.int32)

    base = total // num_cores
    remainder = total % num_cores

    core_info = []
    rw = 0
    blk_offset = 0

    for c in range(num_cores):
        quota = base + (1 if c < remainder else 0)

        if rw >= n or quota == 0:
            core_info.append([n, n, 0, 0])
            continue

        rw_start = rw
        bs = blk_offset
        remaining = quota

        while remaining > 0 and rw < n:
            w_i = int(rw_ptr[rw + 1] - rw_ptr[rw])
            avail = w_i - blk_offset
            if avail <= remaining:
                remaining -= avail
                rw += 1
                blk_offset = 0
            else:
                blk_offset += remaining
                remaining = 0

        if rw > rw_start:
            be = int(rw_ptr[rw] - rw_ptr[rw - 1]) if blk_offset == 0 else blk_offset
            rw_end = rw if blk_offset == 0 else rw + 1
        else:
            rw_end = rw + 1
            be = blk_offset

        core_info.append([rw_start, rw_end, bs, be])

    return np.array(core_info, dtype=np.int32)

def parse_mtx_to_bcsr(file_path, BLOCK_M=16, BLOCK_K=16,num_cores=20):
    """
    Parses a .mtx file to extract matrix and convert to BCSR format.
    
    Args:
        file_path (str): The path to the .mtx file.
        BLOCK_M (int): Block size in rows (default 16)
        BLOCK_K (int): Block size in columns (default 16)
    
    Converts the matrix to BCSR format with block size BLOCK_M x BLOCK_K.
    Saves three binary files:
    - row_ptr.bin (int32): Prefix sum of blocks per row window (size BLOCK_M)
    - col_idx.bin (int32): Starting column index for each block (multiple of BLOCK_K)
    - values.bin (float16): All elements in each block (BLOCK_M*BLOCK_K elements per block, row-major)
    """
    # Read file lines and filter comments
    with open(file_path, 'r') as f:
        lines = [line for line in f if not line.startswith('%') and line.strip()]
    
    if not lines:
        raise ValueError("Empty matrix file or only comments found")
    
    # Parse header (first non-comment line)
    header = lines[0].split()
    if len(header) < 2:
        raise ValueError(f"Invalid header in matrix file: {header}")
    
    # Get dimensions (ignore any additional fields like 'general' or 'symmetric')
    M, K, nnz = map(int, header[:3])
    N = 128  # As per problem description
    data_lines = lines[1:]
    block_rows = (M + BLOCK_M - 1) // BLOCK_M
    block_cols = (K + BLOCK_K - 1) // BLOCK_K
    M_pad=block_rows*BLOCK_M
    K_pad=block_cols*BLOCK_K
    N_pad = 128
    blocks = {}
    # Special case: empty matrix
    if nnz == 0 or len(data_lines) == 0:
        block_rows = (M + BLOCK_M - 1) // BLOCK_M
        row_ptr = np.zeros(block_rows + 1, dtype=np.int32)
        col_idx = np.array([], dtype=np.int32)
        values = np.array([], dtype=np.float16)
        
        
        # Save outputs
        sample_name = os.path.splitext(os.path.basename(file_path))[0]
        output_dir = os.path.join(os.path.dirname(file_path), sample_name)
        os.makedirs(output_dir, exist_ok=True)
        
        row_ptr.tofile(os.path.join(output_dir, 'row_ptr.bin'))
        col_idx.tofile(os.path.join(output_dir, 'col_idx.bin'))
        values.tofile(os.path.join(output_dir, 'values.bin'))
        
        with open(os.path.join(output_dir, 'block_info.txt'), 'w') as f:
            f.write(f"BLOCK_M={BLOCK_M}\n")
            f.write(f"BLOCK_K={BLOCK_K}\n")
            f.write(f"Original_M={M}\n")
            f.write(f"Original_K={K}\n")
            f.write(f"Block_rows={block_rows}\n")
            f.write(f"Block_cols={(K + BLOCK_K - 1) // BLOCK_K}\n")
            f.write(f"Num_blocks=0\n")
            f.write(f"Total_values_stored=0\n")
        
        print(f"{M_pad} {K_pad} {N_pad} {nnz} {block_rows} 0 0")
        return
    
    # Parse data lines
    try:
        data = np.array([list(map(float, line.split())) for line in data_lines])
    except ValueError as e:
        raise ValueError(f"Error parsing data lines: {e}. First problematic line: {data_lines[0]}")
    
    if data.ndim == 1:
        data = data.reshape(1, -1)
    
  
    if(len(data[0])==3):
      rows = data[:, 0].astype(int) - 1
      cols = data[:, 1].astype(int) - 1
      vals = data[:, 2].astype(np.float16)
    elif(len(data[0])==2):
      rows = data[:, 0].astype(int) - 1
      cols = data[:, 1].astype(int) - 1
      vals = np.ones((nnz),np.float16)
    else:
      print("data format is invalid")

      
    csr_row_ptr, csr_col_idx, csr_vals = coo_to_csr(rows, cols, vals, M)
    #MKST重排
    new_csr_row_ptr,new_csr_col_idx,new_csr_vals,reorder_ind_ref=reorder_MinHashLSH_KNN(M,BLOCK_M,nnz,csr_row_ptr,csr_col_idx,csr_vals,N=64,thres_KNN=0.2,num_perm=128,lsh_threshold=0.2)
    
    # #DTC重排
    # new_csr_row_ptr, new_csr_col_idx, new_csr_vals, reorder_ind_ref=reorder_double_DTC(M,BLOCK_M,BLOCK_K,nnz,csr_row_ptr,csr_col_idx,csr_vals)

    block_rows = (M + BLOCK_M - 1) // BLOCK_M
    M_pad=block_rows*BLOCK_M

    #填充
    for padrow in range(M,M_pad):
        reorder_ind_ref.append(padrow)

    new_csr_row_ptr = np.array(new_csr_row_ptr, dtype=np.int32)  
    new_csr_col_idx = np.array(new_csr_col_idx, dtype=np.int32)  
    new_csr_vals = np.array(new_csr_vals, dtype=np.float32)      

    #csr转coo
    rows_new = np.repeat(np.arange(M, dtype=np.int32), np.diff(new_csr_row_ptr))
    cols_new = new_csr_col_idx
    values_new = new_csr_vals
    

    # Populate blocks from csr to bcsr
    for r, c, v in zip(rows_new, cols_new, values_new):
        # Skip elements outside matrix dimensions (shouldn't happen, but safe)
        if r >= M or c >= K:
            continue
        # if(r==1):
        #     print(c)
        block_row = r // BLOCK_M
        block_col = c // BLOCK_K
        local_row = r % BLOCK_M
        local_col = c % BLOCK_K
        
        key = (block_row, block_col)
        if key not in blocks:
            blocks[key] = []
        blocks[key].append((local_row, local_col, v))
    
    # Initialize output arrays
    row_ptr = [0]  # Prefix sum array
    all_block_cols = []  # Starting columns for each block
    all_block_vals = []  # Flattened block values

    # Process each block row
    for br in range(block_rows):
        blocks_in_row = 0
        row_block_cols = []
        row_block_vals = []
        
        # Process each block column in this block row
        for bc in range(block_cols):
            key = (br, bc)
            if key in blocks:
                blocks_in_row += 1
                row_block_cols.append(bc * BLOCK_K)  # Starting column index
                
                # Create dense block with zero padding
                block_data = np.zeros((BLOCK_M, BLOCK_K), dtype=np.float16)
                
                # Fill non-zero elements
                for lr, lc, val in blocks[key]:
                    # Only fill if within original matrix bounds
                    global_row = br * BLOCK_M + lr
                    global_col = bc * BLOCK_K + lc
                    if global_row < M and global_col < K:
                        block_data[lr, lc] = np.float16(val)
                
                # Flatten in row-major order
                row_block_vals.append(block_data.flatten())
        
        # Update prefix sum
        row_ptr.append(row_ptr[-1] + blocks_in_row)
        # Append row data to global arrays
        if row_block_cols:
            all_block_cols.extend(row_block_cols)
            all_block_vals.extend(row_block_vals)
    
    # Convert to numpy arrays
    row_ptr_np = np.array(row_ptr, dtype=np.int32)
    col_idx_np = np.array(all_block_cols, dtype=np.int32)
    values_np = np.concatenate(all_block_vals) if all_block_vals else np.array([], dtype=np.float16)
    reorder_ref_np=np.array(reorder_ind_ref,dtype=np.int32)
    
    # Create output directory
    sample_name = os.path.splitext(os.path.basename(file_path))[0]
    output_dir = os.path.join(os.path.dirname(file_path), sample_name+"_re")
    os.makedirs(output_dir, exist_ok=True)
    
    #Save binary files
    row_ptr_np.tofile(os.path.join(output_dir, 'row_ptr.bin'))
    col_idx_np.tofile(os.path.join(output_dir, 'col_idx.bin'))
    values_np.tofile(os.path.join(output_dir, 'values.bin'))
    reorder_ref_np.tofile(os.path.join(output_dir,'reorder_ref.bin'))


    #不用时注释掉就可以，只是方便查看重排后的结果
    # np.savetxt(os.path.join(output_dir, 'row_ptr.txt'),row_ptr_np,delimiter="\n",fmt="%d")
    # np.savetxt(os.path.join(output_dir, 'col_idx.txt'), col_idx_np,delimiter="\n",fmt="%d")
    # np.savetxt(os.path.join(output_dir, 'values.txt'),values_np,delimiter="\n",fmt="%.10f")
    # np.savetxt(os.path.join(output_dir, 'reorder_ref.txt'), reorder_ref_np,delimiter="\n",fmt="%d")
    # Generate padded B (x2_gm.bin) with deterministic seed based on sample_name
    rng = np.random.default_rng(abs(hash(sample_name)) % (2**32))
    b_pad = np.zeros((K_pad, N_pad), dtype=np.float16)

    # Fill only the meaningful part (original K rows), rest remains 0
    # This avoids any ambiguity if kernel reads padded rows.
    if K > 0:
        b_pad[:K, :N_pad] = rng.integers(1, 11, size=(K, N_pad), dtype=np.int32).astype(np.float16)

    golden_list=[[0 for _ in range(N_pad)] for _ in range(M_pad)]
    for d_idx in range(nnz):
        for db_idx in range(N_pad):
           golden_list[rows_new[d_idx]][db_idx]+=values_new[d_idx]*b_pad[cols_new[d_idx]][db_idx]
    golden=np.array(golden_list,dtype=np.float32)
    
    # Save B and golden
    b_pad.tofile(os.path.join(output_dir, 'x2_gm.bin'))
    golden.tofile(os.path.join(output_dir, 'golden.bin'))
    nozero_rate=round(nnz/(len(all_block_cols)*BLOCK_M*BLOCK_K),2)
    # Save metadata
    with open(os.path.join(output_dir, 'block_info.txt'), 'w') as f:
        f.write(f"BLOCK_M={BLOCK_M}\n")
        f.write(f"BLOCK_K={BLOCK_K}\n")
        f.write(f"Original_M={M}\n")
        f.write(f"Original_K={K}\n")
        f.write(f"Block_rows={block_rows}\n")
        f.write(f"Block_cols={block_cols}\n")
        f.write(f"Num_blocks={len(all_block_cols)}\n")
        f.write(f"Total_values_stored={len(values_np)}\n")
    
    # Print dimensions for calling script
    print(f"{M_pad} {K_pad} {N_pad} {nnz} {block_rows} {len(all_block_cols)} {nozero_rate}")




def parse_mtx_to_bcsr_colcondense(file_path, BLOCK_M=16, BLOCK_K=16,num_cores=20):
    """
    Parses a .mtx file to extract matrix and convert to BCSR format.
    
    Args:
        file_path (str): The path to the .mtx file.
        BLOCK_M (int): Block size in rows (default 16)
        BLOCK_K (int): Block size in columns (default 16)
    
    Converts the matrix to BCSR format with block size BLOCK_M x BLOCK_K.
    Saves three binary files:
    - row_ptr.bin (int32): Prefix sum of blocks per row window (size BLOCK_M)
    - col_idx.bin (int32): Starting column index for each block (multiple of BLOCK_K)
    - values.bin (float16): All elements in each block (BLOCK_M*BLOCK_K elements per block, row-major)
    """
    # Read file lines and filter comments
    with open(file_path, 'r') as f:
        lines = [line for line in f if not line.startswith('%') and line.strip()]
    
    if not lines:
        raise ValueError("Empty matrix file or only comments found")
    
    # Parse header (first non-comment line)
    header = lines[0].split()
    if len(header) < 2:
        raise ValueError(f"Invalid header in matrix file: {header}")
    
    # Get dimensions (ignore any additional fields like 'general' or 'symmetric')
    M, K, nnz = map(int, header[:3])
    N = 128  # As per problem description
    data_lines = lines[1:]
    block_rows = (M + BLOCK_M - 1) // BLOCK_M
    block_cols = (K + BLOCK_K - 1) // BLOCK_K
    M_pad=block_rows*BLOCK_M
    K_pad=block_cols*BLOCK_K
    N_pad = 128
    blocks = {}
    # Special case: empty matrix
    if nnz == 0 or len(data_lines) == 0:
        block_rows = (M + BLOCK_M - 1) // BLOCK_M
        row_ptr = np.zeros(block_rows + 1, dtype=np.int32)
        col_idx = np.array([], dtype=np.int32)
        values = np.array([], dtype=np.float16)
        
        # Save outputs
        sample_name = os.path.splitext(os.path.basename(file_path))[0]
        output_dir = os.path.join(os.path.dirname(file_path), sample_name)
        os.makedirs(output_dir, exist_ok=True)
        
        row_ptr.tofile(os.path.join(output_dir, 'row_ptr.bin'))
        col_idx.tofile(os.path.join(output_dir, 'col_idx.bin'))
        values.tofile(os.path.join(output_dir, 'values.bin'))
        
        with open(os.path.join(output_dir, 'block_info.txt'), 'w') as f:
            f.write(f"BLOCK_M={BLOCK_M}\n")
            f.write(f"BLOCK_K={BLOCK_K}\n")
            f.write(f"Original_M={M}\n")
            f.write(f"Original_K={K}\n")
            f.write(f"Block_rows={block_rows}\n")
            f.write(f"Block_cols={(K + BLOCK_K - 1) // BLOCK_K}\n")
            f.write(f"Num_blocks=0\n")
            f.write(f"Total_values_stored=0\n")
        
        print(f"{M_pad} {K_pad} {N_pad} {nnz} {block_rows} 0 0")
        return
    
    # Parse data lines
    try:
        data = np.array([list(map(float, line.split())) for line in data_lines])
    except ValueError as e:
        raise ValueError(f"Error parsing data lines: {e}. First problematic line: {data_lines[0]}")
    
    if data.ndim == 1:
        data = data.reshape(1, -1)
    
    if(len(data[0])==3):
      rows = data[:, 0].astype(int) - 1
      cols = data[:, 1].astype(int) - 1
      vals = data[:, 2].astype(np.float16)
    elif(len(data[0])==2):
      rows = data[:, 0].astype(int) - 1
      cols = data[:, 1].astype(int) - 1
      vals = np.ones((nnz),np.float16)
    else:
      print("data format is invalid")
    
    csr_row_ptr, csr_col_idx, csr_vals = coo_to_csr(rows, cols, vals, M)
    #MKST重排
    new_csr_row_ptr,new_csr_col_idx,new_csr_vals,reorder_ind_ref=reorder_MinHashLSH_KNN(M,BLOCK_M,nnz,csr_row_ptr,csr_col_idx,csr_vals,N=64,thres_KNN=0.2,num_perm=128,lsh_threshold=0.2)

    # #DTC重排
    #new_csr_row_ptr, new_csr_col_idx, new_csr_vals, reorder_ind_ref=reorder_double_DTC(M,BLOCK_M,BLOCK_K,nnz,csr_row_ptr,csr_col_idx,csr_vals)

    block_rows = (M + BLOCK_M - 1) // BLOCK_M
    M_pad=block_rows*BLOCK_M
    #重排序映射添加填充行映射

    #对M_pad进行填充
    for padrow in range(M,M_pad):
        reorder_ind_ref.append(padrow)

    new_csr_row_ptr = np.array(new_csr_row_ptr, dtype=np.int32)  
    new_csr_col_idx = np.array(new_csr_col_idx, dtype=np.int32)  
    new_csr_vals = np.array(new_csr_vals, dtype=np.float16)      

    #csr转coo
    rows_new = np.repeat(np.arange(M, dtype=np.int32), np.diff(new_csr_row_ptr))
    cols_new = new_csr_col_idx
    values_new = new_csr_vals

    
    
    #column condense start
    sparseAtoB=[0]*nnz*BLOCK_K
    # sparseAtoB=[0]*nnz
    rw_partition = [0]*(block_rows+1)
    TCcolcount_rw=0
    TCcolcount=0
    unique_col={}
    for csr_row in range(M):
        rw_now=csr_row//BLOCK_M
        if(csr_row%BLOCK_M==0):
            all_col_in_rw=[]
            TCcolcount_rw=0
        for csr_ind in range(new_csr_row_ptr[csr_row],new_csr_row_ptr[csr_row+1]):
            all_col_in_rw.append(new_csr_col_idx[csr_ind])
        if((csr_row%BLOCK_M==BLOCK_M-1 or csr_row==M-1) and (len(all_col_in_rw)!=0)):
            lastcol=-1
            all_col_in_rw.sort()
            rw_now=csr_row//BLOCK_M
            for csr_col in range(len(all_col_in_rw)):
                if lastcol!=all_col_in_rw[csr_col]:
                    lastcol=all_col_in_rw[csr_col]
                    sparseAtoB[rw_partition[rw_now]*BLOCK_K+TCcolcount_rw]=all_col_in_rw[csr_col]
                    unique_col[(rw_now,all_col_in_rw[csr_col])]=TCcolcount_rw
                    TCcolcount_rw+=1
            for zerocol in range(TCcolcount_rw,(TCcolcount_rw+BLOCK_K-1)//BLOCK_K*BLOCK_K):
                sparseAtoB[rw_partition[rw_now]*BLOCK_K+zerocol]=sparseAtoB[rw_partition[rw_now]*BLOCK_K+zerocol-1]
        rw_partition[rw_now+1]=rw_partition[rw_now]+(TCcolcount_rw+BLOCK_K-1)//BLOCK_K
    TCcount=rw_partition[block_rows]
    sparseAtoB=sparseAtoB[0:TCcount*BLOCK_K]
    all_block_vals = [0]*(TCcount*BLOCK_M*BLOCK_K)  # Flattened block values
    
    for csr_row in range(M):
        rw_now=csr_row//BLOCK_M 
        for csr_ind in range(new_csr_row_ptr[csr_row],new_csr_row_ptr[csr_row+1]):
            now_col=unique_col[(rw_now,new_csr_col_idx[csr_ind])]
            TCid=rw_partition[rw_now]+now_col//BLOCK_K
            # if(TCid==rw_partition[block_rows]-1):
            #     print("correct")
            all_block_vals[TCid*BLOCK_M*BLOCK_K+(csr_row%BLOCK_M)*BLOCK_K+now_col%BLOCK_K]=new_csr_vals[csr_ind]


    # Convert to numpy arrays
    rw_ptr_np = np.array(rw_partition, dtype=np.int32)
    TC_col_ref_np = np.array(sparseAtoB, dtype=np.int32)
    values_np = np.array(all_block_vals,dtype=np.float16)
    reorder_ref_np=np.array(reorder_ind_ref,dtype=np.int32)
    
    # Create output directory
    sample_name = os.path.splitext(os.path.basename(file_path))[0]
    output_dir = os.path.join(os.path.dirname(file_path), sample_name+"_re_colcondense")
    os.makedirs(output_dir, exist_ok=True)
    
    #Save binary files
    rw_ptr_np.tofile(os.path.join(output_dir, 'rw_ptr.bin'))
    TC_col_ref_np.tofile(os.path.join(output_dir, 'TC_col_ref.bin'))
    values_np.tofile(os.path.join(output_dir, 'values.bin'))
    reorder_ref_np.tofile(os.path.join(output_dir,'reorder_ref.bin'))
    #不用时注释掉就可以，只是方便查看重排后的结果
    # np.savetxt(os.path.join(output_dir, 'rw_ptr.txt'),rw_ptr_np,delimiter="\n",fmt="%d")
    # np.savetxt(os.path.join(output_dir, 'TC_col_ref.txt'),TC_col_ref_np,delimiter="\n",fmt="%d")
    # np.savetxt(os.path.join(output_dir, 'values.txt'),values_np,delimiter="\n",fmt="%.10f")
    # np.savetxt(os.path.join(output_dir, 'reorder_ref.txt'), reorder_ref_np,delimiter="\n",fmt="%d")
    # Generate padded B (x2_gm.bin) with deterministic seed based on sample_name
    rng = np.random.default_rng(abs(hash(sample_name)) % (2**32))
    b_pad = np.zeros((K_pad, N_pad), dtype=np.float16)

    # Fill only the meaningful part (original K rows), rest remains 0
    # This avoids any ambiguity if kernel reads padded rows.

    if K > 0:
        b_pad[:K, :N_pad] = rng.integers(1, 11, size=(K, N_pad), dtype=np.int32).astype(np.float16)

    # Golden: (M_pad x K_pad) @ (K_pad x N_pad) -> (M_pad x N_pad)
    golden_list=[[0.000000 for _ in range(N_pad)] for _ in range(M_pad)]
    for d_idx in range(nnz):
        for db_idx in range(N_pad):
           golden_list[rows_new[d_idx]][db_idx]+=float(values_new[d_idx]*b_pad[cols_new[d_idx]][db_idx])
    golden=np.array(golden_list,dtype=np.float32)

    # Save B and golden
    b_pad.tofile(os.path.join(output_dir, 'x2_gm.bin'))
    golden.tofile(os.path.join(output_dir, 'golden.bin'))
    nozero_rate=round(nnz/(TCcount),2)
    # Save metadata
    with open(os.path.join(output_dir, 'block_info.txt'), 'w') as f:
        f.write(f"BLOCK_M={BLOCK_M}\n")
        f.write(f"BLOCK_K={BLOCK_K}\n")
        f.write(f"Original_M={M}\n")
        f.write(f"Original_K={K}\n")
        f.write(f"Block_rows={block_rows}\n")
        f.write(f"Block_cols={block_cols}\n")
        f.write(f"Num_blocks={TCcount}\n")
        f.write(f"Total_values_stored={len(values_np)}\n")
    content=f"{M_pad} {K_pad} {N_pad} {nnz} {block_rows} {TCcount} {nozero_rate}"

    core_info = optimal_partition_split(rw_ptr_np, num_cores=num_cores)
    core_info.tofile(os.path.join(output_dir, "core_info.bin"))
    np.savetxt(os.path.join(output_dir, "core_info.txt"), core_info, fmt="%d")

    # Print dimensions for calling script
    print(content)
    # with open(os.path.join(output_dir, 'simple_info.txt'),"w") as sf:
    #     sf.write(content+"\n")


def coo_to_csr(rows,cols,values,rowlen):
    csr_row_ptr=np.array([0 for i in range(rowlen+1)])
    csr_col_idx=np.array([-1 for i in range(len(values))])
    csr_vals=np.array([-1.0 for i in range(len(values))],np.float32)
    for row in rows:
        csr_row_ptr[row+1]+=1
    csr_row_ptr=csr_row_ptr.cumsum()
    row_temp_offset=list(csr_row_ptr)
    for (row,col,value) in zip(rows,cols,values):
        csr_col_idx[row_temp_offset[row]]=col
        csr_vals[row_temp_offset[row]]=value
        row_temp_offset[row]+=1
    #verify the correction
    for row in range(rowlen):
        if(row_temp_offset[row]!=csr_row_ptr[row+1]):
            print("[error]:the process of coo transform to csr is error,and the err row is {row}")
    return (csr_row_ptr,csr_col_idx,csr_vals)

# ===============================MKST reorder===================================
def build_minhash_lsh_buckets(num_row, row_ptr, col_idx, num_perm=128, lsh_threshold=0.2):
    lsh = MinHashLSH(threshold=lsh_threshold, num_perm=num_perm)
    minhash_list = []
    row_sets = []

    for i in range(num_row):
        m = MinHash(num_perm=num_perm)
        col_set = []
        for p in range(row_ptr[i], row_ptr[i+1]):
            c = col_idx[p]
            m.update(str(c).encode('utf-8'))
            col_set.append(c)
        lsh.insert(i, m)
        minhash_list.append(m)
        row_sets.append(col_set)

    return lsh, minhash_list, row_sets

def build_knn_graph_from_lsh(num_row, row_ptr, col_idx, lsh, minhash_list, row_sets, N=128, thres_KNN=0.2):
    heaps = [[] for _ in range(num_row)]
    visited_pairs = set()

    def jaccard(l1, l2):
        s1, s2 = set(l1), set(l2)
        if len(s1.union(s2)) == 0:
            return 0.0
        return len(s1.intersection(s2)) / len(s1.union(s2))
    num_zero=0
    for i in range(num_row):
        if row_ptr[i] == row_ptr[i+1]:
            num_zero+=1
            continue
        candidates = lsh.query(minhash_list[i])
        for j in candidates:
            if j == i:
                continue
            key = (i,j) if i < j else (j,i)
            if key in visited_pairs:
                continue
            visited_pairs.add(key)
            sim = jaccard(row_sets[i], row_sets[j])
            if sim < thres_KNN:
                continue
            #row i heap push
            if len(heaps[i]) < N:
                heapq.heappush(heaps[i], (sim, j))
            else:
                if sim > heaps[i][0][0]:
                    heapq.heappushpop(heaps[i], (sim, j))
            #row j heap push
            if len(heaps[j]) < N:
                heapq.heappush(heaps[j], (sim, i))
            else:
                if sim > heaps[j][0][0]:
                    heapq.heappushpop(heaps[j], (sim, i))

    graph = [[] for _ in range(num_row)]
    for i in range(num_row):
        graph[i] = [(j, sim) for sim, j in heaps[i]]
    return graph,num_zero


def build_mst(graph):
    n = len(graph)
    visited = [False]*n
    mst = [[] for _ in range(n)]
    for start in range(n):
        if visited[start]:
            continue
        visited[start] = True
        heap = []
        for nei, w in graph[start]:
            heapq.heappush(heap, (-w, start, nei))
        while heap:
            w, u, v = heapq.heappop(heap)
            if visited[v]:
                continue
            visited[v] = True
            mst[u].append((v, -w))
            mst[v].append((u, -w))
            for nei, w2 in graph[v]:
                if not visited[nei]:
                    heapq.heappush(heap, (-w2, v, nei))
    return mst


def dfs_order_with_block(mst, block_m, row_ptr, col_idx, threshold_union=0.6):
    n = len(mst)
    visited = [False] * n
    order = []

    for i in range(n):
        if not visited[i]:
            current_block = []
            stack = [i]         
            visited[i] = True   

            while stack:
                u = stack.pop()          
                current_block.append(u) 
                neighbors = sorted(mst[u], key=lambda x: -x[1])
                for v, _ in reversed(neighbors):
                    if not visited[v]:
                        visited[v] = True
                        stack.append(v)
            order.extend(current_block)
    return order

def reorder_csr(order, row_ptr, col_idx, vals):
    n = len(order)
    new_row_ptr = np.zeros(n+1, dtype=row_ptr.dtype)
    new_col_idx = []
    new_vals = []
    for i, old_i in enumerate(order):
        start, end = row_ptr[old_i], row_ptr[old_i+1]
        new_col_idx.extend(col_idx[start:end])
        new_vals.extend(vals[start:end])
        new_row_ptr[i+1] = new_row_ptr[i] + (end - start)
    return new_row_ptr, np.array(new_col_idx), np.array(new_vals)


def reorder_MinHashLSH_KNN(num_row, block_m, nnz, row_ptr, col_idx, vals, N=64, thres_KNN=0.5,num_perm=128, lsh_threshold=0.2):

    lsh, minhash_list, row_sets = build_minhash_lsh_buckets(num_row, row_ptr, col_idx, num_perm, lsh_threshold)

    graph,num_zero = build_knn_graph_from_lsh(num_row, row_ptr, col_idx, lsh, minhash_list, row_sets, N=N, thres_KNN=thres_KNN)
    mst = build_mst(graph)
    order = dfs_order_with_block(mst, block_m, row_ptr, col_idx)
    new_row_ptr, new_col_idx, new_vals = reorder_csr(order, row_ptr, col_idx, vals)
    return new_row_ptr, new_col_idx, new_vals, order


    
#copy from code of DTC-SPMM and modify, We reuse codes from  (https://github.com/HPMLL/DTC-SpMM_ASPLOS24)
def reorder_double_DTC(num_row,block_m,block_k,nnz,ptr,idx,vals,per=128,thres=0.2,cluster_thres=0.2,cblock_m=128):
    #print("=== Init lsh ===")
    t0 = time.time()
    lsh = MinHashLSH(threshold=thres, num_perm=per)
    allver = []
    lists = [[] for i in range(num_row)]

    for i in range(num_row):
        m = MinHash(num_perm=per)
        lastcol=-1
        for iter in range((int)(ptr[i]), (int)(ptr[i+1])):
            nowblockcol=idx[iter]
            if(lastcol!=nowblockcol):
              m.update(str(nowblockcol).encode('utf-8'))
              lists[i].append(nowblockcol)
              lastcol=nowblockcol
        lsh.insert(i, m)
        allver.append(m)

    t1 = time.time()
    #print("init LSH time (s)", t1 - t0)

    def root(i):
        if cluster_id[i]!=cluster_id[cluster_id[i]]:
            cluster_id[i]=root(cluster_id[i])
        return cluster_id[i]
    #生成独一无二的pair标识
    def makenum(a, b):
        if a > b:
            tmp = a
            a = b
            b = tmp
        return a * num_row + b
    def jd(l1,l2):
        if len(l1) == 0 or len(l2) == 0:
            return 0
        s1 = set(l1)
        s2 = set(l2)
        return (float)(len(s1.intersection(s2))) / len(s1.union(s2))

    class Pair(object):
        def __init__(self,p1,p2,similarity):
            self.p1 = p1
            self.p2 = p2
            self.simi = similarity
        def __lt__(self,other): # operator < 
            return self.simi > other.simi
        def __str__(self):
            return str(self.p1) + ' ' + str(self.p2) + ' ' + str(self.simi)

    que = queue.PriorityQueue()
    sset = set()

    #print("=== DTC reorder===")
    t2 = time.time()
    for i in range(num_row):
        if ptr[i] == ptr[i + 1]:
            continue
        res = lsh.query(allver[i])
        for simi_row in res:
                if simi_row == i or makenum(simi_row, i) in sset:
                    continue
                que.put(Pair(simi_row, i, jd(lists[i],lists[simi_row])))
                sset.add(makenum(i,simi_row))

    #print("queue size: ", que.qsize())
    t3 = time.time()
    #print("query LSH time (s): ", t3 - t2)
    cluster_id = [i for i in range(num_row)]
    cluster_sz = [1 for i in range(num_row)]
    deleted = [0 for i in range(num_row)]
    num_cluster = num_row

    t4 = time.time()
    while (not que.empty()) and num_cluster > 0:
        item = que.get()
        p1 = item.p1
        p2 = item.p2
        sset.remove(makenum(p1, p2))
        if p1 == cluster_id[p1] and p2 == cluster_id[p2]:
            if deleted[p1] or deleted[p2]:
                continue
            if cluster_sz[p1] < cluster_sz[p2]:
                cluster_id[p1] = p2
                num_cluster = num_cluster - 1
                cluster_sz[p2] = cluster_sz[p1] + cluster_sz[p2]
                if cluster_sz[p2] >= block_m:
                    deleted[p2] = 1
                    num_cluster = num_cluster - 1
            else:
                cluster_id[p2] = p1
                num_cluster = num_cluster - 1
                cluster_sz[p1] = cluster_sz[p1] + cluster_sz[p2]
                if cluster_sz[p1] >= block_m:
                    deleted[p1] = 1
                    num_cluster = num_cluster - 1
        else:
            p1 = root(p1)
            p2 = root(p2)
            if deleted[p1] or deleted[p2]:
                continue
            if p1 != p2 and not makenum(p1, p2) in sset:
                que.put(Pair(p1, p2, jd(lists[p1], lists[p2])))
                sset.add(makenum(p1, p2))
    t5 = time.time()
    #print("clustering time (s): ", t5 - t4)

    clusters = {}
    t6 = time.time()
    for i in range(num_row):
        ro = root(i)
        if ro in clusters:
            clusters[ro].append(i)
        else:
            clusters[ro] = [i]

    t7 = time.time()
    #print("put into clusters time (s): ", t7 - t6)
    cluster_num = len(clusters)
    #print("cluster_num:", cluster_num)

    #print("=== Cache-Aware level clustering ===")
    key = list(clusters.keys())
    ## cluster twice to improve cache behaviour
    def makenum_c(a, b):
        if a > b:
            tmp = a
            a = b
            b = tmp
        return a * cluster_num + b

    per_c = 128   # for 4090
    lsh_c = MinHashLSH(threshold=cluster_thres, num_perm=per_c)
    allver_c = []
    lists_c = [[] for i in range(cluster_num)]   # unique column indices for each cluster lists_c[i]: indices for cluster i
    cnt = 0
    for i in clusters:
        m = MinHash(num_perm=per_c)
        list_cluster_i = [] 
        for node in clusters[i]:
            list_cluster_i = list_cluster_i +  lists[node]
        list_cluster_i = list(set(list_cluster_i))
        lists_c[cnt] = list_cluster_i
        for ind in list_cluster_i:
            m.update(str(ind).encode("utf-8"))
        lsh_c.insert(str(cnt), m)
        allver_c.append(m)
        cnt = cnt + 1
    que_c = queue.PriorityQueue()
    sset_c = set()
    t2 = time.time()
    for i in range(cluster_num):
        if i % 1000 == 0:
            #print("reach cluster: ", i)
            pass
        if(len(lists_c[i])==0):
            continue
        res = lsh_c.query(allver_c[i])
        for item in res:
            if (int)(item) == i or makenum_c(i, (int)(item)) in sset_c:
                continue
            if len(lists_c[(int)(item)]) == 0:
                continue
            que_c.put(Pair(i, (int)(item), jd(lists_c[i], lists_c[(int)(item)])))
            sset_c.add(makenum_c(i, (int)(item)))
    #print("cluster queue size:", que_c.qsize())
    t3 = time.time()
    #print("query cluster LSH time (s): ", t3 - t2)
    cluster_id_c = [i for i in range(cluster_num)]
    cluster_sz_c = [1 for i in range(cluster_num)]
    deleted_c = [0 for i in range(cluster_num)]
    num_cluster_c = cluster_num
    # def root_c(i):
    #     while i != cluster_id_c[i]:
    #         cluster_id_c[i] = cluster_id_c[cluster_id_c[i]]
    #         i = cluster_id_c[i]
    #     return i
    def root_c(i):
        if cluster_id_c[i]!=cluster_id_c[cluster_id_c[i]]:
            cluster_id_c[i]=root_c(cluster_id_c[i])
        return cluster_id_c[i]
    t4 = time.time()
    while (not que_c.empty()) and num_cluster_c > 0:
        item = que_c.get()
        p1 = item.p1
        p2 = item.p2
        sset_c.remove(makenum_c(p1, p2))
        if p1 == cluster_id_c[p1] and p2 == cluster_id_c[p2]:
            if deleted_c[p1] or deleted_c[p2]:
                continue
            if cluster_sz_c[p1] < cluster_sz_c[p2]:
                cluster_id_c[p1] = p2
                num_cluster_c = num_cluster_c - 1
                cluster_sz_c[p2] = cluster_sz_c[p1] + cluster_sz_c[p2]
                if cluster_sz_c[p2] >= cblock_m:
                    deleted_c[p2] = 1
                    num_cluster_c = num_cluster_c - 1
            else:
                cluster_id_c[p2] = p1
                num_cluster_c = num_cluster_c - 1
                cluster_sz_c[p1] = cluster_sz_c[p1] + cluster_sz_c[p2]
                if cluster_sz_c[p1] >= cblock_m:
                    deleted_c[p1] = 1
                    num_cluster_c = num_cluster_c - 1
        else:
            p1 = root_c(p1)
            p2 = root_c(p2)
            if deleted_c[p1] or deleted_c[p2]:
                continue
            if p1 != p2 and not makenum_c(p1, p2) in sset_c:
                que_c.put(Pair(p1, p2, jd(lists_c[p1], lists_c[p2])))
                sset_c.add(makenum_c(p1, p2))
    t5 = time.time()
    #print("cluster clustering time (s): ", t5 - t4)
    clusters_c = {}
    t6 = time.time()
    for i in range(cluster_num):
        ro = root_c(i)
        if ro in clusters_c:
            clusters_c[ro].append(i)
        else:
            clusters_c[ro] = [i]
    cluster_cluster_num = len(clusters_c)
    #print("cluster_of_cluster_num: ", cluster_cluster_num)
    t7 = time.time()
    #print("put clusters into clusters time (s): ", t7 - t6)
    # print(clusters_c)

    #print("=== Save results ===")
    reorder_ind_re = []
    for j in clusters_c:
        for k in clusters_c[j]:
            clustersk = clusters[key[k]]
            for item in clustersk:
                reorder_ind_re.append(item)
    
    reorder_ind=[-1]*num_row
    new_ptr=[0]*(num_row+1)
    for i in range(num_row):
        reorder_ind[reorder_ind_re[i]]=i
        new_ptr[i+1]=new_ptr[i]+ptr[reorder_ind_re[i]+1]-ptr[reorder_ind_re[i]]
    for i in range(num_row):
        if reorder_ind[i]==-1:
            print("[error]: DTC_reorder row is left")
    
    new_idx=[0]*nnz
    new_vals=[0.0]*nnz
    for i in range(num_row):
        new_start_ind=new_ptr[i]
        new_end_ind=new_ptr[i+1]
        start_ind=ptr[reorder_ind_re[i]]
        for ind in range(new_start_ind,new_end_ind):
            new_idx[ind]=idx[start_ind]
            new_vals[ind]=vals[start_ind]
            start_ind+=1


    return (new_ptr,new_idx,new_vals,reorder_ind_re)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python parse_matrix.py <path_to_mtx_file>", file=sys.stderr)
        sys.exit(1)
    
    mtx_file = sys.argv[1]
    parse_mtx_to_bcsr_colcondense(mtx_file)
