#!/bin/bash

# 运行测试
run() {
    mkdir -p $OUTPUT_DIR
    output_file=$OUTPUT_DIR/U4THR-nv.txt
    echo "START" > $output_file

    for dir in `ls $TEST_DIR_u4`
    do
        if [[ -d $TEST_DIR_u4"/"$dir ]]; then
            echo "Processing Directory: $dir"
            echo "DIR: $dir" >> $output_file

            for file in `ls $TEST_DIR_u4"/"$dir`
            do
                if [[ $file == *".u4" ]]; then
                    echo "   FILE: $file"
                    echo "FILE: $file" >> $output_file

                    file_path="$TEST_DIR_u4/$dir/$file"
                    file_size=$(stat -c%s "$file_path")
                    file_KB=$(echo "$file_size / 1024" | bc)
                    echo "SIZE: $file_KB KB" >> $output_file

                    
                    # Step 1: huffman
                    gdeflate_output=$($NVCOMP_PATH/benchmark_gdeflate_chunked -a 0 -f "$file_path")
                    huffman_cmp_thr=$(echo "$gdeflate_output" | grep "compression throughput" | awk 'NR==1 {print $4}')
                    huffman_decmp_thr=$(echo "$gdeflate_output" | grep "decompression throughput" | awk '{print $4}')
                    echo "  NV-Huffman Compressed Thr: $huffman_cmp_thr" >> $output_file
                    echo "  NV-Huffman Decompressed Thr: $huffman_decmp_thr" >> $output_file

                    # Step 2: ans
                    ans_output=$($NVCOMP_PATH/benchmark_ans_chunked -f "$file_path")
                    ans_cmp_thr=$(echo "$ans_output" | grep "compression throughput" | awk 'NR==1 {print $4}')
                    ans_decmp_thr=$(echo "$ans_output" | grep "decompression throughput" | awk '{print $4}')
                    echo "  NV-ANS Compressed Thr: $ans_cmp_thr" >> $output_file
                    echo "  NV-ANS Decompressed Thr: $ans_decmp_thr" >> $output_file

                    # Step 3: ans
                    ans_output=$($PANS_CMP_nv "$file_path" "$file_path.tmp")
                    ans_cmp_time=$(echo "$ans_output" | grep "comp   time" | awk 'NR==1 {print $3}')
                    ans_output=$($PANS_DECMP_nv "$file_path.tmp" "$file_path.out")
                    ans_decmp_time=$(echo "$ans_output" | grep "decomp time" | awk 'NR==1 {print $3}')
                    cmp_thr=$(echo "scale=9; $file_size / 1024 / 1024 / 1024/ (($ans_cmp_time) / 1000)" | bc)
                    decmp_thr=$(echo "scale=9; $file_size / 1024 / 1024 / 1024 / (($ans_decmp_time) / 1000)" | bc)
                    echo "  PANS-CUDA Compressed Thr: $cmp_thr" >> $output_file
                    echo "  PANS-CUDA Decompressed Thr: $decmp_thr" >> $output_file
                    
                    
                    # Step 4: ADM
                    adm_output_path="$file_path.adm"
                    num_ele=$(echo "$file_size / 4" | bc)
                    $ADM32_nv -u4 "$file_path" "$adm_output_path" --dims 1 "$num_ele"
                    adm_output=$($ADM32_nv -u4 "$file_path" "$adm_output_path" --dims 1 "$num_ele")
                    adm_cmp_time=$(echo "$adm_output" | grep "Total Cmp Time: " | awk 'NR==1 {print $4}')
                    adm_decmp_time=$(echo "$adm_output" | grep "Total Decmp Time:" | awk '{print $4}')
                    ans_output=$($PANS_CMP_nv "$adm_output_path" "$adm_output_path.tmp")
                    ans_cmp_time=$(echo "$ans_output" | grep "comp   time" | awk 'NR==1 {print $3}')
                    ans_output=$($PANS_DECMP_nv "$adm_output_path.tmp" "$adm_output_path.out")
                    ans_decmp_time=$(echo "$ans_output" | grep "decomp time" | awk 'NR==1 {print $3}')
                    mans_cmp_thr=$(echo "scale=9; $file_size / 1024 / 1024 / 1024/ (($adm_cmp_time + $ans_cmp_time) / 1000)" | bc)
                    mans_decmp_thr=$(echo "scale=9; $file_size / 1024 / 1024 / 1024 / (($adm_decmp_time + $ans_decmp_time) / 1000)" | bc)
                    echo "  MANS Compressed Thr: $mans_cmp_thr" >> $output_file
                    echo "  MANS Decompressed Thr: $mans_decmp_thr" >> $output_file

                    # echo "  ADM CMP Time: $adm_cmp_time" >> $output_file
                    # echo "  ADM DECMP Time: $adm_decmp_time" >> $output_file
                    # echo "  PANS-GPU CMP Time: $ans_cmp_time" >> $output_file
                    # echo "  PANS-GPU DECMP Time: $ans_decmp_time" >> $output_file


                    # 清理临时文件
                    rm -f  "$adm_output_path" "$adm_output_path.tmp" "$adm_output_path.out"  "$file_path.tmp" "$file_path.out"                  
                fi
            done
            echo "" >> $output_file
        fi
    done

    echo "FINISHED." >> $output_file
}

# 运行测试
run
