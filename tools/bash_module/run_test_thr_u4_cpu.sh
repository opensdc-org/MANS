#!/bin/bash

# 运行测试
run() {
    mkdir -p $OUTPUT_DIR
    output_file=$OUTPUT_DIR/U4THR-cpu.txt
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

                    
                    # Step 1: 16bit huffman 压缩
                    num_ele=$(echo "$file_size / 4" | bc)
                    sz_huffman_output=$(nice -n 20 $SZ_Huffman -I 32 -i "$file_path" -o "$file_path.sz" -1 $num_ele -c $SZ3_CONFIG -M ABS 0.1)
                    huffman_cmp_thr=$(echo "$sz_huffman_output" | grep "Throughput" | awk 'NR==1 {print $5}')
                    huffman_decmp_thr=$(echo "$sz_huffman_output" | grep "Throughput" | awk 'NR==2 {print $9}')
                    echo "  16bit Huffman Compressed Thr: $huffman_cmp_thr" >> $output_file
                    echo "  16bit Huffman Decompressed Thr: $huffman_decmp_thr" >> $output_file

                    # Step 2: FSE -hf 压缩
                    $FSE -hf "$file_path"
                    cmp_output=$(nice -n 20 $FSE -hf "$file_path" 2>&1 >/dev/null)
                    cmp_time=$(echo "$cmp_output" | grep "Compression time" | awk '{print $3}')
                    $FSE -df "$file_path.fse"
                    decmp_output=$(nice -n 20 $FSE -df "$file_path.fse" 2>&1 >/dev/null)
                    decmp_time=$(echo "$decmp_output" | grep "Decompression time" | awk '{print $3}')
                    fse_hf_cmp_thr=$(echo "scale=3; $file_size / 1024 / 1024 / $cmp_time" | bc)
                    fse_hf_decmp_thr=$(echo "scale=3; $file_size / 1024 / 1024 / $decmp_time" | bc)
                    echo "  FSE Huffman Compressed Thr: $fse_hf_cmp_thr" >> $output_file
                    echo "  FSE Huffman Decompressed Thr: $fse_hf_decmp_thr" >> $output_file
                    
                    # # Step 3: FSE -f 压缩
                    $FSE -f "$file_path"
                    cmp_output=$(nice -n 20 $FSE -f "$file_path" 2>&1 >/dev/null)
                    cmp_time=$(echo "$cmp_output" | grep "Compression time" | awk '{print $3}')
                    $FSE -df "$file_path.fse"
                    decmp_output=$(nice -n 20 $FSE -df "$file_path.fse" 2>&1 >/dev/null)
                    decmp_time=$(echo "$decmp_output" | grep "Decompression time" | awk '{print $3}')
                    fse_f_cmp_thr=$(echo "scale=3; $file_size / 1024 / 1024 / $cmp_time" | bc)
                    fse_f_decmp_thr=$(echo "scale=3; $file_size / 1024 / 1024 / $decmp_time" | bc)
                    echo "  FSE ANS Compressed Thr: $fse_f_cmp_thr" >> $output_file
                    echo "  FSE ANS Decompressed Thr: $fse_f_decmp_thr" >> $output_file
                    # echo "  FSE ANS CMP Time: $cmp_time" >> $output_file
                    # echo "  FSE ANS DECMP Time: $decmp_time" >> $output_file


                    # Step 4: MANS -r
                    adm_output_path="$file_path.adm"
                    $ADM32_cpu -u4 "$file_path" "$adm_output_path" --dims 1 "$num_ele"
                    adm_output=$(nice -n 20 $ADM32_cpu -u4 "$file_path" "$adm_output_path" --dims 1 "$num_ele")
                    adm_cmp_time=$(echo "$adm_output" | grep "compress cost" | awk 'NR==1 {print $3}')
                    adm_decmp_time=$(echo "$adm_output" | grep "decompress cost" | awk '{print $3}')
                    $FSE -f "$adm_output_path"
                    cmp_output=$(nice -n 20 $FSE -f "$adm_output_path" 2>&1 >/dev/null)
                    cmp_time=$(echo "$cmp_output" | grep "Compression time" | awk '{print $3}')
                    $FSE -df "$adm_output_path.fse"
                    decmp_output=$(nice -n 20 $FSE -df "$adm_output_path.fse" 2>&1 >/dev/null)
                    decmp_time=$(echo "$decmp_output" | grep "Decompression time" | awk '{print $3}')
                    adm_fse_cmp_thr=$(echo "scale=3; $file_size / 1024 / 1024 / (($adm_cmp_time / 1000) + $cmp_time)" | bc)
                    adm_fse_decmp_thr=$(echo "scale=3; $file_size / 1024 / 1024 / (($adm_decmp_time / 1000) + $decmp_time)" | bc)
                    echo "  MANS -r Compressed Thr: $adm_fse_cmp_thr" >> $output_file
                    echo "  MANS -r Decompressed Thr: $adm_fse_decmp_thr" >> $output_file
                    # echo "  ADM CMP Time: $adm_cmp_time" >> $output_file
                    # echo "  ADM DECMP Time: $adm_decmp_time" >> $output_file
                    # echo "  FSE ANS CMP Time: $cmp_time" >> $output_file
                    # echo "  FSE ANS DECMP Time: $decmp_time" >> $output_file

                    # # Step 5: MANS -p
                    $PANS_CMP_cpu "$adm_output_path" "$adm_output_path.pans"
                    pans_output=$(nice -n 20 $PANS_CMP_cpu "$adm_output_path" "$adm_output_path.pans")
                    pans_cmp_time=$(echo "$pans_output" | grep "comp   time" | awk 'NR==1 {print $3}')
                    $PANS_DECMP_cpu "$adm_output_path.pans" "$adm_output_path.out"
                    pans_output=$(nice -n 20 $PANS_DECMP_cpu "$adm_output_path.pans" "$adm_output_path.out")
                    pans_decmp_time=$(echo "$pans_output" | grep "decomp time" | awk 'NR==1 {print $3}')
                    adm_pans_cmp_thr=$(echo "scale=9; $file_size / 1024 / 1024 / (($adm_cmp_time + $pans_cmp_time) / 1000)" | bc)
                    adm_pans_decmp_thr=$(echo "scale=9; $file_size / 1024 / 1024 / (($adm_decmp_time + $pans_decmp_time) / 1000)" | bc)
                    echo "  MANS -p Compressed Thr: $adm_pans_cmp_thr" >> $output_file
                    echo "  MANS -p Decompressed Thr: $adm_pans_decmp_thr" >> $output_file
                    # echo "  PANS CMP Time: $pans_cmp_time" >> $output_file
                    # echo "  PANS DECMP Time: $pans_decmp_time" >> $output_file

                    # 清理临时文件
                    rm -f "$file_path.sz" "$file_path.u4" "$file_path.fse" "$file_path.bstream" "$file_path.code" "$file_path.code.fse" "$adm_output_path" "$adm_output_path.fse" "$adm_output_path.pans"  "$adm_output_path.out"                    
                fi
            done
            echo "" >> $output_file
        fi
    done

    echo "FINISHED." >> $output_file
}

# 运行测试
run
