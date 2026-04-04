#!/bin/bash

ENABLE_ADT_FSE=${ENABLE_ADT_FSE:-1}

# 运行测试
run() {
    mkdir -p $OUTPUT_DIR
    output_file=$OUTPUT_DIR/U4CR.txt
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
                                         
                    file_path=$TEST_DIR_u4"/"$dir"/"$file
                    file_size=$(stat -c%s "$file_path")
                    file_KB=$(echo "$file_size / 1024" | bc)
                    echo "SIZE: $file_KB KB" >> $output_file

                    # Step 1: 16bit huffman 压缩
                    num_ele=$(echo "$file_size / 4" | bc)
                    sz_huffman_output=$($SZ_Huffman -I 32 -i "$file_path" -o "$file_path.sz" -1 $num_ele -c $SZ3_CONFIG -M ABS 0.1)
                    huffman_ratio=$(echo "$sz_huffman_output" | grep "CR" | awk '{print $2}')
                    echo "  SZ 16bit Huffman Compressed Ratio: $huffman_ratio" >> $output_file

                    # Step 2: FSE -hf 压缩
                    $FSE -hf "$file_path"
                    file_hf_size=$(stat -c%s "$file_path.fse")
                    fse_huffman_ratio=$(echo "scale=3; $file_size / $file_hf_size" | bc)
                    echo "  FSE Huffman Compressed Ratio: $fse_huffman_ratio" >> $output_file

                    # Step 3: FSE -f 压缩
                    $FSE -f "$file_path"
                    file_f_size=$(stat -c%s "$file_path.fse")
                    fse_ans_ratio=$(echo "scale=3; $file_size / $file_f_size" | bc)
                    echo "  FSE ANS Compressed Ratio: $fse_ans_ratio" >> $output_file

                    # # Step 4: ADT-FSE 处理
                    if [[ "$ENABLE_ADT_FSE" == "1" ]]; then
                        python3 $ADT_PATH/short2code.py "$file_path"
                        bstream_size=$(stat -c%s "$file_path.bstream")
                        $FSE -f "$file_path.code"
                        code_f_size=$(stat -c%s "$file_path.code.fse")
                        adt_fse_ratio=$(echo "scale=3; $file_size / ($code_f_size + $bstream_size)" | bc)
                        echo "  ADT-FSE Compressed Ratio: $adt_fse_ratio" >> $output_file
                    else
                        echo "  ADT-FSE Test Skipped" >> $output_file
                    fi

                    # Step 5: MANS -R 
                    adm_output_path="$file_path.adm"
                    $ADM32_cpu -u4 "$file_path" "$adm_output_path" --dims 1 "$num_ele"
                    $FSE -f "$adm_output_path"
                    adm_fse_size=$(stat -c%s "$adm_output_path.fse")
                    adm_fse_cr=$(echo "scale=3; $file_size / $adm_fse_size" | bc)
                    echo "  ADM-FSE Compressed Ratio: $adm_fse_cr" >> $output_file

                    # Step 6: MANS -P
                    $PANS_CMP_cpu "$adm_output_path" "$adm_output_path.pans"
                    adm_pans_size=$(stat -c%s "$adm_output_path.pans")
                    adm_pans_cr=$(echo "scale=3; $file_size / $adm_pans_size" | bc)
                    echo "  ADM-PANS Compressed Ratio: $adm_pans_cr" >> $output_file

                    # Step 7: PANS
                    $PANS_CMP_cpu "$file_path" "$file_path.out"
                    pans_size=$(stat -c%s "$file_path.out")
                    pans_cr=$(echo "scale=3; $file_size / $pans_size" | bc)
                    echo "  PANS Compressed Ratio: $pans_cr" >> $output_file

                    rm -f "$file_path.sz" "$file_path.u4" "$file_path.out" "$file_path.fse" "$file_path.bstream" "$file_path.code" "$file_path.code.fse" "$adm_output_path" "$adm_output_path.fse" "$adm_output_path.pans"

                fi
            done
            echo "" >> $output_file
        fi
    done

    echo "FINISHED." >> $output_file
}

# 运行测试
run
