#!/bin/bash

echo "
        adult_model $PWD/model_weights/adult_model/ 15 $PWD/model_weights/adult_model/NBT_simu_cell_order_sccpm.txt $PWD/model_weights/adult_model/tsp_mRNA_genes.txt
        fetal_model $PWD//model_weights/fetal_model/ 15 $PWD/model_weights/fetal_model/fetal_simu_cell_order_1204.txt $PWD/model_weights/fetal_model/tsp_mRNA_genes.txt
        preg_model $PWD//model_weights/preg_model/ 9 $PWD/model_weights/preg_model/cell_types.tsv $PWD/model_weights/preg_model/mRNA_genes.tsv
" > ./model_list.txt

if [ $# -ne 3 ]
then
    echo 'script parameter error!
    USAGE: bash deconformer_predict.sh $model_name $exp_tsv $out_tsv

    * `$exp_tsv` is the tsv file of the expression matrix.
    * `$out_tsv` is the tsv file of inference result.
    * `$model_name` is the name of the trained model. You can choose from the following three models:
        * `adult_model`: 60 basic cell types; 
        * `fetal_model`: 27 types of cells + 3 types of trophoblast cells + 4 types of fetal cells; 
        * `preg_model`: 60 types of cells + early and late stages of SCT, EVT, VCT, totaling six types of trophoblasts.
        '

    exit 1
fi
        
model=$1
exp_tsv=$2
out_tsv=$3

if [ $(grep -c ${model} ./model_list.txt) -gt 0 ]
then
    run_info=($(grep ${model} ./model_list.txt))
    model_name=${run_info[0]}
    script=$PWD/deconformer_predict.py
    model_path=${run_info[1]}
    epoch=${run_info[2]}
    cell_types=${run_info[3]}
    genes=${run_info[4]}
    rm ./model_list.txt
    echo === RUN: $model_name $(date) ===

    export OMP_NUM_THREADS=20
    python $script $model_path ${exp_tsv} $epoch ${out_tsv} $cell_types $genes
    echo === DONE!!! $(date) ===
else
    echo === ERROR: ${model} is not a available model. $(date) ===
fi
