#!/bin/bash

echo "
        adult_model /deconformer/pathway_model/deconformer_predict_designate.py /deconformer/pathway_model/model_weights/adult_model/ 15
        fetal_model /deconformer/pathway_model/deconformer_predict_designate_fetal.py /deconformer/pathway_model/model_weights/fetal_model/ 15
        preg_model /deconformer/pathway_model/deconformer_predict_designate_preg.py /deconformer/pathway_model/model_weights/preg_model/ 9
" > ./model_list.txt

if [ $# -ne 3 ]
then
    echo 'script parameter error!
    USAGE: docker run --rm \
        -v $workdir:/workspace \
        2303162150/deconformer $model_name $exp_tsv $out_tsv

    According to your actual situation, please replace the `$workdir` `$exp_tsv` `$out_tsv` `$model_name` in the command with a string:
    * `$workdir` is the local synchronization working directory, and the paths for `$exp_tsv` and `$out_tsv` should be relative to this directory.
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
    script=${run_info[1]}
    model_path=${run_info[2]}
    epoch=${run_info[3]}
    echo === RUN: $model_name $(date) ===

    export OMP_NUM_THREADS=20
    python $script $model_path ${exp_tsv} $epoch ${out_tsv}
    echo === DONE!!! $(date) ===
else
    echo === ERROR: ${model} is not a available model. $(date) ===
fi
