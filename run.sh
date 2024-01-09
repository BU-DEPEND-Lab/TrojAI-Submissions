#!/bin/bash


if [[ -n $1 ]]
then 

    if [[ $1 = infer ]]
    then
        ## For inference
        python entrypoint.py infer \
        --model_filepath /home/zwc662/Workspace/TrojAI-Submissions/model/id-00000000/model.pt \
        --result_filepath ./output.txt \
        --scratch_dirpath ./scratch \
        --examples_dirpath /home/zwc662/Workspace/TrojAI-Submissions/model/id-00000000/clean-example-data/ \
        --round_training_dataset_dirpath /mnt/md0/shared/TrojAI-Submissions/trojai-datasets/round14/cyber-apk-nov2023-train//models \
        --learned_parameters_dirpath ./new_learned_parameters \
        --metaparameters_filepath ./metaparameters.json \
        --schema_filepath=./metaparameters_schema.json
    else 
        singularity run \
        --bind /home/zwc662/Workspace/TrojAI-Submissions/model/ \
        --nv \
        ./rl-randomized-lavaworld-aug2023_test_TrinitySRITrojAI-BostonU.simg \
        infer \
        --model_filepath=/home/zwc662/Workspace/TrojAI-Submissions/model/id-00000000/model.pt \
        --result_filepath=./output.txt \
        --scratch_dirpath=./scratch/ \
        --examples_dirpath=/home/zwc662/Workspace/TrojAI-Submissions/model/id-00000000/clean-example-data/ \
        --round_training_dataset_dirpath=/path/to/training/dataset/ \
        --metaparameters_filepath=/metaparameters.json \
        --schema_filepath=/metaparameters_schema.json \
        --learned_parameters_dirpath=/learned_parameters/
    fi
else
    ## For configuration
    python entrypoint.py configure \
    --scratch_dirpath=./scratch/ \
    --metaparameters_filepath=./metaparameters.json \
    --schema_filepath=./metaparameters_schema.json \
    --learned_parameters_dirpath=./new_learned_parameters/ \
    --configure_models_dirpath=/mnt/md0/shared/TrojAI-Submissions/trojai-datasets/round16/cyber-apk-nov2023-train//models
fi


#scp rl-lavaworld-jul2023_test_TrinitySRITrojAI-BostonU.simg weichaozhou@168.122.70.64:~/Downloads/