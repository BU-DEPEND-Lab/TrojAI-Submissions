 

## For configuration
#python entrypoint.py configure \
#--scratch_dirpath=./scratch/ \
#--metaparameters_filepath=./metaparameters.json \
#--schema_filepath=./metaparameters_schema.json \
#--learned_parameters_dirpath=./new_learned_parameters/ \
#--configure_models_dirpath=/mnt/md0/shared/TrojAI-Submissions/trojai-datasets/round14/rl-lavaworld-jul2023-train/models


## For inference
python entrypoint.py infer \
--model_filepath ./model/rl-lavaworld-jul2023-example/model.pt \
--result_filepath ./output.txt \
--scratch_dirpath ./scratch \
--examples_dirpath ./model/rl-lavaworld-jul2023-example/clean-example-data \
--round_training_dataset_dirpath /mnt/md0/shared/TrojAI-Submissions/trojai-datasets/round14/rl-lavaworld-jul2023-train/models \
--learned_parameters_dirpath ./new_learned_parameters \
--metaparameters_filepath ./metaparameters.json \
--schema_filepath=./metaparameters_schema.json

singularity run \
--bind /mnt/md0/shared/TrojAI-Submissions/trojai-datasets/round14/rl-lavaworld-jul2023-train/models \
--nv \
./detector.simg \
infer \
--model_filepath=./model/rl-lavaworld-jul2023-example/model.pt \
--result_filepath=./output.txt \
--scratch_dirpath=./scratch/ \
--examples_dirpath=./model/rl-lavaworld-jul2023-example/clean-example-data/ \
--round_training_dataset_dirpath=/path/to/training/dataset/ \
--metaparameters_filepath=./metaparameters.json \
--schema_filepath=./metaparameters_schema.json \
--learned_parameters_dirpath=./learned_parameters/