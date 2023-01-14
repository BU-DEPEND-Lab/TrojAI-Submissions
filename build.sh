# Feature extraction mode
python entrypoint.py extract \
--round_training_dataset_dirpath /mnt/md0/shared/TrojAI-Submissions/trojai-datasets/round12/ \
--metaparameters_filepath ./metaparameters.json \
--schema_filepath ./metaparameters_schema.json \
--scale_parameters_filepath ./learned_parameters/scale_params.npy

#inference mode
python entrypoint.py infer \
--model_filepath ./model/id-00000002/model.pt \
--result_filepath ./scratch/output.txt \
--scratch_dirpath ./scratch \
--examples_dirpath ./model/id-00000002/clean-example-data \
--round_training_dataset_dirpath /mnt/md0/shared/TrojAI-Submissions/trojai-datasets/round12/ \
--learned_parameters_dirpath ./learned_parameters \
--metaparameters_filepath ./metaparameters.json \
--schema_filepath ./metaparameters_schema.json \
--scale_parameters_filepath ./learned_parameters/scale_params.npy

# self-configure mode
python entrypoint.py configure \
--scratch_dirpath=./scratch/ \
--metaparameters_filepath=./metaparameters.json \
--schema_filepath=./metaparameters_schema.json \
--learned_parameters_dirpath=./new_learned_parameters/ \
--configure_models_dirpath=/mnt/md0/shared/TrojAI-Submissions/trojai-datasets/round12/models \
--scale_parameters_filepath=./new_learned_parameters/scale_params.npy

# test run self-configured model
python entrypoint.py infer \
--model_filepath=./model/id-00000002/model.pt \
--result_filepath=./output.txt \
--scratch_dirpath=./scratch/ \
--examples_dirpath=./model/id-00000002/clean-example-data/ \
--round_training_dataset_dirpath=//mnt/md0/shared/TrojAI-Submissions/trojai-datasets/round12/ \
--metaparameters_filepath=./new_learned_parameters/metaparameters.json \
--schema_filepath=./metaparameters_schema.json \
--learned_parameters_dirpath=./new_learned_parameters/ \
--scale_parameters_filepath=./learned_parameters/scale_params.npy

# Build container
sudo singularity build example_trojan_detector.simg example_trojan_detector.def


# Test run the container in inference mode
 singularity run \
--bind /mnt/md0/shared/TrojAI-Submissions/trojai-datasets/round12/ \
--nv ./example_trojan_detector.simg \
infer --model_filepath=/mnt/md0/shared/TrojAI-Submissions/trojai-datasets/round12/models/id-00000002/model.pt \
--metaparameters_filepath=/metaparameters.json \
--schema_filepath=/metaparameters_schema.json \
--result_filepath=./output.txt \
--scratch_dirpath=./scratch/ \
--examples_dirpath=/mnt/md0/shared/TrojAI-Submissions/trojai-datasets/round12/models/id-00000002/clean-example-data/ \
--round_training_dataset_dirpath=/mnt/md0/shared/TrojAI-Submissions/trojai-datasets/round12/ \
--learned_parameters_dirpath=/learned_parameters/ \
--scale_parameters_filepath=/learned_parameters/scale_params.npy

###### or #######

# Test run the container in self-configure mode
singularity run --nv -B /mnt/md0/shared/TrojAI-Submissions/trojai-datasets/round12:/data/ ./example_trojan_detector.simg \
--learned_parameters_dirpath=/learned_parameters \
--model_filepath=/data/models/id-00000001/model.pt \
--result_filepath=./output.txt \
--scratch_dirpath=./scratch/ \
--examples_dirpath=/data/models/id-00000001/clean-example-data \
--round_training_dataset_dirpath=/data/  \
--metaparameters_filepath=/metaparameters.json  \
--schema_filepath=/metaparameters_schema.json \
--scale_parameters_filepath=/learned_parameters/scale_params.npy