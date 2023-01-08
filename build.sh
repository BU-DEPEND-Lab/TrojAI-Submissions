#python trojan_detector.py --learned_parameters_dirpath=./learned_parameters  --model_filepath=/mnt/md0/shared/TrojAI-Submissions/trojai-datasets/round11/models/id-00000003/model.pt  --metaparameters_filepath=./metaparameters.json  --schema_filepath=./metaparameters_schema.json --result_filepath=./output.txt --scratch_dirpath=./scratch/ --examples_dirpath=/data/models/id-00000003/clean-example-data.json --source_dataset_dirpath /mnt/md0/shared/TrojAI-Submissions/trojai-datasets/round11/image-classification-sep2022-example-source-dataset --round_training_dataset_dirpath=/mnt/md0/shared/TrojAI-Submissions/trojai-datasets/round11/models
python entrypoint.py infer \
--model_filepath ./model/id-00000002/model.pt \
--result_filepath ./scratch/output.txt \
--scratch_dirpath ./scratch \
--examples_dirpath ./model/id-00000002/clean-example-data \
--round_training_dataset_dirpath /mnt/md0/shared/TrojAI-Submissions/trojai-datasets/round12/ \
--learned_parameters_dirpath ./learned_parameters \
--metaparameters_filepath ./metaparameters.json \
--schema_filepath=./metaparameters_schema.json \
--scale_parameters_filepath ./learned_parameters/scale_params.npy

python entrypoint.py configure \
--scratch_dirpath=./scratch/ \
--metaparameters_filepath=./metaparameters.json \
--schema_filepath=./metaparameters_schema.json \
--learned_parameters_dirpath=./new_learned_parameters/ \
--configure_models_dirpath=/mnt/md0/shared/TrojAI-Submissions/trojai-datasets/round12/ \
--scale_parameters_filepath ./new_learned_parameters/scale_params.npy

sudo singularity build example_trojan_detector.simg example_trojan_detector.def
#sudo singularity build test-trojai-r12-weight.simg trojan_detector.def 


#singularity run --nv -B /mnt/md0/shared/TrojAI-Submissions/trojai-datasets/round11:/data/ test-trojai-r10-weight-v2.simg --learned_parameters_dirpath=./learned_parameters --model_filepath=/data/models/id-00000003/model.pt  --result_filepath=./output.txt  --scratch_dirpath=./scratch/  --examples_dirpath=/data/models/id-00000003/clean-example-data.json --source_dataset_dirpath /data/image-classification-sep2022-example-source-dataset  --round_training_dataset_dirpath=/mnt/md0/shared/TrojAI-Submissions/trojai-datasets/round11/models  --metaparameters_filepath=/metaparameters.json  --schema_filepath=/metaparameters_schema.json
 singularity run \
--bind /mnt/md0/shared/TrojAI-Submissions/trojai-datasets/round12/ \
--nv \
./example_trojan_detector.simg \
infer \
--model_filepath=/mnt/md0/shared/TrojAI-Submissions/trojai-datasets/round12/models/id-00000002/model.pt \
--metaparameters_filepath=/metaparameters.json \
--schema_filepath=/metaparameters_schema.json \
--result_filepath=./output.txt \
--scratch_dirpath=./scratch/ \
--examples_dirpath=/mnt/md0/shared/TrojAI-Submissions/trojai-datasets/round12/models/id-00000002/clean-example-data/ \
--round_training_dataset_dirpath=/mnt/md0/shared/TrojAI-Submissions/trojai-datasets/round12/ \
--learned_parameters_dirpath=/learned_parameters/ \
--scale_parameters_filepath=/learned_parameters/scale_params.npy