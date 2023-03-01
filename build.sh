python trojan_detector.py --configure_mode --learned_parameters_dirpath=./learned_parameters  --model_filepath=/mnt/md0/shared/TrojAI-Submissions/trojai-datasets/round11/models/id-00000003/model.pt  --metaparameters_filepath=./metaparameters.json  --schema_filepath=./metaparameters_schema.json --result_filepath=./output.txt --scratch_dirpath=./scratch/ --examples_dirpath=/data/models/id-00000003/clean-example-data.json --source_dataset_dirpath /mnt/md0/shared/TrojAI-Submissions/trojai-datasets/round11/models --round_training_dataset_dirpath=/mnt/md0/shared/TrojAI-Submissions/trojai-datasets/round11/models

#python trojan_detector.py  --model_filepath=./data/round9-train-dataset/models/id-00000105/model.pt  --tokenizer_filepath=./data/round9-train-dataset/tokenizers/google-electra-small-discriminator.pt  --features_filepath=./features.csv  --result_filepath=./output.txt  --scratch_dirpath=./scratch/  --examples_dirpath=./data/round9-train-dataset/models/id-00000105/clean-example-data.json  --round_training_dataset_dirpath=/path/to/training/dataset/  --metaparameters_filepath=./metaparameters.json  --schema_filepath=./metaparameters_schema.json  --learned_parameters_dirpath=./learned_parameters



sudo singularity build test-trojai-r10-weight-v2.simg trojan_detector.def 


singularity run --nv -B /mnt/md0/shared/TrojAI-Submissions/trojai-datasets/round11:/data/ test-trojai-r10-weight-v2.simg --learned_parameters_dirpath=./learned_parameters --model_filepath=/data/models/id-00000003/model.pt  --result_filepath=./output.txt  --scratch_dirpath=./scratch/  --examples_dirpath=/data/models/id-00000003/clean-example-data.json --source_dataset_dirpath /data/image-classification-sep2022-example-source-dataset  --round_training_dataset_dirpath=/mnt/md0/shared/TrojAI-Submissions/trojai-datasets/round11/models  --metaparameters_filepath=/metaparameters.json  --schema_filepath=/metaparameters_schema.json
