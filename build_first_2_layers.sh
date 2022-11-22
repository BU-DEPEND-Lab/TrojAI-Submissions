python trojan_detector.py --learned_parameters_dirpath=./learned_parameters_first_2_layers  --model_filepath=/home/jmfan/packages/trojan-ai//models/id-00000003/model.pt  --result_filepath=./output.txt  --scratch_dirpath=./scratch/  --examples_dirpath=/home/jmfan/packages/trojan-ai//models/id-00000108/clean-example-data.json  --round_training_dataset_dirpath=/path/to/training/dataset/ --source_dataset_dirpath ../trojai-datasets/coco2017/trojai_coco/  --metaparameters_filepath=./metaparameters.json  --schema_filepath=./metaparameters_schema.json

#python trojan_detector.py  --model_filepath=./data/round9-train-dataset/models/id-00000105/model.pt  --tokenizer_filepath=./data/round9-train-dataset/tokenizers/google-electra-small-discriminator.pt  --features_filepath=./features.csv  --result_filepath=./output.txt  --scratch_dirpath=./scratch/  --examples_dirpath=./data/round9-train-dataset/models/id-00000105/clean-example-data.json  --round_training_dataset_dirpath=/path/to/training/dataset/  --metaparameters_filepath=./metaparameters.json  --schema_filepath=./metaparameters_schema.json  --learned_parameters_dirpath=./learned_parameters



sudo singularity build test-trojai-r10-weight-first-2-layers-v2.simg trojan_detector.def 


singularity run --nv -B /home/jmfan/packages/trojan-ai/trojai-datasets:/data/ test-trojai-r10-weight-first-2-layers-v2.simg --learned_parameters_dirpath=./learned_parameters_first_2_layers --model_filepath=/data/round10-train-dataset/models/id-00000003/model.pt  --result_filepath=./output.txt  --scratch_dirpath=./scratch/  --examples_dirpath=/data/round10-train-dataset/models/id-00000003/clean-example-data.json --source_dataset_dirpath /data/coco2017/trojai_coco/  --round_training_dataset_dirpath=/path/to/training/dataset/  --metaparameters_filepath=/metaparameters.json  --schema_filepath=/metaparameters_schema.json
