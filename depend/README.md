# DEPEND package for TrojAI Project

## Installation
* Install the project by `python install setup.py`

## Usage 
* Create  dictionaries to configure:
    1. `algorithm_config`: your detection algorithm parameters, e.g., `weight_analysis`, `mask_generation`, `attribution_analysis`, etc. 
    2. `model_config`: your detector model parameters, e.g, `input_size`, `model_class`, etc.
    3. `learner_config`: detector model training  parameters, e.g., `batch_size`, `epochs`, etc.
    4. `optimizer_config`: detector model optimization  parameters, e.g., `lr`, `momentum`, etc. 
    5. `data_config`: detector dataset  parameters, e.g., `number of splits`, `max_train_samples`, etc.
* Import `depend` library and create a select dependent such as `mask_gen`.
```
from depend import MaskGen, DPConfig
dependent = MaskGen(model_file_location)
config = DP_Config.from_dict(model_config, learner_config, algorithm_config, optimizer_config, data_config)
dependent.configure(config)
```

* To train trojan detector
```
dependent.train_detector()
```

* To infer with pretrained trojan detector, load the detector from path and run.
```
probabilities = dependent.infer(detector_path, target_models_path)
```

## Development
* Install poery `$pip install poetry`
* Use `poetry cache clear --all .` to clean poetry cache if poetry takes too long to search for a library
