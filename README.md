# System Requirements

- Linux (tested on Ubuntu 18.04 LTS)
- CUDA capable NVIDIA GPU (tested on Titan RTX)

Note: This example assumes you are running on a version of Linux (like Ubuntu 18.04 LTS) with a CUDA enabled NVIDIA GPU. Singularity only runs natively on Linux, and most Deep Learning libraries are designed for Linux first. While this Conda setup will install the CUDA drivers required to run PyTorch, the CUDA enabled GPU needs to be present on the system.   

# Summary

This repo contains a minimal working example showing a submission to the TrojAI NIST Evaluation server. This 'solution' loads the model file, inferences 10 random tensors, and then writes a random number to the output file.

A submission singularity container must evaluate a single trained image classification CNN model and determine whether or not the model in question has been poisoned (i.e. a trigger has been embedded).

--------------

# Table of Contents
1. [Challenge Submission](#challenge-submission)
    1. [Compute Resources](#compute-resources)
    2. [Container Handling](#container-handling)
    3. [Container API](#container-api)
2. [How to Build this Minimal Example](#how-to-build-this-minimal-example)
    1. [Install Anaconda Python](#install-anaconda-python)
    2. [Setup the Conda Environment](#setup-the-conda-environment)
    3. [Test Fake Detector Without Containerization](#test-fake-detector-without-containerization)
    4. [Package Solution into a Singularity Container](#package-solution-into-a-singularity-container)
3. [How to Register a TrojAI Team](#how-to-register-a-trojai-team)
    1. [Accounts](#accounts)
    2. [Verify Team Creation](#verify-team-creation)
4. [Container Submission Mechanism](#container-submission-mechanism)
5. [Results](#results)
    1. [Evaluation Server vs Smoke Test Server](#evaluation-server-vs-smoke-test-server)
        1. [Evaluation Server (ES)](#evaluation-server-es)
        2. [Smoke Test Server (STS)](#smoke-test-server-sts)
    2. [Jobs Table](#jobs-table)
    3. [Results Table](#results-table)
    4. [Output Logs](#output-logs)

--------------
# Challenge Submission

Every solution submitted for evaluation must be containerized via [Singularity](https://sylabs.io/docs/) (see this [Singularity tutorial](https://pawseysc.github.io/sc19-containers/)). 

The submitted Singularity container will be run by the NIST test and evaluation server using the API specified in [Container API](#container-api) inside of a virtual machine which has no network capability.

The container submitted to NIST for evaluation must perform trojan detection for a single trained AI model file and output a single probability of the model being poisoned. The test and evaluation infrastructure will iterate over the *N* models for which your container must predict trojan presence. 

For each test data point lacking an output poisoning probability (for example, if you ran out of compute time) will be considered as being probability 0.5 when computing your overall cross entropy loss for the test dataset.


## Compute Resources

Your solution will have the following resources allocated for 24 hours to evaluate the test dataset.

- CPU: 5 physical cores (10 logical) from Intel Xeon Silver 4216 @ 2.10GHz 
- Memory: 24 GB
- GPU: NVidia V100 (32GB GPU memory)
- Disk: 1.5TB SATA SSDs for scratch space


## Container Handling

When you submit a container for evaluation, it will be added to a queue. When it's your container's turn to be run, the container will be copied into a non-networked VM (Ubuntu 18.04 LTS) instance populated with read-only test data and a 1.5TB SATA SSD read/write scratch space drive. The NIST test and evaluation harness will iterate over all elements of the sequestered test dataset and call your container once per data point. Each data point is a trained AI model. Execution will terminate either after 24 hours (the compute time limit) or when it finishes processing all the test data. After your container terminates, NIST will compute the average cross entropy loss between your predictions and the ground truth answers. This score is then posted to the leader-board website.


## Container API

For each point in the test dataset, your container will be launched using the following parameters.

### Parameters Passed to the Container on Launch

- `--model_filepath` = The path to the model file to be evaluated.
- `--result_filepath` = The path to the output result file where the probability [0, 1] (floating point value in the range of 0 to 1 inclusive) of the aforementioned model file being poisoned is to be written as text (not binary). For example, "0.75". No other data should be written to this file. If the test server cannot parse your results file, the default probability of 0.5 will be substituted. If any parse errors occur, they will be listed on the leader-board webpage.
- `--scratch_dirpath` = The path to a directory (empty folder) where temporary data can be written during the evaluation of the model file.
- `--examples_dirpath` = The path to a directory containing a few example png images for each of the classes the model is trained to classify. Names are of the format "class_2_example_35.png".


--------------

# How to Build this Minimal Example

## Install Anaconda Python

[https://www.anaconda.com/distribution/](https://www.anaconda.com/distribution/)

## Setup the Conda Environment

1. `conda create --name fake_detector python=3.6` ([help](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html))
2. `conda activate fake_detector`
3. Install required packages into this conda environment

    1. `conda install numpy`
    2. `conda install pytorch torchvision cudatoolkit=10.1 -c pytorch` ([help](https://pytorch.org/get-started/locally/))

## Test Fake Detector Without Containerization

1.  Clone the repository 
 
    ```
    git clone https://github.com/usnistgov/trojai-example
    cd trojai-example
    ``` 

2. Test the python based `fake_trojan_detector` outside of any containerization to confirm pytorch is setup correctly and can utilize the GPU.

    ```bash
    python fake_trojan_detector.py --model_filepath=./model.pt --result_filepath=./output.txt --scratch_dirpath=./scratch/
    ```

    Example Output:
    
    ```bash
    Trojan Probability: 0.07013004086445151
    ```

## Package Solution into a Singularity Container

Package `fake_trojan_detector.py` into a Singularity container.

1. Install Singularity
    
    - For Ubuntu 18.04 LTS: `sudo apt install singularity-container`
    - For others Linux Distros follow: [https://sylabs.io/guides/3.5/admin-guide/installation.html#installation-on-linux](https://sylabs.io/guides/3.5/admin-guide/installation.html#installation-on-linux)
        
2. Build singularity based on `fake_trojan_detector.def` file: 

    - delete any old copy of output file if it exists: `rm fake_trojan_detector.simg`
    - package container: 
    
      ```bash
      sudo singularity build fake_trojan_detector.simg fake_trojan_detector.def
      ```

    which generates a `fake_trojan_detector.simg` file.

    Example Output:
    ```bash
    $ sudo singularity build fake_trojan_detector.simg fake_trojan_detector.def
    Using container recipe deffile: fake_trojan_detector.def
    Sanitizing environment
    Adding base Singularity environment to container
    tar: ./.exec: implausibly old time stamp -9223372036854775808
    tar: ./.run: implausibly old time stamp -9223372036854775808
    tar: ./.shell: implausibly old time stamp -9223372036854775808
    tar: ./.singularity.d/actions/exec: implausibly old time stamp -9223372036854775808
    tar: ./.singularity.d/actions/run: implausibly old time stamp -9223372036854775808
    tar: ./.singularity.d/actions/shell: implausibly old time stamp -9223372036854775808
    tar: ./.singularity.d/actions/start: implausibly old time stamp -9223372036854775808
    tar: ./.singularity.d/actions/test: implausibly old time stamp -9223372036854775808
    tar: ./.singularity.d/actions: implausibly old time stamp -9223372036854775808
    tar: ./.singularity.d/env/01-base.sh: implausibly old time stamp -9223372036854775808
    tar: ./.singularity.d/env/90-environment.sh: implausibly old time stamp -9223372036854775808
    tar: ./.singularity.d/env/95-apps.sh: implausibly old time stamp -9223372036854775808
    tar: ./.singularity.d/env/99-base.sh: implausibly old time stamp -9223372036854775808
    tar: ./.singularity.d/env: implausibly old time stamp -9223372036854775808
    tar: ./.singularity.d/libs: implausibly old time stamp -9223372036854775808
    tar: ./.singularity.d/runscript: implausibly old time stamp -9223372036854775808
    tar: ./.singularity.d/startscript: implausibly old time stamp -9223372036854775808
    tar: ./.singularity.d: implausibly old time stamp -9223372036854775808
    tar: ./.test: implausibly old time stamp -9223372036854775808
    tar: ./dev: implausibly old time stamp -9223372036854775808
    tar: ./environment: implausibly old time stamp -9223372036854775808
    tar: ./etc/hosts: implausibly old time stamp -9223372036854775808
    tar: ./etc/resolv.conf: implausibly old time stamp -9223372036854775808
    tar: ./etc: implausibly old time stamp -9223372036854775808
    tar: ./home: implausibly old time stamp -9223372036854775808
    tar: ./proc: implausibly old time stamp -9223372036854775808
    tar: ./root: implausibly old time stamp -9223372036854775808
    tar: ./singularity: implausibly old time stamp -9223372036854775808
    tar: ./sys: implausibly old time stamp -9223372036854775808
    tar: ./tmp: implausibly old time stamp -9223372036854775808
    tar: ./var/tmp: implausibly old time stamp -9223372036854775808
    tar: ./var: implausibly old time stamp -9223372036854775808
    tar: .: implausibly old time stamp -9223372036854775808
    Docker image path: index.docker.io/pytorch/pytorch:latest
    Cache folder set to /root/.singularity/docker
    Exploding layer: sha256:16c48d79e9cc2d6cdb79a91e9c410250c1a44102ed4c971fbf24692cc09f2351.tar.gz
    Exploding layer: sha256:3c654ad3ed7d66e3caa5ab60bee1b166359d066be7e9edca6161b72ac06f2008.tar.gz
    Exploding layer: sha256:6276f4f9c29df0a2fc8019e3c9929e6c3391967cb1f610f57a3c5f8044c8c2b6.tar.gz
    Exploding layer: sha256:a4bd43ad48cebce2cad4207b823fe1693e10c440504ce72f48643772e3c98d7a.tar.gz
    Exploding layer: sha256:34cb2ecb4e7e4513ede923e58c6a219e8e025a5f27e9c8e1df37c0f9972cfd9e.tar.gz
    Exploding layer: sha256:1271bead61037d0e1f1e3c7efc63848627a2bd513c884201c3178964c21293a2.tar.gz
    Exploding layer: sha256:913bf197139d82f9984a8417548fee109c096bb7e6dd9672e1a42d8ed8644d59.tar.gz
    Exploding layer: sha256:96e5a748a56a153207ca15202c318e29f61ddfd44784cdcbde95bb7086fa0871.tar.gz
    Exploding layer: sha256:ac87c593cb7de82616275e9ef3b085ebc758b648553381c9e094c70ba54a7bf7.tar.gz
    Exploding layer: sha256:f4cfecb48ca26a9ea56c738af1311b4a44cd075e9e92ac8c1870edffa0f11dfd.tar.gz
    User defined %runscript found! Taking priority.
    Adding files to container
    Copying './fake_trojan_detector.py' to '/'
    Adding runscript
    Finalizing Singularity container
    Calculating final size for metadata...
    Skipping checks
    Building Singularity image...
    Singularity container built: fake_trojan_detector.simg
    Cleaning up...
    ```

3. Test run container: 

    ```bash
    singularity run --nv ./fake_trojan_detector.simg --model_filepath ./model.pt --result_filepath ./output.txt --scratch_dirpath ./scratch
    ```

    Example Output:
    ```bash
    Trojan Probability: 0.7091788412534845
    ```

--------------
# How to Register a TrojAI Team

## Accounts

In order to submit trojan detection solutions to the NIST test and evaluation server you need to contact the NIST team (trojai@nist.gov) and request access. 

Google Drive will be used to submit solutions to the server, so you will need to create or use an existing Google Drive account.

When you email trojai@nist.gov to create a test and evaluation account please include: 
1. Team Name (alpha-numeric, no special characters, no spaces)
2. Google Drive Account email

The Team Names need to be unique across the set of all performers submitting to the NIST test and evaluation server, so you might be requested to pick another team name if a naming conflict arises (team names are first come first serve).

Only containers submitted from the email you notified NIST about will be considered for evaluation, all other container images shared with the NIST Google Drive account will be ignored.

NIST operates two test [servers](#evaluation-server-vs-smoke-test-server). Indicate which (or both) servers you would like accounts on. 

## Verify Team Creation

Once the NIST T&E system creates your team it will show up on the results website [https://pages.nist.gov/trojai/](https://pages.nist.gov/trojai/) Jobs table.

Note: [https://pages.nist.gov/trojai/](https://pages.nist.gov/trojai/) only updates every few minutes. The webpage includes a last updated timestamp near above the tables. 


--------------
# Container Submission Mechanism

Containers are to be submitted for evaluation by sharing them with a functional NIST Google Drive account (trojai@nist.gov) via a team Google Drive account. 

1. Package your solution into a Singularity container.
2. Upload your packaged Singularity container to Google Drive using the account you registered with the NIST T&E Team.
    - **Files from a non-registered email address will be ignored**
    - container names that start with 'test' will be evaluated on the [Smoke Test Server](#evaluation-server-vs-smoke-test-server)
    - container names that do not start with 'test' will be evaluated on the [Evaluation Server](#evaluation-server-vs-smoke-test-server)
    - You can only be sharing 1 file for each server, no matter what the file name. So your Drive account can have up to 2 files shared with the TrojAI Drive account, one starting with 'test' and one which does not. The file count restriction allows the servers to be as agnostic as possible to file naming. 
3. Right click on the container file within Google Drive and select "Share", enter "trojai@nist.gov" and click "Done"
    - For details on how to share a file see [https://support.google.com/drive/answer/2494822](https://support.google.com/drive/answer/2494822) 
    - For details on how to stop sharing a file see [https://support.google.com/drive/answer/2494893](https://support.google.com/drive/answer/2494893)
4. Your container is now visible to the NIST trojai user.
5. Every few minutes (less than 15 minutes) the test and evaluation server will poll the NIST trojai@nist.gov Google Drive account for new submissions.
6. When your submission is detected, your container will be added to the evaluation queue. Your container will run either a) as soon as resources are available, or b) as soon as resources are available after your timeout window has passes. The timeout window is used to limit submission to one per week per team. If you upload another container (which is required to have the same filename) while the previous job is still in the queue (but has not yet been downloaded), your most recent container will be evaluated instead of the container that existed when the submission was entered into the queue.


--------------
# Results

Result Website: [https://pages.nist.gov/trojai/](https://pages.nist.gov/trojai/) 

This website will update with your submitted job status and show results when your submission completes.

A "Last modified" date is included on the page so you know when these tables were last updated. 

## Evaluation Server vs Smoke Test Server

There are two test and evaluation servers being operated by NIST. Both 'servers' operate on the same physical hardware, with identical software setups. The difference between the two 'servers' is which processing queue the job is submitted to, what compute time limits there are, and how much data will be evaluated. Currently there are 2 slots (concurrent executions) on the Smoke Test Server (STS), and 4 slots on the Evaluation Server (ES). These specific numbers might change in the future, but the Evaluation Server (ES) will always have more slots due to the much higher compute time limits. 

The results webpage by default displays the "Evaluation Server" results. You can select the "Smoke Test Server" results by clicking on the server name.

### Evaluation Server (ES)

- Dataset: the sequestered evaluation dataset
- Compute time limit: 24 hours
- Purpose: official evaluation of your container against the sequestered dataset

### Smoke Test Server (STS)

- Dataset: a small publicly available data. 
- Compute time limit: 20 minutes
- Purpose: testing, debugging. 



## Jobs Table

Contains the following fields:

- Team: The team name you selected.
- Execution Date: When your submission was received by the test and evaluation server and added to the input queue.
- File Status: Should be "None" if no file is detected, or "Ok". File status will show "Multiple Files Shared" if you share multiple files with the same name with the TrojAI Drive Account. 
- Job File Date: The timestamp of the file received from your team email which caused a job to be created, entering your submission into the work queue.
- Time Remaining until Next Execution: How long before your next submission will be eligible for entry into the work queue. For Round1 you are allowed one submission per week (7 days).

## Results Table

Contains the following fields:

- Team: The team name you selected.
- Loss (Cross Entropy): Your loss for a given submission.
- Execution Date: When the container was executed on the server.
- File Date: The file modified date of the file executed on the server. This timestamp might change if you changed the file while your job was in the input queue
- Parsing Errors: Any errors which stem from parsing your output results files.
- Launch Errors: Any errors which stem from running your container. 

## Output Logs

When your submission is run, the first 1 MB (megabyte) of output logs are uploaded to the TrojAI NIST Google Drive upon completion. That log file is then shared just with your team email (the logs are not posted publicly). 

The log will be named "<team name>.sts.out" or "<team name>.es.out" depending on which server the job ran on (Smoke Test Server = STS, or Evaluation Server = ES).

The log will be overwritten by subsequent submission evaluation runs. So if you want a persistent copy, download and rename the file from Google Drive before your next submission. 
