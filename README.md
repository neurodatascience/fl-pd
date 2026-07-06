# fl-pd

Federated learning on Parkinson's (and other) disease neuroimaging datasets.

## System requirements

This code requires Python 3.10 or 3.11 along with several other dependencies listed in the `pyproject.toml` file at the root of this repository.

Exact version numberss are detailed in `uv.lock`.

## Installation guide

A typical install is expected to take a few minutes.

### With `uv`

```console
uv sync
```

### Without `uv`

```console
pip install .
```

## Instructions for use

Once all Fed-BioMed nodes have been set up, the main script to run is `scripts/run_fedbiomed_custom_dataset.py`

The commands used to produce the results presented in the paper are in `commands.sh`.
They should be able to be adapted for other use-cases.

## Demo

A sample Nipoppy dataset with simulated FreeSurfer IDPs and annotated tabular phenotypic file can be found in `demo_dataset`.

The instructions below assume that the working directory is the root of the Git repository.

1. Create Fed-BioMed nodes for the individual datasets (`site1` and `site2`) and the combined one (`mega`)

    ```console
    fedbiomed component create -p demo/fedbiomed/node-site1 -c NODE -n site1
    fedbiomed component create -p demo/fedbiomed/node-site2 -c NODE -n site2
    fedbiomed component create -p demo/fedbiomed/node-mega -c NODE -n mega
    ```

2. Add the datasets to the Fed-BioMed nodes

    ```console
    # site1
    fedbiomed node -p demo/fedbiomed/node-site1 dataset add --file demo/fedbiomed/config-site1.json

    # site2
    fedbiomed node -p demo/fedbiomed/node-site2 dataset add --file demo/fedbiomed/config-site2.json

    # mega (one file per prediction task)
    fedbiomed node -p demo/fedbiomed/node-mega dataset add --file demo/fedbiomed/config-mega_site1_site2-nb:Diagnosis.json
    fedbiomed node -p demo/fedbiomed/node-mega dataset add --file demo/fedbiomed/config-mega_site1_site2-nb:Age.json
    fedbiomed node -p demo/fedbiomed/node-mega dataset add --file demo/fedbiomed/config-mega_site1_site2-fl:cognitive_decline_status.json
    ```

3. Start the Fed-BioMed nodes (in different Terminal tabs)

    ```console
    fedbiomed node -p demo/fedbiomed/node-site1 start
    ```

    ```console
    fedbiomed node -p demo/fedbiomed/node-site2 start
    ```

    ```console
    fedbiomed node -p demo/fedbiomed/node-mega start
    ```

4. Run the experiments (takes less than one minute per task with `--sloppy`):

    ```console
    # diagnosis
    ./scripts/run_fedbiomed_custom_dataset.py demo/data ./demo/results demo/fedbiomed demo/data/latest/_stats --tag-mega 'mega_site1_site2' --train-dataset site1 --train-dataset site2 --test-dataset site1 --test-dataset site2 --n-splits 2 --n-null 1 --sloppy --target 'nb:Diagnosis' 

    # age
    ./scripts/run_fedbiomed_custom_dataset.py demo/data ./demo/results demo/fedbiomed demo/data/latest/_stats --tag-mega 'mega_site1_site2' --train-dataset site1 --train-dataset site2 --test-dataset site1 --test-dataset site2 --n-splits 2 --n-null 1 --sloppy --target 'nb:Age'

    # cognitive decline
    ./scripts/run_fedbiomed_custom_dataset.py demo/data ./demo/results demo/fedbiomed demo/data/latest/_stats --tag-mega 'mega_site1_site2' --train-dataset site1 --train-dataset site2 --test-dataset site1 --test-dataset site2 --n-splits 2 --n-null 1 --sloppy --target 'fl:cognitive_decline_status' 
    ```
