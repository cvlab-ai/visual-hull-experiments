# Test environment
for the article "Visual hull-based approach for coronary 3D reconstruction from synthetic heart arteries projections"

## Setting up the environment
Initialize submodules and then
```
python3 -m venv .test-env
source .test-env/bin/activate
pip3 install ./coronaryApp
pip3 install -r requirements.txt
```

## Experiments
Each configuration defined in `config/` directory defines one of the carried out experiments and tests a single independent variable:
- n_projs - number of projections
- angles - angle selection
- noise_translation - artifacts
- general - general method results in perfect conditions

## Generating dataset
```
./generate_data.py -f [config]
```

## Fetching dataset used in the article
Warning! This will overwrite data directory
```
./download_data.py
```

## Running tests
```
./run_experiment -f [config]
```

## Generating figures
```
./figures.py
```
