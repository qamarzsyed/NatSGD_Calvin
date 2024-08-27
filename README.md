# Using NatSGD Dataset with Calvin
This repository takes the repository from the CALVIN benchmark and alters the encoders and decoders from the model to make use of the NatSGD dataset.

### Run Instructions:
1. Download the NatSGD data and place image folder and .npz file in the path referenced in the datasets/natsgd_data_module.py data loader file

2. Run the mcil.py or mcil_*.py files to train the model

3. Evaluate results by running trajectories in the NatSGD simulator
