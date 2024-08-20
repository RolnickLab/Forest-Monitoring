# Tree semantic segmentation from aerial image time series

## Abstract

Earth's forests play an important role in the fight against climate change, and are in turn negatively affected by it. Effective monitoring of different tree species is essential to understanding and improving the health and biodiversity of forests. In this work, we address the challenge of tree species identification by performing semantic segmentation of trees using an aerial image dataset spanning over a year. We compare models trained on single images versus those trained on time series to assess the impact of tree phenology on segmentation performances. We also introduce a simple convolutional block for extracting spatio-temporal features from image time series, enabling the use of popular pretrained backbones and methods. We leverage the hierarchical structure of tree species taxonomy by incorporating a custom loss function that refines predictions at three levels: species, genus, and higher-level taxa. Our findings demonstrate the superiority of our methodology in exploiting the time series modality and confirm that enriching labels using taxonomic information improves the semantic segmentation performance.

Paper link: [https://arxiv.org/abs/2407.13102](Arxiv)

## Getting Started

### Setting Up the Environment

To set up your development environment:

1. Clone this repository:
   ```
   git clone https://github.com/your-username/your-repo-name.git
   cd your-repo-name
   ```

2. Create a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Run the make file to install the data, libraries and clean-up:
   ```
   make all
   ```
   If you only want to download the dataset then you can run:
   ```
   make all
   ```  

### Downloading the Dataset

To get started with this project, you'll need to run bash ```data_download.sh```. By default, the makefile runs the script to download the data, so if you are using the makefile to create the environment then you can skip this step.

The default data download directory can be altered by changing the TARGET_FOLDER variable.

## Training

To train the model:

1. Activate the venv (if using virtual environments) by ```source venv/bin/activate```.
2. There are 4 separate bash files, one for each model. For eg. the one used to train the Processor-UNet is ```processor_unet.sh```.
3. Inside the bash script, the config file used for each model training is given.
   - Eg. In this command ```python treemonitoring/models/trainer.py --cfg treemonitoring/models/configs/processor_unet_7.yaml```, we can edit the hyperparameters like learning rate, batch size and loss etc.
   - To train the model without wand logging, we can use the debug flag like ```python treemonitoring/models/trainer.py --cfg treemonitoring/models/configs/processor_unet_7.yaml --debug```.
4. To train the model with default hyperparameters as used in the paper, run ```bash processor_unet.sh``` (same for other models).

## Inference

To run inference using the trained model:

1. [Explain how to load the trained model]
2. [Provide instructions for running inference]
3. [Include example commands or code snippets]

Example:
```
python inference.py --model_path /path/to/saved_model --input /path/to/input_data
```

## License

This project is licensed under the [Choose a License] License - see the [LICENSE.md](LICENSE.md) file for details.

## Citation

If you use this work in your research, please cite:

```
@misc{ramesh2024treesemanticsegmentationaerial,
      title={Tree semantic segmentation from aerial image time series}, 
      author={Venkatesh Ramesh and Arthur Ouaknine and David Rolnick},
      year={2024},
      eprint={2407.13102},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2407.13102}, 
}
```