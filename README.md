# ASL-Letter-Classifier

### Installation
It is highly recommended that you use Anaconda to set up an environment. If you do not have conda installed, please refer to https://www.anaconda.com/download.

##### Clone the Git repository
Clone the GitHub repository
```bash
# Using https
git clone https://github.com/RileyFrancis/ASL-Letter-Classifier.git

# Using ssh
git clone git@github.com:RileyFrancis/ASL-Letter-Classifier.git
```

##### Set up conda environment
```bash
conda env create -f environment.yml
conda activate asl
```

### Data
There are two different datasets available:
- `dataset` (24 classes. Contains images for all letters excluding J, Z. These letters are excluded due to the movement aspects of the signs themselves that are not able to be captured via image)
- `dataset_small` (2 classes. Contains images for letters A and B)

You can find both of these datasets available on Kaggle at https://kaggle.com/datasets/402fce4b3d34f098f00df0805731050351b604bdeb95dd2debc389a4f9bb4ab1. After downloading the dataset, unzip it to the project root directory.

### Model Training
To train a new model, you can use the `asl_alphabet_classifier.ipynb` Jupyter Notebook file. Before running, change the dataset path accordingly to your unzipped dataset folder (either the full or small dataset). After all cells are executed, a model is saved in the `models` folder.

### Live Prediction
Once a model has been trained, you can run a live prediction program that classifies handshapes in real time. First modify the path in `live_predict.py` to match your trained model. Then execute the following command:
```python
python live_predict.py
```