# Chlamy-IMPI

This repository contains the image processing and data analysis pipeline used to study photosynthesis in a large
number of Chlamydomonas reinhardtii mutants under various growth conditions.

Note: this repository is still under development, so things might break and these instructions might be incomplete.

## Image processing

### 1. Well segmentation

After setting up the environment (see below), download all image data from the google drive folder,
located at: https://drive.google.com/drive/folders/1rU8VOIdwBuDX_N6MTn0Bg5SYYb-Ov8zv and place all .tif files
in the `data` directory in this project. Then run the image processing pipeline:

```
python run_well_segmentation_preprocessing.py
```

This should store a bunch of `.npy` files in the `output` directory. Note- you can skip this step, and download 
the `.npy` files directly from a shared cache here: https://drive.google.com/drive/folders/1LB1znkc95zbgKAPVU2Rz4MMwbdcjtsBK


### 2. Database creation

Run 

```
python main.py
```

in the `database_creation` directory. This has a hardcoded input directory where it looks for .npy and corresponding .csv files, which you might need to alter.
You can also skip this step, and download a pre-created database directly from this folder: https://drive.google.com/drive/folders/1hclnhGfkmy8Rh1l_703z4dsmeCx1DbVz 


## Data analysis

Coming soon.


## Poetry instructions:

First install poetry: https://python-poetry.org/docs/#installation

Next, install the dependencies and activate the virtual environment:

```
$ poetry install
$ poetry shell
```

## Streamlit instructions:

```
$ poetry shell
$ streamlit run chlamy_impi/interactive.demo.py
```
