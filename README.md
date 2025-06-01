# Statistical models for Mitochondria and Golgi Apparatus

This is the code used for master's thesis with title "Procedural generation of mitochondria and Golgi apparatus shapes and their distribution within the cell with the use of statistical models" by Aljaž Šmaljcelj

Initial volumetric data are obtained from the [Urocell dataset](https://github.com/MancaZerovnikMekuc/UroCell)

## Code files

* utils.py:
* math_utils.py:
* bezier.py
  * a

### Preprocessing

* extract_mitochondria.py
  * extracts individual mitochondria instances in voxel representation in nitfl format
  * accepts parameters
    * `data_directory`: relative path to directory where all the instances in `.nii.gz`
    * `extracted_data_directory`: path to directory where the results are to be saved
* skeletonization.py
  * skeletonizes objects in the given directory; results are stored in 'skeletons' folder
* extract_golgi.py

### Statistical analysis

* sampling.py
  * samples distances in mitochondria objects from skeletons to the 

### Generating new objects

### Evaluation

## Mitochondria

In order to obtain new mitochondria shapes, perform the following:
* create 

## Golgi apparatus





