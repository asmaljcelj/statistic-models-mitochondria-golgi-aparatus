# Statistical models for Mitochondria and Golgi Apparatus

This is the code used for master`s thesis with title "Procedural generation of mitochondria and Golgi apparatus shapes and their distribution within the cell with the use of statistical models" by Aljaž Šmaljcelj

Initial volumetric data are obtained from the [Urocell dataset](https://github.com/MancaZerovnikMekuc/UroCell)

## Code files

* `utils.py`: contains utility methods
* `math_utils.py`: contains methods that perform mathematical calculations 
* `bezier.py`: utility methods for extracting bezier curve

### Preprocessing

* `mitochondria_extract.py`
  * extracts individual mitochondria instances in voxel representation in nitfl format
  * volumetric `.nii.gz` files are taken from `data` directory
  * instances are saved in `extracted_data` directory where they are split into `learning` and `testing` directories
* `mitochondria_skeletonization.py`
  * skeletonizes objects in the `extracted_data` directories; results are stored in `skeletons` folder
  * keeps the skeletons from learning and testing sets separate
* `ga_extract.py`
  * extracts individual Golgi apparatus instances in voxel representation in nitfl format
  * volumetric `.nii.gz` files are taken from `data_ga/approximate` directory
  * instances are saved in `ga_instances` directory, together with their PCA components in a separate file ending with `_ev`

### Statistical analysis

* `mitochondria_sampling.py`
  * samples distances in mitochondria objects from skeletons to the surface
  * parameters:
    * `num_of_skeleton_points`: number of points on the skeleton that are used for sampling origins
    * `n`: degree of Bezier curve
    * `num_of_samples`: number of sampled directions used for hemisphere sampling
    * `angle_increment`: increment of the angle used to sample new direction
      * has to be divisible by 360
  * saves measurements in `measurements` directory
    * learning distances are grouped together
    * testing instances are saved individually
* `ga_statistics.py`
  * samples individual cisternae from each extracted Golgi apparatus instance
  * reads instances from `ga_instances` directory
  * parameters:
    * `split_percentage`: percentage of instances being placed in learning set, others go to testing set
    * `num_of_distance_vectors`: number of sampling distances in 2D space used to measure distances
  * saves the measurements in `mesurements_ga` directory
    * performs a split into learning and testing sets
      * learning set cisternae distances are grouped together
      * testing set cisternae instance are stored individually

### Generating new objects

* `mitochondria_new_instance.py`: generates new mitochondria instance
  * parameters can be given in command line:
    * `-c`: curvature value
      * if only 1 value is given, it applies to all skeleton points, otherwise one value to each skeleton point 
    * `-l`: length of skeleton (number of skeleton points)
    * `-t`: torsion value
      * if only 1 value is given, it applies to all skeleton points, otherwise one value to each skeleton point 
    * `-s`: seed for random number generator
    * `-sl`: sigma value for sampling new length value
    * `-ss`: sigma value for sampling points around skeleton
    * `-sst`: sigma value for sampling points around one instance end
    * `-se`: sigma value for sampling points around other instance end
    * `-sc`: sigma value for sampling curvature values
    * `-st`: sigma value for sampling torsion values
    * `-it`: number of smoothing iterations performed at the end
  * creates new mesh `mesh.obj` file in `results` directory
* `ga_new_instance.py`: generates new Golgi apparatus instance
  * parameters can be given in command line:
    * `-c`: number of cisternae in a stack
    * `-l`: parameter `interval_width` for sampling new distances
      * if only 1 value is given, it applies to all samples, otherwise it is used for every direction 
    * `-s`: seed for random number generator
    * `-sc`: parameter `scale` for scaling the entire object
      * if only 1 value is given, it applies to all samples, otherwise it is used for every direction 
    * `-it`: number of smoothing iterations performed at the end
  * creates new `mesh_ga.obj` file in `results` directory

### Evaluation

* `instance_evaluation.py`
  * calculates RMSE for specified mitochondria or Golgi apparatus instances
  * specify the path to the testing instance and path to all the testing instances measurements
  * prints result in console

## Mitochondria

In order to obtain new mitochondria shapes, perform the following:
* save the data in `.nii.gz` files to `data` directory
* run `mitochondria_extract.py`
* run `mitochondria_skeletonization.py`
* run `mitochondria_sampling.py`
* run `mitochondria_new_instance.py` with desired parameters

## Golgi apparatus

In order to obtain new Golgi apparatus shapes, perform the following:
* save the data in `.nii.gz` files to `data_ga` directory
* run `ga_extract.py`
* run `ga_statistics.py`
* run `ga_new_instance.py` with desired parameters
