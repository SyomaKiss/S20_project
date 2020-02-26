# Data
### Intro

One of the challenges of using machine learning techniques with medical data is the frequent dearth of source image data on which to train. A representative example is automated lung cancer diagnosis, where nodule images need to be classified as suspicious or benign. In this project we investigate generative deep learning models for an ability to create realistic nodule images.

### Data used

* NIH dataset: https://www.kaggle.com/nih-chest-xrays/data  
Review: https://lukeoakdenrayner.wordpress.com/2017/12/18/the-chestxray14-dataset-problems/  
Contains 79 images labeled with bboxes encompassing lung nodules (possibly only one of several presented on the picture)

### Data which could be potentially used in future work

* Manual labeling of NIH images. It contains much more nodule images without bbox labeling than with.
* JSRT http://db.jsrt.or.jp/eng.php 
It contains no more than one nodule per image, which is the fact making it suitable for uni-object-detection.
Issues:
nodules are extremely small, sometimes non-recognizable by the human eye; 2) labels are represented as circles, whereas the nodules on the images do not always span the square shape(need rectangular labeling)
label circles are often unpredictably smaller than nodule itself, which makes it impossible to span the whole nodule with a black box(in order to fit into GAN to predict the nodule in that place) because the margin you need to add to circle-label varies significantly inside the dataset.
* (not verified) https://litfl.com/clinical-cases/ 
* (not verified) https://radiopaedia.org/encyclopaedia/cases/all?lang=us 

