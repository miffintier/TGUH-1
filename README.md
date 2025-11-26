# TGUH
Text-Guided Unsupervised Hashing with
Community Exploration for Image Retrieval.![image](/framework.png)

# Main Dependencies
-python 3.9.12
-pytorch 1.13.1
-torchvision 0.14.1
-numpy 1.21.5

# Data
The VOC2012, FLICKR25K and MSCOCO datasets are all publicly available.
You can download VOC2012 at https://pjreddie.com/media/files/VOCtrainval_11-May-2012.tar
You can download FLICKR25K at http://press.liacs.nl/mirflickr/mirflickr25k.v3b/mirflickr25k.zip
You can download MSCOCO at http://images.cocodataset.org/zips/train2014.zip   http://images.cocodataset.org/zips/val2014.zip
# Training
If you want to run the project quickly, simply prepare the Flickr25K dataset, modify the data path accordingly, and execute python run.py.
We have prepared the files required for running the project using the Flickr dataset as an example.
For further experiments, please follow the steps outlined below. Fisrt, generate the object pseudo-Label for each dataset.

TGUH requires generating textual descriptions for the images in the dataset, and you can use the OFA model or other models to accomplish this.
First, extract features from the dataset and the texts. We use the CLIP model to extract features separately for the database, training set, and test set, while also extracting features from the generated textual descriptions.
''''
$ cd TGUH/clip_feature
python clip_img_feature.py
python clip_text_feature.py
''''


Subsequently, we construct a similarity matrix from the image and text features, and run Create_Matrix.py twice to generate two matrices.
''''
$ cd TGUH/train_Martix
python Create_Martix.py
''''

Then, we explore community relationships based on the aforementioned information. 
''''
python create_community.py
''''
Finally, assign the output of create_community.py to community75in run.py as the community partition, then run run.py to complete the execution.
''''
python run.py
''''

The relevant parameters can be adjusted in the file's config.