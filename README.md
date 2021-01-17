# NICE

## NICE-Dataset
 NICE-Dataset is a vision-language dataset for image commenting. Given an image, models are required to generate human-like comments grounded on the image. NICE-Dataset has two settings. You can download images and related data for each setting at [here](https://drive.google.com/drive/folders/1V6M1W8x9vCKgabE-1dCfpoHMnj0TM00z?usp=sharing).

 There are three folders in the link: images, Setting1, and Setting2. "images" folder contains image data. "Setting1" folder

 images: We extract the image-comment pairs from Reddit and the time period for the data is from 2011-2012. Each zip file with the year prefix (2011 or 2012) has a set of images.

 Setting1:

 Setting2: For the two trainval_liwc_6x6_<year>_NoBadimg_Cleaned.tsv files, the format for each line follows:
 ```
 imgid	cmt1	liwc1	cmt2	liwc2	cmt3	liwc3	cmt4	liwc4	cmt5	liwc5	cmt6	liwc6
 ```



## paper
 This is Pytorch implementation of MAGIC model for paper "NICE: Neural Image Commenting Evaluation with an Emphasis on Emotion and Empathy" and the NICE dataset. Code and dataset will be published soon.
