# NICE

## NICE-Dataset
 NICE-Dataset is a vision-language dataset for image commenting. Given an image, models are required to generate human-like comments grounded on the image. NICE-Dataset has two settings. You can download images and related data for each setting at [here](https://drive.google.com/drive/folders/1V6M1W8x9vCKgabE-1dCfpoHMnj0TM00z?usp=sharing).

 There are three folders in the link: images, Setting1, and Setting2. "images" folder contains image data. "Setting1" folder

 images: We extract the image-comment pairs from Reddit and the time period for the data is from 2011-2012. Each zip file with the year prefix (2011 or 2012) has a set of images.

 Setting1: For this setting, a subset of data is labeled by mechanical turkers with 7 different categories scored from 1-7: Appropriate, Emotional, Empathetic, Engaging, Relevant, Offensive, Selected. We provide a set of data for validation. The format for the "labeledDataValidation_liwc_NoBadImg.tsv" is:
 ```
 imgid	cmt1	liwc1	cmt2	liwc2	cmt3	liwc3	cmt4	liwc4	cmt5	liwc5	cmt6	liwc6	Appropriate	Emotional	Empathetic	Engaging	Relevant	Offensive	Selected
 ```

 Setting2: For the two trainval_liwc_6x6_<year>_NoBadimg_Cleaned.tsv files, the format for each line follows:
 ```
 imgid	cmt1	liwc1	cmt2	liwc2	cmt3	liwc3	cmt4	liwc4	cmt5	liwc5	cmt6	liwc6
 ```
 For both files, the first line is the column names. "processed_data_152_<year>.zip" contains json files for encoded comments using GPT2 tokenizer from [huggingface/transformers](https://github.com/huggingface/transformers). After unzipping the file, "imgid_cleaned.json" has the image_id for each data sample, "input_ids_cleaned.json" has the encoded comments starting with the special token "[SEP]", "label_ids_cleaned.json" has the encoded comments ending with the special token "[SEP]", and "liwc_cleaned_json" has the liwc features for each comments in each data sample.

