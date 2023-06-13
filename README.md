# **Understanding Visual Features of Nature Images Based on their Likeability**

In this project we have examined the  visual features of nature images and their relationship to the the number of upvotes (similar to likes) they receive. The objective is to understand which features contribute to an increased number of likes indicating their visual appeal to humans. This is investigated in a conceptual manner with two different methods: With a Convolutional Neural Network (CNN) the aim is to predict the number of upvotes (similar to likes) of nature images i.e. predict its popularity. It is assumed that a higher level of popularity is due to more individuals finding the image visually appealing. By extracting and visualizing the layers of the CNN regression model it is possible to analyse the advantageous features for likeability.
With a Generative Adversarial Network (GAN) the goal is to generate nature images similar to the ones in our dataset.  The network is trained on the 10\% least liked images, the 10\% most liked images and all the images. The idea is to look for differences between the generated images from the three models suggesting different themes or features.

In the following we will give an overview of this Github Repository.

## NOTE
Several scripts uses CUDA to utilize the cores of NVIDIA GPUs. To use these scripts you need NVIDIA GPU, CUDA software and the CUDA version of PyTorch.

GitHub limits the size of files and folder to 100 MB. As a result of this rule, some folders and files are not uploaded to this repository. The following structure should be implemented for the code to work flawless:
- A folder called Data has to hold submission.csv which has the columns "Score" (upvotes) and "ID" and a folder resized_images256 containing the images as .jpg files with the dimensions 256x256 (which can be produced by the files in the Prepare_Data folder). The data can be found here under the names submissions86000.csv and resized_images.zip: https://drive.google.com/drive/folders/1R4O8abUPCkbYx5PcVdN6dWPocCMfRSwY?usp=sharing
- The full dataset of 600x600 images can be found through the following link: https://1drv.ms/f/s!AjOdB1PhR86mgc0FZ528w563sSZRqg?e=NIrcHM
- In the following path: /StyleGAN3/stylegan3-main/datasets the folder is supposed to contain the datasets used for training the pretrained StyleGAN3. These includes all_data256.zip, best_data256.zip and worst_data256.zip. The datasets are uploaded to Google Drive and can be found here: https://drive.google.com/drive/folders/1R4O8abUPCkbYx5PcVdN6dWPocCMfRSwY?usp=sharing
- We used a pretrained StyleGAN3 by using a snapshot of a pretrained network on nature images from https://github.com/justinpinkney/awesome-pretrained-stylegan3. The snapshot can be found at: https://drive.google.com/drive/folders/19o_drFWln1zSuqQGGWSRC3BpnprizbP_?usp=sharing. Our network snapshots after training on the different datasets (all, best, worst) can be found here and used by anyone: https://drive.google.com/drive/folders/12nuJKrh5_-gxZWHsTWGiEMDDD5O_XrX2?usp=sharing.

## Prepare Data
The folder prepare data contains the files associated with acquiring the data found in the "Data"-folder. In the file postDownloader.py the API Pushshift was used to obtain 420.000 IDs of images from the subreddit "EartPorn". 
In the file main.py the functions of DownloadRedditPictures.py are called. This function uses Reddit's own API praw to download images from the subreddit via the IDs obtained earlier. 
The images were resized with Image_Resizer.py. The desired dimensions turned out to be 256x256 as both our models had these as input dimensions. The figure below displays the examples of images found in the dataset.

![Image Link](https://github.com/VeryThankYou/FagProject/blob/Organized/readme_images/Data.png)

## ResNet
HFFeatureExtractor is the file for extracting features from the CNN. The CNN is a pretrained ResNet-50 made for image classification found on huggingface. We modified the network to a CNN regression by editing the last layer of the network and fine-tuned the network by training it on our dataset. The RegressionEvaluation.py calculates the performance of the CNN compared to a baseline and FeatureVisualizer.py can visualize layers of the network. 

![Image Link](https://github.com/VeryThankYou/FagProject/blob/Organized/readme_images/FeatureExtraction.png)

## GAN
The GAN folder contains a failed attempt (Keras.py) to build a GAN from scratch. The results from one of its runs (see figure below) are discussed in the report and made us explore other possibilities. 

![Image Link](https://github.com/VeryThankYou/FagProject/blob/Organized/readme_images/10epochs.png)

## StyleGAN3
The StyleGAN3 folder contains the files necessary to train the pretrained StyleGAN3. The stylegan3-main holds the snapshot of the pretrained StyleGAN3 (.pkl file) and a folder called datasets which contains the zipped dataset. 
The splitter splits the resulting snapshots into individual images and combined the images with the same seed across the models trained on different datasets.

![Image Link](https://github.com/VeryThankYou/FagProject/blob/Organized/readme_images/combined164.png)

## Questionnaire
The preparations for the questionnaire included a randomization of the order of the individual images in the combined images across the models trained on different datasets. This was done in order to avoid additional bias. The results from the questionnaire could be found as a .csv-file and a script using statistical analysis called Questionnaire_Stats.py was used to assess the significance of the results. 
