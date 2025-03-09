# Bachelor Thesis: Music Emotion Recognition: Uni- and Multi-modal Machine Learning Approaches based on Audio and Symbolic Data

This is the repository of the project 
> bachelorThesis\_NinaBraunmiller\_k11923286.pdf

date: 2023 September 15
author: Braunmiller Nina

Used languages:
* Python 3.7.3 is the main language to go through all the notebooks and py files. Also the imported modules are usable via python when it is not stated differently. Exceptions: version 3.7.10 in stage1\_data\_collecting\_phase/folder audio2midi\_converter/audio2midi\_Wang when generating MIDI files; version 3.8.10 in stage3_models/multimodal\_model 
* Java 1.8.0_362 to start jSymbolic in stage2\_feature\_extraction/symbolicFeatureExtraction
* Octave 7.1.0 to make use of the package and notebook in stage1\_data\_collecting\_phase/folder audio2midi\_converter/audio2midi\_Pearce

## thesis overview

The thesis deals with the field of Music Emotion Recognition (MER). It consits of different stages which build up on each other. The thesis grapples with all of them. They are are closer explained in the upcoming sections. 

### stage-1

Here are tools provided which can be used by every stage. Also embedded images of this README file are stored there. In overall\_used\_tools/requirements\_check.py you can see the used versions for the project py files and notebooks. However, everytime you run such a file it should give feedback whether you use the same versions as they are originally used.

### stage0_provided_information

This folder contains the sample information and belonging GEMS annotations. They are already provided and not part of the achivements of this thesis.  
The central information file is:
> gems-emotion-tags-main/data/ratings\_iccs\_per\_track.xlsx

features.md lists part of the features which are in stage 2 extracted out of MIDI files in name of 
> Renato Panda, Ricardo Malheiro, and Rui Pedro Paiva. “Novel Audio Features for Music Emotion Recognition”. In: IEEE Transactions on Affective Computing 11 (2020), pp. 614–626. DOI: 10.1109/TAFFC.2018.2820691.

Also the provided mp3 tracks are stored here.



### stage1_data_collecting_phase

Here are listed three different possibilities to collect or generate MIDI data.
1. folder audio2midi\_converter: Here three different converters are tried out. Also the tracks are prepared by changing the data format from mp3 to wav and removing the singer's voice of them.
2. folder Lakh\_MIDI\_dataset: When MIDI files are collected instead of generated, it can happen that not all files can be found. Therefore, the Lakh MIDI dataset is used to check whether missing MIDI files can be added.
3. folder webscraping: This approach downloads MIDI data from the internet.


### stage2_feature_extraction

In this stage the critical features for emotion prediction of music files shall be extracted.
1. folder audioFeatureExtraction: From audio files features can be extracted with help of a varity of modules. The outputs are fusioned.
2. folder symbolicFeatureExtraction: MIDI files are used as base of symbolic feature extraction with help of the software jSymbolic 2.2. Also relevant features of Panda et al. (2020) are considered in an extra notebook. Both outputs are merged.

### stage3_models

Finally, the models for emotion predictions are implemented.
1. multimodal\_model: a model is implemented with following architecture:
![stage-1/irrelevant_readme_files/multimodal.jpg](attachment:stage-1/irrelevant_readme_files/multimodal.jpg)
2. unimodal\_scalarFeatures\_model: Different MIDI sources (converter and webscraped ones) are used as inputs and compared by the performance of sklearn models. Only scalar features are relevant. The used models for comparison are kNN, SVM, and RF which are used for each label individually. Also all are executed for categorical and regressive labels. 




