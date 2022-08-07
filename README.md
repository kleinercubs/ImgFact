# ImgFact

Code and data for paper "ImgFact: Triplet Fact Grounding on Images"


<img src="imgs/motivation.jpg"/>

We introduce a new task of triplet fact grounding, which aims to collect images that embody entities and their relation. For example, given a triplet fact (**David_Beckham**, **Spouse**, **Victoria_Beckha**), we expect to find intimate images of **David_Beckham** and **Victoria_Beckha**, as shown in Fig. 

## Download
The dataset can be accessed by [GoogleDrive](https://drive.google.com/drive/folders/17MWnf1hQFuOLJ-8iIe0w7Culhy2DJBzE?usp=sharing).

There are 30 different `.zip` files. Take `Triplelist001.zip` as an example, the images inside is named followed the rule `relation/head_entity tail_entity/idx.jpg`.

## Construction Codes
All the codes related to the dataset construction pipeline are in [data_construction](https://github.com/kleinercubs/ImgFact/tree/main/dataset_construction).

## Evaluation Codes
All the codes related to the dataset evaluation are in [evaluation](https://github.com/kleinercubs/ImgFact/tree/main/evaluation).

## License
[![](https://i.creativecommons.org/l/by/4.0/88x31.png)](http://creativecommons.org/licenses/by/4.0/)

This work is licensed under a [Creative Commons Attribution 4.0 International License](http://creativecommons.org/licenses/by/4.0/)