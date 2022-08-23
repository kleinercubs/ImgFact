# ImgFact

This is the official github repository for the paper "ImgFact: Triplet Fact Grounding on Images".

We presented our implementation of ImgFact's construction pipeline and the experiments, and released the ImgFact dataset.

## Contents

- [Overview](#overview)
- [Data Format](#data-format)
- [Download](#download)
- [Dataset Construction](#dataset-construction)
- [Dataset Evaluation](#dataset-evaluation)
- [License](#license)
- [Citation](#citation)

## Overview

<img src="imgs/motivation.jpg"/>

ImgFact is a dataset constructed for a new task that we introduced: triplet fact grounding, which aims to collect images that embody entities and their relation. For example, given a triplet fact (**David_Beckham**, **Spouse**, **Victoria_Beckha**), we expect to find intimate images of **David_Beckham** and **Victoria_Beckha**, as shown in Fig.

In ImgFact, we present a large amount of image-triplet pairs, where the images embody the two entites and the relation between them of the correlated triplet.

## Data Format

Here we describe how ImgFact is stored and organized. The ImgFact dataset is split into 30 subsets and each subset is compressed into a `.zip` file named as `TriplelistXXX.zip` (XXX is the index ranging from 001 to 030) .

In each subset of ImgFact, The files are organized as follows:

    |-TriplelistXXX
        |-relation1
            |-"Entity1 Entity2"
                |-1.jpg
                |-2.jpg
                |-3.jpg
                ...
        |-relation2
        |-relation3
        ...
    ...

The name of the subdirectories, for example "realation1" or "relation2", in the triplelist root directory indicates the relation of the triplet that the images in it embody, and the name of the second-level subdirectories, like "Entity1 Entity2", is composed of two entity names splitted by a space meaning the two entities of the triplet that the images in it embody.

## Download

Here we provide a release version of ImgFact. The dataset can be accessed by [GoogleDrive](https://drive.google.com/drive/folders/17MWnf1hQFuOLJ-8iIe0w7Culhy2DJBzE?usp=sharing).

The dataset is splitted into 30 different `.zip` chunk files, each one is about 1GB large. Take `Triplelist001.zip` as an example, the images inside is named followed the rule `relation/head_entity tail_entity/idx.jpg`.

## Dataset Construction

All the codes related to the dataset construction pipeline are in [data_construction](https://github.com/kleinercubs/ImgFact/tree/main/dataset_construction).

- Entity Filtering: Run `inference.py` to filter entities with a trained classifier.
- Relation Filtering: First run `filter_tuples.py` , then run `gen_sample_tuples.py`, after that run `gen_candidate_relations.py`, and finally run `gen_visual_relations.py` and apply pre-defined thresholds to get the result.
- Entity-based Image Filtering: Run `ptuningfilter.py` and `ptuningfilter_ent.py` respectively and aggregate the results by getting their intersection as the filter result.
- Relation-based Image Filtering:
- Clustering:

Our implementation of the pipeline can be found here, in which all the steps except image collection is included in this repo. For image collection, we refer to this [repo]() for reference.

## Dataset Evaluation

All the codes related to the dataset evaluation are in [evaluation](https://github.com/kleinercubs/ImgFact/tree/main/evaluation).

## License

[![](https://i.creativecommons.org/l/by/4.0/88x31.png)](http://creativecommons.org/licenses/by/4.0/)

This work is licensed under a [Creative Commons Attribution 4.0 International License](http://creativecommons.org/licenses/by/4.0/)

## Citation
