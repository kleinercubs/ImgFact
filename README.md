# ImgFact

This is the official github repository for the paper "ImgFact: Triplet Fact Grounding on Images".

We presented our implementation of ImgFact's construction pipeline and the experiments, and released the ImgFact dataset.

## Contents

- [ImgFact](#imgfact)
  - [Contents](#contents)
  - [Overview](#overview)
  - [Download](#download)
  - [ImgFact API](#imgfact-api)
  - [Data Format](#data-format)
  - [Dataset Construction](#dataset-construction)
  - [Dataset Evaluation and Application](#dataset-evaluation-and-application)
  - [License](#license)

## Overview

<img src="imgs/motivation.jpg"/>


In ImgFact, we aim at grounding triplet facts in KGs on images to construct a new MMKG, where these images reflect not only head and tail entities, but also their relations.

For example, given a triplet fact (**David_Beckham**, **Spouse**, **Victoria_Beckha**), we expect to find intimate images of **David_Beckham** and **Victoria_Beckha**.

## Download

Here we provide a release version of ImgFact. The full dataset including all the images and the corresponding triplets can be accessed by [GoogleDrive](https://drive.google.com/drive/folders/17MWnf1hQFuOLJ-8iIe0w7Culhy2DJBzE?usp=sharing).

The triplets to path map file is [triplet_path_mapping.json](https://github.com/kleinercubs/ImgFact/blob/main/triplet_path_mapping.json).

## ImgFact API

 Here we provide a easy-to-use API to enable easy access of ImgFact data. Before using the ImgFact api, you should download both the dataset and the `triplet_path_mapping.json` into one directory. You can use the api to explore ImgFact by:

```python
>>> from imgfact_api import ImgFactDataset
>>> dataset = ImgFactDataset(root_dir=".")
>>> data = dataset.load_data()
```

To list all the relations and entities in ImgFact, use:

```python
>>> relations = imgfact.load_relations()
>>> entities = imgfact.load_entities()
```

The ImgFact api supports different image browsing method, you can retrieve image by the triplet that it embodies. There are three methods to access images:

```python
# Retrieve images by entity
>>> imgs = get_entity_img(head_entity="Ent1", tail_entity="Ent2")

# Retrieve images by relation
>>> imgs = get_relation_img(relation="relation1")

# Retrieve images by triplet
>>> imgs = get_triplet_img(triplet="Ent1 relation Ent2")
```


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

For example, the image `Triplelist001/relation/head_ent tail_ent/1.jpg` means that the image embodies the triplet `head_ent relation tail_ent` in it.

## Dataset Construction

All the codes related to the dataset construction pipeline are in [data_construction](https://github.com/kleinercubs/ImgFact/tree/main/dataset_construction). 
Our implementation of the pipeline can be found here, in which all the steps except image collection is included in this repo. For image collection, we refer to this [AutoCrawler](https://github.com/YoongiKim/AutoCrawler) for reference.
 The construction pipeline should run by the following order:

- Entity Filtering: Filter entities with a trained classifier.

```
python inference.py
```

- Relation Filtering: Run following commands in order and apply pre-defined thresholds to get the result.

```
python filter_tuples.py
python gen_sample_tuples.py
python gen_candidate_relations.py
python gen_visual_relations.py
```

- Entity-based Image Filtering: Run following codes respectively and aggregate the results by getting their intersection as the filter result.

```
python ptuningfilter.py
python ptuningfilter_ent.py
```

- Image Collection: Apply any toolbox that can collect images from search engines.
- Relation-based Image Filtering: Run following codes for training and inference.

```
python CPgen.py --do_train
python CPgen.py --do_predict --file {XXX}
```

Note: `XXX` denotes the 3 digit file id, starts with leading zero, e.g. `001`.

- Clustering: Get the final clustering result.

```
python cluster.py
```

## Dataset Evaluation and Application

All the codes related to the dataset evaluation and application are in [eval_and_app](https://github.com/kleinercubs/ImgFact/tree/main/eval_and_app).

The evaluation and application are similar. The only difference is the information the model received.

- Generate sub-task datasets by simply run script `generation.sh`.
- Training and evaluation with different models on different sub-task:

On ViLT:

```
python vilt.py --dataset {TASK_NAME} --epochs 150 --lr 3e-5 --optimizer ranger
```

On BERT+ResNet:

```
python multimodal_naive.py --dataset {TASK_NAME} --epochs 150 --lr 3e-5 --optimizer ranger
```

Note: If you want to perform the experiments by using only text information, use:

```
python multimodal_naive.py --dataset {TASK_NAME} --epochs 150 --lr 3e-5 --optimizer ranger --modality text
```

Default `TASK_NAME` includes `predict_s/spo`, `predict_s/p`, `predict_s/o`, `predict_s/messy`, `predict_p/spo`, `predict_p/s`, `predict_p/o`, `predict_p/messy`, `predict_o/spo`, `predict_o/s`, `predict_o/p` and `predict_o/messy`. 

The specific task name follows the naming rules: `predict_{predict target}/{known information}`. For examples, `predict_s/spo` means given the images containing all the information of the triplets and want the model to predict the missing head entity. 

## License

[![](https://licensebuttons.net/l/by-nc/4.0/88x31.png)](https://creativecommons.org/licenses/by-nc/4.0/)

This work is licensed under a [Creative Commons Attribution-NonCommercial 4.0 International Public License](https://creativecommons.org/licenses/by-nc/4.0/)

