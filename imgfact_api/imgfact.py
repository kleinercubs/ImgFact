import os
import json

def load_data_line(path:str) -> list:
    '''
    load data inline
    '''

    resultdata = []
    with open(path, "r", encoding="utf-8") as f:
        data = f.readlines()
        for item in data:
            if item[-1] == "\n":
                resultdata.append(item[:-1])
            else:
                resultdata.append(item)
    return resultdata

class ImgFactDataset():
    '''
    The Imgfact main class 
    '''

    def __init__(self, root_dir) -> None:
        self.mapping = dict()
        self.image_mapping = dict()
        self.entitylist = list()
        self.relationlist = list()

        print("Loading ImageFact data...")
        with open(os.path.join(root_dir, "triplet_path_mapping.json"), "r", encoding="utf-8") as f:
            self.mapping = json.load(f)
        
        entities = set()
        relations = set()
        for triplet in self.mapping:
            triplet_path = os.path.join(root_dir, self.mapping[triplet])
            if not is.path.exists(triplet_path):
                continue
            images = os.listdir(triplet_path)
            self.image_mapping[tuple(triplet.split(" "))] = [os.path.join(triplet_path, img) for img in images]
            ent1, relation, ent2 = triplet.split(" ")
            entities.add(ent1)
            entities.add(ent2)
            relations.add(relation)
        self.entitylist = list(entities)
        self.relationlist = list(relaitons)


    def load_data(self, root_dir, use_sample=False):
        '''
        Load Imgfact image data
        '''
        
        if use_sample:
            data = 0
        return self.image_mapping


    def load_entities(self):
        '''
        Load Imgfact entities
        '''

        return self.entitylist

    def load_relations(self):
        '''
        Load Imgfact relations
        '''
        
        return self.relationlist

    def get_entity_img(self, head_entity = None, tail_entity = None) -> list:
        '''
        Get the images that embody the input entity
        '''

        if head_entity is None and tail_entity is None:
            print("Please specify the head entity or tail entity to filter images")
            return
        
        entity_data = []
        for triplet in self.image_mapping:
            if triplet[0] != head_entity and head_entity is not None:
                continue
            if triplet[2] != tail_entity and tail_entity is not None:
                continue
            entity_data.append((triplet, self.image_mapping[triplet]))
        return entity_data

    def get_relation_img(self, relation):
        '''
        Get the images that embody the input relation
        '''
        relation_data = []
        for triplet in self.image_mapping:
            if triplet[1] != relation:
                continue
            relation_data.append((triplet, self.image_mapping[triplet]))
        
        return relation_data

    def get_triplet_img(self, triplet:tuple):
        '''
        Get the images that embody the input triplet
        '''
        
        triplet_data = []
        for triplet in self.image_mapping:
            if triplet != triplet:
                continue
            triplet_data.append((triplet, self.image_mapping[triplet]))
        
        return triplet_data
