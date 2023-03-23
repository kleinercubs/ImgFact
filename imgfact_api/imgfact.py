import os
import json
from tqdm import tqdm

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

        print("Loading ImageFact data...")
        with open(os.path.join(root_dir, "triplet_path_mapping.json"), "r", encoding="utf-8") as f:
            self.mapping = json.load(f)
        self.entities = set()
        self.relations = set()

        for triplet in tqdm(self.mapping):
            triplet_path = os.path.join(root_dir, self.mapping[triplet])
            if not os.path.exists(triplet_path):
                continue
            images = os.listdir(triplet_path)
            self.image_mapping[tuple(triplet.split(" "))] = [os.path.join(triplet_path, img) for img in images]
            ent1, relation, ent2 = triplet.split(" ")
            self.entities.add(ent1)
            self.entities.add(ent2)
            self.relations.add(relation)
        
        print(f"Total Triplets:{len(self.mapping)} Loaded Triplets:{len(self.image_mapping)}")


    def load_entities(self):
        '''
        Load Imgfact entities
        '''

        return list(self.entities)


    def load_relations(self):
        '''
        Load Imgfact relations
        '''
        
        return list(self.relations)


    def retrieve_img_from_entity(self, head_entity = None, tail_entity = None) -> list:
        '''
        Get the images that embody the input entity
        '''

        assert head_entity is not None or tail_entity is not None, "Please specify the head entity or tail entity to filter images"
        assert head_entity is None or head_entity in self.entities, f"entity {head_entity} not found in loaded triplets"
        assert tail_entity is None or tail_entity in self.entities, f"entity {tail_entity} not found in loaded triplets"

        entity_data = []
        for triplet in self.image_mapping:
            if triplet[0] != head_entity and head_entity is not None:
                continue
            if triplet[2] != tail_entity and tail_entity is not None:
                continue
            entity_data.append((triplet, self.image_mapping[triplet]))
        return entity_data


    def retrieve_img_from_relation(self, relation:str):
        '''
        Get the images that embody the input relation
        '''

        assert relation in self.relations, f"relation {relation} not found in loaded triplets"

        relation_data = []
        for triplet in self.image_mapping:
            if triplet[1] != relation:
                continue
            relation_data.append((triplet, self.image_mapping[triplet]))
        
        return relation_data


    def retrieve_img_from_triplet(self, triplet:tuple):
        '''
        Get the images that embody the input triplet
        '''

        assert triplet in self.image_mapping, f"triplet {triplet} not found in loaded triplets"
        
        triplet_data = []
        for triplet in self.image_mapping:
            if triplet != triplet:
                continue
            triplet_data.append((triplet, self.image_mapping[triplet]))
        
        return triplet_data
