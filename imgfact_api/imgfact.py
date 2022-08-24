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

    def __init__(self) -> None:
        self.data = []

    def load_data(self, use_sample=False):
        '''
        Load Imgfact image data
        '''

        if use_sample:
            data = 0
        return


    def load_entities(self):
        '''
        Load Imgfact entities
        '''

        entitypath = ""
        return load_data_line(entitypath)

    def load_relations(self):
        '''
        Load Imgfact relations
        '''
        
        relationpath = ""
        return load_data_line(relationpath)

    def get_entity_img(self, head_entity = None, tail_entity = None) -> list:
        '''
        Get the images that embody the input entity
        '''

        if head_entity is None and tail_entity is None:
            print("Please specify the head entity or tail entity to filter images")
            return
        
        return

    def get_relation_img(self, relation):
        '''
        Get the images that embody the input relation
        '''
        
        return

    def get_triplet_img(self, triplet:tuple):
        '''
        Get the images that embody the input triplet
        '''
        
        return