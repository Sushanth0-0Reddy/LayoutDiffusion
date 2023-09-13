import os
import random
from collections import defaultdict
import torch
import torchvision.transforms as T
import numpy as np
import h5py
import PIL
import pickle
from layout_diffusion.dataset.util import image_normalize
from layout_diffusion.dataset.augmentations import RandomSampleCrop, RandomMirror
from torch.utils.data import Dataset
import json
from PIL import Image

from omegaconf import OmegaConf
#from layout_diffusion.dataset.data_loader import build_loaders
import matplotlib.pyplot as plt

def labels_from_class_and_part(X_train_vector,class_v_vector):
    # Broadcasting to match dimensions
    X_train_expanded = np.expand_dims(X_train_vector, axis=0)
    class_v_expanded = np.expand_dims(class_v_vector, axis=1)
    # Perform the dor product
    return np.dot(class_v_expanded,X_train_expanded)

def filtered_labels(labels_v,class_comb_dict):
    label = []
    for i, x in enumerate(class_comb_dict.values()):
        #print(x)
        label = label + list(labels_v[i,:len(x.keys())])
        #print(len(x.keys()))
        
    #print(len(label),label)
    return np.array(label)

def get_class_and_part_names(index_class_dict,class_comb_dict,index_part_list,X, class_v):
    x = np.where(class_v == 1)
    class_name = index_class_dict[int(x[0])]
    
    labels_v = labels_from_class_and_part(X[ :, 0], class_v)
                
    final_labels = filtered_labels(labels_v,class_comb_dict)
    
    
    #labels_v = labels_v.flatten()[0:16]
    #print(len(class_comb_dict[class_name].keys()))
    
    y = np.where(X[:, 0] == 1)
    
    all_part_names = []
    for i, t in enumerate(class_comb_dict[class_name]):
        #print(i,t)
        part_names = []
        if i in list(y[0]):
            for parts in class_comb_dict[class_name][t]:
                #print(parts)
                part_names.append(index_part_list[int(x[0])][parts])
                #print(index_part_list[int(x[0])][parts])
            #print("\n")
        if len(part_names)>0:
            all_part_names.append(part_names)
    return class_name, all_part_names , final_labels


class MeronymnetDataset(Dataset):
    
    
    def __init__(self, images , X, class_v, index_class_dict, index_part_list, class_comb_dict, image_size=(256, 256),
                 max_objects_per_image=16, max_num_samples=None, mask_size=32,  
                 left_right_flip=False, min_object_size=1/128, use_MinIoURandomCrop=False,
                 return_origin_image=False, specific_image_ids=[]
                 ):
        super(MeronymnetDataset, self).__init__()

        self.return_origin_image = return_origin_image
        if self.return_origin_image:
            self.origin_transform = T.Compose([
                T.ToTensor(),
                image_normalize()
            ])

        self.images = images
        self.X = X
        self.class_v = class_v
        self.index_class_dict = index_class_dict
        self.index_part_list = index_part_list
        self.class_comb_dict = class_comb_dict
        self.mask_size = mask_size
        self.image_size = image_size
        self.min_object_size = min_object_size
        # self.vocab = vocab
        self.num_objects = 94
        self.max_objects_per_image = max_objects_per_image
        self.max_num_samples = max_num_samples
        self.left_right_flip = left_right_flip
        if left_right_flip:
            self.random_flip = RandomMirror()

        self.use_MinIoURandomCrop = use_MinIoURandomCrop
        if use_MinIoURandomCrop:
            self.MinIoURandomCrop = RandomSampleCrop()

        self.transform = T.Compose([
            T.ToTensor(),
            T.Resize(size=image_size, antialias=True),
            image_normalize()
        ])

        self.total_num_bbox = 0
        self.total_num_invalid_bbox = 0
        
                     
        '''
        with h5py.File(h5_path, 'r') as f:
            for k, v in f.items():
                if k == 'image_paths':
                    # self.image_
                    self.image_paths = [str(path, encoding="utf-8") for path in list(v)]
                else:
                    self.data[k] = torch.IntTensor(np.asarray(v))
        
        '''
        
        self.data = {}
        self.data['images'] = []
        self.data['objects_per_image'] = []
        self.data['object_boxes'] =[]
        self.data['object_names'] =  []
        self.data['object_class']=[]
                     
        for i in range(len(images)):
            #print(i)
            boxes = []
            self.data['images'].append(images[i])
            for j in range(len(X[i])):
                if(self.X[i,j,0]==1):
                    boxes.append(X[i,j,1:])
            self.data['object_boxes'].append(np.asarray(boxes))
            class_name,all_part_names,labels_v = get_class_and_part_names(self.index_class_dict,self.class_comb_dict ,self.index_part_list ,self.X[i,:,:],class_v[i])
            object_names = []
            for i in range(len(all_part_names)):
                object_names.append(class_name+' '+' '.join(all_part_names[i]))
            self.data['object_names'].append(object_names)
            self.data['objects_per_image'].append(len(object_names))
            self.data['object_class'].append(np.where(labels_v==1)[0]+1) # 1 to 112 
            
            #self.data['relationship_subjects'] = 

        '''
        self.vocab = {
            'object_name_to_idx': {},
            'pred_name_to_idx': {},
        }
        object_idx_to_name = {}
        all_instance_categories = []
        for category_data in instances_data['categories']:
            category_id = category_data['id']
            category_name = category_data['name']
            all_instance_categories.append(category_name)
            object_idx_to_name[category_id] = category_name
            self.vocab['object_name_to_idx'][category_name] = category_id
        all_stuff_categories = []
        if stuff_data:
            for category_data in stuff_data['categories']:
                category_name = category_data['name']
                category_id = category_data['id']
                all_stuff_categories.append(category_name)
                object_idx_to_name[category_id] = category_name
                self.vocab['object_name_to_idx'][category_name] = category_id

        '''            
        # get specific image ids or get specific number of images
        selected_idx = []
        self.specific_image_ids = specific_image_ids
        if self.specific_image_ids:
            specific_image_ids_set = set(specific_image_ids)
            for idx, image_path in enumerate(self.image_paths):
                if image_path in specific_image_ids_set:
                    selected_idx.append(idx)
                    specific_image_ids_set.remove(image_path)
                if len(specific_image_ids_set) == 0:
                    break

            if len(specific_image_ids_set) > 0:
                for image_path in list(specific_image_ids_set):
                    print('image path: {} is not found'.format(image_path))

            assert len(specific_image_ids_set) == 0
        elif self.max_num_samples:
            selected_idx = [idx for idx in range(self.max_num_samples)]

        if selected_idx:
            print('selected_idx = {}'.format(selected_idx))
            #self.image_paths = [self.image_paths[idx] for idx in selected_idx]
            for k in list(self.data.keys()):
                self.data[k] = [self.data[k][idx] for idx in selected_idx]

    def check_with_relation(self, image_index):
        '''
        :param obj_idxs: the idxs of objects of image
        :return: with_relations = [True, False, ....], shape=(O,), O is the number of objects
        '''
        obj_idxs = range(self.data['objects_per_image'][image_index].item())
        with_relations = [False for i in obj_idxs]
        for r_idx in range(self.data['relationships_per_image'][image_index]):
            s = self.data['relationship_subjects'][image_index, r_idx].item()
            o = self.data['relationship_objects'][image_index, r_idx].item()
            with_relations[s] = True
            with_relations[o] = True
        without_relations = [not i for i in with_relations]
        return with_relations, without_relations

    def filter_invalid_bbox(self, H, W, bbox, is_valid_bbox, verbose=False):

        for idx, obj_bbox in enumerate(bbox):
            if not is_valid_bbox[idx]:
                continue
            
            self.total_num_bbox += 1
            x, y, w, h = obj_bbox
            #print(obj_bbox)
            if (x >= W) or (y >= H):
                is_valid_bbox[idx] = False
                print(self.total_num_invalid_bbox)
                self.total_num_invalid_bbox += 1
                if verbose:
                    print(
                        'total_num = {}, invalid_num = {}, x = {}, y={}, w={}, h={}, W={}, H={}'.format(
                            self.total_num_bbox, self.total_num_invalid_bbox, x, y, w, h, W, H,
                        )
                    )
                continue

            x0, y0, x1, y1 = x, y, x + w, y + h
            x1 = np.clip(x + w, 0, W)
            y1 = np.clip(y + h, 0, H)
            #print(x0,y0,x1,y1)
            if (y1 - y0 < self.min_object_size) or (x1 - x0 < self.min_object_size):
                print(y1-y0,x1-x0)
                print("invalid-",idx)
                is_valid_bbox[idx] = False
                self.total_num_invalid_bbox += 1
                if verbose:
                    print(
                        'total_num = {}, invalid_num = {}, x = {}, y={}, w={}, h={}, W={}, H={}'.format(
                            self.total_num_bbox, self.total_num_invalid_bbox, x, y, w, h, W, H,
                        )
                    )
                continue
            bbox[idx][0], bbox[idx][1], bbox[idx][2], bbox[idx][3] = x0, y0, x1, y1

        return bbox, is_valid_bbox

    def get_init_meta_data(self, image_index):
        layout_length = self.max_objects_per_image + 2
        meta_data = {
            'obj_bbox': torch.zeros([layout_length, 4]),
            'obj_class': torch.LongTensor(layout_length).fill_(self.num_objects + 1),
            'obj_class_name':['_null_']*layout_length,
            'is_valid_obj': torch.zeros([layout_length]),
        }

        # The first object will be the special __image__ object
        meta_data['obj_bbox'][0] = torch.FloatTensor([0, 0, 1, 1])
        meta_data['obj_class'][0] = 0
        meta_data['is_valid_obj'][0] = 1.0
        meta_data['obj_class_name'][0]= '__image__'

        return meta_data

    def __len__(self):
        num = len(self.data['object_names'])
        assert num == len(self.data['object_class'])
        return num

    def __getitem__(self, index):
        """
        Returns a tuple of:
        - image: FloatTensor of shape (C, H, W)
        - objs: LongTensor of shape (O,)
        - boxes: FloatTensor of shape (O, 4) giving boxes for objects in
          (x0, y0, x1, y1) format, in a [0, 1] coordinate system.
        """

        # Figure out which objects appear in relationships and which don't
        # with_relations, without_relations = self.check_with_relation(image_index=index)  # (O,)

        image = self.data['images'][index]
        if self.return_origin_image:
            origin_image = np.array(image, dtype=np.float32) / 255.0
        
        image = np.array(image, dtype=np.float32) / 255.0
        H, W, _ = 1,1,1
        num_obj = self.data['objects_per_image'][index]
        obj_bbox = self.data['object_boxes'][index][:num_obj]
        #print(obj_bbox)
        obj_class = self.data['object_class'][index][:num_obj]
        obj_class_name = self.data['object_names'][index][:num_obj]
        is_valid_obj = (obj_class >= 0)
        #print(obj_bbox)
        #get meta data
        meta_data = self.get_init_meta_data(image_index=index)
        meta_data['width'], meta_data['height'] = W, H
        #meta_data['with_relations'] = with_relations

        # filter invalid bbox
        
        obj_bbox, is_valid_obj = self.filter_invalid_bbox(H=H, W=W, bbox=obj_bbox, is_valid_bbox=is_valid_obj)
        #print(obj_bbox)
        #print(is_valid_obj)
        
        # flip
        if self.left_right_flip:
            #plt.imshow(image)
            #plt.show()
            image, obj_bbox, obj_class = self.random_flip(image, obj_bbox, obj_class)

        # random crop image and its bbox
        if self.use_MinIoURandomCrop:
            #plt.imshow(image)
            #plt.show()
            image, updated_obj_bbox, updated_obj_class, tmp_is_valid_obj = self.MinIoURandomCrop(image, obj_bbox[is_valid_obj], obj_class[is_valid_obj])

            tmp_idx = 0
            tmp_tmp_idx = 0
            for idx, is_valid in enumerate(is_valid_obj):
                if is_valid:
                    if tmp_is_valid_obj[tmp_idx]:
                        obj_bbox[idx] = updated_obj_bbox[tmp_tmp_idx]
                        tmp_tmp_idx += 1
                    else:
                        is_valid_obj[idx] = False
                    tmp_idx += 1

            meta_data['new_height'] = image.shape[0]
            meta_data['new_width'] = image.shape[1]
            H, W, _ = image.shape
        
        #print(obj_bbox,is_valid_obj)
        obj_bbox = torch.FloatTensor(obj_bbox[is_valid_obj])
        obj_class = torch.LongTensor(obj_class[is_valid_obj])

        #obj_bbox[:, 0::2] = obj_bbox[:, 0::2] 
        #obj_bbox[:, 1::2] = obj_bbox[:, 1::2] 
        #print(obj_bbox)

        num_selected = min(obj_bbox.shape[0], self.max_objects_per_image)
        #print(num_selected)
        selected_obj_idxs = range(0, num_selected)
        meta_data['obj_bbox'][1:1 + num_selected] = obj_bbox[selected_obj_idxs]
        meta_data['obj_class'][1:1 + num_selected] = obj_class[selected_obj_idxs]
        meta_data['is_valid_obj'][1:1 + num_selected] = 1.0
        meta_data['num_selected'] = num_selected
        meta_data['num_add'] = 0
        meta_data['obj_class_name'][1:1+num_selected] = obj_class_name
        meta_data['num_obj'] = meta_data['num_selected'] + meta_data['num_add'] - 1
        #plt.imshow(np.asarray(self.transform(image)).transpose(1,2,0))
        #plt.show()
        #print(np.asarray(self.transform(image)).transpose(1,2,0).max())
        if self.return_origin_image:
            meta_data['origin_image'] = self.origin_transform(origin_image)
        
        return self.transform(image), meta_data

'''
        if num_selected < self.max_objects_per_image and self.use_orphaned_objects:
            num_add = min(self.max_objects_per_image - num_selected, obj_bbox_without_relations.shape[0])
            if num_add > 0:
                selected_obj_idxs = random.sample(range(obj_bbox_without_relations.shape[0]), num_add)
                meta_data['obj_bbox'][1 + num_selected:1 + num_selected + num_add] = obj_bbox_without_relations[selected_obj_idxs]
                meta_data['obj_class'][1 + num_selected:1 + num_selected + num_add] = obj_class_without_relations[selected_obj_idxs]
                meta_data['is_valid_obj'][1 + num_selected:1 + num_selected + num_add] = 1.0
                meta_data['num_add'] = num_add
'''
def mn_collate_fn_for_layout(batch):
    """
    Collate function to be used when wrapping CocoSceneGraphDataset in a
    DataLoader. Returns a tuple of the following:

    - imgs: FloatTensor of shape (N, C, H, W)
    - objs: LongTensor of shape (N, L) giving object categories
    - masks: FloatTensor of shape (N, L, H, W)
    - is_valid_obj: FloatTensor of shape (N, L)
    """

    all_meta_data = defaultdict(list)
    all_imgs = []

    for i, (img, meta_data) in enumerate(batch):
        all_imgs.append(img[None])
        for key, value in meta_data.items():
            all_meta_data[key].append(value)

    all_imgs = torch.cat(all_imgs)
    for key, value in all_meta_data.items():
        if key in ['obj_bbox', 'obj_class', 'is_valid_obj'] or key.startswith('labels_from_layout_to_image_at_resolution'):
            all_meta_data[key] = torch.stack(value)

    return all_imgs, all_meta_data




def load_data(mode):
    data_path = '/content/gdrive/MyDrive/meronymnet/meronymnet/data_np_16/'
    part_data_post_fix = '_scaled_sqr'
    obj_data_postfix = '_obj_boundary_sqr'
    file_postfix = '_combined_mask_data'

    if mode == 'train':
        X_file = 'X_train' + part_data_post_fix + '.np'
        class_v_file = 'class_v' + file_postfix + '.np'
        images_file = 'images_train' + file_postfix + '.np'
    elif mode == 'val':
        X_file = 'X_train_val' + part_data_post_fix + '.np'
        class_v_file = 'class_v_val' + file_postfix + '.np'
        images_file = 'images_val' + file_postfix + '.np'
    elif mode == 'test':
        X_file = 'X_test' + part_data_post_fix + '.np'
        class_v_file = 'class_v_test' + file_postfix + '.np'
        images_file = 'images_test' + file_postfix + '.np'
    else:
        raise ValueError("Invalid mode. Mode should be 'train', 'val', or 'test'.")

    with open(data_path + X_file, 'rb') as pickle_file:
        X = pickle.load(pickle_file)

    with open(data_path + class_v_file, 'rb') as pickle_file:
        class_v = pickle.load(pickle_file)

    with open(data_path + images_file, 'rb') as pickle_file:
        images = pickle.load(pickle_file)

    return X, class_v, images


def build_mn_dsets(cfg, mode='train'):
    assert mode in ['train', 'val', 'test']
    X,class_v,images = load_data(mode)
    #print(X[0])
    X[:, :, 4] = X[:, :, 4] - X[:, :, 2]
    X[:, :, 3] = X[:, :, 3] - X[:, :, 1]

    params = cfg.data.parameters
    
    '''with open(os.path.join(params.root_dir, params.vocab_json), 'r') as f:
        vocab = json.load(f)
    '''
    #object_names = ['bird', 'cat', 'cow', 'dog', 'horse', 'sheep', 'person']

    #part labels
    bird_labels = {'head':1, 'leye':2, 'reye':3, 'beak':4, 'torso':5, 'neck':6, 'lwing':7, 'rwing':8, 'lleg':9, 'lfoot':10, 'rleg':11, 'rfoot':12, 'tail':13}

    cat_labels = {'head':1, 'leye':2, 'reye':3, 'lear':4, 'rear':5, 'nose':6, 'torso':7, 'neck':8, 'lfleg':9, 'lfpa':10, 'rfleg':11, 'rfpa':12, 'lbleg':13, 'lbpa':14, 'rbleg':15, 'rbpa':16, 'tail':17}

    cow_labels = {'head':1, 'leye':2, 'reye':3, 'lear':4, 'rear':5, 'muzzle':6, 'lhorn':7, 'rhorn':8, 'torso':9, 'neck':10, 'lfuleg':11, 'lflleg':12, 'rfuleg':13, 'rflleg':14, 'lbuleg':15, 'lblleg':16, 'rbuleg':17, 'rblleg':18, 'tail':19}

    dog_labels = {'head':1, 'leye':2, 'reye':3, 'lear':4, 'rear':5, 'nose':6, 'torso':7, 'neck':8, 'lfleg':9, 'lfpa':10, 'rfleg':11, 'rfpa':12, 'lbleg':13, 'lbpa':14, 'rbleg':15, 'rbpa':16, 'tail':17, 'muzzle':18}

    horse_labels = {'head':1, 'leye':2, 'reye':3, 'lear':4, 'rear':5, 'muzzle':6, 'lfho':7, 'rfho':8, 'torso':9, 'neck':10, 'lfuleg':11, 'lflleg':12, 'rfuleg':13, 'rflleg':14, 'lbuleg':15, 'lblleg':16, 'rbuleg':17, 'rblleg':18, 'tail':19, 'lbho':20, 'rbho':21}
    
    sheep_labels = cow_labels

    person_labels = {'head':1, 'leye':2,  'reye':3, 'lear':4, 'rear':5, 'lebrow':6, 'rebrow':7,  'nose':8,  'mouth':9,  'hair':10, 'torso':11, 'neck': 12, 'llarm': 13, 'luarm': 14, 'lhand': 15, 'rlarm':16, 'ruarm':17, 'rhand': 18, 'llleg': 19, 'luleg':20, 'lfoot':21, 'rlleg':22, 'ruleg':23, 'rfoot':24}


    bird_comb = {
        1 : [1, 2, 3, 4],
        5 : [5],
        6 : [6],
        7 : [7],
        8 : [8],
        9 : [9],
        10 : [10],
        11 : [11],
        12 : [12],
        13 : [13]
    }

    cat_comb = {
        1 : [1, 2, 3, 4, 5, 6],
        7 : [7],
        8 : [8],
        9 : [9],
        10 : [10],
        11 : [11],
        12 : [12],
        13 : [13],
        14 : [14],
        15 : [15],
        16 : [16],
        17 : [17]
    }
    cow_comb = {
        1 : [1, 2, 3, 4, 5, 6],
        7 : [7],
        8 : [8],
        9 : [9],
        10 : [10],
        11 : [11],
        12 : [12],
        13 : [13],
        14 : [14],
        15 : [15],
        16 : [16],
        17 : [17],
        18 : [18],
        19 : [19]
    }
    dog_comb = {
        1 : [1, 2, 3, 4, 5, 6],
        7 : [7],
        8 : [8],
        9 : [9],
        10 : [10],
        11 : [11],
        12 : [12],
        13 : [13],
        14 : [14],
        15 : [15],
        16 : [16],
        17 : [17],
        18 : [18]
    }
    horse_comb = {
        1 : [1, 2, 3, 4, 5, 6],
        7 : [7],
        8 : [8],
        9 : [9],
        10 : [10],
        11 : [11],
        12 : [12],
        13 : [13],
        14 : [14],
        15 : [15],
        16 : [16],
        17 : [17],
        18 : [18],
        19 : [19],
        20 : [20],
        21 : [21]
    }
    sheep_comb = {
        1 : [1, 2, 3, 4, 5, 6],
        7 : [7],
        8 : [8],
        9 : [9],
        10 : [10],
        11 : [11],
        12 : [12],
        13 : [13],
        14 : [14],
        15 : [15],
        16 : [16],
        17 : [17],
        18 : [18],
        19 : [19]
    }
    person_comb = {
        1 : [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        11 : [11],
        12 : [12],
        13 : [13],
        14 : [14],
        15 : [15],
        16 : [16],
        17 : [17],
        18 : [18],
        19 : [19],
        20 : [20],
        21 : [21],
        22 : [22],
        23 : [23],
        24 : [24],
    }


    part_dict = {
        'cow' : cow_labels,
        'sheep' : sheep_labels,
        'bird' : bird_labels,
        'person' : person_labels,
        'cat' : cat_labels,
        'dog' : dog_labels,
        'horse' : horse_labels

    }

    class_comb_dict = {
        'cow' : cow_comb,
        'sheep' : sheep_comb,
        'bird' : bird_comb,
        'person' : person_comb,
        'cat' : cat_comb,
        'dog' : dog_comb,
        'horse' : horse_comb 
    }
    class_dict = {'cow':0,'sheep':1,'bird':2,'person':3,'cat':4,'dog':5,'horse':6,'aeroplane':7,'motorbike':8,'bicycle':9,'car':10} # should be in this order
    labels_v = []
    index_class_dict = {index: name for index, name in enumerate(class_comb_dict)}
    part_dict_list  = [ part_dict[name] for index, name in enumerate(part_dict)]
    index_part_list = []
    for i,part_dict in enumerate(part_dict_list):
        index_part_list.append({index+1: name for index, name in enumerate(part_dict)})
            
    #print(vocab)
    #vocab['object_name_to_idx']['__image__'] = 0
    #vocab['object_name_to_idx']['__null__'] = 179
    #vocab['object_idx_to_name'].append('__null__')

    dataset = MeronymnetDataset(
        images = images,
        X = X,
        class_v = class_v,
        index_class_dict=index_class_dict,
        index_part_list=index_part_list,
        class_comb_dict=class_comb_dict,
        image_size=(params.image_size, params.image_size),
        mask_size=params.mask_size_for_layout_object,
        max_num_samples=params[mode].max_num_samples,
        max_objects_per_image=params.max_objects_per_image,
        left_right_flip=params[mode].left_right_flip,
        use_MinIoURandomCrop=params[mode].use_MinIoURandomCrop,
        return_origin_image=params.return_origin_image,
        specific_image_ids=params[mode].specific_image_ids
    )

    num_imgs = len(dataset)
    print('%s dataset has %d images' % (mode, num_imgs))

    return dataset