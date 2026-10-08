import io
import os
import random

import pyarrow as pa
import torch
from PIL import Image

import SimpleITK as sitk
import re
import numpy as np
from ..transforms import keys_to_transforms

from collections import OrderedDict


import json  

import pandas as pd

import csv
from scipy.ndimage import rotate, zoom

from transformers import RobertaConfig, RobertaModel, BertTokenizer, AutoConfig, AutoTokenizer
import pickle

import torch.nn.functional as F

from nltk.tokenize import sent_tokenize

from .data_loader import load_case, load_masks

from typing import Dict, Optional

_ORGAN_HEADERS = [
    "Spleen:",
    "Liver:",
    "Pancreas:",
    "Kidney:",
    "Colon:",
]

# Lesion sections
_LESION_HEADERS = [
    "Spleen lesions:",
    "Liver lesions:",
    "Pancreas lesions:",  # also "Pancreatic lesions:"
    "Pancreatic lesions:",
    "Kidney lesions:",
    "Colon lesions:",
]

_ALL_HEADERS = _ORGAN_HEADERS + _LESION_HEADERS + ["IMPRESSION:"]
_HEADER_PATTERN = re.compile(
    r"^(" + "|".join(re.escape(h) for h in _ALL_HEADERS) + r")",
    re.MULTILINE,
)

class CTRGDataset_RATE_hr_sim_fea(torch.utils.data.Dataset):
    def __init__(
            self,
            config,
            split: str = "train",
            image_only: bool = False,
    ):
        super().__init__()
        # Hyper-Parameters
        self.image_only = image_only
        self.usage = split
        self.max_length = 380
        self.target_size = [240, 480, 480]
        self.config = config


        if self.usage == "train":
            self.text_path = config["text_path_train"]
            self.image_path = config["image_path_train"] 
            self.label_path = config["label_path_train"] 
            self.feature_path = config["imgfea_path_train"] 
            self.mask_path = config["mask_path_train"]
            self.mask_fast_path = config["mask_fast_path_train"]

        else:
            self.text_path = config["text_path_test"]
            self.image_path = config["image_path_test"] 
            self.label_path = config["label_path_test"] 
            self.feature_path = config["imgfea_path_test"]  
            self.mask_path = config["mask_path_test"]
            self.mask_fast_path = config["mask_fast_path_test"]



        self.text = {}
        self.label = {}
        self.get_text_and_label()


    def get_text_and_label(self):
        all_text = {}
        all_label = {}

        # get all text 
        files = open(self.text_path, "r")
        reader = csv.reader(files)
        

        for cline in reader:
            if cline[3] == "Findings_EN":
                continue
            
            if cline[0].split("_")[-1][:1] != "1" and self.usage == "val":
                continue
            
            all_text[cline[0]] = cline[3]

        files.close()

        # get all label
        files = open(self.label_path, "r")
        reader = csv.reader(files)

        for cline in reader:
            if cline[0] == "VolumeName":
                continue
            
            if cline[0].split("_")[-1][:1] != "1" and self.usage == "val":
                continue
            
            all_label[cline[0]] = torch.tensor(np.array([int(l) for l in cline[1:]]))

        files.close()

        if self.usage == "val":
            limit = 100
        else:
            limit = 10000000

        for l in list(all_text.keys()):
            key = l
            # if len(all_text[l]) < 500:
            #     continue
            # image_path = os.path.join(self.feature_path, "_".join(key.split("_")[:2]), "_".join(key.split("_")[:2]) + key.split("_")[2],  key.split(".")[0] + ".npz" + "")
            image_path = os.path.join(self.feature_path, key.split(".")[0] + ".npz" + "selected_patch.npy")

            #print(image_path)
            if os.path.exists(image_path) and limit > 0:
                limit -= 1
                self.text[key] = all_text[key]
                self.label[key] = all_label[key]

        print(len(self.text.keys()))

    @property
    def corpus(self):
        return [text for texts in self.all_texts for text in texts]

    def __len__(self):
        return len(self.text.keys())

    def get_raw_image(self, index, image_key="image"):
        
        key = list(self.text.keys())[index]

        image_path = os.path.join(self.image_path, "_".join(key.split("_")[:2]), "_".join(key.split("_")[:2]) + key.split("_")[2],  key.split(".")[0] + ".npz")

        return np.load(image_path)['arr_0'], image_path
    
    def get_image_fea(self, index):
        
        key = list(self.text.keys())[index]

        image_path = os.path.join(self.feature_path,  key.split(".")[0] + ".npz" + "selected_patch.npy")
        all_feature = os.path.join(self.feature_path,  key.split(".")[0] + ".npz" + "image_feature.npy")

        return np.load(image_path), image_path, np.load(all_feature)

    def pad_or_crop(self, tensor, target_size):
        """
        将输入的三维张量填充或裁剪到目标大小。
        
        参数:
        tensor (numpy.ndarray): 输入的三维张量，形状为 (depth, height, width)。
        target_size (tuple): 目标大小，形状为 (target_depth, target_height, target_width)。
        mode (str): 模式，'train' 表示训练模式，'test' 表示测试模式。
        
        返回:
        numpy.ndarray: 填充或裁剪后的张量，形状为 target_size。
        """
        input_depth, input_height, input_width = tensor.shape
        target_depth, target_height, target_width = target_size

        # 初始化输出张量
        output_tensor = np.zeros(target_size, dtype=tensor.dtype)

        if self.usage == 'train':
            # 随机裁剪或填充
            start_depth = np.random.randint(0, max(target_depth - input_depth, 1))
            start_height = np.random.randint(0, max(target_height - input_height, 1))
            start_width = np.random.randint(0, max(target_width - input_width, 1))
        else:
            # 中心裁剪或填充
            start_depth = max((target_depth - input_depth) // 2, 0)
            start_height = max((target_height - input_height) // 2, 0)
            start_width = max((target_width - input_width) // 2, 0)

        end_depth = start_depth + min(input_depth, target_depth)
        end_height = start_height + min(input_height, target_height)
        end_width = start_width + min(input_width, target_width)

        # 计算输入张量的起始和结束索引
        input_start_depth = max((input_depth - target_depth) // 2, 0)
        input_start_height = max((input_height - target_height) // 2, 0)
        input_start_width = max((input_width - target_width) // 2, 0)

        input_end_depth = input_start_depth + min(input_depth, target_depth)
        input_end_height = input_start_height + min(input_height, target_height)
        input_end_width = input_start_width + min(input_width, target_width)

        # 将输入张量填充或裁剪到输出张量
        output_tensor[start_depth:end_depth, start_height:end_height, start_width:end_width] = \
            tensor[input_start_depth:input_end_depth, input_start_height:input_end_height, input_start_width:input_end_width]

        return output_tensor

    def augment_image(self, image_array):
        """
        对3D图像进行数据增强。
        
        参数:
        image_array (numpy.ndarray): 3D图像数据
        
        返回:
        augmented_image (numpy.ndarray): 增强后的图像
        """
        # 随机旋转
        angle = np.random.uniform(-10, 10)  # 随机选择旋转角度
        rotated_image = rotate(image_array, angle, axes=(1, 2), reshape=False)
        
        ## 随机缩放
        #zoom_factor = np.random.uniform(0.9, 1.1)  # 随机选择缩放因子
        #zoomed_image = zoom(rotated_image, (1, zoom_factor, zoom_factor))
        
        return rotated_image

    
    def pad_or_crop2(self, tensor, target_size):
        """
        将输入的三维张量填充或裁剪到目标大小。

        参数:
        tensor (numpy.ndarray): 输入的三维张量，形状为 (depth, height, width)。
        target_size (tuple): 目标大小，形状为 (target_depth, target_height, target_width)。

        返回:
        numpy.ndarray: 填充或裁剪后的张量，形状为 target_size。
        """
        input_depth, input_height, input_width = tensor.shape
        target_depth, target_height, target_width = target_size

        # 初始化输出张量
        output_tensor = np.zeros(target_size, dtype=tensor.dtype)

        # 计算填充或裁剪的起始和结束索引
        start_depth = max((target_depth - input_depth) // 2, 0)
        start_height = max((target_height - input_height) // 2, 0)
        start_width = max((target_width - input_width) // 2, 0)

        end_depth = start_depth + min(input_depth, target_depth)
        end_height = start_height + min(input_height, target_height)
        end_width = start_width + min(input_width, target_width)

        # 计算输入张量的起始和结束索引
        input_start_depth = max((input_depth - target_depth) // 2, 0)
        input_start_height = max((input_height - target_height) // 2, 0)
        input_start_width = max((input_width - target_width) // 2, 0)

        input_end_depth = input_start_depth + min(input_depth, target_depth)
        input_end_height = input_start_height + min(input_height, target_height)
        input_end_width = input_start_width + min(input_width, target_width)

        # 将输入张量填充或裁剪到输出张量
        output_tensor[start_depth:end_depth, start_height:end_height, start_width:end_width] = \
            tensor[input_start_depth:input_end_depth, input_start_height:input_end_height, input_start_width:input_end_width]


        return output_tensor
    

    def image_prepro(self, image):

        # maybe crop like this
        # image = self.pad_or_crop(self.augment_image(image), target_size = self.target_size)
        image = self.pad_or_crop2(image, target_size = self.target_size)
        image = image.transpose(0,2,1)
        image = torch.tensor(image).float() 

        return image

    def get_mask(self, index, img_shape = None):
        key = list(self.text.keys())[index]
        fast_path = os.path.join(self.mask_fast_path, key.split(".")[0][:-1] + "1" + ".npy")
        if not os.path.exists(fast_path):
            mask = torch.tensor(self.get_raw_mask(index).transpose(0, 3, 1, 2)).unsqueeze(0)
            # print(image.shape)
            mask_shape = mask.shape
            # image_shape = img_shape
            
            mask = F.interpolate(mask, scale_factor=[img_shape[0] / mask_shape[2], img_shape[1] / mask_shape[3], img_shape[2] / mask_shape[4]], mode='nearest').squeeze()

            # print(mask.shape, image.shape)

            mask = self.pad_or_crop3(mask, self.target_size)

            c, d, h, w = mask.shape
            mask = (mask.reshape(9, d // 15, 15, h // 30, 30, w // 30, 30).permute(0,1,3,5,2,4,6).sum((4,5,6)) > 900).float()
            other = (mask.sum(0) == 0).float().unsqueeze(0)
            mask = torch.cat([mask, other], 0)
            np.save(fast_path, mask.numpy().astype(np.uint8))
            return mask
        else:
            # print("!")
            return torch.tensor(np.load(fast_path))

    def get_image(self, index, image_key="image"):
        
        if False:
            image, path = self.get_raw_image(index, image_key=image_key)
            image_tensor = self.image_prepro(image)
        else:
            image_tensor, path, all_image = self.get_image_fea(index)
            #print(image_tensor.shape)
            image_tensor = torch.tensor(image_tensor)

        mask = self.get_mask(index)

        return {
            "image": image_tensor,
            "img_index": list(self.text.keys())[index],
            "cap_index": 0,
            "raw_index": index,
            "data_name": path.split("/")[-1],
            "obver_label": self.label[list(self.label.keys())[index]],
            "all_image": all_image,
            "mask": mask,
        }

    def seg_text(self, text):
    
        follow_segtence = []


        keywords =  [["trachea", " bronchie", " bronchi ", " bronc", "tracheostomy", "tracheost", "nasogastric"], 
                    ["heart", "mediastinum", "mediastinal", "cardiac", "ventricle", "brachiocephalic", "vena", "aorta", "aortic", "artery", "thymus", "mediastinal tissue", "prevascular", "Pericardial", "vascular", " CTO", "arteries", " LAD", "cardiothoracic", "paratracheal", "atria", "mitral valve"],
                    ["lung", "pulmonary", "bilateral hilar", "emphysema", "pneumonic", "pneumonia", "Hilar", "consolidation", "interlobular"],
                    ["esophagus", "inlet", "Cricopharyngeal", "Esophag", "hiatal hernia"], 
                    ["Pleura","thorax", "membrane", "diaphragm"], # "thoracic"
                    [" rib", "spine", "sternum", "bone", "spinal", "vertebrae", "Clavicle", "Scapula", "Humerus", "Femur", "Cartilage", "Sternum", "Tube Bone", "Vertebral", "fractures", "costochondral", "Vertebra", "sternoclavicular"], 
                    ["thyroid"],
                    ["breast", "mammary", "chest", "armpits", "armpit", "axilla", "retroareolar", "gynecomastia", "thoracic wall"],
                    ["Abdomen", "Abdominal", "Adrenal", "Colon", "Duodenum", "Pericholecytic", "Gallbladder", "Intestine", "bowel", "kidney", "perinephric", "liver", "intrahepatic", "hepatic", "Caudate", "Pancreas", "Portal Vein", "Splenic Vein", "Rectum", "Renal", "Spleen", "Stomach", "Celiac", "hepatosteatosis", "peritoneum", "retrocrural", "gall bladder"], 
                    ["foramina", "pleuroparenchymal", "appropriate", " bladder", "Perivesical", "prostate", "catheter", "scalene"]]
        # The caudate lobe and left lobe are hypertrophic, and the liver contours are irregular
        ##### pre seg
        # The patient has a port catheter.
        #Focal nodular opacity with vascular enlargement is observed in the right lung middle lobe adjacent to the major fissure, in the left lung lower lobe basal segment and lower lobe superior segment, 
        #and in the right lung lower lobe mediobasal segment, and it is suspicious for ultra-early Covid-19 pneumonia
        ###### pre seg
        
        segs = sent_tokenize(text)

        ## remove the short sentence
        # segs = [l.strip() for l in segs if len(l.split(" ")) > 3]


        idx_label = [[] for seg_idx in segs]

        for seg_idx, seg in enumerate(segs):
            for keyidx, keyword in enumerate(keywords):
                matched = False
                for key in keyword:
                    if key.lower() in seg.lower():
                        idx_label[seg_idx].append(keyidx)
                        matched = True
                        break
                #if matched:
                #    break
        
        

        ## final check
        you = False

        for iidx in range(len(idx_label)):
            # print(idx_label[iidx], idx_label[iidx] == [], idx_label[iidx] is [])
            if idx_label[iidx] == [] or idx_label[iidx] == [9]:

                if segs[iidx].lower().strip().startswith("the nodules") or segs[iidx].strip()[:3].lower() == "and" or segs[iidx].strip()[:11].lower() == "the largest" or segs[iidx].strip()[:11].lower() == "in addition" or segs[iidx].strip()[:14].lower() == "the appearance" or segs[iidx].strip()[:7].lower() == "however" or segs[iidx].strip()[:5].lower() == "again" or segs[iidx].strip()[:2].lower() == "it" or segs[iidx].strip()[:4].lower() == "with" or segs[iidx].strip()[:11].lower() == "surrounding" or segs[iidx].strip()[:11].lower() == "the nodules" or segs[iidx].strip()[:5].lower() == "which" or segs[iidx].strip()[:5].lower() == "about" or segs[iidx].strip()[:10].lower() == "especially" or segs[iidx].strip()[:5].lower() == "after" or segs[iidx].strip()[:5].lower() == "signs" or segs[iidx].strip()[:4].lower() == "left" or segs[iidx].strip()[:8].lower() == "the size" or segs[iidx].strip()[:4].lower() == "size" or segs[iidx].strip()[:8].lower() == "ct value" or segs[iidx].strip()[:11].lower() == "ct diameter":
                    idx_label[iidx] = idx_label[iidx - 1]
                if iidx > 0 and (segs[iidx].strip()[-8:].lower() == "detected" or segs[iidx].strip()[-8:].lower() == "observed"):
                    idx_label[iidx] = idx_label[iidx - 1]

                elif iidx > 0 and segs[iidx-1].strip()[-1] == ",":
                    idx_label[iidx] = idx_label[iidx - 1]
                elif segs[iidx] in follow_segtence:
                    idx_label[iidx] = idx_label[iidx - 1]
                elif (iidx!=0 and iidx!=len(idx_label)-1):
                    for kkk in range(iidx + 1, len(idx_label)):
                        # print(idx_label[iidx - 1], idx_label[kkk])
                        if idx_label[iidx - 1] == idx_label[kkk]:
                            idx_label[iidx] = idx_label[iidx - 1]
                            #print(aaa)
                            break
                        elif idx_label[kkk] != []:
                            #print(bbb)
                            break
                else:
                    if not you:
                        continue

            else:
                you = True

        ## inte segs
        final_seg = ["" for l in range(10)]
        for iidx in range(len(idx_label)):
            if idx_label[iidx] is []:
                final_seg[9] += (" " + segs[iidx])  # merge all other, include those which have no keyword.
                continue

            for count, tidx in enumerate(list(set(idx_label[iidx]))):

                if tidx < 9:
                    # if len(self.tokenizer(final_seg[tidx])["input_ids"]):
                    final_seg[tidx] += (" " + segs[iidx])
                elif len(list(set(idx_label[iidx]))) == 1:
                    final_seg[9] += (" " + segs[iidx])  # merge all other

        
        
        for iidx, seg in enumerate(final_seg):
            if len(seg) == 0:
                if iidx == 0:
                    final_seg[iidx] += ("No abnormality found in trachea.")
                elif iidx == 1:
                    final_seg[iidx] += ("No abnormality found in mediastinum and heart.")
                elif iidx == 2:
                    final_seg[iidx] += ("No abnormality found in lung.")
                elif iidx == 3:
                    final_seg[iidx] += ("No abnormality found in esophagus.")
                elif iidx == 4:
                    final_seg[iidx] += ("No abnormality found in pleural.")
                elif iidx == 5:
                    final_seg[iidx] += ("No abnormalities in rib.")
                elif iidx == 6:
                    final_seg[iidx] += ("No abnormalities in thyroid.")
                elif iidx == 7:
                    final_seg[iidx] += ("No abnormalities in chest.")
                elif iidx == 8:
                    final_seg[iidx] += ("No abnormalities in abdomen organs.")
                elif iidx > 8: # others
                    final_seg[iidx] += ("No abnormalities in other organs.")
            
                continue

        final_text = final_seg[0][:-1] + "."
        for l in final_seg[1:]:
            l = l[:-1] + "."
            final_text += " " + l
        # print(final_text)
        final_token_merge = self.tokenizer(final_text,
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_special_tokens_mask=True,
            )

        final_idx = [0]

        for tokenidx, token in enumerate(final_token_merge["input_ids"][:-1]):
            if token == 4 and final_token_merge["input_ids"][tokenidx + 1] == 1437:
                final_idx.append(tokenidx)
        final_idx.append(len(final_token_merge["input_ids"]))
        
        while len(final_idx) < 11:
            final_idx.append(self.max_length)
            # print(final_text)
        return final_token_merge, final_text, final_idx, final_seg

    def get_text(self, raw_index):
        # index, caption_index = self.index_mapper[raw_index]
        text_ori = self.text[list(self.text.keys())[raw_index]]

       
        if len(text_ori.split(" ") ) > 240:
            text = " ".join(text_ori.split(" ")[:220]) + "."
        else:
            text = text_ori
        encoding, text, final_idx, final_seg = self.seg_text(text)

        prompt_ids = self.tokenizer("The", return_tensors="pt").input_ids

        if True:
            rg_encoding = self.tokenizer(text_ori, padding="max_length",
                    truncation=True,
                    max_length=self.max_length,
                    return_special_tokens_mask=True,
                    return_tensors="pt")
        else:  
            rg_encoding = self.tokenizer(text, padding="max_length",
                    truncation=True,
                    max_length=self.max_length,
                    return_special_tokens_mask=True,
                    return_tensors="pt")

        if self.usage == "train":
            return {
                "text": (text_ori, encoding),
                "img_index": raw_index,
                "cap_index": 0,
                "raw_index": raw_index,
                "seg_index": final_idx,
                "prompt_ids": prompt_ids,
                "rg_encoding": rg_encoding.input_ids,
                "rg_attn": rg_encoding.attention_mask,
                'final_seg':final_seg,
            }
        else:
            return {
                "text": (text_ori, encoding),
                "img_index": raw_index,
                "cap_index": 0,
                "raw_index": raw_index,
                "seg_index": final_idx,
                "prompt_ids": prompt_ids,
                "rg_encoding": rg_encoding.input_ids,
                "rg_attn": rg_encoding.attention_mask,
                'final_seg':final_seg,
            }

    def get_suite(self, index):
        result = None
        while result is None:
            #try:
            ret = dict()
            ret.update(self.get_image(index))
            if not self.image_only:
                txt = self.get_text(index)
                ret.update({"replica": True if txt["cap_index"] > 0 else False})
                ret.update(txt)

            result = True

        name = list(self.text.keys())[ret["img_index"]]
        name_list = name.split(".")[0].split("_")
        
        return ret
    
    def __getitem__(self, index):

        
        ret = self.get_suite(index)

        new_ret = {}
        new_ret["text_ids"] = ret["rg_encoding"]
        new_ret["text_masks"] = ret["rg_attn"]
        new_ret["seg_index"] = torch.tensor(ret["seg_index"])
        new_ret["obver_label"] = ret["obver_label"].long()
        new_ret["prompt_ids"] = ret["prompt_ids"]
        new_ret["all_image"] = ret["all_image"]

        final_seg = ret["final_seg"]

        all_encoding = []
        for seg in final_seg:
            all_encoding.append(self.tokenizer(seg, padding="max_length",
                truncation=True,
                max_length=200,
                return_special_tokens_mask=True,
                return_tensors="pt"))
        all_encodings = torch.cat([l.input_ids for l in all_encoding])
        all_maps = torch.cat([l.attention_mask for l in all_encoding])
        #print(all_encodings.shape)
        new_ret["all_encodings"] = all_encodings
        new_ret["all_maps"] = all_maps
        new_ret["image"] = ret["image"]

        new_ret["data_name"] = ret["data_name"]
        new_ret["final_seg"] = ret["final_seg"]
        new_ret["text_ori"] = ret["text"][0]
        
        new_ret['mask'] = ret["mask"]
        
        return new_ret
    



class CTRGDataset_RATE_hr_sim(torch.utils.data.Dataset):
    def __init__(
            self,
            config,
            split: str = "train",
            image_only: bool = False,
    ):
        super().__init__()
        # Hyper-Parameters
        self.image_only = image_only
        self.usage = split
        self.max_length = 380
        self.target_size = [240, 480, 480]
        self.config = config

        # Image Transformations
        # not implement yet

        if self.usage == "train":

            self.text_path = config["text_path_train"]
            self.text_embedding_path = config["text_emb_path_train"]
            self.image_path = config["image_path_train"] 
            self.label_path = config["label_path_train"] 
            self.mask_path = config["mask_path_train"] # "/apdcephfs_cq10/share_1290796/lh/M3AE-master/CTRATE/data_volumes/mask_processed/train"
            self.mask_fast_path = config["mask_fast_path_train"] # "/apdcephfs_cq10/share_1290796/lh/M3AE-master/CTRATE/data_volumes/mask_processed/train_fast"

            self.lesion_mask_path = config["lesion_mask_path"]
            self.lesion_mask_json = config["lesion_mask_json"]

            self.feature_path = config["imgfea_path_train"]
            self.json_path = config["json_path_train"]
            self.abnormality_text_embedding_path = config["abnormality_text_embedding_path"]

        else:
        
            self.text_path = config["text_path_test"]
            self.text_embedding_path = config["text_emb_path_test"]
            self.image_path = config["image_path_test"] 
            self.label_path = config["label_path_test"] 
            self.mask_path = config["mask_path_test"] #  "/apdcephfs_cq10/share_1290796/lh/M3AE-master/CTRATE/data_volumes/mask_processed/valid"
            self.mask_fast_path = config["mask_fast_path_test"] # "/apdcephfs_cq10/share_1290796/lh/M3AE-master/CTRATE/data_volumes/mask_processed/valid_fast"

            self.lesion_mask_path = config["lesion_mask_path"]
            self.lesion_mask_json = config["lesion_mask_json"]

            self.json_path = config["json_path_test"]
            self.feature_path = config["imgfea_path_test"]

            self.abnormality_text_embedding_path = config["abnormality_text_embedding_path"]

        self.get_text_and_label()

    def get_text_and_label(self):

        with open(self.lesion_mask_json, "r", encoding="utf-8") as f:
            lesion_json = json.load(f)
        self.lesion_json = lesion_json[self.usage]

        print(len(self.lesion_json))
        self.name_have_lesion = []
        for data in self.lesion_json:
            self.name_have_lesion.append(data["name"])

        self.text = {}
        self.label = {}
        self.text_embedding = []
        all_text = {}
        all_label = {}

        # get all text 
        files = open(self.text_path, "r")
        reader = csv.reader(files)
        

        for cline in reader:
            if cline[3] == "Findings_EN":
                continue
            
            # if cline[0].split("_")[-1][:1] != "1" and self.usage != "train":
            #     continue
            
            all_text[cline[0]] = cline[3]

        files.close()

        # get all label
        files = open(self.label_path, "r")
        reader = csv.reader(files)

        for cline in reader:
            if cline[0] == "VolumeName":
                continue
            
            # if cline[0].split("_")[-1][:1] != "1" and self.usage != "train":
            #     continue
            
            all_label[cline[0]] = torch.tensor(np.array([int(l) for l in cline[1:]]))

        files.close()

        if self.usage == "val":
            limit = 300
        else:
            limit = 300000
        start_id = 0
        end_id = 250000
        for idx, l in enumerate(list(all_text.keys())):
            if idx < start_id or idx > end_id:
                continue
            key = l

            image_path = os.path.join(self.image_path, "_".join(key.split("_")[:2]), "_".join(key.split("_")[:2]) + key.split("_")[2],  key.split(".")[0] + ".npz")
            patient_name = image_path.split("/")[-1]
            # Optional: skip volumes whose image feature has already been
            # extracted into ``skip_existing_feature_path``. Empty = disabled.
            store_path = self.config.get("skip_existing_feature_path", "")
            if store_path and os.path.exists(os.path.join(store_path, patient_name + "qwen_spatial_feature" + ".npy")):
                continue

            tpath = os.path.join(self.abnormality_text_embedding_path, "_".join(patient_name.split(".")[0].split("_")[:3]) + "_1" + ".npz")
            feapath = os.path.join(self.text_embedding_path, key.split(".")[0][:-1] + "1" + ".npz_text_feature.npy")
            
            if os.path.exists(image_path) and limit > 0 and os.path.exists(feapath) and os.path.exists(tpath):

            # if os.path.exists(image_path) and limit > 0:
                limit -= 1
                self.text[key] = all_text[key]
                self.label[key] = all_label[key]
                self.text_embedding.append(np.load(feapath))
            else:
                print(image_path, os.path.exists(feapath), os.path.exists(image_path))

            if limit <= 0:
                break
        self.text_embedding = np.array(self.text_embedding)

        print(len(self.text.keys()))


    @property
    def corpus(self):
        return [text for texts in self.all_texts for text in texts]

    def __len__(self):
        return len(self.text.keys())

    def get_raw_image(self, index, image_key="image"):
        
        key = list(self.text.keys())[index]

        image_path = os.path.join(self.image_path, "_".join(key.split("_")[:2]), "_".join(key.split("_")[:2]) + key.split("_")[2],  key.split(".")[0] + ".npz")

        return np.load(image_path)['arr_0'], image_path

    def pad_or_crop(self, tensor, target_size):
        """
        将输入的三维张量填充或裁剪到目标大小。
        
        参数:
        tensor (numpy.ndarray): 输入的三维张量，形状为 (depth, height, width)。
        target_size (tuple): 目标大小，形状为 (target_depth, target_height, target_width)。
        mode (str): 模式，'train' 表示训练模式，'test' 表示测试模式。
        
        返回:
        numpy.ndarray: 填充或裁剪后的张量，形状为 target_size。
        """
        input_depth, input_height, input_width = tensor.shape
        target_depth, target_height, target_width = target_size

        # 初始化输出张量
        output_tensor = np.zeros(target_size, dtype=tensor.dtype)

        if self.usage == 'train':
            # 随机裁剪或填充
            start_depth = np.random.randint(0, max(target_depth - input_depth, 1))
            start_height: int = np.random.randint(0, max(target_height - input_height, 1))
            start_width = np.random.randint(0, max(target_width - input_width, 1))
        else:
            # 中心裁剪或填充
            start_depth = max((target_depth - input_depth) // 2, 0)
            start_height = max((target_height - input_height) // 2, 0)
            start_width = max((target_width - input_width) // 2, 0)

        end_depth = start_depth + min(input_depth, target_depth)
        end_height = start_height + min(input_height, target_height)
        end_width = start_width + min(input_width, target_width)

        # 计算输入张量的起始和结束索引
        input_start_depth = max((input_depth - target_depth) // 2, 0)
        input_start_height = max((input_height - target_height) // 2, 0)
        input_start_width = max((input_width - target_width) // 2, 0)

        input_end_depth = input_start_depth + min(input_depth, target_depth)
        input_end_height = input_start_height + min(input_height, target_height)
        input_end_width = input_start_width + min(input_width, target_width)

        # 将输入张量填充或裁剪到输出张量
        output_tensor[start_depth:end_depth, start_height:end_height, start_width:end_width] = \
            tensor[input_start_depth:input_end_depth, input_start_height:input_end_height, input_start_width:input_end_width]

        return output_tensor

    def augment_image(self, image_array):
        """
        对3D图像进行数据增强。
        
        参数:
        image_array (numpy.ndarray): 3D图像数据
        
        返回:
        augmented_image (numpy.ndarray): 增强后的图像
        """
        # 随机旋转
        angle = np.random.uniform(-10, 10)  # 随机选择旋转角度
        rotated_image = rotate(image_array, angle, axes=(1, 2), reshape=False)
        
        ## 随机缩放
        # zoom_factor = np.random.uniform(0.9, 1.1)  # 随机选择缩放因子
        # zoomed_image = zoom(rotated_image, (1, zoom_factor, zoom_factor))
        
        return rotated_image

    
    def pad_or_crop2(self, tensor, target_size):
        """
        将输入的三维张量填充或裁剪到目标大小。

        参数:
        tensor (numpy.ndarray): 输入的三维张量，形状为 (depth, height, width)。
        target_size (tuple): 目标大小，形状为 (target_depth, target_height, target_width)。

        返回:
        numpy.ndarray: 填充或裁剪后的张量，形状为 target_size。
        """
        input_depth, input_height, input_width = tensor.shape
        target_depth, target_height, target_width = target_size

        # 初始化输出张量
        output_tensor = np.zeros(target_size, dtype=tensor.dtype)

        # 计算填充或裁剪的起始和结束索引
        start_depth = max((target_depth - input_depth) // 2, 0)
        start_height = max((target_height - input_height) // 2, 0)
        start_width = max((target_width - input_width) // 2, 0)

        end_depth = start_depth + min(input_depth, target_depth)
        end_height = start_height + min(input_height, target_height)
        end_width = start_width + min(input_width, target_width)

        # 计算输入张量的起始和结束索引
        input_start_depth = max((input_depth - target_depth) // 2, 0)
        input_start_height = max((input_height - target_height) // 2, 0)
        input_start_width = max((input_width - target_width) // 2, 0)

        input_end_depth = input_start_depth + min(input_depth, target_depth)
        input_end_height = input_start_height + min(input_height, target_height)
        input_end_width = input_start_width + min(input_width, target_width)

        # 将输入张量填充或裁剪到输出张量
        output_tensor[start_depth:end_depth, start_height:end_height, start_width:end_width] = \
            tensor[input_start_depth:input_end_depth, input_start_height:input_end_height, input_start_width:input_end_width]


        return output_tensor  

    def image_prepro(self, image):

        # maybe crop like this
        # image = self.pad_or_crop(self.augment_image(image), target_size = self.target_size)
        image = self.pad_or_crop2(image, target_size = self.target_size)
        image = image.transpose(0,2,1)
        image = torch.tensor(image).float() 

        return image

        

    def get_mask(self, index, img_shape = None):
        key = list(self.text.keys())[index]
        fast_path = os.path.join(self.mask_fast_path, key.split(".")[0][:-1] + "1" + ".npy")
        if not os.path.exists(fast_path):
            mask = torch.tensor(self.get_raw_mask(index).transpose(0, 3, 1, 2)).unsqueeze(0)
            # print(image.shape)
            mask_shape = mask.shape
            # image_shape = img_shape
            
            mask = F.interpolate(mask, scale_factor=[img_shape[0] / mask_shape[2], img_shape[1] / mask_shape[3], img_shape[2] / mask_shape[4]], mode='nearest').squeeze()

            # print(mask.shape, image.shape)

            mask = self.pad_or_crop3(mask, self.target_size)

            c, d, h, w = mask.shape
            mask = (mask.reshape(9, d // 15, 15, h // 30, 30, w // 30, 30).permute(0,1,3,5,2,4,6).sum((4,5,6)) > 900).float()
            other = (mask.sum(0) == 0).float().unsqueeze(0)
            mask = torch.cat([mask, other], 0)
            np.save(fast_path, mask.numpy().astype(np.uint8))
            return mask
        else:
            # print("!")
            return torch.tensor(np.load(fast_path))

    def get_lesion_mask(self, index):
        key = list(self.text.keys())[index]
        if key in self.name_have_lesion:
            lesion_mask_path = os.path.join(self.lesion_mask_path, key.split(".")[0] + ".npz")
            lesion_mask = np.load(lesion_mask_path)['arr_0']

            final_lesion_mask = np.zeros(lesion_mask.shape[1:])
            for l in range(lesion_mask.shape[0]):
                final_lesion_mask[lesion_mask[l] > 0] = l+1

            text_query = self.lesion_json[self.name_have_lesion.index(key)]["findings"]

            return final_lesion_mask, text_query
        else:
            return None, None

    def random_flip_3d(self, volume, p_flip_d=0.2, p_flip_h=0.2, p_flip_w=0.2):
        """
        对 3D volume (D, H, W) 进行随机翻转（前后、上下、左右）。
        
        参数:
            volume (np.ndarray): 输入的 3D 体积，形状为 (D, H, W)
            p_flip_d (float): 沿 D 轴（前后/深度方向）翻转的概率，默认 0.5
            p_flip_h (float): 沿 H 轴（上下）翻转的概率，默认 0.5
            p_flip_w (float): 沿 W 轴（左右）翻转的概率，默认 0.5

        返回:
            np.ndarray: 增强后的 3D volume，形状不变
        """
        if np.random.rand() < p_flip_d:
            volume = np.flip(volume, axis=0)  # 前后翻转（D 轴）
        if np.random.rand() < p_flip_h:
            volume = np.flip(volume, axis=1)  # 上下翻转（H 轴）
        if np.random.rand() < p_flip_w:
            volume = np.flip(volume, axis=2)  # 左右翻转（W 轴）
        return volume    

    def get_image_fea(self, index):
        
        key = list(self.text.keys())[index]
        # self.feature_path = self.config["imgfea_path_train"]
        image_path = os.path.join(self.feature_path,  key.split(".")[0] + ".npz" + "selected_patch.npy")
        all_feature = os.path.join(self.feature_path,  key.split(".")[0] + ".npz" + "image_feature.npy")

        return np.load(all_feature), image_path

    def get_image(self, index, image_key="image"):

        if True:
            image, path = self.get_raw_image(index, image_key=image_key)
            
            # print(image.shape)
            # image = self.random_flip_3d(image)
            mask = self.get_mask(index, image.shape)

            image = image*1000
            hu_min, hu_max = -1000, 200
            image = np.clip(image, hu_min, hu_max)

            image = (((image+400 ) / 600)).astype(np.float32)
            image_tensor = self.image_prepro(image)
        else:
            image, path = self.get_image_fea(index)
            image_tensor = torch.tensor(image)
            mask = self.get_mask(index)

        # Lesion masks (ReXGroundingCT) are optional -- disabled by default,
        # which is the behaviour of the original snapshot.  Set
        # ``with_lesion=True`` to enable the lesion-alignment branch of the
        # pretrain module.
        if self.config.get("with_lesion", False):
            lesion_mask, text_query = self.get_lesion_mask(index)

            if lesion_mask is None:
                lesion_mask = np.zeros(self.target_size)

                text_query = {
                    "0": "kkkkkkkkkkk"
                }

            lesion_mask = self.image_prepro(lesion_mask)
        else:
            lesion_mask = 1
            text_query = {
                    "0": "kkkkkkkkkkk"
                }

        

        return {
            "image": image_tensor,
            "image_downsample": mask, 
            "mask": mask,
            "img_index": list(self.text.keys())[index],
            "cap_index": 0,
            "raw_index": index,
            "data_name": path.split("/")[-1],
            "obver_label": self.label[list(self.label.keys())[index]],

            "text_query": text_query,
            "lesion_mask": lesion_mask
        }
    
    def seg_text(self, text):
    
        follow_segtence = []


        keywords =  [["trachea", " bronchie", " bronchi ", " bronc", "tracheostomy", "tracheost", "nasogastric"], 
                    ["heart", "mediastinum", "mediastinal", "cardiac", "ventricle", "brachiocephalic", "vena", "aorta", "aortic", "artery", "thymus", "mediastinal tissue", "prevascular", "Pericardial", "vascular", " CTO", "arteries", " LAD", "cardiothoracic", "paratracheal", "atria", "mitral valve"],
                    ["lung", "pulmonary", "bilateral hilar", "emphysema", "pneumonic", "pneumonia", "Hilar", "consolidation", "interlobular"],
                    ["esophagus", "inlet", "Cricopharyngeal", "Esophag", "hiatal hernia"], 
                    ["Pleura","thorax", "membrane", "diaphragm"], # "thoracic"
                    [" rib", "spine", "sternum", "bone", "spinal", "vertebrae", "Clavicle", "Scapula", "Humerus", "Femur", "Cartilage", "Sternum", "Tube Bone", "Vertebral", "fractures", "costochondral", "Vertebra", "sternoclavicular"], 
                    ["thyroid"],
                    ["breast", "mammary", "chest", "armpits", "armpit", "axilla", "retroareolar", "gynecomastia", "thoracic wall"],
                    ["Abdomen", "Abdominal", "Adrenal", "Colon", "Duodenum", "Pericholecytic", "Gallbladder", "Intestine", "bowel", "kidney", "perinephric", "liver", "intrahepatic", "hepatic", "Caudate", "Pancreas", "Portal Vein", "Splenic Vein", "Rectum", "Renal", "Spleen", "Stomach", "Celiac", "hepatosteatosis", "peritoneum", "retrocrural", "gall bladder"], 
                    ["foramina", "pleuroparenchymal", "appropriate", " bladder", "Perivesical", "prostate", "catheter", "scalene"]]
        # The caudate lobe and left lobe are hypertrophic, and the liver contours are irregular
        ##### pre seg
        # The patient has a port catheter.
        #Focal nodular opacity with vascular enlargement is observed in the right lung middle lobe adjacent to the major fissure, in the left lung lower lobe basal segment and lower lobe superior segment, 
        #and in the right lung lower lobe mediobasal segment, and it is suspicious for ultra-early Covid-19 pneumonia
        ###### pre seg
        
        segs = sent_tokenize(text)

        ## remove the short sentence
        # segs = [l.strip() for l in segs if len(l.split(" ")) > 3]


        idx_label = [[] for seg_idx in segs]

        for seg_idx, seg in enumerate(segs):
            for keyidx, keyword in enumerate(keywords):
                matched = False
                for key in keyword:
                    if key.lower() in seg.lower():
                        idx_label[seg_idx].append(keyidx)
                        matched = True
                        break
                #if matched:
                #    break
        
        

        ## final check
        you = False

        for iidx in range(len(idx_label)):
            # print(idx_label[iidx], idx_label[iidx] == [], idx_label[iidx] is [])
            if idx_label[iidx] == [] or idx_label[iidx] == [9]:

                if segs[iidx].lower().strip().startswith("the nodules") or segs[iidx].strip()[:3].lower() == "and" or segs[iidx].strip()[:11].lower() == "the largest" or segs[iidx].strip()[:11].lower() == "in addition" or segs[iidx].strip()[:14].lower() == "the appearance" or segs[iidx].strip()[:7].lower() == "however" or segs[iidx].strip()[:5].lower() == "again" or segs[iidx].strip()[:2].lower() == "it" or segs[iidx].strip()[:4].lower() == "with" or segs[iidx].strip()[:11].lower() == "surrounding" or segs[iidx].strip()[:11].lower() == "the nodules" or segs[iidx].strip()[:5].lower() == "which" or segs[iidx].strip()[:5].lower() == "about" or segs[iidx].strip()[:10].lower() == "especially" or segs[iidx].strip()[:5].lower() == "after" or segs[iidx].strip()[:5].lower() == "signs" or segs[iidx].strip()[:4].lower() == "left" or segs[iidx].strip()[:8].lower() == "the size" or segs[iidx].strip()[:4].lower() == "size" or segs[iidx].strip()[:8].lower() == "ct value" or segs[iidx].strip()[:11].lower() == "ct diameter":
                    idx_label[iidx] = idx_label[iidx - 1]
                if iidx > 0 and (segs[iidx].strip()[-8:].lower() == "detected" or segs[iidx].strip()[-8:].lower() == "observed"):
                    idx_label[iidx] = idx_label[iidx - 1]

                elif iidx > 0 and segs[iidx-1].strip()[-1] == ",":
                    idx_label[iidx] = idx_label[iidx - 1]
                elif segs[iidx] in follow_segtence:
                    idx_label[iidx] = idx_label[iidx - 1]
                elif (iidx!=0 and iidx!=len(idx_label)-1):
                    for kkk in range(iidx + 1, len(idx_label)):
                        # print(idx_label[iidx - 1], idx_label[kkk])
                        if idx_label[iidx - 1] == idx_label[kkk]:
                            idx_label[iidx] = idx_label[iidx - 1]
                            #print(aaa)
                            break
                        elif idx_label[kkk] != []:
                            #print(bbb)
                            break
                else:
                    if not you:
                        continue

            else:
                you = True

        ## inte segs
        final_seg = ["" for l in range(10)]
        for iidx in range(len(idx_label)):
            if idx_label[iidx] is []:
                final_seg[9] += (" " + segs[iidx])  # merge all other, include those which have no keyword.
                continue

            for count, tidx in enumerate(list(set(idx_label[iidx]))):

                if tidx < 9:
                    # if len(self.tokenizer(final_seg[tidx])["input_ids"]):
                    final_seg[tidx] += (" " + segs[iidx])
                elif len(list(set(idx_label[iidx]))) == 1:
                    final_seg[9] += (" " + segs[iidx])  # merge all other

        
        
        for iidx, seg in enumerate(final_seg):
            if len(seg) == 0:
                if iidx == 0:
                    final_seg[iidx] += ("No abnormality found in trachea.")
                elif iidx == 1:
                    final_seg[iidx] += ("No abnormality found in mediastinum and heart.")
                elif iidx == 2:
                    final_seg[iidx] += ("No abnormality found in lung.")
                elif iidx == 3:
                    final_seg[iidx] += ("No abnormality found in esophagus.")
                elif iidx == 4:
                    final_seg[iidx] += ("No abnormality found in pleural.")
                elif iidx == 5:
                    final_seg[iidx] += ("No abnormalities in rib.")
                elif iidx == 6:
                    final_seg[iidx] += ("No abnormalities in thyroid.")
                elif iidx == 7:
                    final_seg[iidx] += ("No abnormalities in chest.")
                elif iidx == 8:
                    final_seg[iidx] += ("No abnormalities in abdomen organs.")
                elif iidx > 8: # others
                    final_seg[iidx] += ("No abnormalities in other organs.")
            
                continue

        final_text = final_seg[0][:-1] + "."
        for l in final_seg[1:]:
            l = l[:-1] + "."
            final_text += " " + l
        # print(final_text)
        final_token_merge = self.tokenizer(final_text,
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_special_tokens_mask=True,
            )

        final_idx = [0]

        for tokenidx, token in enumerate(final_token_merge["input_ids"][:-1]):
            if token == 4 and final_token_merge["input_ids"][tokenidx + 1] == 1437:
                final_idx.append(tokenidx)
        final_idx.append(len(final_token_merge["input_ids"]))
        
        while len(final_idx) < 11:
            final_idx.append(self.max_length)
            # print(final_text)
        return final_token_merge, final_text, final_idx, final_seg

    

    def get_seg_text(self, patient_name):

        cregion_report = self.name_2_region[patient_name]

        if 'mediastinum' in cregion_report.keys():
            if "heart" in cregion_report.keys():
                cregion_report["heart"] += cregion_report["mediastinum"]
            else:
                cregion_report["heart"] = cregion_report["mediastinum"]
    
        regions = ['trachea and bronchie', 'heart', 'lung', 'esophagus', 'pleura', 'bone', 'thyroid', 'abdomen', 'breast', 'others']

        for r in regions:
            if r not in cregion_report.keys():
                cregion_report[r] = "No abnormality found in " + r + "." 

        final_seg = [cregion_report[r] for r in regions]

        final_text = final_seg[0][:-1] + "."
        for l in final_seg[1:]:
            l = l[:-1] + "."
            final_text += " " + l
        # print(final_text)
        final_token_merge = self.tokenizer(final_text,
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_special_tokens_mask=True,

            )
        # print(final_seg)
        # print(final_text)
        # print(tokenizer(final_text))

        final_idx = [0]

        for tokenidx, token in enumerate(final_token_merge["input_ids"][:-1]):
            if token == 4 and final_token_merge["input_ids"][tokenidx + 1] == 1437:
                final_idx.append(tokenidx)
        final_idx.append(len(final_token_merge["input_ids"]))

        #print(final_idx, final_token_merge)
        #for iidx in range(len(final_idx)-4):
        #    print(final_text[final_token_merge["offset_mapping"][final_idx[iidx]][0]:final_token_merge["offset_mapping"][final_idx[iidx+1]][0]])
        
        while len(final_idx) < 11:
            final_idx.append(self.max_length)
            # print(final_text)
        return final_token_merge, final_text, final_idx, final_seg

    def get_label_similarity(self, vector1, vector2):
        intersection = torch.sum((vector1 == 1) & (vector2 == 1)).item()
        union = torch.sum((vector1 == 1) | (vector2 == 1)).item()
        
        if union == 0:
            return 1
        jaccard_similarity = intersection / union
        return jaccard_similarity

    def get_abnormality_embedding(self, raw_index):
        
        patient_name = list(self.text.keys())[raw_index]

        all_text_embedding = self.text_embedding[raw_index]

        # print(all_text_embedding.shape)

        abnorm_text_embedding = all_text_embedding.copy()

        abnorm_label = [0] * 10

        tpath = os.path.join(self.abnormality_text_embedding_path, "_".join(patient_name.split(".")[0].split("_")[:3]) + "_1" + ".npz")
        data = np.load(tpath)
        structural_keys = ['trachea and bronchie', 'mediastinum_heart', 'lung', 'esophagus', 'pleura', 'bone', 'thyroid', 'breast', 'abdomen', 'others']

        for index, struc_name in enumerate(structural_keys):
            # print(data[struc_name].dtype)
            if data[struc_name].dtype != "int64":
                abnorm_text_embedding[index] = data[struc_name]
                abnorm_label[index] = 1
        # print(abnorm_label)
        return all_text_embedding, abnorm_text_embedding, abnorm_label

    def extract_abnorm(self, jpath):
        with open(jpath, 'r', encoding='utf-8') as file:
            data = json.load(file)

        organ_keys = ["trachea and bronchie", "mediastinum", "heart", "lung", "esophagus", "pleura", "bone", "thyroid", "breast", "abdomen", "others"]
        return_sentence = []
        return_sentence_ab = []
        for organ in organ_keys:
            csentence = ""
            if len(data["hierarchical_abnormalities"][organ]["attributes"]["abnormality"]) != 0:
                if len(csentence) == 0:
                    csentence = organ + ", "
                csentence += ", ".join(data["hierarchical_abnormalities"][organ]["attributes"]["abnormality"]) + "."
            ab_sentence = csentence
            if len(data["hierarchical_abnormalities"][organ]["attributes"]["non_abnormality"]) != 0 or False:
                if len(csentence) == 0:
                    csentence = organ + ", "
                csentence += ", ".join(data["hierarchical_abnormalities"][organ]["attributes"]["non_abnormality"]) + "."
            if csentence != "":
                return_sentence.append(csentence)
            if ab_sentence != "":
                return_sentence_ab.append(ab_sentence)

            # print("all", csentence, "ab", ab_sentence)
        
        return_sentence.extend(data["global_abnormalities"])
        return_sentence.extend(data["impression_disorders"])

        return_sentence_ab.extend(data["global_abnormalities"])
        return_sentence_ab.extend(data["impression_disorders"])

        # final_abnorm_sentence = ". ".join(return_sentence) + "."
        final_abnorm_sentence = " ".join(return_sentence)
        final_abnorm_sentence_ab = " ".join(return_sentence_ab)
        # print(final_abnorm_sentence)
        return final_abnorm_sentence, final_abnorm_sentence_ab

    def get_text(self, raw_index):
        # index, caption_index = self.index_mapper[raw_index]
        key = list(self.text.keys())[raw_index]
        text_ori = self.text[list(self.text.keys())[raw_index]]
        current_ab_text, current_ab_text_ab = self.extract_abnorm(os.path.join(self.json_path, "_".join(key.split(".")[0].split("_")[:-1]) + "_1.json"))

       
        if len(text_ori.split(" ") ) > 240:
            text = " ".join(text_ori.split(" ")[:220]) + "."
        else:
            text = text_ori
        encoding, text, final_idx, final_seg = self.seg_text(text)
        # encoding, text, final_idx, final_seg = self.get_seg_text(key)
 

        prompt_ids = self.tokenizer("The", return_tensors="pt").input_ids

        if True:
            rg_encoding = self.tokenizer(text_ori, padding="max_length",
                    truncation=True,
                    max_length=self.max_length,
                    return_special_tokens_mask=True,
                    return_tensors="pt")
        else:  
            rg_encoding = self.tokenizer(text, padding="max_length",
                    truncation=True,
                    max_length=self.max_length,
                    return_special_tokens_mask=True,
                    return_tensors="pt")

        all_text_embedding, abnorm_text_embedding, abnorm_label = self.get_abnormality_embedding(raw_index)
        
        if self.usage == "train":
            return {
                "text": (text_ori, encoding),
                "img_index": raw_index,
                "cap_index": 0,
                "raw_index": raw_index,
                "seg_index": final_idx,
                "prompt_ids": prompt_ids,
                "rg_encoding": rg_encoding.input_ids,
                "rg_attn": rg_encoding.attention_mask,
                'final_seg':final_seg,
                "text_embedding": all_text_embedding,
                "text_embedding_ab": abnorm_text_embedding,
                "abnorm_label": abnorm_label,
                'impression': current_ab_text_ab, # impression,

            }
        else:
            return {
                "text": (text_ori, encoding),
                "img_index": raw_index,
                "cap_index": 0,
                "raw_index": raw_index,
                "seg_index": final_idx,
                "prompt_ids": prompt_ids,
                "rg_encoding": rg_encoding.input_ids,
                "rg_attn": rg_encoding.attention_mask,
                'final_seg':final_seg,
                "text_embedding": all_text_embedding,
                "text_embedding_ab": abnorm_text_embedding,
                "abnorm_label": abnorm_label,
                'impression': current_ab_text_ab, # impression,
            }

    def get_suite(self, index):
        # result = None
        # while result is None:
            #try:
        ret = dict()
        ret.update(self.get_image(index))
        if not self.image_only:
            txt = self.get_text(index)
            ret.update({"replica": True if txt["cap_index"] > 0 else False})
            ret.update(txt)
            #except Exception as e:
            #    print(f"Error while read file idx {index} in {self.names[0]} -> {e}")
            #    index = random.randint(0, len(self.index_mapper) - 1)
        
        return ret
    
    def __getitem__(self, index):
        
        ret = self.get_suite(index)

        new_ret = {}
        new_ret["text_ids"] = ret["rg_encoding"]
        new_ret["text_masks"] = ret["rg_attn"]
        new_ret["seg_index"] = torch.tensor(ret["seg_index"])
        new_ret["obver_label"] = ret["obver_label"].long()
        new_ret["prompt_ids"] = ret["prompt_ids"]

        final_seg = ret["final_seg"]

        all_encoding = []
        for seg in final_seg:
            all_encoding.append(self.tokenizer(seg, padding="max_length",
                truncation=True,
                max_length=200,
                return_special_tokens_mask=True,
                return_tensors="pt"))
                
        all_encodings = torch.cat([l.input_ids for l in all_encoding])
        all_maps = torch.cat([l.attention_mask for l in all_encoding])
        #print(all_encodings.shape)
        new_ret["all_encodings"] = all_encodings
        new_ret["all_maps"] = all_maps
        new_ret["image"] = ret["image"]
        new_ret["image_downsample"] = ret["image_downsample"]

        new_ret["data_name"] = ret["data_name"]
        new_ret["final_seg"] = ret["final_seg"]
        new_ret["text"] = ret["text"][0]

        new_ret["text_embedding"] = torch.tensor(ret["text_embedding"])
        new_ret["text_embedding_ab"] = torch.tensor(ret["text_embedding_ab"])
        new_ret["abnorm_label"] = torch.tensor(ret["abnorm_label"])
        
        new_ret['mask'] = ret["mask"]

        new_ret["text_query"] = ret["text_query"]
        new_ret["lesion_mask"] = ret["lesion_mask"]

        new_ret['impression'] = ret["impression"]

        return new_ret
    
    def collate(self, batch):
        batch_size = len(batch)
        keys = set([key for b in batch for key in b.keys()])
        dict_batch = {k: [dic[k] if k in dic else None for dic in batch] for k in keys}
        # print(dict_batch.keys())
        img_keys = [k for k in list(dict_batch.keys()) if "image" in k]
        for img_key in img_keys:
            # print(dict_batch[img_key][0].shape)
            cimage = torch.cat([l.unsqueeze(0) for l in dict_batch[img_key]], 0).unsqueeze(1)

            dict_batch[img_key] = cimage
        
        # exit(0)

        txt_keys = [k for k in list(dict_batch.keys()) if "text" in k]
        
        if len(txt_keys) != 0:

            encodings = [[d[1] for d in dict_batch[txt_key]] for txt_key in txt_keys]
            flatten_encodings = [e for encoding in encodings for e in encoding]
            flatten_mlms = flatten_encodings
            for i, txt_key in enumerate(txt_keys):
                texts, encodings = ([d[0] for d in dict_batch[txt_key]], [d[1] for d in dict_batch[txt_key]])
                #mlm_ids, mlm_labels = (
                #    flatten_mlms["input_ids"][batch_size * (i): batch_size * (i + 1)],
                #    flatten_mlms["labels"][batch_size * (i): batch_size * (i + 1)],
                #)

                input_ids = []
                attention_mask = []
                for _i, encoding in enumerate(encodings):
                    _input_ids, _attention_mask = (
                        torch.tensor(encoding["input_ids"]),
                        torch.tensor(encoding["attention_mask"]),
                    )
                    input_ids.append( _input_ids.unsqueeze(0))
                    attention_mask.append( _attention_mask.unsqueeze(0))

                input_ids = torch.cat(input_ids, 0)
                attention_mask = torch.cat(attention_mask, 0)

                dict_batch[txt_key] = texts
                dict_batch[f"{txt_key}_ids"] = input_ids
                dict_batch[f"{txt_key}_labels"] = torch.full_like(input_ids, -100)
                #dict_batch[f"{txt_key}_ids_mlm"] = mlm_ids
                #dict_batch[f"{txt_key}_labels_mlm"] = mlm_labels
                dict_batch[f"{txt_key}_masks"] = attention_mask
                dict_batch[f"{txt_key}_ori"] = encoding
        
        return dict_batch

class CTRGDataset_RATE_hr_sim_fea_qwen(CTRGDataset_RATE_hr_sim):
    def __init__(
            self,
            config,
            split: str = "train",
            image_only: bool = False,
    ):
        self.config = config

        if split == "train":
            self.normalized_label_path = self.config["normalized_label_path_train"]
        else:
            self.normalized_label_path = self.config["normalized_label_path_val"]
    
        super().__init__(config, split, image_only)

    def get_text_and_label(self):
        
        '''
        with open(self.lesion_mask_json, "r", encoding="utf-8") as f:
            lesion_json = json.load(f)
        self.lesion_json = lesion_json[self.usage]

        print(len(self.lesion_json))
        self.name_have_lesion = []
        for data in self.lesion_json:
            self.name_have_lesion.append(data["name"])
        '''

        self.text = {}
        self.label = {}
        self.text_embedding = []
        all_text = {}
        all_label = {}

        # get all text 
        files = open(self.text_path, "r")
        reader = csv.reader(files)
        
        self.all_dict = self.build_volume_abnormality_dict(self.normalized_label_path)

        # ── imp_chain_mode: bare-finding-name-only dict for the <imp_name>
        # segment of the 3-stage chained <imp_name>/<imp_loc>/<answer>
        # generation target (see forward() in set_prediction_stage2_trainer_v5.py).
        # Built from the SAME csv_path (so the min_finding_freq filter and
        # row/volume merge order are identical) but forcing
        # remove_location=True, remove_attrs=True regardless of the base
        # config flags -- this dict is ONLY consumed by the <imp_name>
        # rendering, never by <imp_loc> (which keeps using self.all_dict,
        # governed by the normal remove_abnormality_location/attributes
        # config knobs as before).
        self.imp_chain_mode = bool(self.config.get("imp_chain_mode", False))
        if self.imp_chain_mode:
            self.all_dict_name_only = self.build_volume_abnormality_dict(
                self.normalized_label_path, remove_location=True, remove_attrs=True,
            )
        # Per-sample planning order dropout probability (train only). 0 = off.
        # Mutually exclusive with order_imp_by_anatomy (fixed anatomy order);
        # if both are set, dropout wins and we warn, since a fixed order and a
        # randomised order cannot both apply.
        self._imp_order_dropout = float(self.config.get("imp_order_dropout", 0.0) or 0.0)
        self._imp_finding_dropout = float(self.config.get("imp_finding_dropout", 0.0) or 0.0)
        _any_dropout = (self._imp_order_dropout > 0.0) or (self._imp_finding_dropout > 0.0)
        if _any_dropout and bool(self.config.get("order_imp_by_anatomy", False)):
            print("[CTRGDataset] WARN: planning dropout (order/finding) and "
                  "order_imp_by_anatomy=True are both set; these are "
                  "contradictory. Base order falls back to raw CSV and dropout "
                  "is applied per sample; the anatomy sort is ignored.")
        if _any_dropout:
            print(f"[CTRGDataset] planning perturbation (usage={self.usage}): "
                  f"imp_order_dropout={self._imp_order_dropout}, "
                  f"imp_finding_dropout={self._imp_finding_dropout}.")

        for cline in reader:
            if cline[3] == "Findings_EN":
                continue
            
            if (cline[0].split("_")[-1][:1] != "1" or len(cline[0].split("_")[-1].split(".")[0]) != 1) and self.config.get("only_use_1", False):
                continue
            
            all_text[cline[0]] = cline[3]

        files.close()

        # get all label
        files = open(self.label_path, "r")
        reader = csv.reader(files)

        for cline in reader:
            if cline[0] == "VolumeName":
                continue
            
            if (cline[0].split("_")[-1][:1] != "1" or len(cline[0].split("_")[-1].split(".")[0]) != 1) and self.config.get("only_use_1", False):
                continue
            
            all_label[cline[0]] = torch.tensor(np.array([int(l) for l in cline[1:]]))

        files.close()

        if self.usage == "val":
            limit = self.config.get("val_limit", 1500)
        else:
            limit = self.config.get("train_limit", 15000000)
        start_id = 0
        end_id = 250000

        print(len(list(all_text.keys())))
        for idx, l in enumerate(list(all_text.keys())):
            if idx < start_id or idx > end_id:
                continue
            key = l
            image_path = os.path.join(self.feature_path, key.split(".")[0] + ".npz" + "qwen_spatial_feature.npy")
            # image_path = os.path.join(self.image_path, "_".join(key.split("_")[:2]), "_".join(key.split("_")[:2]) + key.split("_")[2],  key.split(".")[0] + ".npz")
            #patient_name = image_path.split("/")[-1]
            #store_path = "/jizhicfs/datalh/M3AE/img_feature/qwen3_4B_vl_multigpu"
            #if os.path.exists(os.path.join(store_path, patient_name + "qwen_spatial_feature" + ".npy")):
            #    continue
            if key.split(".")[0] + ".nii.gz" not in self.all_dict:
                print(key.split(".")[0] + ".nii.gz", "not in abnormality dict")
                # print(list(self.all_dict.keys())[:10])
                continue
            tpath = os.path.join(self.abnormality_text_embedding_path, "_".join(key.split(".")[0].split("_")[:3]) + "_1" + ".npz")
            textpath = os.path.join(self.text_embedding_path, key.split(".")[0][:-1] + "1" + ".npz_text_feature.npy")
            fast_path = os.path.join(self.mask_fast_path, key.split(".")[0][:-1] + "1" + ".npy")
            
            if os.path.exists(image_path) and limit > 0 and os.path.exists(textpath) and os.path.exists(tpath) and os.path.exists(fast_path):

            # if os.path.exists(image_path) and limit > 0:
                limit -= 1
                self.text[key] = all_text[key]
                self.label[key] = all_label[key]
                self.text_embedding.append(np.load(textpath))
            else:
                print(image_path, os.path.exists(textpath), os.path.exists(image_path))

            if limit <= 0:
                break
        self.text_embedding = np.array(self.text_embedding)

        print(len(self.text.keys()))
        
    def get_image_fea(self, index):
        key = list(self.text.keys())[index]
        image_path = os.path.join(self.feature_path, key.split(".")[0] + ".npz" + "qwen_spatial_feature.npy")
        image_fea = np.load(image_path)
        return image_fea, image_path, image_fea

    def get_image(self, index, image_key="image"):
        image_fea, image_path, all_image = self.get_image_fea(index)

        mask = self.get_mask(index)
        return {
            "image": image_fea,
            "img_index": list(self.text.keys())[index],
            "mask": mask,
            "cap_index": 0,
            "raw_index": index,
            "data_name": image_path.split("/")[-1],
            "obver_label": self.label[list(self.label.keys())[index]],
        }

    def _compute_finding_freq_for_imp(self, csv_path: str) -> Dict[str, int]:
        """
        Count, per canonical ``finding`` name, the number of DISTINCT
        Volumenames (across the WHOLE csv_path file -- typically the
        TRAIN csv, passed in explicitly via ``min_finding_freq_csv_path``
        below, or falls back to ``csv_path`` itself) in which that finding
        appears at least once in ``abnormality_structured``.

        This mirrors ``set_prediction_trainer_v5.py::_collect_finding_freq``
        (per-VOLUME occurrence, not per-row) so that a ``min_finding_freq``
        threshold applied here selects the SAME long-tail cutoff as the
        Stage-1 set-prediction top-K restriction, keeping the Stage-2 <imp>
        target text consistent with what the frozen Stage-1 decoder was
        actually supervised to predict.

        The result depends ONLY on ``csv_path`` (not on remove_location /
        remove_attrs), yet build_volume_abnormality_dict is called twice per
        dataset (located + name-only), each triggering this full-file scan.
        We therefore memoise per ``csv_path`` on the instance so the large
        CSV is parsed once instead of twice.
        """
        cache = getattr(self, "_freq_cache", None)
        if cache is None:
            cache = self._freq_cache = {}
        if csv_path in cache:
            return cache[csv_path]

        freq: Dict[str, int] = {}
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            vol_seen: Dict[str, set] = {}
            for row in reader:
                vol = row['Volumename'].strip()
                raw = (row.get('abnormality_structured') or '').strip()
                if not raw or raw == '[]':
                    continue
                try:
                    items = json.loads(raw)
                except (json.JSONDecodeError, TypeError, ValueError):
                    continue
                if not isinstance(items, list):
                    continue
                seen = vol_seen.setdefault(vol, set())
                for it in items:
                    if not isinstance(it, dict):
                        continue
                    finding = (it.get('finding') or '').strip().lower()
                    if not finding or finding in seen:
                        continue
                    seen.add(finding)
                    freq[finding] = freq.get(finding, 0) + 1
        cache[csv_path] = freq
        return freq

    def build_volume_abnormality_dict(
        self,
        csv_path: str,
        remove_location: Optional[bool] = None,
        remove_attrs: Optional[bool] = None,
    ) -> OrderedDict:
        """
        Read the CSV file and build a dictionary:
        key: Volumename
        value: combined abnormality string (semicolon-separated across all organs)

        Text is assembled from ``abnormality_structured`` (the JSON list of
        ``{finding, location, attributes}`` produced by the stage4 pipeline,
        e.g. ``stage4_renorm_v2.normalize_finding``) rather than read
        pre-formatted from ``abnormality_normalized``. This makes IMP text
        generation share the SAME canonical finding names / normalization
        fixes as the set-prediction supervision (they used to diverge: the
        old ``abnormality_normalized`` column was produced by an earlier,
        now-superseded normalization pass and never picked up later
        ``stage4_renorm_v2.py`` fixes or the LLM-assisted long-tail merges).

        Each item is rendered as ``"{finding} in {location}"``, optionally
        followed by a parenthesized attribute list (``" ({...})"``) when
        ``remove_attrs`` is False. When it is empty (``location`` missing),
        the ``" in {location}"`` suffix is dropped. Rows with an
        empty/unparseable ``abnormality_structured`` (``""``/``"[]"``/not a
        list) are skipped. Multiple rows for the same Volumename are merged,
        in CSV row order.

        If ``remove_attrs`` is True, the parenthesized attribute suffix is
        omitted entirely (mirrors the old behaviour of stripping
        ``"(...)"`` from ``abnormality_normalized``).

        If ``remove_location`` is True, the ``" in {location}"`` suffix is
        omitted entirely and each item is rendered as just ``"{finding}"``
        (still followed by the attribute suffix unless ``remove_attrs`` is
        also True).

        When ``config["group_imp_by_finding"]`` is True, repeated instances of
        the SAME canonical finding are grouped instead of rendered as repeated
        indistinguishable names. This removes the ambiguous training target
        that caused autoregressive loops such as ``lung nodule; lung nodule;
        ...`` while preserving every location/attribute instance:

          name-only: ``lung nodule; fibrotic sequela``
          located:   ``lung nodule: right upper lobe (5 mm) | left lower lobe
                     (3 mm); fibrotic sequela: bilateral apices (scar)``

        The group order follows each finding's first occurrence; duplicate
        location/attribute details within a group are removed stably. The
        option is disabled by default for backward compatibility and must be
        enabled when training the new grouped-planning version.

        ``remove_location`` / ``remove_attrs`` default to
        ``config["remove_abnormality_location"]`` /
        ``config["remove_abnormality_attributes"]`` when not passed
        explicitly (``None``), but can be overridden per-call -- e.g. by
        ``imp_chain_mode`` (see ``get_text_and_label``) to build a SECOND,
        bare-finding-name-only dict alongside the normal one without
        touching the base config flags.

        ── Frequency-based finding filter (min_finding_freq) ──
        If ``config["min_finding_freq"] > 0`` (and
        ``config["filter_imp_by_min_finding_freq"]`` is not explicitly
        False), any ``finding`` whose train-set volume-frequency is BELOW
        that threshold is dropped from the rendered <imp> text (same
        long-tail cutoff Stage 1's ``restrict_to_topk_findings`` applies to
        the set-prediction GT). The frequency table is always computed from
        ``config["normalized_label_path_train"]`` (the TRAIN csv) so the
        cutoff is identical whether this method is called for the train or
        the val split -- otherwise a finding common in train but rare in
        val (or vice-versa) would get inconsistently filtered. This filter
        is independent of ``remove_location``/``remove_attrs`` overrides, so
        two dicts built from the same ``csv_path`` (e.g. name-only vs
        name+location) always keep/drop the exact same set of findings and
        stay item-for-item aligned.
        """
        if remove_attrs is None:
            remove_attrs = self.config.get("remove_abnormality_attributes", False)
        if remove_location is None:
            remove_location = self.config.get("remove_abnormality_location", False)

        min_finding_freq = int(self.config.get("min_finding_freq", 0) or 0)
        filter_imp = bool(self.config.get("filter_imp_by_min_finding_freq", True))
        # New grouped planning format: <imp_name> is a true SET of unique
        # finding names, while <imp_loc> groups every location/attribute
        # instance under that name. Disabled by default for old checkpoints.
        group_by_finding = bool(self.config.get("group_imp_by_finding", False))
        allowed_findings: Optional[set] = None

        # ── Anatomy-order sorting of the planning findings (ROUGE_L fix) ─────
        # The GT report is organised in a FIXED 10-segment anatomical / exam-
        # flow order (see seg_text: trachea -> mediastinum/heart -> lung ->
        # esophagus -> pleura -> bone -> thyroid -> chest/breast -> abdomen ->
        # others). The <imp_name>/<imp_loc> planning text, however, lists
        # findings in the RAW order of the CSV `abnormality_structured` JSON
        # array, which is unrelated to that anatomical order. The 3-stage
        # model then learns to emit the report following the planning order,
        # so its sentence order drifts from GT -> shorter LCS -> lower
        # ROUGE_L (diagnosed: Kendall tau 0.740 vs baseline 0.767).
        #
        # When order_imp_by_anatomy=True we STABLE-sort each volume's findings
        # by the anatomical segment their `finding`+`location` text maps to,
        # so the planning prior matches GT's report order. Findings that hit
        # no keyword keep a large rank (go last, like seg 9 "others"), and
        # ties preserve their original CSV order (stable) so behaviour inside
        # a segment is unchanged. Default False => byte-for-byte legacy order.
        # Order dropout and anatomy sort are mutually exclusive. If dropout is
        # on, force the BASE order to the raw CSV order (anat sort off) so the
        # located and name-only lists share an identical base ordering and a
        # single shuffle permutation stays index-aligned across both segments.
        _dropout_on = (float(self.config.get("imp_order_dropout", 0.0) or 0.0) > 0.0) or \
                      (float(self.config.get("imp_finding_dropout", 0.0) or 0.0) > 0.0)
        order_by_anat = bool(self.config.get("order_imp_by_anatomy", False)) and not _dropout_on
        # Same keyword table / order as seg_text's 10-segment split.
        _anat_keywords = [
            ["trachea", " bronchie", " bronchi ", " bronc", "tracheostomy", "tracheost", "nasogastric"],
            ["heart", "mediastinum", "mediastinal", "cardiac", "ventricle", "brachiocephalic", "vena", "aorta", "aortic", "artery", "thymus", "prevascular", "pericardial", "vascular", "arteries", "cardiothoracic", "paratracheal", "atria", "mitral valve"],
            ["lung", "pulmonary", "bilateral hilar", "emphysema", "pneumonic", "pneumonia", "hilar", "consolidation", "interlobular", "nodule", "atelectas", "infiltrat"],
            ["esophagus", "inlet", "cricopharyngeal", "esophag", "hiatal hernia"],
            ["pleura", "thorax", "membrane", "diaphragm", "effusion"],
            [" rib", "spine", "sternum", "bone", "spinal", "vertebrae", "clavicle", "scapula", "humerus", "femur", "cartilage", "tube bone", "vertebral", "fractures", "costochondral", "vertebra", "sternoclavicular"],
            ["thyroid"],
            ["breast", "mammary", "chest", "armpits", "armpit", "axilla", "retroareolar", "gynecomastia", "thoracic wall"],
            ["abdomen", "abdominal", "adrenal", "colon", "duodenum", "pericholecytic", "gallbladder", "intestine", "bowel", "kidney", "perinephric", "liver", "intrahepatic", "hepatic", "caudate", "pancreas", "portal vein", "splenic vein", "rectum", "renal", "spleen", "stomach", "celiac", "hepatosteatosis", "peritoneum", "retrocrural", "gall bladder"],
            ["foramina", "pleuroparenchymal", "bladder", "perivesical", "prostate", "scalene"],
        ]
        _ANAT_MISS_RANK = len(_anat_keywords)  # unmatched -> after all named segs

        def _anat_rank(finding_txt: str, location_txt: str) -> int:
            hay = f"{finding_txt} {location_txt}".lower()
            for rank, kws in enumerate(_anat_keywords):
                for kw in kws:
                    if kw.strip().lower() in hay:
                        return rank
            return _ANAT_MISS_RANK
        if filter_imp and min_finding_freq > 0:
            freq_csv_path = self.config.get("normalized_label_path_train", csv_path)
            freq = self._compute_finding_freq_for_imp(freq_csv_path)
            allowed_findings = {f for f, c in freq.items() if c >= min_finding_freq}
            print(f"[build_volume_abnormality_dict] min_finding_freq={min_finding_freq} "
                  f"(freq computed from {freq_csv_path!r}): "
                  f"{len(allowed_findings)}/{len(freq)} canonical findings kept for <imp> text")

        volume_dict = OrderedDict()
        n_dropped_by_freq = 0

        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                vol = row['Volumename'].strip()
                raw = (row.get('abnormality_structured') or '').strip()

                # Ensure every volume has an entry
                if vol not in volume_dict:
                    volume_dict[vol] = []

                # Skip empty values: empty string, "[]", or just whitespace
                if not raw or raw == '[]':
                    continue

                try:
                    items = json.loads(raw)
                except (json.JSONDecodeError, TypeError, ValueError):
                    continue
                if not isinstance(items, list):
                    continue

                for it in items:
                    if not isinstance(it, dict):
                        continue
                    finding = (it.get('finding') or '').strip()
                    if not finding:
                        continue

                    if allowed_findings is not None and finding.lower() not in allowed_findings:
                        n_dropped_by_freq += 1
                        continue

                    location = (it.get('location') or '').strip()
                    attrs = it.get('attributes') or {}
                    attr_str = ""
                    if not remove_attrs and isinstance(attrs, dict) and attrs:
                        # Preserve JSON insertion order, but drop empty values
                        # and duplicate attribute strings within one instance.
                        attr_values = []
                        for value in attrs.values():
                            value = str(value).strip()
                            if value and value not in attr_values:
                                attr_values.append(value)
                        attr_str = ', '.join(attr_values)

                    seq_idx = len(volume_dict[vol])
                    rank = _anat_rank(finding, location) if order_by_anat else 0

                    if group_by_finding:
                        # Store a structured instance. Rendering happens only
                        # after all rows are read, when instances sharing the
                        # same canonical finding can be grouped together.
                        # `detail` deliberately excludes the finding name:
                        #   location + optional attributes -> "RUL (5 mm)"
                        # The name-only call (remove_location=True,
                        # remove_attrs=True) consequently has an empty detail
                        # and renders exactly one unique name per group.
                        detail = "" if remove_location else location
                        if attr_str:
                            detail = f"{detail} ({attr_str})" if detail else f"({attr_str})"
                        volume_dict[vol].append(
                            (rank, seq_idx, finding.lower(), finding, detail)
                        )
                    else:
                        # Legacy one-item-per-instance rendering.
                        if remove_location:
                            part = finding
                        else:
                            part = f"{finding} in {location}" if location else finding
                        if attr_str:
                            part = f"{part} ({attr_str})"
                        volume_dict[vol].append((rank, seq_idx, part))

        if allowed_findings is not None:
            print(f"[build_volume_abnormality_dict] {csv_path!r}: dropped "
                  f"{n_dropped_by_freq} finding items below min_finding_freq")

        # Join all parts with "; " for each volume; use "NO abnormality" for empty ones.
        # Each element is (anat_rank, seq_idx, part). When order_imp_by_anatomy
        # is on, stable-sort by (anat_rank, seq_idx) so findings follow GT's
        # anatomical/exam-flow order; otherwise (anat_rank==0 for all) the sort
        # is a no-op on seq_idx and preserves the exact legacy CSV order.
        #
        # We ALSO stash the ordered per-volume parts LIST on self so that
        # __getitem__ can optionally re-shuffle them per sample (imp_order_dropout).
        # The list is keyed by an attribute name derived from remove_location so
        # the <imp_loc> (located) and <imp_name> (name-only) dicts keep separate
        # ordered lists that stay index-for-index aligned (same findings, same
        # base order) -- a single shared permutation applied to both keeps the
        # two planning segments consistent.
        result = OrderedDict()
        parts_by_vol = OrderedDict()
        n_grouped_instances = 0
        for vol, records in volume_dict.items():
            if records:
                if group_by_finding:
                    # Group by canonical finding key while preserving the first
                    # occurrence's rank/order/spelling. OrderedDict makes this
                    # deterministic across train/val and across DDP workers.
                    groups = OrderedDict()
                    for rank, seq_idx, finding_key, finding, detail in records:
                        if finding_key not in groups:
                            groups[finding_key] = {
                                "rank": rank,
                                "seq_idx": seq_idx,
                                "finding": finding,
                                "details": [],
                            }
                        # Stable detail de-duplication: repeated identical
                        # location+attribute instances carry no extra signal.
                        if detail and detail not in groups[finding_key]["details"]:
                            groups[finding_key]["details"].append(detail)

                    ordered_groups = sorted(
                        groups.values(), key=lambda g: (g["rank"], g["seq_idx"])
                    )
                    parts = []
                    for group in ordered_groups:
                        finding = group["finding"]
                        details = group["details"]
                        # name-only call: details == [] -> one unique finding.
                        # located call: group every retained loc(attr) instance
                        # under that finding using a clear intra-group '|'.
                        part = f"{finding}: {' | '.join(details)}" if details else finding
                        parts.append(part)
                    n_grouped_instances += len(records) - len(parts)
                else:
                    ordered = sorted(records, key=lambda t: (t[0], t[1]))
                    parts = [p for _, _, p in ordered]

                result[vol] = '; '.join(parts)
                parts_by_vol[vol] = parts
            else:
                result[vol] = 'NO abnormality'
                parts_by_vol[vol] = []

        # Route the ordered parts list to the right cache (located vs name-only)
        # so __getitem__ can rebuild a shuffled string on the fly.
        if not hasattr(self, "_imp_parts_cache"):
            self._imp_parts_cache = {}
        cache_key = "name_only" if remove_location else "located"
        self._imp_parts_cache[cache_key] = parts_by_vol

        if group_by_finding:
            mode = "name-only" if remove_location and remove_attrs else "located"
            print(f"[build_volume_abnormality_dict] group_imp_by_finding=True "
                  f"({mode}): collapsed {n_grouped_instances} repeated finding "
                  f"instances into unique finding groups; all distinct "
                  f"location/attribute details are preserved in <imp_loc>.")
        if order_by_anat:
            print("[build_volume_abnormality_dict] order_imp_by_anatomy=True: "
                  "planning findings sorted into GT's 10-segment anatomical order.")

        return result

    def __getitem__(self, index):
        
        ret = self.get_suite(index)

        new_ret = {}
        new_ret["text_ids"] = ret["rg_encoding"]
        new_ret["text_masks"] = ret["rg_attn"]
        new_ret["seg_index"] = torch.tensor(ret["seg_index"])
        new_ret["obver_label"] = ret["obver_label"].long()
        new_ret["prompt_ids"] = ret["prompt_ids"]

        final_seg = ret["final_seg"]

        all_encoding = []
        for seg in final_seg:
            all_encoding.append(self.tokenizer(seg, padding="max_length",
                truncation=True,
                max_length=200,
                return_special_tokens_mask=True,
                return_tensors="pt"))
                
        all_encodings = torch.cat([l.input_ids for l in all_encoding])
        all_maps = torch.cat([l.attention_mask for l in all_encoding])
        #print(all_encodings.shape)
        new_ret["all_encodings"] = all_encodings
        new_ret["all_maps"] = all_maps
        new_ret["image"] = ret["image"]
        #new_ret["image_downsample"] = ret["image_downsample"]

        new_ret["data_name"] = ret["data_name"]
        new_ret["final_seg"] = ret["final_seg"]
        new_ret["text"] = ret["text"][0]
        new_ret["text_ori"] = ret["text"][0]

        new_ret["text_embedding"] = torch.tensor(ret["text_embedding"])
        new_ret["text_embedding_ab"] = torch.tensor(ret["text_embedding_ab"])
        new_ret["abnorm_label"] = torch.tensor(ret["abnorm_label"])
        
        new_ret['mask'] = ret["mask"]

        _vol_key = ret["data_name"].split(".")[0] + ".nii.gz"
        new_ret["abnormalization"] = self.all_dict[_vol_key]
        if getattr(self, "imp_chain_mode", False):
            new_ret["abnormalization_name_only"] = self.all_dict_name_only[_vol_key]

        # ── Planning perturbation (train-only ROUGE_L / over-reliance fix) ──
        # Two independent, composable perturbations of the <imp_loc>/<imp_name>
        # planning text, applied PER SAMPLE (fresh randomness each __getitem__
        # -> varies across epochs; NEVER at val/test). Both are driven by a
        # single shared index list `keep` so the located and name-only segments
        # stay finding-for-finding aligned.
        #
        #   imp_finding_dropout (p_fd): each finding is INDEPENDENTLY dropped
        #     from the planning with prob p_fd (at least 1 kept). The GT report
        #     (<answer> target) is left UNCHANGED, so this teaches <answer> that
        #     planning is only a hint -- it must still recover findings the
        #     planning omits (mimics imperfect inference-time planning, reduces
        #     over-reliance / copying).
        #   imp_order_dropout (p_od): with prob p_od, SHUFFLE the (kept)
        #     findings' order, so the model cannot learn "write the report in
        #     planning order" (the diagnosed cause of the 3-stage ROUGE_L gap).
        #
        # We only add finding names / drop them -- we never INJECT findings the
        # case does not have (that would train the model to ignore planning and
        # tends to hurt F1). order dropout is mutually exclusive with
        # order_imp_by_anatomy (handled in build_volume_abnormality_dict, which
        # falls back to raw CSV base order whenever any dropout is on).
        _p_od = float(getattr(self, "_imp_order_dropout", 0.0) or 0.0)
        _p_fd = float(getattr(self, "_imp_finding_dropout", 0.0) or 0.0)
        _is_train = getattr(self, "usage", "train") == "train"
        if _is_train and (_p_od > 0.0 or _p_fd > 0.0):
            cache = getattr(self, "_imp_parts_cache", {})
            loc_parts = cache.get("located", {}).get(_vol_key, None)
            if loc_parts and len(loc_parts) >= 1:
                n = len(loc_parts)
                # 1) finding dropout -> which indices to keep
                if _p_fd > 0.0 and n >= 2:
                    keep = [i for i in range(n) if random.random() > _p_fd]
                    if not keep:                       # never drop everything
                        keep = [random.randrange(n)]
                else:
                    keep = list(range(n))
                # 2) order dropout -> shuffle the kept indices
                if _p_od > 0.0 and len(keep) >= 2 and random.random() < _p_od:
                    random.shuffle(keep)
                # Rebuild only if something actually changed vs the cached str.
                changed = (keep != list(range(n)))
                if changed:
                    new_ret["abnormalization"] = '; '.join(loc_parts[i] for i in keep)
                    if getattr(self, "imp_chain_mode", False):
                        name_parts = cache.get("name_only", {}).get(_vol_key, None)
                        if name_parts and len(name_parts) == n:
                            new_ret["abnormalization_name_only"] = '; '.join(
                                name_parts[i] for i in keep)

        #new_ret["text_query"] = ret["text_query"]
        #new_ret["lesion_mask"] = ret["lesion_mask"]
        new_ret['impression'] = ret["impression"]
        return new_ret



        return image_fea, image_path, image_fea

class VQAdataset(CTRGDataset_RATE_hr_sim_fea):

    def __init__(self, config, split: str = "train",):
        
        self.usage = split
        self.with_lesion = False
        super().__init__(config, split = split)

        
        # self.get_text_and_label()

    def get_text_and_label(self):
        ####################
        if self.with_lesion:
            self.lesion_mask_path = self.config["lesion_mask_path"]
            self.lesion_mask_json = self.config["lesion_mask_json"]

            with open(self.lesion_mask_json, "r", encoding="utf-8") as f:
                lesion_json = json.load(f)
            self.lesion_json = lesion_json[self.usage]

            print(len(self.lesion_json))
            self.name_have_lesion = []
            for data in self.lesion_json:
                self.name_have_lesion.append(data["name"])
        ######################


        if self.usage == "train":
            self.data_list = pd.read_csv(self.config["vqa_data_train_path"])
        elif self.usage == "val":
            df = pd.read_csv(self.config["vqa_data_test_path"])
            self.data_list = df.sample(n=4000, random_state=42).reset_index(drop=True)
        elif self.usage == "test":
            self.data_list = pd.read_csv(self.config["vqa_data_test_path"])
        else:
            print("The mode is not desired!")
        
        keep_mask = pd.Series([True] * len(self.data_list), index=self.data_list.index)

        print(len(self))

        print("xxk")

        for idx, row in self.data_list.iterrows():
            
            # row 是一个 pandas Series，可通过列名访问每一项
            key = row["VolumeName"].replace("test", "valid")

            # if row["question_type"] != "Task2_Anomaly_detection":
            #     continue

            image_path = os.path.join(self.feature_path,  key.split(".")[0] + ".npz" + "selected_patch.npy")
            fast_path = os.path.join(self.mask_fast_path, key.split(".")[0][:-1] + "1" + ".npy")
            if not os.path.exists(image_path) or not os.path.exists(fast_path): #  or key.split(".")[0] != "valid_526_a_1" or row["source_folder"] != "Task1_Image_Observation":
                keep_mask[idx] = False

        # 一次性保留满足条件的行
        self.data_list = self.data_list[keep_mask].reset_index(drop=True)

        print(len(self))

    def __len__(self):

        return len(self.data_list)

    def get_image_fea(self, index):
        
        key = self.data_list.iloc[index]["VolumeName"].replace("test", "valid")

        image_path = os.path.join(self.feature_path,  key.split(".")[0] + ".npz" + "selected_patch.npy")

        all_feature = os.path.join(self.feature_path,  key.split(".")[0] + ".npz" + "image_feature.npy")

        return np.load(image_path), image_path, np.load(all_feature)
    
    def get_mask(self, index, img_shape = None):
        key = self.data_list.iloc[index]["VolumeName"].replace("test", "valid")

        fast_path = os.path.join(self.mask_fast_path, key.split(".")[0][:-1] + "1" + ".npy")
        # print(key, fast_path)
        if not os.path.exists(fast_path):
            return None
        else:
            # print("!")
            return torch.tensor(np.load(fast_path))

    def get_lesion_mask(self, index):
        key = self.data_list.iloc[index]["VolumeName"].replace("test", "valid")

        if key in self.name_have_lesion:
            lesion_mask_path = os.path.join(self.lesion_mask_path, key.split(".")[0] + ".npz")
            lesion_mask = np.load(lesion_mask_path)['arr_0']

            final_lesion_mask = np.zeros(lesion_mask.shape[1:])
            for l in range(lesion_mask.shape[0]):
                final_lesion_mask[lesion_mask[l] > 0] = l+1

            text_query = self.lesion_json[self.name_have_lesion.index(key)]["findings"]

            return final_lesion_mask, text_query
        else:
            return None, None

    def get_raw_image(self, index, image_key="image"):
        
        key = self.data_list.iloc[index]["VolumeName"].replace("test", "valid")


        image_path = os.path.join(self.image_path, "_".join(key.split("_")[:2]), "_".join(key.split("_")[:2]) + key.split("_")[2],  key.split(".")[0] + ".npz")

        return np.load(image_path)['arr_0'], image_path

    def get_image(self, index, image_key="image"):
        
        if False:
            
            image, path = self.get_raw_image(index, image_key=image_key)
            image_tensor_reso = self.image_prepro(image)
        else:
            image_tensor_reso = torch.tensor(np.array(1))

        image_tensor, path, all_image = self.get_image_fea(index)
        #print(image_tensor.shape)
        mask = self.get_mask(index)
        # print(mask.shape)
        image_tensor = torch.tensor(image_tensor)

        if self.with_lesion:

            lesion_mask, text_query = self.get_lesion_mask(index)

            if lesion_mask is None:
                lesion_mask = np.zeros(self.target_size)

                text_query = {
                    "0": "kkkkkkkkkkk"
                }

            lesion_mask = self.image_prepro(lesion_mask)

            return {
                "image": image_tensor,
                "image_tensor_reso": image_tensor_reso,
                "raw_index": index,
                "data_name": path.split("/")[-1],
                "all_image": all_image,
                "mask": mask,
                "lesion_mask": lesion_mask,
                "text_query": text_query
            }
        else:
            return {
                "image": image_tensor,
                "image_tensor_reso": image_tensor_reso,
                "raw_index": index,
                "data_name": path.split("/")[-1],
                "all_image": all_image,
                "mask": mask,
                # "lesion_mask": lesion_mask,
                # "text_query": text_query
            }

    



    def get_text(self, raw_index):

        data = self.data_list.iloc[raw_index]
        # question example:Task,Subtask,VolumeName,Question,Answer,QuestionType,AnswerChoice,Choice A,Choice B,Choice C,Choice D
        # :                4,2,test_1000_a_1.nii.gz,Does the CT scan exhibit atelectasis?,No,Close,B,Yes,No,-,-
        # :                5,4,test_331_e_1.nii.gz,"In light of the current CT imaging, how should we classify the hiatal hernia?","Resolved Lesion (Previously present or recurrent, now absent)",Close,B,"Refractory Lesion (Persistent or recurrent, now present)","Resolved Lesion (Previously present or recurrent, now absent)","New Lesion (Absent previously, now present)",No Abnormality (Always absent)


        parsed_organs = data["Auto_Organ"]
        parsed_organs = str(parsed_organs).strip()
            
        # 针对大模型可能产生的额外引号进行清洗
        # 如果字符串看起来像 "['...']" (被双引号包裹)，先去掉外层引号
        if parsed_organs.startswith('"') and parsed_organs.endswith('"'):
            parsed_organs = parsed_organs[1:-1]
        # 或者是单引号包裹
        if parsed_organs.startswith("'") and parsed_organs.endswith("'"):
            parsed_organs = parsed_organs[1:-1]
        



        shuffle_choices = self.usage == "train"
        # print(shuffle_choices)

        if data["QuestionType"] == "Close":
            question = data["Question"]
        
            # Determine if 2-choice or 4-choice
            is_four_choice = (data["Choice C"] != "-")
            
            if is_four_choice:
                original_choices = [
                    ("A", data["Choice A"]),
                    ("B", data["Choice B"]),
                    ("C", data["Choice C"]),
                    ("D", data["Choice D"])
                ]
            else:
                original_choices = [
                    ("A", data["Choice A"]),
                    ("B", data["Choice B"])
                ]
            


            if shuffle_choices:
                # 🔥 Only shuffle during TRAINING
                shuffled_choices = original_choices.copy()
                random.shuffle(shuffled_choices)
                
                # Build question with shuffled choices
                choice_str = " ".join(f"{label}. {text}" for label, text in shuffled_choices)
                question += " Choose one from following choices: " + choice_str
                
                # Remap answer label based on text matching
                correct_old_label = data["AnswerChoice"]
                correct_text = None
                for label, text in original_choices:
                    if label == correct_old_label:
                        correct_text = text
                        break
                if correct_text is None:
                    raise ValueError(f"AnswerChoice '{correct_old_label}' not found in choices.")
                
                # Find new label in shuffled list
                new_label = None
                for label, text in shuffled_choices:
                    if text == correct_text:
                        new_label = label
                        break
                if new_label is None:
                    raise ValueError("Correct answer text not found after shuffle.")
                    
                answer = f"{new_label}. {correct_text}"
                
            else:
                # 🧪 During VALIDATION / TESTING: keep original order
                if is_four_choice:
                    choice_str = "Choose one from following choices: " \
                            f"A. {data['Choice A']} B. {data['Choice B']} " \
                            f"C. {data['Choice C']} D. {data['Choice D']}"
                else:
                    choice_str = "Choose one from following choices: " \
                            f"A. {data['Choice A']} B. {data['Choice B']}"
                question += " " + choice_str
                answer = f"{data['AnswerChoice']}. {data['Answer']}"
        else:
            question = data["Question"]
            answer = str(data["Answer"])

        return {"question": question,
                "answer": answer,
                "question_type": data["source_folder"],
                "source_file": data["source_file"],
                "parsed_organs": parsed_organs}


    def get_suite(self, index):

        result = None
        while result is None:
            #try:
            ret = dict()
            ret.update(self.get_image(index))
            if not self.image_only:
                txt = self.get_text(index)
                ret.update(txt)

            result = True

        
        return ret


    def __getitem__(self, index):

        ret = self.get_suite(index)

        new_ret = {}

        new_ret["image"] = ret["image"]

        new_ret["data_name"] = ret["data_name"]
        new_ret["question"] = ret["question"]
        new_ret["answer"] = ret["answer"]
        new_ret["question_type"] = ret["question_type"]
        new_ret["source_file"] = ret["source_file"]

        new_ret["all_image"] = ret["all_image"]

        new_ret['mask'] = ret["mask"]

        new_ret['parsed_organs'] = ret["parsed_organs"]

        if self.with_lesion:
            new_ret["lesion_mask"] = ret["lesion_mask"]
            new_ret["text_query"] = ret["text_query"]

        # new_ret['image_tensor_reso'] = ret["image_tensor_reso"]
        
        
        for k, v in new_ret.items():
            if v is None:
                print(f"Warning: {k} is None for index {index} in data_name {new_ret['data_name']}")    

        return new_ret



class VQAdataset_qwen(VQAdataset):
    
    def __init__(self, config, split: str = "train",):

        super().__init__(config, split = split)
    
    def get_text_and_label(self):
        ####################
        if self.with_lesion:
            self.lesion_mask_path = self.config["lesion_mask_path"]
            self.lesion_mask_json = self.config["lesion_mask_json"]

            with open(self.lesion_mask_json, "r", encoding="utf-8") as f:
                lesion_json = json.load(f)
            self.lesion_json = lesion_json[self.usage]

            print(len(self.lesion_json))
            self.name_have_lesion = []
            for data in self.lesion_json:
                self.name_have_lesion.append(data["name"])
        ######################

        if self.usage == "train":
            self.data_list = pd.read_csv(self.config["vqa_data_train_path"])
        elif self.usage == "val":
            df = pd.read_csv(self.config["vqa_data_test_path"])
            self.data_list = df.sample(n=4000, random_state=42).reset_index(drop=True)
        elif self.usage == "test":
            self.data_list = pd.read_csv(self.config["vqa_data_test_path"])
        else:
            print("The mode is not desired!")
        
        keep_mask = pd.Series([True] * len(self.data_list), index=self.data_list.index)

        print(len(self))

        print("xxk")

        for idx, row in self.data_list.iterrows():
            
            # row 是一个 pandas Series，可通过列名访问每一项
            key = row["VolumeName"].replace("test", "valid")

            # if row["question_type"] != "Task2_Anomaly_detection":
            #     continue

            image_path = os.path.join(self.feature_path,  key.split(".")[0] + ".npz" + "qwen_spatial_feature.npy")
            fast_path = os.path.join(self.mask_fast_path, key.split(".")[0][:-1] + "1" + ".npy")
            if not os.path.exists(image_path) or not os.path.exists(fast_path): #  or key.split(".")[0] != "valid_526_a_1" or row["source_folder"] != "Task1_Image_Observation":
                keep_mask[idx] = False
                print(fast_path)

        # 一次性保留满足条件的行
        self.data_list = self.data_list[keep_mask].reset_index(drop=True)

        print(len(self))

    def get_image_fea(self, index):
        key = self.data_list.iloc[index]["VolumeName"].replace("test", "valid")
        image_path = os.path.join(self.feature_path, key.split(".")[0] + ".npz" + "qwen_spatial_feature.npy")
        image_fea = np.load(image_path)
        return image_fea, image_path, image_fea