import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
import json
import h5py
import numpy as np
import os
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from tsv_file import TSVFile

class ICEDataset(Dataset):
    def __init__(self, args, imgid_file: str, liwc_file: str, input_file: str, label_file: str, img_path: str, data_split: str):
        assert os.path.isfile(imgid_file)
        assert os.path.isfile(liwc_file)
        assert os.path.isfile(input_file)
        assert os.path.isfile(label_file)
        self.args = args

        if data_split!='train' and data_split!='val' and data_split!='test':
            raise "Data split must be train/val/test"
        self.data_split = data_split

        with open(imgid_file, 'r') as f:
            self.imgid = json.load(f)
        with open(liwc_file, 'r') as f:
            self.liwc = json.load(f)
        with open(input_file, 'r') as f:
            self.input_ids = json.load(f)
        with open(label_file, 'r') as f:
            self.label_ids = json.load(f)

        self.img_path = img_path
        with open(os.path.join(img_path, 'img_feat_dict.json'), 'r', encoding='utf8') as f:
            self.img_feat_dict = json.load(f)


    def __len__(self):
        return len(self.imgid)

    def __getitem__(self, item):
        imgid = self.imgid[item]
        imgid = imgid.replace('/', '*')
        imgfeatfile = self.img_feat_dict[imgid]
        with h5py.File(os.path.join(self.img_path, imgfeatfile), 'r') as imgf:
            img = torch.from_numpy(np.array(imgf[imgid])).float()


        curliwc = torch.from_numpy(np.array(self.liwc[item])).float()
        curinput_ids = torch.from_numpy(np.array(self.input_ids[item]))
        curlabel_ids = torch.from_numpy(np.array(self.label_ids[item]))

        return img, curliwc, curinput_ids, curlabel_ids


class ICEDatasetNew(Dataset):
    def __init__(self, args, imgid_file: str, liwc_file: str, title_file: str, cmt_file: str, img_path: str, data_split: str):
        assert os.path.isfile(imgid_file)
        assert os.path.isfile(liwc_file)
        assert os.path.isfile(title_file)
        assert os.path.isfile(cmt_file)
        self.args = args

        if data_split!='train' and data_split!='val' and data_split!='test':
            raise "Data split must be train/val/test"

        self.data_split = data_split

        with open(imgid_file, 'r') as f:
            self.imgid = json.load(f)
        with open(liwc_file, 'r') as f:
            self.liwc = json.load(f)
        with open(title_file, 'r') as f:
            self.title_ids = json.load(f)
        with open(cmt_file, 'r') as f:
            self.cmt_ids = json.load(f)

        self.img_path = img_path
        with open(os.path.join(img_path, 'img_feat_2011', 'img_feat_dict.json'), 'r', encoding='utf8') as f:
            self.img_feat_dict_2011 = json.load(f)

        with open(os.path.join(img_path, 'img_feat_2012', 'img_feat_dict.json'), 'r', encoding='utf8') as f:
            self.img_feat_dict_2012 = json.load(f)


    def __len__(self):
        return len(self.imgid)

    def __getitem__(self, item):
        imgid = self.imgid[item]
        imgid = imgid.replace('*','/')
        if imgid in self.img_feat_dict_2011:
            imgfeatfile = self.img_feat_dict_2011[imgid]
            with h5py.File(os.path.join(self.img_path, 'img_feat_2011', imgfeatfile), 'r') as imgf:
                img = torch.from_numpy(np.array(imgf[imgid])).float()
        else:
            imgfeatfile = self.img_feat_dict_2012[imgid]
            with h5py.File(os.path.join(self.img_path, 'img_feat_2012', imgfeatfile), 'r') as imgf:
                img = torch.from_numpy(np.array(imgf[imgid])).float()

        curliwc = torch.tensor(self.liwc[item]).float().unsqueeze(0)

        curtitle = self.title_ids[item]
        curtitle = curtitle[:25] + [50256]*(25-len(curtitle))
        curtitle = torch.tensor(curtitle).long()

        curcmt = self.cmt_ids[item][:29]
        curcmt = curcmt + [50256]*(29-len(curcmt))

        curinput_ids = torch.tensor([50256]+curcmt).long()
        curlabel_ids = torch.tensor(curcmt+[50256]).long()


        return img, curliwc, curtitle, curinput_ids, curlabel_ids



class NICEDatasetResnet(Dataset):
    def __init__(self, args, tokenizer, data_split: str):
        self.args = args
        self.tokenizer = tokenizer

        assert data_split in ['train', 'val', 'test']

        self.data_split = data_split

        self.corpus = TSVFile(os.path.join(args.data_dir, data_split+'.tsv'))

        self.img_path = args.img_path
        with open(os.path.join(self.img_path, 'img_feat_2011', 'img_feat_dict.json'), 'r', encoding='utf8') as f:
            self.img_feat_dict_2011 = json.load(f)

        with open(os.path.join(self.img_path, 'img_feat_2012', 'img_feat_dict.json'), 'r', encoding='utf8') as f:
            self.img_feat_dict_2012 = json.load(f)

    def __len__(self):
        return len(self.corpus)

    def __getitem__(self, item):
        corpus_line = self.corpus.seek(item)
        imgid = corpus_line[0]
        imgid = imgid.replace('/', '*')
        if imgid in self.img_feat_dict_2011:
            imgfeatfile = self.img_feat_dict_2011[imgid]
            with h5py.File(os.path.join(self.img_path, 'img_feat_2011', imgfeatfile), 'r') as imgf:
                img = torch.from_numpy(np.array(imgf[imgid])).float()
        else:
            imgfeatfile = self.img_feat_dict_2012[imgid]
            with h5py.File(os.path.join(self.img_path, 'img_feat_2012', imgfeatfile), 'r') as imgf:
                img = torch.from_numpy(np.array(imgf[imgid])).float()

        data_line = corpus_line[2:]
        curliwc = []
        curinput_ids = []
        curlabel_ids = []
        for cmt, liwc in zip(data_line[0::2], data_line[1::2]):
            if cmt == "-":
                cmt = "<|endoftext|>"
            curliwc.append(eval(liwc))
            cmt = self.tokenizer.encode(cmt)
            curcmt = cmt[:self.args.cmt_len-1] + [50256]*(self.args.cmt_len-1-len(cmt))

            curinput_ids.append(50256)
            for l in curcmt:
                curinput_ids.append(l)

            for l in curcmt:
                curlabel_ids.append(l)
            curlabel_ids.append(50256)


        curliwc = torch.tensor(curliwc).float()
        curinput_ids = torch.tensor(curinput_ids).long()
        curlabel_ids = torch.tensor(curlabel_ids).long()

        return img, curliwc, curinput_ids, curlabel_ids


class ICEDatasetSplit(Dataset):
    def __init__(self, args, imgid_file: str, liwc_file: str, input_file: str, label_file: str, img_path: str, data_split: str):
        assert os.path.isfile(imgid_file)
        assert os.path.isfile(liwc_file)
        assert os.path.isfile(input_file)
        assert os.path.isfile(label_file)
        self.args = args
        self.data_split = data_split

        with open(imgid_file, 'r') as f:
            self.imgid = json.load(f)
        with open(liwc_file, 'r') as f:
            self.liwc = json.load(f)
        with open(input_file, 'r') as f:
            self.input_ids = json.load(f)
        with open(label_file, 'r') as f:
            self.label_ids = json.load(f)

        self.img_path = img_path
        with open(os.path.join(img_path, 'img_feat_dict.json'), 'r', encoding='utf8') as f:
            self.img_feat_dict = json.load(f)


    def __len__(self):
        return len(self.imgid)

    def __getitem__(self, item):
        imgid = self.imgid[item]
        imgid = imgid.replace('*','/')
        imgfeatfile = self.img_feat_dict[imgid]
        with h5py.File(os.path.join(self.img_path, imgfeatfile), 'r') as imgf:
            img = np.array(imgf[imgid])

        curliwc = np.array(self.liwc[item])
        curinput_ids = np.array(self.input_ids[item])
        curlabel_ids = np.array(self.label_ids[item])
        return img, curliwc, curinput_ids, curlabel_ids


class ICEDatasetImgRegion(Dataset):
    def __init__(self, args, imgid_file: str, liwc_file: str, cmt_file: str, title_file: str, img_path_2011: str, img_path_2012: str, data_split: str):
        assert os.path.isfile(imgid_file)
        assert os.path.isfile(liwc_file)
        assert os.path.isfile(title_file)
        assert os.path.isfile(cmt_file)
        self.args = args

        if data_split!='train' and data_split!='val' and data_split!='test':
            raise "Data split must be train/val/test"

        self.data_split = data_split

        with open(imgid_file, 'r') as f:
            self.imgid = json.load(f)
        with open(liwc_file, 'r') as f:
            self.liwc = json.load(f)
        with open(title_file, 'r') as f:
            self.title_ids = json.load(f)
        with open(cmt_file, 'r') as f:
            self.cmt_ids = json.load(f)

        self.img_path_2012 = img_path_2012
        self.img_path_2011 = img_path_2011
        with open(os.path.join(img_path_2012, 'img_dict_feat_2012_kz.json'), 'r', encoding='utf8') as f:
            self.img_feat_dict_2012 = json.load(f)
        with open(os.path.join(img_path_2011, '2011keys.json'), 'r', encoding='utf8') as f:
            self.img_feat_dict_2011 = json.load(f)

    def __len__(self):
        return len(self.imgid)

    def __getitem__(self, item):
        imgid = self.imgid[item]
        imgid = imgid.replace('*','/')
        year = imgid[:4]
        if year == "2011":
            imggroup = self.img_feat_dict_2011[imgid]
            img_path = self.img_path_2011
            imgfeatfile = os.path.join(img_path, 'feat_cls_1000', 'imacom_detectron_100dets_train2011_feat'+imggroup[-3:]+'.h5')
            imgclsfile = os.path.join(img_path, 'feat_cls_1000', 'imacom_detectron_100dets_train2011_cls'+imggroup[-3:]+'.h5')
            imgbboxfile = os.path.join(img_path, 'feat_cls_1000', 'imacom_detectron_100dets_train2011_bbox'+imggroup[-3:]+'.h5') 
        else:
            imggroup = self.img_feat_dict_2012[imgid]
            img_path = self.img_path_2012
            imgfeatfile = os.path.join(img_path, 'feat_cls_1000', 'nice_detection_vg_100dets_checkpoint_trainval_feat'+imggroup[-3:]+'.h5')
            imgclsfile = os.path.join(img_path, 'feat_cls_1000', 'nice_detection_vg_100dets_checkpoint_trainval_cls'+imggroup[-3:]+'.h5')
            imgbboxfile = os.path.join(img_path, 'feat_cls_1000', 'nice_detection_vg_100dets_checkpoint_trainval_bbox'+imggroup[-3:]+'.h5') 

        with h5py.File(imgfeatfile, 'r') as imgfeat:
            img = torch.tensor(imgfeat[imggroup][:]).float()

        
        with h5py.File(imgclsfile, 'r') as imgclsf:
            imgcls = torch.tensor(imgclsf[imggroup][:]).float()

           
        with h5py.File(imgbboxfile, 'r') as img_bboxs:
            imgbbox = torch.tensor(img_bboxs[imggroup][:]).float()


        curliwc = torch.tensor(self.liwc[item]).float().unsqueeze(0)

        curtitle = self.title_ids[item]
        curtitle = curtitle[:25] + [50256]*(25-len(curtitle))
        curtitle = torch.tensor(curtitle).long()

        curcmt = self.cmt_ids[item][:29]
        curcmt = curcmt + [50256]*(29-len(curcmt))

        curinput_ids = torch.tensor([50256]+curcmt).long()
        curlabel_ids = torch.tensor(curcmt+[50256]).long()


        return img, imgcls, imgbbox, curliwc, curtitle, curinput_ids, curlabel_ids
