import nltk
import os
import torch
import torch.utils.data as data
from vocabulary import Vocabulary
from PIL import Image
from pycocotools.coco import COCO
import numpy as np
from tqdm import tqdm
import random
import json

def get_loader_val(transform,
               annotations_file,
               img_folder,
               batch_size=1,
               vocab_threshold=None,
               vocab_file='./vocab.pkl',
               start_word="<start>",
               end_word="<end>",
               unk_word="<unk>",
               vocab_from_file=True,
               num_workers=0,
               cocoapi_loc='/opt'):
    """Returns the data loader.
    Args:
      transform: Image transform.
      mode: One of 'train' or 'test'.
      batch_size: Batch size (if in testing mode, must have batch_size=1).
      vocab_threshold: Minimum word count threshold.
      vocab_file: File containing the vocabulary. 
      start_word: Special word denoting sentence start.
      end_word: Special word denoting sentence end.
      unk_word: Special word denoting unknown words.
      vocab_from_file: If False, create vocab from scratch & override any existing vocab_file.
                       If True, load vocab from from existing vocab_file, if it exists.
      num_workers: Number of subprocesses to use for data loading 
      cocoapi_loc: The location of the folder containing the COCO API: https://github.com/cocodataset/cocoapi
    """
    
    # Based on mode (train, val, test), obtain img_folder and annotations_file.

    assert vocab_from_file==True, "Change vocab_from_file to True."
#     img_folder = os.path.join(cocoapi_loc, 'cocoapi/images/test2014/')
#     annotations_file = os.path.join(cocoapi_loc, 'cocoapi/annotations/captions_val2014.json')

    # COCO caption dataset.
    dataset = CoCoDatasetValid(transform=transform,
                          batch_size=batch_size,
                          vocab_threshold=vocab_threshold,
                          vocab_file=vocab_file,
                          start_word=start_word,
                          end_word=end_word,
                          unk_word=unk_word,
                          annotations_file=annotations_file,
                          vocab_from_file=vocab_from_file,
                          img_folder=img_folder)


    # data loader for COCO dataset.
    data_loader = data.DataLoader(dataset=dataset, 
                                  num_workers=num_workers,
                                  batch_size=4)

    return data_loader


class CoCoDatasetValid(data.Dataset):
    
    def __init__(self, transform, batch_size, vocab_threshold, vocab_file, start_word, 
        end_word, unk_word, annotations_file, vocab_from_file, img_folder):
        self.transform = transform
        self.batch_size = batch_size
        self.vocab = Vocabulary(vocab_threshold, vocab_file, start_word,
            end_word, unk_word, annotations_file, vocab_from_file)
        self.img_folder = img_folder

        self.coco = COCO(annotations_file)
        self.ids = list(self.coco.anns.keys())
        print('Obtaining caption lengths...')
        
        all_tokens = [nltk.tokenize.word_tokenize(str(self.coco.anns[self.ids[index]]['caption']).lower()) for index in tqdm(np.arange(len(self.ids)))]
        self.caption_lengths = [len(token) for token in all_tokens]
        
        self.i = 0

    def debug(self, index):
        # obtain image and caption if in training mode
        ann_id = self.ids[index]
        caption = self.coco.anns[ann_id]['caption']
        img_id = self.coco.anns[ann_id]['image_id']
        path = self.coco.loadImgs(img_id)[0]['file_name']

        # Convert image to tensor and pre-process using transform
        image = Image.open(os.path.join(self.img_folder, path)).convert('RGB')
        image = self.transform(image)

        # Convert caption to tensor of word ids.
        tokens = nltk.tokenize.word_tokenize(str(caption).lower())
        caption = []
        caption.append(self.vocab(self.vocab.start_word))
        caption.extend([self.vocab(token) for token in tokens])
        caption = caption[:12]
        l = len(caption)
        for i in range(len(caption), 13):
            caption.append(self.vocab(self.vocab.end_word))
        caption = torch.Tensor(caption).long()

        # return pre-processed image and caption tensors
        return image, caption
        
    def __getitem__(self, index):
        # obtain image and caption if in training mode
        ann_id = self.ids[index]
        caption = self.coco.anns[ann_id]['caption']
        img_id = self.coco.anns[ann_id]['image_id']
        path = self.coco.loadImgs(img_id)[0]['file_name']

        # Convert image to tensor and pre-process using transform
        image = Image.open(os.path.join(self.img_folder, path)).convert('RGB')
        image = self.transform(image)

        # Convert caption to tensor of word ids.
        tokens = nltk.tokenize.word_tokenize(str(caption).lower())
        caption = []
        caption.append(self.vocab(self.vocab.start_word))
        caption.extend([self.vocab(token) for token in tokens])
        caption = caption[:12]
        for i in range(len(caption), 13):
            caption.append(self.vocab(self.vocab.end_word))
        caption = torch.Tensor(caption).long()

        # return pre-processed image and caption tensors
        return image, caption


    def __len__(self):
        return len(self.ids)