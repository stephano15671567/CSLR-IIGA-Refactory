#############################################
#                                           #
# Load sequential data from PHOENIX-2014    #
#                                           #
#############################################

from __future__ import print_function, division
import os
from re import T
import torch
import pandas as pd
import numpy as np
import _pickle as pickle
import matplotlib.pyplot as plt
import cv2
import random
from skimage import io
import gzip

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

from tools.phoenix_cleanup import clean_phoenix_2014
from tools.indexs_list import idxs


#Ignore warnings
import warnings
warnings.filterwarnings("ignore")

def collate_fn(data, fixed_padding=None, pad_index=1232):
    """Creates mini-batch tensors w/ same length sequences by performing padding to the sequecenses.
    We should build a custom collate_fn to merge sequences w/ padding (not supported in default).
    Seqeuences are padded to the maximum length of mini-batch sequences (dynamic padding), else pad
    all Sequences to a fixed length.

    Returns:
        hand_seqs: torch tensor of shape (batch_size, padded_length).
        hand_lengths: list of length (batch_size); 
        src_seqs: torch tensor of shape (batch_size, padded_length).
        src_lengths: list of length (batch_size); 
        trg_seqs: torch tensor of shape (batch_size, padded_length).
        trg_lengths: list of length (batch_size); 
    """

    def pad(sequences, t):
        lengths = [len(seq) for seq in sequences]

        #For sequence of images
        if(t=='source'):
            #Retrieve shape of single sequence
            #(seq_length, channels, n_h, n_w)
            seq_shape = sequences[0].shape
            if(fixed_padding):
                padded_seqs = fixed_padding
                padded_seqs = torch.zeros(len(sequences), fixed_padding, seq_shape[1], seq_shape[2], seq_shape[3]).type_as(sequences[0])
            else:
                padded_seqs = torch.zeros(len(sequences), max(lengths), seq_shape[1], seq_shape[2], seq_shape[3]).type_as(sequences[0])

        #For sequence of words
        elif(t=='target'):
            padded_seqs = np.full((len(sequences), max(lengths)), fill_value=pad_index, dtype=int)

        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = seq[:end]

        return padded_seqs, lengths

    src_seqs = []
    trg_seqs = []
    right_hands = []
    left_hands = []

    for element in data:
        src_seqs.append(element['images'])
        trg_seqs.append(element['translation'])

        right_hands.append(element['right_hands'])

    #pad sequences
    src_seqs, src_lengths = pad(src_seqs, 'source')
    trg_seqs, trg_lengths = pad(trg_seqs, 'target')

    #pad hand sequences
    if(type(right_hands[0]) != type(None)):
        hand_seqs, hand_lengths = pad(right_hands, 'source')
    else:
        hand_seqs = None
        hand_lengths = None

    return src_seqs, src_lengths, trg_seqs, trg_lengths, hand_seqs, hand_lengths


#From abstract function Dataset
class PhoenixDataset(Dataset):
    """Sequential Sign language images dataset."""

    def __init__(self, csv_file, root_dir, segment_path, lookup_table, random_drop, uniform_drop, istrain, transform=None,rescale=224, sos_index=1, eos_index=2, unk_index=0, fixed_padding=None, hand_dir=None, hand_transform=None, channels=3):

        #Get data
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.segment_path= segment_path
        self.hand_dir = hand_dir
        self.random_drop = random_drop
        self.uniform_drop = uniform_drop
        self.transform = transform
        self.hand_transform = hand_transform
        self.istrain = istrain
        self.rescale = rescale

        self.channels = channels

        #index used for eos token and unk
        self.eos_index = eos_index
        self.unk_index = unk_index
        self.sos_index = sos_index

        #Retrieve lookup table dic from path
        with open(lookup_table, 'rb') as pickle_file:
            self.lookup_table = pickle.load(pickle_file)


    def __len__(self):
        #Return size of dataset
        return len(self.annotations)

    def __getitem__(self, idx):
        # 1. Obtener el nombre y construir rutas
        name = self.annotations.iloc[idx, 0].split('|')[0]
        path_con_uno = os.path.join(self.root_dir, name, '1')
        path_directo = os.path.join(self.root_dir, name)
        
        # Elegir la ruta que sí existe
        seq_name = path_con_uno if os.path.exists(path_con_uno) else path_directo
        segments_name = os.path.join(self.segment_path, name)

        # --- INICIALIZACIÓN CRÍTICA (Evita el UnboundLocalError) ---
        trsf_images = None
        hand_images = None
        
        # 2. Verificar existencia de archivos
        if not os.path.exists(seq_name):
            raise FileNotFoundError(f"No existe la carpeta: {seq_name}")
            
        files = sorted([f for f in os.listdir(seq_name) if f.endswith('.png')])
        if not files:
            raise RuntimeError(f"Carpeta vacía: {seq_name}")

        # 3. Determinar índices y longitudes
        if self.istrain:
            indexs = idxs(len(files), random_drop=self.random_drop, uniform_drop=self.uniform_drop)
        else:
            drop_val = self.random_drop if self.random_drop else self.uniform_drop
            indexs = idxs(len(files), random_drop=None, uniform_drop=drop_val)
        
        seq_length = len(indexs)

        # 4. Crear los tensores base
        trsf_images = torch.zeros((seq_length, self.channels, self.rescale, self.rescale))
        if self.hand_dir:
            hand_path = os.path.join(self.hand_dir, name)
            hand_images = torch.zeros((seq_length, self.channels, 112, 112))

        # 5. Parámetros de Crop aleatorio (solo si es train)
        w1 = random.randint(0, 256 - 224)
        h1 = random.randint(0, 256 - 224)

        # 6. Bucle de carga de imágenes
        for i, ind in enumerate(indexs):
            img = files[ind]
            img_name = os.path.join(seq_name, img)
            
            # Cargar imagen principal
            image = cv2.imread(img_name)
            if image is None: continue
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Cargar segmentación
            seg_name = os.path.join(segments_name, img.replace('.png', '.npy.gz'))
            if os.path.exists(seg_name):
                with gzip.open(seg_name, 'rb') as f:
                    segmentation = np.load(f)
                segmentation = cv2.resize(segmentation.astype(np.uint8), (224, 224), interpolation=cv2.INTER_NEAREST)
                segm_2class = np.repeat(segmentation[..., np.newaxis], 3, axis=2)
                image_resized = cv2.resize(image, (224, 224))
                annotated_image = (image_resized * segm_2class).astype(np.uint8)
            else:
                annotated_image = cv2.resize(image, (224, 224))

            # Aplicar Resize final a 256 y Crop a 224
            annotated_image = cv2.resize(annotated_image, (256, 256))
            if self.istrain:
                annotated_image = annotated_image[h1:h1 + 224, w1:w1 + 224, :]
            else:
                annotated_image = annotated_image[16:16 + 224, 16:16 + 224, :]
            
            trsf_images[i] = self.transform(annotated_image)

            # Lógica de manos
            if self.hand_dir:
                hand_file = os.path.join(hand_path, f'images{ind:04d}.png')
                if os.path.exists(hand_file):
                    h_img = io.imread(hand_file)
                    if h_img.ndim == 2: h_img = np.stack([h_img] * 3, axis=-1)
                    hand_images[i] = self.hand_transform(h_img[:, :, :self.channels])

        # 7. Procesar traducción (Glosas)
        translation = self.annotations.iloc[idx, 0].split('|')[-1]
        translation = clean_phoenix_2014(translation).split(' ')
        trans = [self.lookup_table.get(word, self.unk_index) for word in translation]

        return {'images': trsf_images, 'right_hands': hand_images, 'translation': trans}


# Helper function to show a batch
def show_batch(sample_batched):
    """Show sequence of images with translation for a batch of samples."""

    images_batch, images_length, trans_batch, trans_length = \
            sample_batched
    batch_size = len(images_batch)
    im_size = images_batch.size(2)

    #Show only one sequence of the batch
    grid = utils.make_grid(images_batch[0, :images_length[0]])
    grid = grid.numpy()
    return np.transpose(grid, (1,2,0))


#Use this to subtract mean from each pixel measured from PHOENIX-T dataset
#Note: means has been subtracted from 227x227 images, this has been provided by camgoz
class SubtractMeans(object):
    def __init__(self, path, rescale):
        #NOTE: Newest np versions default value allow_pickle=False
        self.mean = np.load(path, allow_pickle=True)
        self.mean = self.mean.astype('uint8')
        self.rescale = rescale

    def __call__(self, image):

        #No need to resize (take long time..)
        #image = cv2.resize(image,(self.mean.shape[0], self.mean.shape[1]))
        assert image.shape == self.mean.shape
        image -= self.mean
        #image = cv2.resize(image,(self.rescale, self.rescale))

        return image


def loader(csv_file, root_dir, segment_path, lookup, rescale, batch_size, num_workers, random_drop, uniform_drop, show_sample, istrain=False, mean_path='FulFrame_Mean_Image_227x227.npy', fixed_padding=None, hand_dir=None, data_stats=None, hand_stats=None, channels=3):

    #Note: when using random cropping, this with reshape images with randomCrop size instead of rescale
    if(istrain):
        if(data_stats):
            trans = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomAffine(10),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                transforms.Resize((rescale, rescale)),
                transforms.ToTensor(),
                transforms.Normalize(mean=data_stats['mean'], std=data_stats['std'])
                ])
        else:
            trans = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomAffine(10),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                transforms.Resize((rescale, rescale)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
            
        if(hand_stats):
            hand_trans = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.RandomAffine(10),
                    transforms.Resize((112, 112)),
                    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=hand_stats['mean'], std=hand_stats['std'])
                    ])
        else:
            hand_trans = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.RandomAffine(10),
                    transforms.Resize((112, 112)),
                    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
                
    else:
        if(data_stats):
            trans = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((rescale, rescale)),
                transforms.ToTensor(),
                transforms.Normalize(mean=data_stats['mean'], std=data_stats['std'])
                ])
            
        else:
             trans = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((rescale, rescale)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])

        if(hand_stats):
            hand_trans = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize((112, 112)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=hand_stats['mean'], std=hand_stats['std'])
                    ])
        else:
            hand_trans = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize((112, 112)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                    ])

    ##Iterate through the dataset and apply data transformation on the fly

    #Apply data augmentation to avoid overfitting
    transformed_dataset = PhoenixDataset(csv_file=csv_file,
                                            root_dir=root_dir,
                                            segment_path=segment_path,
                                            lookup_table=lookup,
                                            random_drop=random_drop,
                                            uniform_drop=uniform_drop,
                                            transform=trans,
                                            rescale=rescale,
                                            istrain=istrain,
                                            hand_dir=hand_dir,
                                            hand_transform=hand_trans,
                                            channels = channels
                                            )

    size = len(transformed_dataset)

    #Iterate in batches
    #Note: put num of workers to 0 to avoid memory saturation. Fix shuffle to be istrain
    dataloader = DataLoader(transformed_dataset, batch_size=batch_size,
                            shuffle=istrain, num_workers=num_workers, collate_fn=collate_fn)

    #Show a sample of the dataset
    if(show_sample and istrain):
        for i_batch, sample_batched in enumerate(dataloader):
            #plt.figure()
            img = show_batch(sample_batched)
            plt.axis('off')
            plt.ioff()
            plt.imshow(img)
            #plt.show()
            plt.savefig('data_sample.png')
            break

    return dataloader, size