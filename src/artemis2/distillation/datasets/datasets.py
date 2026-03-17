import os
import csv
import glob
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils import data
from itertools import compress
import ast


class VideoRecord(object):
    def __init__(self, row):
        self._data = row

    @property
    def path(self):
        return self._data[0]

    @property
    def num_frames(self):
        return int(self._data[1])

    @property
    def label(self):
        return self._data[2]
    

class ActionDataset(data.Dataset):
    def __init__(self, total_length):
        self.total_length = total_length
        self.video_list = []
        self.random_shift = False

    def _sample_indices(self, num_frames):
        if num_frames <= self.total_length:
            indices = np.linspace(0, num_frames - 1, self.total_length, dtype=int)
        else:
            ticks = np.linspace(0, num_frames, self.total_length + 1, dtype=int)
            if self.random_shift:
                indices = ticks[:-1] + np.random.randint(ticks[1:] - ticks[:-1])
            else:
                indices = ticks[:-1] + (ticks[1:] - ticks[:-1]) // 2
        return indices

    @staticmethod
    def _load_image(directory, image_name):
        return [Image.open(os.path.join(directory, image_name)).convert('RGB')]
    
    def __getitem__(self, index):
        record = self.video_list[index]
        image_names = self.file_list[index]
        indices = self._sample_indices(record.num_frames)
        return self._get(record, image_names, indices)
    
    def __len__(self):
        return len(self.video_list)


"""class ActionDataset(data.Dataset):
    def __init__(self, total_length):
        self.total_length = total_length
        self.video_list = []
        self.random_shift = False
        self.ak_flow_indexes = pd.read_csv('ak_cosine_clip_indexes.csv')

    def _sample_indices(self, video_id):
        row = self.ak_flow_indexes[self.ak_flow_indexes['id'] == video_id].iloc[0]
        indexes = ast.literal_eval(row['indexes'])
        #indexes = indexes[1:]
        #if len(indexes) < 16:
        #    indexes = np.linspace(0, len(indexes) - 1, 16)
        #    indexes = np.round(indexes).astype(int)
        last_index = indexes[-1]
        while len(indexes) < 16:
            indexes.append(last_index)
        return indexes

    @staticmethod
    def _load_image(directory, image_name):
        return [Image.open(os.path.join(directory, image_name)).convert('RGB')]

    def __getitem__(self, index):
        record = self.video_list[index]
        image_names = self.file_list[index]
        indices = self._sample_indices(record.path.split('/')[-1])
        return self._get(record, image_names, indices)

    def __len__(self):
        return len(self.video_list)"""
        

class AnimalKingdom(ActionDataset):
    def __init__(self, path, act_dict, total_length=12, transform=None, random_shift=False, mode='train'):
        super(AnimalKingdom, self).__init__(total_length)  # TODO: added
        self.path = path
        self.total_length = total_length
        self.transform = transform
        self.random_shift = random_shift
        self.mode = mode
        self.anno_path = os.path.join(self.path, 'action_recognition', 'annotation', mode + '_light.csv')
        print(self.anno_path, flush=True)
        self.act_dict = act_dict
        self.num_classes = len(act_dict)
        try:
            self.video_list, self.file_list = self._parse_annotations()
        except OSError:
            print('ERROR: Could not read annotation file "{}"'.format(self.anno_path))
            raise

    def _parse_annotations(self):
        video_list = []
        file_list = []
        with open(self.anno_path) as f:
            if self.anno_path.split('/')[-1] in ['train_light.csv', 'val_light.csv']:
                print(f, flush=True)
                reader = csv.DictReader(f, delimiter=';')
            else:
                reader = csv.DictReader(f, delimiter=',')
            #reader = csv.DictReader(f, delimiter=';')
            for row in reader:
                ovid = row['video_id']
                labels = row['labels']
                path = os.path.join(self.path, 'action_recognition', 'dataset', 'image', ovid)
                files = sorted(os.listdir(path))
                file_list += [files]
                count = len(files)
                labels = [int(l) for l in labels.split(',')]
                video_list += [VideoRecord([path, count, labels])]
        return video_list, file_list

    def _get(self, record, image_names, indices):
        images = list()
        summaries_align = np.load(
            os.path.join('ak_embeddings_align_sharegpt4', image_names[0].split('.')[0].split('_')[0] + '.npy'))
        summaries = np.load(os.path.join('ak_embeddings_flava_sharegpt4', image_names[0].split('.')[0].split('_')[0] + '.npy'))
        for idx in indices:
            try:
                img = self._load_image(record.path, image_names[idx])
            except:
                print('ERROR: Could not read image "{}"'.format(os.path.join(record.path, image_names[idx])))
                print('invalid indices: {}'.format(indices))
                raise
            images.extend(img)
        process_data = self.transform(images)
        process_data = process_data.view((self.total_length, -1) + process_data.size()[-2:])
        label = np.zeros(self.num_classes)
        label[record.label] = 1.0
        return process_data, np.array(summaries_align), np.array(summaries), label


class MammalNet(ActionDataset):
    def __init__(self, path, act_dict, total_length=12, transform=None, random_shift=False, mode='train'):
        self.path = path
        self.total_length = total_length
        self.transform = transform
        self.random_shift = random_shift
        self.mode = mode

        if mode.isnumeric():
            self.anno_path = os.path.join(self.path, f'annotation/{mode}_light.csv')
        else:
            self.anno_path = os.path.join(self.path, f'annotation/{mode}ing.csv')

        self.act_dict = act_dict
        self.num_classes = len(act_dict)
        try:
            self.video_list, self.file_list = self._parse_annotations()
        except OSError:
            print('ERROR: Could not read annotation file "{}"'.format(self.anno_path))
            raise

    def _parse_annotations(self):
        video_list = []
        file_list = []
        with open(self.anno_path) as f:
            reader = csv.DictReader(f, delimiter=',')
            for row in reader:
                ovid = row['video_id']
                labels = row['behavior']
                path = os.path.join(self.path, 'dataset', 'image', ovid)
                files = sorted(os.listdir(path))
                file_list += [files]
                count = len(files)
                labels = int(labels)
                video_list += [VideoRecord([path, count, labels])]
        return video_list, file_list

    def _get(self, record, image_names, indices):
        images = list()
        summaries = np.load(os.path.join('mammalnet_embeddings_align_sharegpt', record.path.split('/')[-1] + '.npy'))
        for idx in indices:
            try:
                img = self._load_image(record.path, image_names[idx])
                #print('CORRECT: ', record.path, image_names[idx])
            except:
                print(record.path, image_names, idx, flush=True)
                print('ERROR: Could not read image "{}"'.format(os.path.join(record.path, image_names[idx])))
                print('invalid indices: {}'.format(indices))
                raise
            images.extend(img)
        process_data = self.transform(images)
        process_data = process_data.view((self.total_length, -1) + process_data.size()[-2:])
        #label = np.zeros(self.num_classes)  # need to fix this hard number
        #label[record.label] = 1.0
        return process_data, np.array(summaries), record.label


class BaboonLandDataset(ActionDataset):
    def __init__(self, path, act_dict, total_length=12, transform=None, random_shift=False, mode='train'):
        """
        Initialize the BaboonLand dataset.

        Args:
            path (str): Base path to the dataset
            act_dict (dict): Dictionary mapping action labels to integer indices
            total_length (int, optional): Number of frames to sample. Defaults to 12.
            transform (callable, optional): Optional transform to be applied on a list of images. Defaults to None.
            random_shift (bool, optional): Whether to randomly shift frame sampling. Defaults to False.
            mode (str, optional): Dataset mode, either 'train' or 'val'. Defaults to 'train'.
        """
        # Call the parent class constructor
        super().__init__(total_length)

        self.path = path
        self.transform = transform
        self.random_shift = random_shift
        self.mode = mode

        # Construct annotation path
        if mode.isnumeric():
            self.anno_path = os.path.join(self.path, f'annotation/{mode}_light.csv')
        else:
            self.anno_path = os.path.join(self.path, f'annotation/{mode}.csv')

        self.act_dict = act_dict
        self.num_classes = len(act_dict)

        try:
            self.video_list, self.file_list = self._parse_annotations()
        except OSError:
            print(f'ERROR: Could not read annotation file "{self.anno_path}"')
            raise

    def _parse_annotations(self):
        """
        Parse the annotation CSV file and prepare video records.

        Returns:
            Tuple of video_list and file_list
        """
        # Read the annotation CSV
        if self.mode.isnumeric():
            df = pd.read_csv(self.anno_path, delimiter=',')
        else:
            df = pd.read_csv(self.anno_path, delimiter=' ')
        video_list = []
        file_list = []

        # Group by unique video_id to handle videos with multiple frames
        for video_id, video_group in df.groupby('video_id'):
            # Sort frames by frame_id to ensure correct order
            video_group = video_group.sort_values('frame_id')

            # Get unique video path (assuming first row represents the video path)
            video_path = os.path.dirname(video_group['path'].iloc[0])

            # Get all frame filenames for this video
            frame_files = video_group['path'].apply(os.path.basename).tolist()

            # Prepare labels
            labels = video_group['labels'].values
            unique_labels = np.unique(labels)

            # Create one-hot encoded labels for all frames
            num_frames = len(frame_files)
            label_matrix = np.zeros((num_frames, self.num_classes), dtype=bool)
            for i, label in enumerate(labels):
                label_matrix[i, label] = 1

            # Add to lists
            file_list.append(frame_files)
            video_list.append(VideoRecord(
                [
                    os.path.join(self.path, 'dataset', 'image', video_path),
                    num_frames,
                    label_matrix
                ]
            ))

        return video_list, file_list

    def _get(self, record, image_names, indices):
        """
        Retrieve and process images for the given indices.

        Args:
            record (VideoRecord): Video record containing path and metadata
            image_names (list): List of image filenames
            indices (np.ndarray): Indices of frames to retrieve

        Returns:
            Tuple of processed images and corresponding labels
        """
        images = []
        summaries = np.load(os.path.join('baboonland_embeddings_align_sharegpt', record.path.split('/')[-1] + '.npy'))
        for idx in indices:
            try:
                img = self._load_image(record.path, image_names[idx])
            except OSError:
                print(f'ERROR: Could not read image "{record.path}/{image_names[idx]}"')
                print(f'Invalid indices: {indices}')
                raise
            images.extend(img)

        # Apply transformations
        process_data = self.transform(images)
        process_data = process_data.view((self.total_length, -1) + process_data.size()[-2:])

        # Get labels - take the max across sampled frames (any frame with the label)
        label = record.label[indices].any(0).astype(np.float32)

        return process_data, np.array(summaries), label


class Charades(ActionDataset):
    def __init__(self, path, act_dict, total_length=12, transform=None, random_shift=False, mode='train'):
        self.path = path
        self.total_length = total_length
        self.transform = transform
        self.random_shift = random_shift
        self.mode = mode
        self.anno_path = os.path.join(self.path, 'Charades', 'Charades_v1_' + mode + '.csv')
        self.act_dict = act_dict
        self.num_classes = len(act_dict)
        try:
            self.video_list, self.file_list = self._parse_annotations()
        except OSError:
            print('ERROR: Could not read annotation file "{}"'.format(self.anno_path))
            raise

    @staticmethod
    def _cls2int(x):
        return int(x[1:])

    def _parse_annotations(self):
        video_list = []
        file_list = []
        with open(self.anno_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                actions = row['actions']
                if actions == '': continue
                vid = row['id']
                path = os.path.join(self.path, 'Charades_v1_rgb', vid)
                files = sorted(os.listdir(path))
                num_frames = len(files)
                fps = num_frames / float(row['length'])
                labels = np.zeros((num_frames, self.num_classes), dtype=bool)
                actions = [[self._cls2int(c), float(s), float(e)] for c, s, e in [a.split(' ') for a in actions.split(';')]]
                for frame in range(num_frames):
                    for ann in actions:
                        if frame/fps > ann[1] and frame/fps < ann[2]: labels[frame, ann[0]] = 1
                idx = labels.any(1)
                num_frames = idx.sum()
                file_list += [list(compress(files, idx.tolist()))]
                video_list += [VideoRecord([path, num_frames, labels[idx]])]
        return video_list, file_list

    def _get(self, record, image_names, indices):
        images = list()
        for idx in indices:
            try:
                img = self._load_image(record.path, image_names[idx])
            except OSError:
                print('ERROR: Could not read image "{}"'.format(record.path))
                print('invalid indices: {}'.format(indices))
                raise
            images.extend(img)
        process_data = self.transform(images)
        process_data = process_data.view((self.total_length, -1) + process_data.size()[-2:])
        label = record.label[indices].any(0).astype(np.float32)
        return process_data, label


class Hockey(ActionDataset):
    def __init__(self, path, act_dict, total_length=12, transform=None, random_shift=False, mode='train', test_part=6, stride=10):
        self.path = path
        self.total_length = total_length
        self.transform = transform
        self.random_shift = random_shift
        self.mode = mode
        self.stride = stride
        self.clip_length = 30
        self.act_dict = act_dict
        self.num_classes = len(act_dict)
        self._parse_annotations(test_part, mode)

    def _parse_annotations(self, test_part, mode):
        all_dirs = glob.glob(os.path.join(self.path, 'period*-gray'))
        test_dirs = glob.glob(os.path.join(self.path, 'period*-' + str(test_part) + '-gray'))
        train_dirs = list(set(all_dirs) - set(test_dirs))
        if mode == 'train':
            self.video_list, self.file_list = self._files_labels(train_dirs)
        else:
            self.video_list, self.file_list = self._files_labels(test_dirs)

    def _files_labels(self, dirs):
        video_list = []
        file_list = []
        for dir in dirs:
            files = sorted(os.listdir(dir))
            with open(dir + '-label.txt') as f: labels = [[int(x) for x in line.split(',')] for line in f]
            for j in range(0, (len(files) - self.clip_length) + 1, self.stride):
                file_list += [files[j : j + self.clip_length]]
                video_list += [VideoRecord([dir, self.clip_length, np.array(labels[j : j + self.clip_length], dtype=bool)])]
        return video_list, file_list
    
    def _get(self, record, image_names, indices):
        images = list()
        for idx in indices:
            try:
                img = self._load_image(record.path, image_names[idx])
            except OSError:
                print('ERROR: Could not read image "{}"'.format(record.path))
                print('invalid indices: {}'.format(indices))
                raise
            images.extend(img)
        process_data = self.transform(images)
        process_data = process_data.view((self.total_length, -1) + process_data.size()[-2:])
        label = record.label[indices].any(0).astype(np.float32)
        return process_data, label


class Thumos14(ActionDataset):
    def __init__(self, path, act_dict, total_length=12, transform=None, random_shift=False, mode='train'):
        self.path = path
        self.total_length = total_length
        self.transform = transform
        self.random_shift = random_shift
        self.mode = mode
        self.act_dict = act_dict
        self.num_classes = len(act_dict)
        if mode == 'train':
            self.video_list, self.file_list = self._parse_train_annotations()
        else:
            self.video_list, self.file_list = self._parse_testval_annotations(mode)

    def _parse_train_annotations(self):
        path_frames = os.path.join(self.path, 'frames', 'train')
        paths_videos = glob.glob(os.path.join(path_frames, '*/*'))
        file_list = []
        video_list = []
        for path in paths_videos:
            files = sorted(os.listdir(path))
            num_frames = len(files)
            cls = self.act_dict.get(path.split('/')[8])
            file_list += [files]
            video_list += [VideoRecord([path, num_frames, cls])]
        return video_list, file_list

    def _parse_testval_annotations(self, mode):
        path_frames = os.path.join(self.path, 'frames', mode)
        paths_videos = sorted(glob.glob(os.path.join(path_frames, '*')))
        path_ants = os.path.join(self.path, 'annotations', mode)

        # consider the fps from the meta data
        from scipy.io import loadmat
        if mode == 'val':
            file_meta_data = os.path.join(path_ants, 'validation_set.mat')
            meta_key = 'validation_videos'
        elif mode == 'test':
            file_meta_data = os.path.join(path_ants, 'test_set_meta.mat')      
            meta_key = 'test_videos'
        fps = loadmat(file_meta_data)[meta_key][0]['frame_rate_FPS'].astype(int)

        video_fps = {}
        video_frames = {}
        video_num_frames = {}
        for i, path in enumerate(paths_videos):
            vid = path.split('/')[-1]
            files = sorted(os.listdir(path))
            video_fps[vid] = fps[i]
            video_frames[vid] = files
            num_frames = len(files)
            video_num_frames[vid] = num_frames
        file_list = []
        video_list = [] 
                
        for cls in self.act_dict.keys():
            path_ants_cls = os.path.join(path_ants, cls + '_' + mode + '.txt')
            with open(path_ants_cls, 'r') as f:
                lines = f.read().splitlines()
                for lin in lines:
                    vid, _, strt_sec, end_sec = lin.split(' ')
                    strt_frme = np.ceil(float(strt_sec) * video_fps[vid]).astype(int)
                    end_frme = np.floor(float(end_sec) * video_fps[vid]).astype(int)
                    frames_ = video_frames[vid][strt_frme:end_frme + 1]
                    num_frames = end_frme - strt_frme + 1
                    if len(frames_) != num_frames:
                        continue
                        # breakpoint()
                    file_list += [frames_]
                    path = os.path.join(path_frames, vid)
                    video_list += [VideoRecord([path, num_frames, self.act_dict.get(cls)])]
        return video_list, file_list

    def _get(self, record, image_names, indices):
        images = list()
        for idx in indices:
            try:
                img = self._load_image(record.path, image_names[idx])
            except OSError:
                print('ERROR: Could not read image "{}"'.format(record.path))
                print('invalid indices: {}'.format(indices))
                raise
            images.extend(img)
        process_data = self.transform(images)
        process_data = process_data.view((self.total_length, -1) + process_data.size()[-2:])
        return process_data, record.label
    

class Volleyball(ActionDataset):
    def __init__(self, path, act_dict, total_length=12, transform=None, random_shift=False, mode='train'):
        self.path = path
        self.total_length = total_length
        self.transform = transform
        self.random_shift = random_shift
        self.mode = mode
        self.act_dict = act_dict
        self.num_classes = len(act_dict)
        
        # dataset split
        train_set = [1, 3, 6, 7, 10, 13, 15, 16, 18, 22, 23, 31, 32, 36, 38, 39, 40, 41, 42, 48, 50, 52, 53, 54]
        val_set = [0, 2, 8, 12, 17, 19, 24, 26, 27, 28, 30, 33, 46, 49, 51]
        test_set = [4, 5, 9, 11, 14, 20, 21, 25, 29, 34, 35, 37, 43, 44, 45, 47]

        # clip_length = 41
        self.clip_length = 41

        # set the split
        if self.mode == 'train':
            self.set = train_set
        elif self.mode == 'val':
            self.set = val_set
        elif self.mode == 'test':
            self.set = test_set
        else:
            print('ERROR: invalid split')

        self.video_list, self.file_list = self._parse_annotations(self.path)

    def _parse_annotations(self, path):        
        video_list = []
        file_list = []
        for c in self.set:
            file = os.path.join(path, 'volleyball', str(c), 'annotations.txt')
            with open(file, 'r') as f: lines = f.read().splitlines()
            for line in lines:
                video_path = os.path.join(path, 'volleyball', str(c), line.split()[0].split('.')[0])
                labels = [self.act_dict.get(c.capitalize()) for c in list(set(line.split()[6::5]))]
                video_list += [VideoRecord([video_path, self.clip_length, labels])]
                file_list += [sorted(os.listdir(video_path))]
        return video_list, file_list

    def _get(self, record, image_names, indices):
        images = list()
        for idx in indices:
            try:
                img = self._load_image(record.path, image_names[idx])
            except OSError:
                print('ERROR: Could not read image "{}"'.format(record.path))
                print('invalid indices: {}'.format(indices))
                raise
            images.extend(img)
        process_data = self.transform(images)
        process_data = process_data.view((self.total_length, -1) + process_data.size()[-2:])
        label = np.zeros(self.num_classes)
        label[record.label] = 1.0
        return process_data, label
    

if __name__ == '__main__':
    path = '/home/adutta/Workspace/Datasets/Volleyball'
    volleyball_dataset = Volleyball(path, total_length=12, random_shift=True, mode='train')
    # import torchvision.transforms as transforms
    # transform = transforms.ToTensor()
    # thumos_dataset = Thumos14(path, total_length=15, random_shift=True, mode='train')
    # ind = thumos_dataset._sample_indices(150)
    # print(ind)
    # print(len(thumos_dataset))

    # charades_dataset.__getitem__(0)