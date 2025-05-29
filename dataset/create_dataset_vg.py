import os
import math
import numpy as np
import hdf5plugin
import h5py
from tqdm import tqdm
import argparse
import torch
# This script creates the vg files in npy format for faster opening.
# TODO: make this script indipendent from the type of event representation, so not only VGs.
 
# params TODO: create yaml file for these.
IMAGE_CHANGE_RANGE = 1
RECTIFY_EVENTS = True
EVENT_HEIGHT = 480
EVENT_WIDTH = 640
EVENT_BINS = 1
EVENT_CLIP_RANGE = None
OUTPUT_NUM = 1

def tensor_normalize_to_range(tensor, min_val, max_val):
    tensor_min = torch.min(tensor)
    tensor_max = torch.max(tensor)
    tensor = (tensor - tensor_min) / (tensor_max - tensor_min + 1e-8) * (max_val - min_val) + min_val
    return tensor

def events_norm(events, clip_range=1.0, final_range=1.0, enforce_no_events_zero=False):
    # assert clip_range > 0

    if clip_range == 'auto':
        n_mean = events[events < 0].mean() * 1.5  # tensor(-0.7947)
        p_mean = events[events > 0].mean() * 1.5  # tensor(1.2755)
    else:
        nonzero_ev = (events != 0)
        num_nonzeros = nonzero_ev.sum()
        if num_nonzeros > 0:
            mean = events.sum() / num_nonzeros
            stddev = torch.sqrt((events ** 2).sum() / num_nonzeros - mean ** 2)
            mask = nonzero_ev.float()
            events = mask * (events - mean) / (stddev + 1e-8)
        n_mean = -clip_range
        p_mean = clip_range
    '''mask = torch.nonzero(events, as_tuple=True)
    if mask[0].size()[0] > 0:
        mean = events[mask].mean()
        std = events[mask].std()
        if std > 0:
            events[mask] = (events[mask] - mean) / std
        else:
            events[mask] = events[mask] - mean'''

    if enforce_no_events_zero:
        events_smaller_0 = events.detach().clone()
        events[events < 0] = 0
        # events = torch.clamp(events, 0, clip_range)
        events = torch.clamp(events, 0, p_mean)
        events = tensor_normalize_to_range(events, min_val=0, max_val=final_range)
        events_smaller_0[events_smaller_0 > 0] = 0
        # events_smaller_0 = torch.clamp(events_smaller_0, -clip_range, 0)
        events_smaller_0 = torch.clamp(events_smaller_0, n_mean, 0)

        events_smaller_0 = tensor_normalize_to_range(events_smaller_0, min_val=-final_range, max_val=0)
        events += events_smaller_0
    else:
        events = torch.clamp(events, -clip_range, clip_range) * final_range
        events = events / clip_range * final_range
    return events

def events_to_voxel_grid(time, x, y, pol, width, height, num_bins, normalize_flag=False):
    """
    This function generates a voxel grid from events.

    Args:
        time: (N,) tensor of event timestamps
        x: (N,) tensor of event x coordinates
        y: (N,) tensor of event y coordinates
        pol: (N,) tensor of event polarities (0 or 1)
        width: width of the voxel grid
        height: height of the voxel grid
        num_bins: number of bins in the voxel grid
        normalize_flag: whether to normalize the voxel grid

    Returns:
        voxel_grid: (num_bins, height, width) tensor of voxel grid
    """
    assert x.shape == y.shape == pol.shape == time.shape
    assert x.ndim == 1

    voxel_grid = torch.zeros((num_bins, height, width), dtype=torch.float, requires_grad=False)
    C, H, W = voxel_grid.shape

    with torch.no_grad():
        voxel_grid = voxel_grid.to(pol.device)
        voxel_grid = voxel_grid.clone()

        t_norm = time
        t_norm = (C - 1) * (t_norm - t_norm[0]) / (t_norm[-1] - t_norm[0])

        x0 = x.int()
        y0 = y.int()
        t0 = t_norm.int()

        value = 2 * pol - 1

        for xlim in [x0, x0 + 1]:
            for ylim in [y0, y0 + 1]:
                for tlim in [t0, t0 + 1]:
                    mask = (xlim < W) & (xlim >= 0) & (ylim < H) & (ylim >= 0) & (tlim >= 0) & (tlim < num_bins)
                    interp_weights = value * (1 - (xlim - x).abs()) * (1 - (ylim - y).abs()) * (
                                1 - (tlim - t_norm).abs())

                    index = H * W * tlim.long() + \
                            W * ylim.long() + \
                            xlim.long()

                    voxel_grid.put_(index[mask], interp_weights[mask], accumulate=True)

        if normalize_flag:
            mask = torch.nonzero(voxel_grid, as_tuple=True)
            if mask[0].size()[0] > 0:
                mean = voxel_grid[mask].mean()
                std = voxel_grid[mask].std()
                if std > 0:
                    voxel_grid[mask] = (voxel_grid[mask] - mean) / std
                else:
                    voxel_grid[mask] = voxel_grid[mask] - mean

    return voxel_grid

def get_events_vg(events_h5, rectify_map, events_finish_index, events_start_index):
        events_t = np.asarray(events_h5['events/{}'.format('t')][events_start_index: events_finish_index + 1])
        events_x = np.asarray(events_h5['events/{}'.format('x')][events_start_index: events_finish_index + 1])
        events_y = np.asarray(events_h5['events/{}'.format('y')][events_start_index: events_finish_index + 1])
        events_p = np.asarray(events_h5['events/{}'.format('p')][events_start_index: events_finish_index + 1])

        events_t = (events_t - events_t[0]).astype('float32')
        events_t = torch.from_numpy((events_t / events_t[-1]))
        events_p = torch.from_numpy(events_p.astype('float32'))
        if RECTIFY_EVENTS:
            assert len(rectify_map) > 0
            xy_rect = rectify_map[events_y, events_x]
            events_x = xy_rect[:, 0]
            events_y = xy_rect[:, 1]
        events_x = torch.from_numpy(events_x.astype('float32'))
        events_y = torch.from_numpy(events_y.astype('float32'))
        events_vg = events_to_voxel_grid(events_t, events_x, events_y, events_p, EVENT_WIDTH, EVENT_HEIGHT,
                                         num_bins=EVENT_BINS, normalize_flag=False)

        if EVENT_CLIP_RANGE is not None:
            events_clip_range = random.uniform(self.events_clip_range[0], self.events_clip_range[1])
        else:
            events_clip_range = (events_finish_index - events_start_index) / 500000 * 1.5
            # events_clip_range = 'auto'
        # print('events_clip_range: {}'.format(events_clip_range))
        events_vg = events_norm(events_vg, clip_range=events_clip_range, final_range=1.0, enforce_no_events_zero=True)
        return events_vg

def generate_voxel_grid_cache(dst_path, events_num, image_change_num=1, warp_images_flag=False):

    file_list = os.listdir(dst_path)
    file_list.sort()
    # images_to_events_index_path, events_h5_path, events_num, events_path_txt

    for file_name in file_list:
        if 'zurich_city_' not in file_name:
            continue
        city_name = file_name.split('zurich_city_')[-1]
        print('processing {}...'.format(file_name))

        images_to_events_index_path = '{}{}/images/images_to_events_index.txt'.format(dst_path, file_name)
        images_to_events_index = np.loadtxt(images_to_events_index_path, dtype='int64')

        events = h5py.File('{}{}/events/left/events.h5'.format(dst_path, file_name), 'r')
        rectify_map = []
        if RECTIFY_EVENTS:
            rectify_map = h5py.File('{}{}/events/left/rectify_map.h5'.format(dst_path, file_name), 'r')
            rectify_map = np.asarray(rectify_map['rectify_map'])

        events_total_num = int(events['events/t'].shape[0])

        images_list_path = '{}{}/images/left/rectified/'.format(dst_path, file_name)
        if not warp_images_flag:
            images_list = os.listdir(images_list_path)
            images_list.sort()

            assert images_to_events_index.shape[0] == len(images_list)
        else:
            images_list = []
            for i in range(images_to_events_index.shape[0]):
                images_list.append('{:06d}.png'.format(i))

        vg_save_path = '{}{}/events/events_vg/'.format(dst_path, file_name)
        if not os.path.isdir(vg_save_path):
            os.mkdir(vg_save_path)
            print("created directory for saving events in: " + file_name)
        with tqdm(total=len(images_list), desc="generating cache...") as pbar:
            for i, images_name in enumerate(images_list):
                if warp_images_flag:
                    if not os.path.isfile('{}{}'.format(images_list_path, images_name).replace('images/left/rectified', 'warp_images')):
                        continue
                if events_num < images_to_events_index[i] and i >= image_change_num:
                    images_path = '{}{}'.format(images_list_path, images_name)
                    event_finish_index = int(images_to_events_index[i])
                    event_start_index = int(images_to_events_index[i - IMAGE_CHANGE_RANGE ])
                    event_vg = torch.zeros((OUTPUT_NUM, EVENT_BINS, EVENT_HEIGHT, EVENT_WIDTH))
                    event_vg[0,:] = get_events_vg(events, rectify_map, event_finish_index, event_start_index)
                    event_file_path = vg_save_path + images_name.split('.')[0] + '.npy'
                    np.save(event_file_path, event_vg)
                    pbar.update(1)
                    #dataset_txt.write(images_path + ' ' + str(images_to_events_index[i]) + '\n')


if __name__ == '__main__':
    print('Creating npy files for events voxel grid')

    # root path of the DSEC_Night dataset

    dir_path = os.path.dirname(os.path.realpath(__file__))
    root_dir = dir_path.split("dataset")[0] + "/data/DSEC_Night/"

    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, default=root_dir)
    opt = parser.parse_args()

    generate_voxel_grid_cache(dst_path=opt.root_dir,
                        events_num=0, image_change_num=1,
                        warp_images_flag=False)
