import math
import torch
import numpy as np


def reserve_schedule_sampling_exp(itr, batch_size, args):
    if itr < args.r_sampling_step_1:
        r_eta = 0.5
    elif itr < args.r_sampling_step_2:
        r_eta = 1.0 - 0.5 * math.exp(-float(itr - args.r_sampling_step_1) / args.r_exp_alpha)
    else:
        r_eta = 1.0

    if itr < args.r_sampling_step_1:
        eta = 0.5
    elif itr < args.r_sampling_step_2:
        eta = 0.5 - (0.5 / (args.r_sampling_step_2 - args.r_sampling_step_1)) * (itr - args.r_sampling_step_1)
    else:
        eta = 0.0

    r_random_flip = np.random.random_sample(
        (batch_size, args.pre_seq_length - 1))
    r_true_token = (r_random_flip < r_eta)

    random_flip = np.random.random_sample(
        (batch_size, args.aft_seq_length - 1))
    true_token = (random_flip < eta)

    # One decision per sample/time step; spatial and channel axes broadcast.
    tokens = np.concatenate((r_true_token, true_token), axis=1)
    return torch.as_tensor(tokens, dtype=torch.float32, device=args.device).reshape(
        batch_size, args.total_length - 2, 1, 1, 1)


def schedule_sampling(eta, itr, batch_size, args, update_eta=True):
    if not args.scheduled_sampling:
        return 0.0, torch.zeros(
            (batch_size, args.aft_seq_length - 1, 1, 1, 1), device=args.device)

    if itr < args.sampling_stop_iter:
        if update_eta:
            eta -= args.sampling_changing_rate
    else:
        eta = 0.0
    random_flip = np.random.random_sample(
        (batch_size, args.aft_seq_length - 1))
    true_token = (random_flip < eta)
    return eta, torch.as_tensor(true_token, dtype=torch.float32, device=args.device).reshape(
        batch_size, args.aft_seq_length - 1, 1, 1, 1)


def reshape_patch(img_tensor, patch_size):
    assert 5 == img_tensor.ndim
    batch_size, seq_length, img_height, img_width, num_channels = img_tensor.shape
    a = img_tensor.reshape(batch_size, seq_length,
                                img_height//patch_size, patch_size,
                                img_width//patch_size, patch_size,
                                num_channels)
    b = a.transpose(3, 4)
    patch_tensor = b.reshape(batch_size, seq_length,
                                  img_height//patch_size,
                                  img_width//patch_size,
                                  patch_size*patch_size*num_channels)
    return patch_tensor


def reshape_patch_back(patch_tensor, patch_size):
    batch_size, seq_length, patch_height, patch_width, channels = patch_tensor.shape
    img_channels = channels // (patch_size*patch_size)
    a = patch_tensor.reshape(batch_size, seq_length,
                                  patch_height, patch_width,
                                  patch_size, patch_size,
                                  img_channels)
    b = a.transpose(3, 4)
    img_tensor = b.reshape(batch_size, seq_length,
                                patch_height * patch_size,
                                patch_width * patch_size,
                                img_channels)
    return img_tensor
