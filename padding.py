import torch.nn.functional as F

def pad_to_size(img, target_height=160, target_width=288):
    _, h, w = img.shape
    pad_h = target_height - h
    pad_w = target_width - w
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    return F.pad(img, (pad_left, pad_right, pad_top, pad_bottom))  # (left, right, top, bottom)


def remove_padding(tensor, original_height=160, original_width=272, target_height=160, target_width=288):
    pad_h = target_height - original_height
    pad_w = target_width - original_width
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    return tensor[..., pad_top:target_height - pad_bottom, pad_left:target_width - pad_right]
