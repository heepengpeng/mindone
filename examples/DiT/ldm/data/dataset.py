import mindspore.dataset as ds
import numpy as np
from PIL import Image
from mindspore.dataset import ImageFolderDataset, vision
from mindspore.dataset.transforms import transforms

from utils.util import Lambda


def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    pil_image = Image.fromarray(pil_image)
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


def build_dataset(args, rank_id, device_num):
    if args.distribute:
        sampler = ds.DistributedSampler(device_num, rank_id)
        dataset = ImageFolderDataset(args.data_path, sampler=sampler,
                                     num_parallel_workers=args.num_workers)
    else:
        dataset = ImageFolderDataset(args.data_path, shuffle=True)
    dataset = dataset.map(operations=[vision.Decode()], input_columns=['image'])
    dataset = dataset.map(
        operations=transforms.Compose([Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size))]),
        input_columns=['image'])
    dataset = dataset.map(operations=transforms.Compose([vision.RandomHorizontalFlip()]),
                          input_columns=['image'])
    dataset = dataset.map(
        operations=transforms.Compose([vision.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], is_hwc=True)]),
        input_columns=['image'])
    dataset = dataset.map(operations=transforms.Compose([vision.ToTensor()]),
                          input_columns=['image'])
    dataset = dataset.batch(batch_size=int(args.global_batch_size // device_num))
    return dataset
