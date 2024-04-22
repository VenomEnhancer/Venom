from torchvision import transforms
from PIL import Image
from math import sqrt
import torch
import random

def get_trigger_mask(img_size, total_pieces, masked_pieces):
    div_num = sqrt(total_pieces)
    step = int(img_size // div_num)
    candidate_idx = random.sample(list(range(total_pieces)), k=masked_pieces)
    mask = torch.ones((img_size, img_size))
    for i in candidate_idx:
        x = int(i % div_num)  # column
        y = int(i // div_num)  # row
        mask[x * step: (x + 1) * step, y * step: (y + 1) * step] = 0
    return mask


class adaBlendedImageAttack(object):

    @classmethod
    def add_argument(self, parser):
        parser.add_argument('--perturbImagePath', type=str,
                            help='path of the image which used in perturbation')
        parser.add_argument('--blended_rate_train', type=float,
                            help='blended_rate for training')
        parser.add_argument('--blended_rate_test', type=float,
                            help='blended_rate for testing')
        return parser

    def __init__(self, train=True):
        trigger_path = '/home/Data/backdoorbench/utils/bd_img_transform/img/hellokitty_32.png'
        self.trigger_transform = transforms.Compose([ transforms.ToTensor()])
        trigger = Image.open(trigger_path).convert("RGB")
        trigger = self.trigger_transform(trigger) #.cpu().numpy().transpose(1, 2, 0) * 255

        self.pieces=16
        self.mask_rate=0.5
        self.alpha=0.2
        self.img_size=32
        self.masked_pieces = round(self.mask_rate * self.pieces)
        self.mask = get_trigger_mask(self.img_size, self.pieces, self.masked_pieces)
        self.target_image = trigger
        self.train=train

    def __call__(self, img, target = None, image_serial_id = None):
        return self.add_trigger(img)

    def add_trigger(self, img):
        img = self.trigger_transform(img)
        if self.train:
            img = img + self.alpha * self.mask * (self.target_image - img)
        else:
            img = (1 - self.alpha) * img + self.alpha *  self.target_image
        return img.cpu().numpy().transpose(1, 2, 0) * 255
        # return (1-self.blended_rate) * img + (self.blended_rate) * self.target_image
