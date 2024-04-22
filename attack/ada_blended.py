'''
this script is for adaptive_blended attack

modified from https://github.com/Unispac/Circumventing-Backdoor-Defenses 
'''


import argparse
import logging
import os
import sys
import torch
import numpy as np
from torchvision import transforms

os.chdir(sys.path[0])
sys.path.append('../')
os.getcwd()

from attack.badnet import BadNet, add_common_attack_args
from utils.aggregate_block.dataset_and_transform_generate import get_num_classes, get_input_shape
from utils.backdoor_generate_poison_index import generate_poison_index_from_label_transform
from utils.aggregate_block.bd_attack_generate import bd_attack_img_trans_generate, bd_attack_label_trans_generate
from copy import deepcopy
from utils.bd_dataset_v2 import prepro_cls_DatasetBD_v2, dataset_wrapper_with_transform


class Ada_Blended(BadNet):

    def set_bd_args(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser = add_common_attack_args(parser)
        parser.add_argument('--attack_train_replace_imgs_path', type=str)
        parser.add_argument('--attack_test_replace_imgs_path', type=str)
        parser.add_argument('--bd_yaml_path', type=str, default='../config/attack/ada_blended/default.yaml', help='path for yaml file provide additional default attributes')
        return parser

    def process_args(self, args):

        args.terminal_info = sys.argv
        args.num_classes = get_num_classes(args.dataset)
        args.input_height, args.input_width, args.input_channel = get_input_shape(args.dataset)
        args.img_size = (args.input_height, args.input_width, args.input_channel)
        args.dataset_path = f"{args.dataset_path}/{args.dataset}"
        
        args.attack_train_replace_imgs_path = f"../resource/ada_blended/{args.dataset}_ada_blended_train_b1.npy"    
        args.attack_test_replace_imgs_path = f"../resource/ada_blended/{args.dataset}_ada_blended_test_b1.npy"
        return args
    
    def stage1_non_training_data_prepare(self):
        logging.info(f"stage1 start")

        assert 'args' in self.__dict__
        args = self.args

        train_dataset_without_transform, \
        train_img_transform, \
        train_label_transform, \
        test_dataset_without_transform, \
        test_img_transform, \
        test_label_transform, \
        clean_train_dataset_with_transform, \
        clean_train_dataset_targets, \
        clean_test_dataset_with_transform, \
        clean_test_dataset_targets \
            = self.benign_prepare()

        # load replace img from npy    
        train_bd_img_transform, test_bd_img_transform = bd_attack_img_trans_generate(args)
        bd_label_transform = bd_attack_label_trans_generate(args)

        # choose posion and cover index
        args.pratio = args.p_ratio
        targets = clean_train_dataset_targets
        targets = np.array(targets)

        poison_index = np.zeros(len(targets))
        cover_index = np.zeros(len(targets))

        p_num = int(args.p_ratio * len(targets))
        non_zero_array = np.random.choice(np.arange(len(targets)), p_num, replace = False)
        arr = list(non_zero_array)
        p_arr = sorted(arr[:int(p_num/2)])
        c_arr = sorted(arr[int(p_num/2):])
        poison_index[p_arr] = 1
        cover_index[c_arr] = 1

        train_poison_index = poison_index

        logging.debug(f"poison train idx is saved")
        torch.save(train_poison_index,
                   args.save_path + '/train_poison_index_list.pickle')

        ### generate train dataset for backdoor attack
        bd_train_dataset = prepro_cls_DatasetBD_v2(
            deepcopy(train_dataset_without_transform),
            poison_indicator=train_poison_index,
            bd_image_pre_transform=train_bd_img_transform,
            bd_label_pre_transform=bd_label_transform,
            save_folder_path=f"{args.save_path}/bd_train_dataset",
            cover_indicator= cover_index)

        bd_train_dataset_with_transform = dataset_wrapper_with_transform(
            bd_train_dataset,
            train_img_transform,
            train_label_transform)

        ### all test index except target
        test_poison_index = generate_poison_index_from_label_transform(
            clean_test_dataset_targets,
            label_transform=bd_label_transform,
            train=False)

        ### generate test dataset for ASR
        bd_test_dataset = prepro_cls_DatasetBD_v2(
            deepcopy(test_dataset_without_transform),
            poison_indicator=test_poison_index,
            bd_image_pre_transform=test_bd_img_transform,
            bd_label_pre_transform=bd_label_transform,
            save_folder_path=f"{args.save_path}/bd_test_dataset",
            cover_indicator=None)

        ### remove target from bd_test
        bd_test_dataset.subset(
            np.where(test_poison_index == 1)[0])
        
        bd_test_dataset_with_transform = dataset_wrapper_with_transform(
            bd_test_dataset,
            test_img_transform,
            test_label_transform)

        self.stage1_results = clean_train_dataset_with_transform, \
                              clean_test_dataset_with_transform, \
                              bd_train_dataset_with_transform, \
                              bd_test_dataset_with_transform

if __name__ == '__main__':
    attack = Ada_Blended()
    parser = argparse.ArgumentParser(description=sys.argv[0])
    parser = attack.set_args(parser)
    parser = attack.set_bd_args(parser)
    args = parser.parse_args()
    attack.add_bd_yaml_to_args(args)
    attack.add_yaml_to_args(args)
    args = attack.process_args(args)
    attack.prepare(args)
    os.environ["CUDA_VISIBLE_DEVICES"] = "%s" % args.device[-1]
    args.device = 'cuda:0'
    attack.stage1_non_training_data_prepare()
    attack.stage2_training()
    logging.info("===End ada_blended===")
