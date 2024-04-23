import os
import sys
import yaml
import json
import time
import math

os.chdir(sys.path[0])
sys.path.append('../')
os.getcwd()

import argparse
import logging
import torch
import torch.nn as nn
from copy import deepcopy
import numpy as np

from collections import Counter
from CKA import CudaCKA
from attack.prototype import NormalCase
from utils.aggregate_block.model_trainer_generate import generate_cls_model
from utils.aggregate_block.bd_attack_generate import bd_attack_img_trans_generate, bd_attack_label_trans_generate
from utils.bd_dataset_v2 import prepro_cls_DatasetBD_v2, dataset_wrapper_with_transform
from utils.backdoor_generate_poison_index import generate_poison_index_from_label_transform
from utils.trainer_cls import PureCleanModelTrainer, get_target_dataset_loader
from utils.aggregate_block.train_settings_generate import argparser_criterion, argparser_opt_scheduler
from utils.aggregate_block.dataset_and_transform_generate import dataset_and_transform_generate, get_num_classes
from utils.couple_utils import *

def get_target_layer_(model_name):
    if model_name == 'vgg19_bn':
        return 'features.49'
    elif model_name == 'preactresnet18':
        return 'layer4.1.conv2'
    elif model_name == 'preactresnet50':
        return 'layer4.2.conv2' # 'layer4.2.conv3'
    elif model_name == 'vit_b_16':
        return '1.encoder.layers.encoder_layer_11.dropout'
    elif model_name == 'convnext_tiny':
        return 'features.7.2.block.0'
        # return 'features.7.2.block' # blk
    else:
        return NotImplementedError("Unsupported Model Structure")

def get_args(parser):
    parser.add_argument('--yaml_path', default='sim.yaml', type=str,
                        help='yaml path')
    parser.add_argument('--save_path', default='results/cifar10/vgg19_bn', type=str,
                        help='save path')
    parser.add_argument('--cka', default='cosine', type=str,
                        help='choose type of cka')
    parser.add_argument('--mode', default='train', type=str,
                        help='train/ calculate')
    parser.add_argument('--model', default='vgg19_bn', type=str)
    parser.add_argument('--device', default='cuda:0', type=str)
    parser.add_argument('--dataset', default='cifar10', type=str)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--frequency_save', default=0, type=int)
    args = parser.parse_args()
    return args 

class CalcSim(NormalCase):
    
    def __init__(self):
        super().__init__()
    
    def add_yaml_to_args(self, args):
        return super().add_yaml_to_args(args)
    
    def prepare_clean_data(self):
        _, _, _, _, _, _, clean_train_dataset_with_transform2, _, clean_test_dataset_with_transform2, _ = self.benign_prepare()
        return clean_train_dataset_with_transform2, clean_test_dataset_with_transform2
          
    def prepare_bd_data(self):
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
            
        train_bd_img_transform, test_bd_img_transform = bd_attack_img_trans_generate(args)
        ### get the backdoor transform on label
        bd_label_transform = bd_attack_label_trans_generate(args)

        ### set the backdoor attack data and backdoor test data
        train_poison_index = generate_poison_index_from_label_transform(
            clean_train_dataset_targets,
            label_transform=bd_label_transform,
            train=True,
            pratio=args.pratio if 'pratio' in args.__dict__ else None,
            p_num=args.p_num if 'p_num' in args.__dict__ else None,
        )

        ### generate train dataset for backdoor attack
        bd_train_dataset = prepro_cls_DatasetBD_v2(
            deepcopy(train_dataset_without_transform),
            poison_indicator=train_poison_index,
            bd_image_pre_transform=train_bd_img_transform,
            bd_label_pre_transform=bd_label_transform,
            save_folder_path=f"{args.save_path}/bd_train_dataset",
        )

        bd_train_dataset_with_transform = dataset_wrapper_with_transform(
            bd_train_dataset,
            train_img_transform,
            train_label_transform,
        )

        ### decide which img to poison in ASR Test
        test_poison_index = generate_poison_index_from_label_transform(
            clean_test_dataset_targets,
            label_transform=bd_label_transform,
            train=False,
        )

        ### generate test dataset for ASR
        bd_test_dataset = prepro_cls_DatasetBD_v2(
            deepcopy(test_dataset_without_transform),
            poison_indicator=test_poison_index,
            bd_image_pre_transform=test_bd_img_transform,
            bd_label_pre_transform=bd_label_transform,
            save_folder_path=f"{args.save_path}/bd_test_dataset",
        )

        bd_test_dataset.subset(
            np.where(test_poison_index == 1)[0]
        )

        bd_test_dataset_with_transform = dataset_wrapper_with_transform(
            bd_test_dataset,
            test_img_transform,
            test_label_transform,
        )
        return bd_train_dataset_with_transform, bd_test_dataset_with_transform
    
    def set_vars(self, args):
        self.add_yaml_to_args(args)
        args = self.process_args(args)
        self.cuda_cka = CudaCKA(args.device)
        self.args = args
    
        # global variables
        self.compute_cka = {'linear':self.cuda_cka.linear_CKA,'kernel':self.cuda_cka.kernel_CKA,'cosine':self.cuda_cka.cosine}
        self.activation = {}
        self.vgg_layers = {}
        self.hooks = []
        self.sim4layers = {}

        self.activation1 = {}
        self.activation2 = {}
        
        # model
        self.model = generate_cls_model(
            model_name=args.model,
            num_classes=args.num_classes,
            image_size=args.img_size[0],
        )
        
        # prepare clean data
        self.clean_train_dataset_with_transform, self.clean_test_dataset_with_transform = self.prepare_clean_data()
            
    def get_activation(self,name):
        def hook(model, input, output):
            self.activation[name] = output.clone().detach()
        return hook

    def get_activation_couple(self,name):
        def hook(model, input, output):
            self.activation1[name] = output.clone().detach()
        return hook
    
    def get_activation_pure(self,name):
        def hook(model, input, output):
            self.activation2[name] = output.clone().detach()
        return hook

    def register_layers(self):
        self.vgg_layers = init_layers(self.args.model, self.model)
        layer_name = get_target_layer_(self.args.model)
        self.hooks.append(self.vgg_layers[layer_name].register_forward_hook(self.get_activation(layer_name)))
        self.sim4layers[layer_name] = []

    def remove_hooks(self):
        # remove hooks
        for hook in self.hooks:
            hook.remove()  
            self.hooks = [] 
        torch.cuda.empty_cache()        

    # filter element greater than threshold
    def filter_tensor(self, tensor, threshold):
        indices = torch.nonzero(tensor > threshold)
        values = tensor[indices]
        return indices, values

    # calculate path for label m
    def calc_path_for_label_m(self, label_m, model, K=100, batch_size=64, iter_num=1):
        self.register_layers()
        args = self.args
        
        # prepare target dataset
        indexs = [i for i, (_, label,*_) in enumerate(self.clean_train_dataset_with_transform) if label == label_m]
        class_m_dataset = torch.utils.data.Subset(self.clean_train_dataset_with_transform, indexs)

        # target dataset length
        target_len = len(class_m_dataset)

        # load data        
        if iter_num == 0:
            train_target_loader = torch.utils.data.DataLoader(class_m_dataset, batch_size=target_len,num_workers=args.num_workers, shuffle=True)
            print(f'label:{label_m}\tsize:{target_len}')
            batch_size = target_len
        else:
            train_target_loader = torch.utils.data.DataLoader(class_m_dataset, batch_size=batch_size,num_workers=args.num_workers, shuffle=True)
        
        iter_count = 0

        # target layer
        target_layer_name = get_target_layer_(args.model)

        # calc
        for X_target, *_ in train_target_loader:
        # X_target, *_ = iter(train_target_loader).__next__()
            X_target = X_target.to(args.device)
            _ = model(X_target)

            a = self.activation[target_layer_name].clone().detach() # B*C*A*A
            if args.model == 'vit_b_16':
                a = a[:,1:,:] # Bx196x768
                a = a.reshape(a.shape[0],a.shape[1]*3,-1) # BxCxAA
                a = a.permute(1,0,2) # C*B*AA
            else:
                a = a.permute(1,0,2,3) # C*B*A*A
                kernal_n = a.shape[3]
                a = a.view(a.shape[0],a.shape[1],-1) # C*B*AA
            if args.is_norm:
                a = torch.nn.functional.normalize(a, p=2, dim=2) # normalize dim of AA
            lp_matrix = torch.cdist(a,a,p=2) # C*B*B
            B = batch_size
            channel_mean_lp = lp_matrix.sum(dim=(1,2)) / (B**2-B) # C*1
            self.sim4layers[target_layer_name].append(channel_mean_lp.tolist())
            iter_count+=1
            if iter_count == iter_num:
                break
                
        # calculate mean similarity and filter topK
        layer_sim = {}
        channel_mean_sim = torch.tensor(self.sim4layers[target_layer_name])
        channel_mean_sim = torch.mean(channel_mean_sim,dim=0)

        # print all sim
        if label_m == args.attack_target:
            target_sims =[[i,sim.item()] for i, sim  in enumerate(channel_mean_sim)]
            target_sims = sorted(target_sims, key=lambda x:x[1])
            with open(args.save_path+'/similarity/target_sims.json','w+') as f:
                json.dump(target_sims, f, indent=4)

        values, indexs = torch.topk(channel_mean_sim,k=K,largest=False, sorted=True)
        tmp = []
        for i,v in zip(indexs, values):
            tmp.append([i.item(), v.item()])
        layer_sim[target_layer_name] = tmp
        self.remove_hooks()
        return layer_sim # {layer_name:[[index,sim_value]]}

    # calculate similarity to find the common neurons
    def calc_sim_of_common_neurons(self, model, K=100, batch_size=512, iter_num=1):
        self.register_layers()
        args = self.args
        
        # prepare common dataset
        if args.dataset == 'tiny' and args.part_common:
            # part_classes
            target_classes = get_target_class(args.dataset)
            indexs = []
            for label_m in target_classes:
                indexs += [i for i, (_, label,*_) in enumerate(self.clean_train_dataset_with_transform) if label == label_m]
            part_dataset = torch.utils.data.Subset(self.clean_train_dataset_with_transform, indexs)
            train_target_loader = torch.utils.data.DataLoader(part_dataset, batch_size=batch_size,drop_last=False, num_workers=args.num_workers, shuffle=True) 
        else: 
            # all classes
            train_target_loader = torch.utils.data.DataLoader(self.clean_train_dataset_with_transform, batch_size=batch_size,drop_last=False, num_workers=args.num_workers, shuffle=True)            

        iter_count = 0

        for X_target, *_ in train_target_loader:
            X_target = X_target.to(args.device)
            _ = model(X_target)
            layer_name = get_target_layer_(args.model)
            a = self.activation[layer_name].clone().detach() # BxCxAxA
            if args.model == 'vit_b_16':
                a = a[:,1:,:] # Bx196x768
                a = a.reshape(a.shape[0],a.shape[1]*3,-1) # BxCxAA
                a = a.permute(1,0,2) # C*B*AA
            else:
                a = a.permute(1,0,2,3) # C*B*A*A
                kernal_n = a.shape[3]
                a = a.view(a.shape[0],a.shape[1],-1) # C*B*AA
            if args.is_norm:
                a = torch.nn.functional.normalize(a, p=2, dim=2) # normalize dim of AA
            lp_matrix = torch.cdist(a,a,p=2) # C*B*B
            B = batch_size
            channel_mean_lp = lp_matrix.sum(dim=(1,2)) / (B**2-B) # C*1
            # if not args.is_norm:
            #     channel_mean_lp = channel_mean_lp / kernal_n # remove the impact of n
            self.sim4layers[layer_name].append(channel_mean_lp.tolist())
            iter_count+=1
            if iter_count == iter_num:
                break
          
        # calculate mean similarity 
        layer_sim = {}
        layer_name = get_target_layer_(args.model)
        # for layer_name in self.vgg_layers.keys():
        channel_mean_sim = torch.tensor(self.sim4layers[layer_name])
        channel_mean_sim = torch.mean(channel_mean_sim,dim=0)
        values, indexs = torch.topk(channel_mean_sim,k=K,largest=False, sorted=True)
        tmp = []
        for i,v in zip(indexs, values):
            tmp.append([i.item(), v.item()])
        layer_sim[layer_name] = tmp
        self.remove_hooks()
        return layer_sim # {layer_name:[[index,sim_value]]}     

    def exclude_common_neurons(self, couple_path, common_path):
        couple_path = f'../similarity/results/{args.dataset}/{args.model}/similarity/{couple_path}'
        common_path = f'../similarity/results/{args.dataset}/{args.model}/similarity/{common_path}'
        calc_path_drop = f'../similarity/results/{args.dataset}/{args.model}/similarity/TCDP_drop_common.json'
        calc_path_save = f'../similarity/results/{args.dataset}/{args.model}/similarity/TCDP_save_common.json'
        dic2 = {}
        dic3 = {}
        dic_drop = {}
        dic_save = {}
        with open(couple_path,'r') as f:
            dic2 = json.load(f)
        with open(common_path,'r') as f:
            dic3 = json.load(f)
        for layer_name in dic2.keys():
            index_value_count2 = dic2[layer_name]
            index_value3 = dic3[layer_name]
            index_value3 = {str(i[0]):i[1] for i in index_value3}
            drops = []
            saves = []
            for index_value_count in index_value_count2:
                if str(index_value_count[0]) in index_value3.keys():
                    index_value_count.append(index_value3[str(index_value_count[0])])
                    drops.append(index_value_count)
                else:
                    saves.append(index_value_count)
            dic_drop[layer_name] = drops
            dic_save[layer_name] = saves
        with open(calc_path_drop,'w+') as f:
            json.dump(dic_drop,f,indent=4)
        with open(calc_path_save,'w+') as f:
            json.dump(dic_save,f,indent=4)
        
        target_layer = get_target_layer_(args.model)
        TCDP = dic_save[target_layer][:7]+dic_save[target_layer][-3:]
        TCDP = {args.attack_target: {get_target_layer_(args.model): sorted([i[0] for i in TCDP])} }
        return TCDP

    def calc_TCDP(self, TCDP_name='single_deep_conv_10'):
        args = self.args 
        if not args.mode == 'calc':
            return 
        logging.info(args)

        self.device  = torch.device(
                    (
                        f"cuda:{[int(i) for i in args.device[5:].split(',')][0]}" if "," in args.device else args.device
                    ) if torch.cuda.is_available() else "cpu"
                )

        #  load model
        model = self.model
        model_path = args.save_path + '/attack_result.pt'
        load_file = torch.load(model_path)
        model.load_state_dict(load_file['model'])

        if ',' in args.device:
            model = torch.nn.DataParallel(
                model,
                device_ids=[int(i) for i in args.device[5:].split(",")]  # eg. "cuda:2,3,7" -> [2,3,7]
            )
        else:
            model.to(args.device)
        args.device = self.device
        model.eval()

        start_time = time.time()

        #### compute target path
        target_label = args.attack_target
        target_path = self.calc_path_for_label_m(target_label, model, K=args.target_K, batch_size=args.target_batch_size, iter_num=args.target_iter_num) 
        with open(args.save_path+'/similarity/target_path.json','w+') as f:
            json.dump(target_path, f, indent=4)
        end_time = time.time() # target path time
        print('Target Path: %s s' % (end_time-start_time))

        #### calculate couple neurons and its similarity
        if args.dataset == 'tiny':
            target_classes = get_target_class(args.dataset)
        else:
            target_classes = range(get_num_classes(args.dataset))
        # for layer_name in self.vgg_layers.keys():
        layer_name = get_target_layer_(args.model)
        path0=target_path[layer_name]
        totoal_neurons = []
        for i in target_classes:
            if i==target_label:
                continue
            path = self.calc_path_for_label_m(i, model, K=args.other_K, batch_size=args.other_batch_size, iter_num=args.other_iter_num)[layer_name] 
            totoal_neurons.extend([i[0] for i in path])
        neuron_count = Counter(totoal_neurons)
        for n in path0:
            n.append(neuron_count[n[0]])
        target_path[layer_name] = sorted(path0,key=lambda x:(-x[2],-x[1]))
        with open(args.save_path+'/similarity/couple_path.json','w+') as f:
            json.dump(target_path, f, indent=4)
        last_time = end_time
        end_time = time.time() # couple neurons time
        print('Couple Neurons: %s s' % (end_time-last_time))

        #### calculate common neurons and its similarity
        common_path = self.calc_sim_of_common_neurons(model,K=args.common_K,batch_size=args.common_batch_size,iter_num=args.common_iter_num)
        with open(args.save_path+f'/similarity/common_neurons_K{args.common_K}.json','w+') as f:
            json.dump(common_path, f, indent=4)
        last_time = end_time
        end_time = time.time() # common neurons time
        print('Common Neurons: %s s' % (end_time-last_time))
        
        #### exclude common neurons
        TCDP = self.exclude_common_neurons('couple_path.json',f'common_neurons_K{args.common_K}.json')
        with open(f'{args.save_path}/json/{TCDP_name}.json','w+') as f:
            json.dump(TCDP, f, indent=4)
        last_time = end_time
        end_time = time.time() # end time
        print('TCDP: %s s' % (end_time-last_time))

    def calc_mean_act_of_TCDP(self,mode='clean'):
        args = self.args    
        if not args.mode == 'calc':
            return  
        logging.info(args)
        self.register_layers()
        
        # prepare target dataset
        indexs = [i for i, (_, label,*_) in enumerate(self.clean_train_dataset_with_transform) if label == args.attack_target]
        target_class_dataset = torch.utils.data.Subset(self.clean_train_dataset_with_transform, indexs)
        self.train_target_loader = torch.utils.data.DataLoader(target_class_dataset, batch_size=args.batch_size, drop_last=False, num_workers=args.num_workers, shuffle=True)
        
        #  load model
        model = self.model     
        model_path = args.save_path + '/attack_result.pt'
        load_file = torch.load(model_path)
        model.load_state_dict(load_file['model'])
        model.to(args.device)
        model.eval()
       
        # load TCDP
        critical_path = {}
        with open(f'{args.save_path}/json/{mode}.json','r') as f:
            critical_path = json.load(f)[str(args.attack_target)]

        # init
        target_acts = {}
        for layer_name in critical_path.keys():
            target_acts[layer_name] = None

        for X_target, *_ in self.train_target_loader:
            X_target = X_target.to(args.device)
            _ = model(X_target)
            
            with torch.no_grad():
                for layer_name in critical_path.keys():
                    a = self.activation[layer_name].clone().detach().to('cpu')
                    if args.model == 'vit_b_16':
                        a = a[:,1:,:] # Bx196x768
                        a = a.reshape(a.shape[0],a.shape[1]*3,-1) # BxCxAA
                    a = torch.mean(a,dim=0)
                    channels = critical_path[layer_name]
                    if target_acts[layer_name] == None:
                        target_acts[layer_name] = a[channels]
                    else:
                        b = target_acts[layer_name]
                        target_acts[layer_name] = torch.mean(torch.stack([a[channels], b], dim=0), dim=0)
                        
        # save
        torch.save(target_acts,f'{args.save_path}/act/{mode}.pth')
        # {"layer1":tentor}        
        self.remove_hooks()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculate Similarity')
    args = get_args(parser)
    calc_sim = CalcSim()
    #### whether calc common neurons in part classes
    args.part_common = True
    #### whether normalization
    args.is_norm = False
    #### params to calc sim
    args.target_K=100
    args.target_batch_size=64
    args.target_iter_num=0 # <0 whole dataset；=0 whole dataset at once； >0 certain iterations
    args.other_K=50
    args.other_batch_size=64
    args.other_iter_num=0 # <0 whole dataset；=0 whole dataset at once； >0 certain iterations
    args.common_K=50
    args.common_batch_size=512
    args.common_iter_num=0 # <=0, whole dataset; >0 certain iterations
    # os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
    # os.environ["CUDA_VISIBLE_DEVICES"] = "%s" % args.device[-1]
    # args.device = 'cuda:0'
    calc_sim.set_vars(args)

    # calculate TCDP
    with torch.no_grad():
        calc_sim.calc_TCDP(TCDP_name='test')

    # calculate mean act for TCDP
    calc_sim.calc_mean_act_of_TCDP('test')

    
'''
### calculate TCDP using matrix operation
model="vgg19_bn"
device="cuda:6"
dataset="cifar10"
save_path="results/${dataset}/${model}"
log_path="similarity/results/${dataset}/${model}/similarity/calc_TCDP.log"
mkdir -p similarity/results/${dataset}/${model}/similarity/
mkdir -p similarity/results/${dataset}/${model}/json/
mkdir -p similarity/results/${dataset}/${model}/act/
nohup python similarity/calc_sim_matrix.py --mode calc --save_path ${save_path}  --model ${model} --device ${device} --dataset ${dataset} > ${log_path} 2>&1 &
''' 
    