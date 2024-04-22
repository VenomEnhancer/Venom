import os
import sys
import yaml
import json

os.chdir(sys.path[0])
sys.path.append('../')
os.getcwd()

import argparse
import logging
import torch
import torch.nn as nn
from copy import deepcopy
import numpy as np

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
        _, train_img_transform1, _, _, _, _, clean_train_dataset_with_transform1, _, _, _ = self.benign_prepare()
        _, _, _, _, _, _, clean_train_dataset_with_transform2, _, clean_test_dataset_with_transform2, _ = self.benign_prepare()
        clean_dataset = prepro_cls_DatasetBD_v2(clean_train_dataset_with_transform1.wrapped_dataset)
        clean_train_dataset_with_transform2.wrapped_dataset = clean_dataset
        clean_train_dataset_with_transform2.wrap_img_transform = train_img_transform1
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
    
        for layer_name in self.vgg_layers.keys():
            self.hooks.append(self.vgg_layers[layer_name].register_forward_hook(self.get_activation(layer_name)))
            self.sim4layers[layer_name] = []

    def remove_hooks(self):
        # remove hooks
        for hook in self.hooks:
            hook.remove()  
            self.hooks = []         

    def train_clean_model(self):
        args = self.args
        if not args.mode == 'train':
            return
        logging.info(args)
        
        # prepare data
        self.train_clean_loader = torch.utils.data.DataLoader(self.clean_train_dataset_with_transform,batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        self.test_clean_loader = torch.utils.data.DataLoader(self.clean_test_dataset_with_transform,batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        _, bd_test_dataset_with_transform = self.prepare_bd_data()
        self.test_bd_loader = torch.utils.data.DataLoader(bd_test_dataset_with_transform,batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        
        target_test_dataloader = get_target_dataset_loader(args, self.clean_train_dataset_with_transform, bd_test_dataset_with_transform[0][1])
        
        trainer = PureCleanModelTrainer(self.model)
        optimizer, scheduler = argparser_opt_scheduler(self.model, self.args)
        criterion = argparser_criterion(args)
        trainer.train_with_test_each_epoch_on_mix(
            self.train_clean_loader,
            self.test_clean_loader,
            self.test_bd_loader,
            args.epochs,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=args.device,
            frequency_save=args.frequency_save,
            save_folder_path=args.save_path,
            save_prefix='clean',
            amp=args.amp,
            prefetch=args.prefetch,
            prefetch_transform_attr_name="ori_image_transform_in_loading", # since we use the preprocess_bd_dataset
            non_blocking=args.non_blocking,
            
            target_test_dataloader=target_test_dataloader,
        )
        save_dict = {
            'model_name': args.model,
            'model': self.model.cpu().state_dict(),
        }
        torch.save(
            save_dict,
            f'{args.save_path}/attack_result.pt',
        )
    
    # filter element greater than threshold
    def filter_tensor(self, tensor, threshold):
        indices = torch.nonzero(tensor > threshold)
        values = tensor[indices]
        return indices, values

    # calculate path for label m
    def calc_path_for_label_m(self, label_m, model, sim_calculator, threshold=0.9):
        self.register_layers()
        args = self.args
        
        # prepare target dataset
        indexs = [i for i, (_, label,*_) in enumerate(self.clean_train_dataset_with_transform) if label == label_m]
        train_target_loader = torch.utils.data.DataLoader(self.clean_train_dataset_with_transform, batch_size=args.batch_size, sampler=torch.utils.data.sampler.SubsetRandomSampler(indexs),drop_last=False, num_workers=args.num_workers)     

        for X_target, *_ in train_target_loader:
            X_target = X_target.to(args.device)
            _ = model(X_target)

            batch_num = X_target.shape[0]
            num_couple = int(batch_num / 2)
            if args.dataset == 'cfp' or args.dataset == 'gtsrb':
                for layer_name in self.vgg_layers.keys():
                    a = self.activation[layer_name].clone().detach()
                    for i in range(batch_num):
                        for j in range(i+1, batch_num):    
                            sample0 = a[i]
                            sample1 = a[j]
                            self.sim4layers[layer_name].append(sim_calculator(sample0,sample1).tolist())
            else:
                for layer_name in self.vgg_layers.keys():
                    a = self.activation[layer_name].clone().detach()
                    for i in range(num_couple):
                        sample0 = a[2*i]
                        sample1 = a[2*i+1]
                        self.sim4layers[layer_name].append(sim_calculator(sample0,sample1).tolist())
                    
        # calculate mean similarity and filter by threshold
        layer_sim = {}
        for layer_name in self.vgg_layers.keys():
            tensor = torch.tensor(self.sim4layers[layer_name])
            tensor = torch.mean(tensor,dim=0)
            indexs, values = self.filter_tensor(tensor, threshold)
            tmp = []
            for i,v in zip(indexs, values):
                tmp.append([i.item(), v.item()])
            tmp = sorted(tmp, key=lambda x:-x[1])
            layer_sim[layer_name] = tmp
        self.remove_hooks()
        return layer_sim # {layer_name:[[index,sim_value]]}

    # calculate similarity to find the common neurons
    # specify threshold for different datasets
    # we suggest 0.7 for cifar10 and cifar100, 0.9 for gtsrb
    def calc_sim_of_common_neurons(self, model, sim_calculator, threshold=0.9):
        self.register_layers()
        args = self.args
        
        # prepare all dataset
        train_target_loader = torch.utils.data.DataLoader(self.clean_train_dataset_with_transform, batch_size=args.batch_size,drop_last=False, num_workers=args.num_workers)

        for X_target, *_ in train_target_loader:
            X_target = X_target.to(args.device)
            _ = model(X_target)
            
            batch_num = X_target.shape[0]
            num_couple = int(batch_num / 2)
            if args.dataset == 'cfp' or args.dataset == 'gtsrb':
                for layer_name in self.vgg_layers.keys():
                    a = self.activation[layer_name].clone().detach()
                    for i in range(batch_num):
                        for j in range(i+1, batch_num):    
                            sample0 = a[i]
                            sample1 = a[j]
                            self.sim4layers[layer_name].append(sim_calculator(sample0,sample1).tolist())
            else:
                for layer_name in self.vgg_layers.keys():
                    a = self.activation[layer_name].clone().detach()
                    for i in range(num_couple):
                        sample0 = a[2*i]
                        sample1 = a[2*i+1]
                        self.sim4layers[layer_name].append(sim_calculator(sample0,sample1).tolist())
                    
        # calculate mean similarity and filter by threshold
        layer_sim = {}
        for layer_name in self.vgg_layers.keys():
            tensor = torch.tensor(self.sim4layers[layer_name])
            tensor = torch.mean(tensor,dim=0)
            indexs, values = self.filter_tensor(tensor, threshold)
            tmp = []
            for i,v in zip(indexs, values):
                tmp.append([i.item(), v.item()])
            tmp = sorted(tmp, key=lambda x:-x[1])
            layer_sim[layer_name] = tmp
        self.remove_hooks()
        return layer_sim # {layer_name:[[index,sim_value]]}     

    def exclude_common_neurons(self, couple_path, common_path, threshold):
        couple_path = f'../similarity/results/{args.dataset}/{args.model}/similarity/{couple_path}'
        common_path = f'../similarity/results/{args.dataset}/{args.model}/similarity/{common_path}'
        calc_path_drop = f'../similarity/results/{args.dataset}/{args.model}/similarity/TCDP_drop_{threshold}.json'
        calc_path_save = f'../similarity/results/{args.dataset}/{args.model}/similarity/TCDP_save_{threshold}.json'
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
        
        targe_layer = get_target_layer(args.model)
        TCDP = dic_save[targe_layer][:7]+dic_save[targe_layer][-3:]
        TCDP = {args.attack_target: {get_target_layer(args.model): sorted([i[0] for i in TCDP])} }
        return TCDP

    def calc_TCDP(self, save_threshold=0.7, TCDP_name='single_deep_conv_10'):
        args = self.args 
        if not args.mode == 'calc':
            return 
        logging.info(args)

        #  load model
        model = self.model
        model_path = args.save_path + '/attack_result.pt'
        load_file = torch.load(model_path)
        model.load_state_dict(load_file['model'])
        model.to(args.device)
        model.eval()
       
        # set calculator
        sim_calculator = self.compute_cka[self.args.cka]

        # calculate paths for different labels
        critical_paths = []
        num_classes = get_num_classes(args.dataset)
        for i in range(num_classes):
            critical_paths.append(self.calc_path_for_label_m(i, model, sim_calculator))

        # save target path
        target_label = args.attack_target
        target_path = critical_paths[target_label]
        with open(args.save_path+'/similarity/target_path.json','w+') as f:
            json.dump(target_path, f, indent=4)

        # calculate couple neurons and its similarity
        for layer_name in self.vgg_layers.keys():
            path0=target_path[layer_name]
            for n in path0:
                n.append(0)
                for i in range(num_classes):
                    if i==target_label:
                        continue
                    path = critical_paths[i][layer_name]
                    for ni in path:
                        if n[0]==ni[0]:
                            n[2]+=1
            target_path[layer_name] = sorted(path0,key=lambda x:(-x[2],-x[1]))
        with open(args.save_path+'/similarity/couple_path.json','w+') as f:
            json.dump(target_path, f, indent=4)

        # calculate common neurons and its similarity
        common_path = self.calc_sim_of_common_neurons(model,sim_calculator,threshold=save_threshold)
        with open(args.save_path+f'/similarity/common_neurons_{save_threshold}.json','w+') as f:
            json.dump(common_path, f, indent=4)
        
        TCDP = self.exclude_common_neurons('couple_path.json',f'common_neurons_{save_threshold}.json',save_threshold)
        with open(f'{args.save_path}/json/{TCDP_name}.json','w+') as f:
            json.dump(TCDP, f, indent=4)

    def calc_mean_act_of_TCDP(self,mode='clean'):
        args = self.args    
        if not args.mode == 'calc':
            return  
        logging.info(args)
        self.register_layers()
        
        # prepare target dataset
        indexs = [i for i, (_, label,*_) in enumerate(self.clean_train_dataset_with_transform) if label == args.attack_target]
        self.train_target_loader = torch.utils.data.DataLoader(self.clean_train_dataset_with_transform, batch_size=args.batch_size, sampler=torch.utils.data.sampler.SubsetRandomSampler(indexs),drop_last=False, num_workers=args.num_workers)
        
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
    calc_sim.set_vars(args)
    os.environ["CUDA_VISIBLE_DEVICES"] = "%s" % args.device[-1]
    args.device = 'cuda:0'

    # train/micro_train clean model 
    calc_sim.train_clean_model()
    # calculate TCDP
    calc_sim.calc_TCDP(save_threshold=0.7, TCDP_name='single_deep_conv_10')
    # calculate mean act for TCDP
    calc_sim.calc_mean_act_of_TCDP('single_deep_conv_10')

    
'''
### train clean model
model="vgg19_bn"
device="cuda:0"
dataset="cifar10"
save_path="results/${dataset}/${model}"
log_path="similarity/results/${dataset}/${model}/similarity/test.log"
nohup python similarity/calc_similarity.py --mode train --save_path ${save_path} --model ${model} --device ${device} --dataset ${dataset}> ${log_path} 2>&1 &

### micro-train clean model
epochs=2
frequency_save=1
model="vgg19_bn"
device="cuda:0"
dataset="cifar10"
save_path="results/${dataset}/half_train/${model}/${epochs}"
log_path="similarity/results/${dataset}/${model}/similarity/half_train.log"
nohup python similarity/calc_similarity.py --mode train --frequency_save ${frequency_save} --epochs ${epochs} --save_path ${save_path} --model ${model} --device ${device} --dataset ${dataset}> ${log_path} 2>&1 &

### calculate TCDP
model="vgg19_bn"
device="cuda:0"
dataset="cifar10"
save_path="results/${dataset}/${model}"
log_path="similarity/results/${dataset}/${model}/similarity/test.log"
mkdir -p similarity/results/${dataset}/${model}/similarity/
nohup python similarity/calc_similarity.py --mode calc --save_path ${save_path}  --model ${model} --device ${device} --dataset ${dataset} > ${log_path} 2>&1 &
''' 
    