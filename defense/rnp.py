'''
This file is modified based on the following source:
link : https://github.com/bboylyg/RNP
The defense method is called rnp.
'''

import argparse
import os,sys
import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.parameter import Parameter
import torchvision.models as models
import torchvision.transforms as transforms
from collections import OrderedDict
import pandas as pd
import copy

sys.path.append('../')
sys.path.append(os.getcwd())

from pprint import  pformat
import yaml
import logging
import time
from scipy.stats import median_abs_deviation as MAD
from scipy.stats import gamma

from defense.base import defense
from utils.aggregate_block.train_settings_generate import argparser_criterion, argparser_opt_scheduler
from utils.trainer_cls import PureCleanModelTrainer, get_target_dataset_loader, given_dataloader_test, Metric_Aggregator
from utils.choose_index import choose_index
from utils.aggregate_block.fix_random import fix_random
from utils.aggregate_block.model_trainer_generate import generate_cls_model
from utils.log_assist import get_git_info
from utils.aggregate_block. dataset_and_transform_generate import get_input_shape, get_num_classes, get_transform
from utils.save_load_attack import load_attack_result, save_defense_result
from utils.bd_dataset_v2 import prepro_cls_DatasetBD_v2


class MaskBatchNorm2d(nn.BatchNorm2d):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super(MaskBatchNorm2d, self).__init__(
            num_features, eps, momentum, affine, track_running_stats)
        self.neuron_mask = Parameter(torch.Tensor(num_features))
        self.neuron_noise = Parameter(torch.Tensor(num_features))
        self.neuron_noise_bias = Parameter(torch.Tensor(num_features))
        init.ones_(self.neuron_mask)

    def forward(self, input: Tensor) -> Tensor:
        self._check_input_dim(input)

        # exponential_average_factor is set to self.momentum
        # (when it is available) only so that it gets updated
        # in ONNX graph when this node is exported to ONNX.
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            # TODO: if statement only here to tell the jit to skip emitting this when it is None
            if self.num_batches_tracked is not None:  # type: ignore
                self.num_batches_tracked = self.num_batches_tracked + 1  # type: ignore
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        r"""
        Decide whether the mini-batch stats should be used for normalization rather than the buffers.
        Mini-batch stats are used in training mode, and in eval mode when buffers are None.
        """
        if self.training:
            bn_training = True
        else:
            bn_training = (self.running_mean is None) and (self.running_var is None)

        r"""
        Buffers are only updated if they are to be tracked and we are in training mode. Thus they only need to be
        passed when the update should occur (i.e. in training mode when they are tracked), or when buffer stats are
        used for normalization (i.e. in eval mode when buffers are not None).
        """
        assert self.running_mean is None or isinstance(self.running_mean, torch.Tensor)
        assert self.running_var is None or isinstance(self.running_var, torch.Tensor)

        coeff_weight = self.neuron_mask
        coeff_bias = 1.0

        return F.batch_norm(
            input,
            # If buffers are not to be tracked, ensure that they won't be updated
            self.running_mean if not self.training or self.track_running_stats else None,
            self.running_var if not self.training or self.track_running_stats else None,
            self.weight * coeff_weight, self.bias * coeff_bias,
            bn_training, exponential_average_factor, self.eps)
        
def replace_bn_module(module, parent=None, name=None):
    """
    replace BatchNorm2d as MaskBatchNorm2d
    """
    for child_name, child_module in module.named_children():
        replace_bn_module(child_module, parent=module, name=child_name)

    if isinstance(module, nn.BatchNorm2d):
        new_module = MaskBatchNorm2d(module.num_features)
        if parent is not None:
            setattr(parent, name, new_module)
            
def load_state_dict(net, orig_state_dict):
    if 'state_dict' in orig_state_dict.keys():
        orig_state_dict = orig_state_dict['state_dict']

    new_state_dict = OrderedDict()
    for k, v in net.state_dict().items():
        if k in orig_state_dict.keys():
            new_state_dict[k] = orig_state_dict[k]
        else:
            new_state_dict[k] = v
    net.load_state_dict(new_state_dict)
    
def read_data(file_name):
    tempt = pd.read_csv(file_name, sep='\s+', skiprows=1, header=None)
    layer = tempt.iloc[:, 1]
    idx = tempt.iloc[:, 2]
    value = tempt.iloc[:, 3]
    mask_values = list(zip(layer, idx, value))
    return mask_values
        
class rnp(defense):
    
    def __init__(self, args):
        with open(args.yaml_path, 'r') as f:
            defaults = yaml.safe_load(f)
        defaults.update({k:v for k,v in args.__dict__.items() if v is not None})
        
        args.__dict__ = defaults

        args.terminal_info = sys.argv

        args.num_classes = get_num_classes(args.dataset)
        args.input_height, args.input_width, args.input_channel = get_input_shape(args.dataset)
        args.img_size = (args.input_height, args.input_width, args.input_channel)
        args.dataset_path = f"{args.dataset_path}/{args.dataset}"

        self.args = args
        
        if 'result_file' in args.__dict__ :
            if args.result_file is not None:
                self.set_result(args.result_file)
        
    @staticmethod
    def add_arguments(parser):
        parser.add_argument('--device', type=str, help='cuda, cpu')
        parser.add_argument("-pm","--pin_memory", type=lambda x: str(x) in ['True', 'true', '1'], help = "dataloader pin_memory")
        parser.add_argument("-nb","--non_blocking", type=lambda x: str(x) in ['True', 'true', '1'], help = ".to(), set the non_blocking = ?")
        parser.add_argument('--checkpoint_load', type=str, help='the location of load model')
        parser.add_argument('--checkpoint_save', type=str, help='the location of checkpoint where model is saved')
        parser.add_argument('--log', type=str, help='the location of log')
        parser.add_argument("--dataset_path", type=str, help='the location of data')
        parser.add_argument('--dataset', type=str, help='mnist, cifar10, cifar100, gtrsb, tiny') 
        parser.add_argument('--result_file', type=str, help='the location of result')
        parser.add_argument('--amp', type=lambda x: str(x) in ['True','true','1'])
        parser.add_argument('--frequency_save', type=int, help=' frequency_save, 0 is never')
        parser.add_argument("-pf", '--prefetch', type=lambda x: str(x) in ['True', 'true', '1'], help='use prefetch')
        
        parser.add_argument('--model', type=str, help='resnet18')
        parser.add_argument('--random_seed', type=int, help='random seed')
        parser.add_argument('--yaml_path', type=str, default="./config/defense/rnp/config.yaml", help='the path of yaml')
        parser.add_argument('--client_optimizer', type=int)
        parser.add_argument('--sgd_momentum', default=0.9, type=float)
        parser.add_argument('--wd', type=float, default=5e-4, help='weight decay of sgd')
        parser.add_argument('--lr_scheduler', type=str, help='the scheduler of lr')
        parser.add_argument('--steplr_milestones', type=int, nargs='+', default=[10, 20])
        parser.add_argument('--steplr_gamma', type=int, nargs='+', default=0.1)
        # RNP
        parser.add_argument('--ratio', type=float, default=0.01, help='ratio of defense data')
        parser.add_argument('--index', type=str, help='index of clean data')
        parser.add_argument('--batch_size', type=int)
        parser.add_argument('--alpha', type=float, default=0.2)
        parser.add_argument('--clean_threshold', type=float, default=0.2, help='threshold of unlearning accuracy')
        parser.add_argument('--lr', type=float, default=0.01, help='the learning rate for neuron unlearning')
        parser.add_argument('--recovering_lr', type=float, default=0.2, help='the learning rate for mask optimization')
        parser.add_argument('--unlearning_epochs', type=int, default=20, help='the number of epochs for unlearning')
        parser.add_argument('--recovering_epochs', type=int, default=20, help='the number of epochs for recovering')
        parser.add_argument('--mask_file', type=str, default=None, help='The text file containing the mask values')
        parser.add_argument('--pruning-by', type=str, default='threshold', choices=['number', 'threshold'])
        parser.add_argument('--pruning-max', type=float, default=0.90, help='the maximum number/threshold for pruning')
        parser.add_argument('--pruning-step', type=float, default=0.05, help='the step size for evaluating the pruning')
    
    def set_result(self, result_file):
        attack_file = 'record/' + result_file
        save_path = 'record/' + result_file + '/defense/rnp/'
        if not (os.path.exists(save_path)):
            os.makedirs(save_path)
        # assert(os.path.exists(save_path))    
        self.args.save_path = save_path
        if self.args.checkpoint_save is None:
            self.args.checkpoint_save = save_path + 'checkpoint/'
            if not (os.path.exists(self.args.checkpoint_save)):
                os.makedirs(self.args.checkpoint_save) 
        if self.args.log is None:
            self.args.log = save_path + 'log/'
            if not (os.path.exists(self.args.log)):
                os.makedirs(self.args.log)  
        self.result = load_attack_result(attack_file + '/attack_result.pt')
        
    def set_logger(self):
        args = self.args
        logFormatter = logging.Formatter(
            fmt='%(asctime)s [%(levelname)-8s] [%(filename)s:%(lineno)d] %(message)s',
            datefmt='%Y-%m-%d:%H:%M:%S',
        )
        logger = logging.getLogger()

        fileHandler = logging.FileHandler(args.log + '/' + time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()) + '.log')
        fileHandler.setFormatter(logFormatter)
        logger.addHandler(fileHandler)

        consoleHandler = logging.StreamHandler()
        consoleHandler.setFormatter(logFormatter)
        logger.addHandler(consoleHandler)

        logger.setLevel(logging.INFO)
        logging.info(pformat(args.__dict__))

        try:
            logging.info(pformat(get_git_info()))
        except:
            logging.info('Getting git info fails.')
            
    def set_devices(self):
        # self.device = torch.device(
        #     (
        #         f"cuda:{[int(i) for i in self.args.device[5:].split(',')][0]}" if "," in self.args.device else self.args.device
        #         # since DataParallel only allow .to("cuda")
        #     ) if torch.cuda.is_available() else "cpu"
        # )
        self.device = self.args.device
    
    def set_trainer(self, model):
        self.trainer = PureCleanModelTrainer(
            model,
        )
    
    def train_step_unlearning(self, args, model, criterion, optimizer, data_loader):
        model.train()
        total_correct = 0
        total_loss = 0.0
        for i, (images, labels, *_) in enumerate(data_loader):
            images, labels = images.to(self.args.device), labels.to(self.args.device)
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            # print("loss {}".format(loss.item()))
            pred = output.data.max(1)[1]
            total_correct += pred.eq(labels.view_as(pred)).sum()
            total_loss += loss.item()

            nn.utils.clip_grad_norm_(model.parameters(), max_norm=20, norm_type=2)
            (-loss).backward()
            optimizer.step()

        loss = total_loss / len(data_loader)
        acc = float(total_correct) / len(data_loader.dataset)
        return loss, acc
    
    def test(self, model, criterion, data_loader):
        model.eval()
        total_correct = 0
        total_loss = 0.0
        with torch.no_grad():
            for i, (images, labels, *_) in enumerate(data_loader):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                output = model(images)
                total_loss += criterion(output, labels).item()
                pred = output.data.max(1)[1]
                total_correct += pred.eq(labels.data.view_as(pred)).sum()
        loss = total_loss / len(data_loader)
        acc = float(total_correct) / len(data_loader.dataset)
        return loss, acc
        
    def save_checkpoint(self, state, file_path):
        # filepath = os.path.join(args.output_dir, args.arch + '-unlearning_epochs{}.tar'.format(epoch))
        torch.save(state, file_path)
    
    def pruning(self, net, neuron):
        state_dict = net.state_dict()
        weight_name = '{}.{}'.format(neuron[0], 'weight')
        state_dict[weight_name][int(neuron[1])] = 0.0
        net.load_state_dict(state_dict)
       
    def save_mask_scores(self, state_dict, file_name):
        mask_values = []
        count = 0
        for name, param in state_dict.items():
            if 'neuron_mask' in name:
                for idx in range(param.size(0)):
                    neuron_name = '.'.join(name.split('.')[:-1])
                    mask_values.append('{} \t {} \t {} \t {:.4f} \n'.format(count, neuron_name, idx, param[idx].item()))
                    count += 1
        with open(file_name, "w") as f:
            f.write('No \t Layer Name \t Neuron Idx \t Mask Score \n')
            f.writelines(mask_values)
            
    def evaluate_by_number(self, model, logger, mask_values, pruning_max, pruning_step, criterion, clean_loader, poison_loader, ori_cl_acc):
        results = []
        nb_max = int(np.ceil(pruning_max))
        nb_step = int(np.ceil(pruning_step))
        for start in range(0, nb_max + 1, nb_step):
            lastmodel = copy.deepcopy(model.state_dict())
            i = start
            for i in range(start, start + nb_step):
                self.pruning(model, mask_values[i])
            layer_name, neuron_idx, value = mask_values[i][0], mask_values[i][1], mask_values[i][2]
            cl_loss, cl_acc = self.test(model=model, criterion=criterion, data_loader=clean_loader)
            po_loss, po_acc = self.test(model=model, criterion=criterion, data_loader=poison_loader)
            logger.info('{} \t {} \t {} \t {} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f}'.format(
                i+1, layer_name, neuron_idx, value, po_loss, po_acc, cl_loss, cl_acc))
            results.append('{} \t {} \t {} \t {} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f}'.format(
                i+1, layer_name, neuron_idx, value, po_loss, po_acc, cl_loss, cl_acc))
            
            # early stop
            if abs(cl_acc - ori_cl_acc) / ori_cl_acc >= 0.05:
                logging.info("{}, {}, {}".format(cl_acc, ori_cl_acc, abs(cl_acc - ori_cl_acc) / ori_cl_acc))
                model.load_state_dict(lastmodel)
                break
            
        return results


    def evaluate_by_threshold(self, model, logger, mask_values, pruning_max, pruning_step, criterion, clean_loader, poison_loader, ori_cl_acc):
        results = []
        thresholds = np.arange(0, pruning_max + pruning_step, pruning_step)
        start = 0
        for threshold in thresholds:
            lastmodel = copy.deepcopy(model.state_dict())
            idx = start
            for idx in range(start, len(mask_values)):
                if float(mask_values[idx][2]) <= threshold:
                    self.pruning(model, mask_values[idx])
                    start += 1
                else:
                    break
            layer_name, neuron_idx, value = mask_values[idx][0], mask_values[idx][1], mask_values[idx][2]
            cl_loss, cl_acc = self.test(model=model, criterion=criterion, data_loader=clean_loader)
            po_loss, po_acc = self.test(model=model, criterion=criterion, data_loader=poison_loader)
            
            logger.info('{:.2f} \t {} \t {} \t {} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f}'.format(
                start, layer_name, neuron_idx, threshold, po_loss, po_acc, cl_loss, cl_acc))
            results.append('{:.2f} \t {} \t {} \t {} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f}\n'.format(
                start, layer_name, neuron_idx, threshold, po_loss, po_acc, cl_loss, cl_acc))
            
            # early stop
            if abs(cl_acc - ori_cl_acc) / ori_cl_acc >= 0.05:
                logging.info("{}, {}, {}".format(cl_acc, ori_cl_acc, abs(cl_acc - ori_cl_acc) / ori_cl_acc))
                model.load_state_dict(lastmodel)
                break
            
        return results
        
    def clip_mask(self, unlearned_model, lower=0.0, upper=1.0):
        params = [param for name, param in unlearned_model.named_parameters() if 'neuron_mask' in name]
        with torch.no_grad():
            for param in params:
                param.clamp_(lower, upper)
    
    def train_step_recovering(self, args, unlearned_model, criterion, mask_opt, data_loader):
        unlearned_model.train()
        total_correct = 0
        total_loss = 0.0
        nb_samples = 0
        for i, (images, labels, *_) in enumerate(data_loader):
            images, labels = images.to(self.args.device), labels.to(self.args.device)
            nb_samples += images.size(0)

            mask_opt.zero_grad()
            output = unlearned_model(images)
            loss = criterion(output, labels)
            loss = args.alpha * loss

            pred = output.data.max(1)[1]
            total_correct += pred.eq(labels.view_as(pred)).sum()
            total_loss += loss.item()
            loss.backward()
            mask_opt.step()
            self.clip_mask(unlearned_model)

        loss = total_loss / len(data_loader)
        acc = float(total_correct) / nb_samples
        return loss, acc    

    def save_mask_scores(self, state_dict, file_name):
        mask_values = []
        count = 0
        for name, param in state_dict.items():
            if 'neuron_mask' in name:
                for idx in range(param.size(0)):
                    neuron_name = '.'.join(name.split('.')[:-1])
                    mask_values.append('{} \t {} \t {} \t {:.4f} \n'.format(count, neuron_name, idx, param[idx].item()))
                    count += 1
        with open(file_name, "w") as f:
            f.write('No \t Layer Name \t Neuron Idx \t Mask Score \n')
            f.writelines(mask_values)   
        
    def mitigation(self):
        self.set_devices()
        fix_random(self.args.random_seed)
        
        # Prepare model, optimizer, scheduler
        model = generate_cls_model(self.args.model,self.args.num_classes)
        model.load_state_dict(self.result['model'])
        if "," in self.device:
            model = torch.nn.DataParallel(
                model,
                device_ids=[int(i) for i in self.args.device[5:].split(",")]  # eg. "cuda:2,3,7" -> [2,3,7]
            )
            self.args.device = f'cuda:{model.device_ids[0]}'
            model.to(self.args.device)
        else:
            model.to(self.args.device)
        criterion = argparser_criterion(args)
        
        train_tran = get_transform(self.args.dataset, *([self.args.input_height,self.args.input_width]) , train = True)
        test_tran = get_transform(self.args.dataset, *([self.args.input_height,self.args.input_width]) , train = False)
        
        logging.info('----------- Data Initialization --------------')
        
        defense_data_set = self.result['clean_train']
        clean_dataset = prepro_cls_DatasetBD_v2(self.result['clean_train'].wrapped_dataset)
        data_all_length = len(clean_dataset)
        ran_idx = choose_index(self.args, data_all_length) 
        log_index = self.args.log + 'index.txt'
        np.savetxt(log_index, ran_idx, fmt='%d')
        clean_dataset.subset(ran_idx)
        data_set_without_tran = clean_dataset
        data_set_o = self.result['clean_train']
        data_set_o.wrapped_dataset = data_set_without_tran
        data_set_o.wrap_img_transform = train_tran
        defense_data_loader = torch.utils.data.DataLoader(data_set_o, batch_size=self.args.batch_size, num_workers=self.args.num_workers,drop_last=False,pin_memory=args.pin_memory)
        
        data_bd_testset = self.result['bd_test']
        data_bd_testset.wrap_img_transform = test_tran
        data_bd_loader = torch.utils.data.DataLoader(data_bd_testset, batch_size=self.args.batch_size, num_workers=self.args.num_workers,drop_last=False,pin_memory=args.pin_memory)

        data_clean_testset = self.result['clean_test']
        data_clean_testset.wrap_img_transform = test_tran
        data_clean_loader = torch.utils.data.DataLoader(data_clean_testset, batch_size=self.args.batch_size, num_workers=self.args.num_workers,drop_last=False,pin_memory=args.pin_memory)

        target_test_dataloader = get_target_dataset_loader(args, data_clean_testset, data_bd_testset[0][1])
        original_tacc = given_dataloader_test(model, target_test_dataloader, criterion, args.non_blocking, args.device)[0]['test_acc']
        
        logging.info('----------- Backdoor Model Initialization --------------')
        
        model = generate_cls_model(self.args.model,self.args.num_classes)
        model.load_state_dict(self.result['model'])
        if "," in self.device:
            model = torch.nn.DataParallel(
                model,
                device_ids=[int(i) for i in self.args.device[5:].split(",")]  # eg. "cuda:2,3,7" -> [2,3,7]
            )
            self.args.device = f'cuda:{model.device_ids[0]}'
            model.to(self.args.device)
        else:
            model.to(self.args.device)
            
        criterion = torch.nn.CrossEntropyLoss().to(self.args.device)
        optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr, momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.args.steplr_milestones, gamma=0.1)       
        # optimizer, scheduler = argparser_opt_scheduler(model, self.args)
        # criterion = argparser_criterion(args)
        
        logging.info('----------- Model Unlearning --------------')
        logging.info('Epoch \t lr \t Time \t TrainLoss \t TrainACC \t PoisonLoss \t PoisonACC \t CleanLoss \t CleanACC')

        for epoch in range(0, args.unlearning_epochs + 1):
            start = time.time()
            lr = optimizer.param_groups[0]['lr']
            train_loss, train_acc = self.train_step_unlearning(args=self.args, model=model, criterion=criterion, optimizer=optimizer,
                                        data_loader=defense_data_loader)
            cl_test_loss, cl_test_acc = self.test(model=model, criterion=criterion, data_loader=data_clean_loader)
            po_test_loss, po_test_acc = self.test(model=model, criterion=criterion, data_loader=data_bd_loader)
            scheduler.step()
            end = time.time()
            logging.info(
                '%d \t %.3f \t %.1f \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f',
                epoch, lr, end - start, train_loss, train_acc, po_test_loss, po_test_acc,
                cl_test_loss, cl_test_acc)
            
            if train_acc <= args.clean_threshold or epoch == args.unlearning_epochs:
            # save the last checkpoint
                file_path = os.path.join(self.args.save_path, f'unlearned_model_last.tar')
                # torch.save(net.state_dict(), os.path.join(args.output_dir, 'unlearned_model_last.tar'))
                self.save_checkpoint({
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'clean_acc': cl_test_acc,
                    'bad_acc': po_test_acc,
                    'optimizer': optimizer.state_dict(),
                }, file_path)
                break
            
        logging.info('----------- Model Recovering --------------')
        # Step 2: load unleanred model checkpoints
        checkpoint = torch.load(file_path, map_location=self.args.device)
        logging.info('Unlearned Model: {}, {}, {}'.format(checkpoint['epoch'], checkpoint['clean_acc'], checkpoint['bad_acc']))
        
        unlearned_model = generate_cls_model(self.args.model,self.args.num_classes)
        # print(unlearned_model)
        replace_bn_module(unlearned_model)
        # print(unlearned_model)
        
        load_state_dict(unlearned_model, orig_state_dict=checkpoint['state_dict'])
        unlearned_model = unlearned_model.to(self.args.device)
        criterion = torch.nn.CrossEntropyLoss().to(self.args.device)

        parameters = list(unlearned_model.named_parameters())
        mask_params = [v for n, v in parameters if "neuron_mask" in n]
        mask_optimizer = torch.optim.SGD(mask_params, lr=args.recovering_lr, momentum=0.9)
        
        # Recovering
        logging.info('Epoch \t lr \t Time \t TrainLoss \t TrainACC \t PoisonLoss \t PoisonACC \t CleanLoss \t CleanACC')
        for epoch in range(1, args.recovering_epochs + 1):
            start = time.time()
            lr = mask_optimizer.param_groups[0]['lr']
            train_loss, train_acc = self.train_step_recovering(args=self.args, unlearned_model=unlearned_model, criterion=criterion, data_loader=defense_data_loader,
                                            mask_opt=mask_optimizer)
            cl_test_loss, cl_test_acc = self.test(model=unlearned_model, criterion=criterion, data_loader=data_clean_loader)
            po_test_loss, po_test_acc = self.test(model=unlearned_model, criterion=criterion, data_loader=data_bd_loader)
            end = time.time()
            logging.info('{} \t {:.3f} \t {:.1f} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f}'.format(
                epoch, lr, end - start, train_loss, train_acc, po_test_loss, po_test_acc,
                cl_test_loss, cl_test_acc))
        self.save_mask_scores(unlearned_model.state_dict(), os.path.join(self.args.save_path, 'mask_values.txt'))
        
        del unlearned_model, model
        logging.info('----------- Backdoored Model Pruning --------------')
        net = generate_cls_model(self.args.model,self.args.num_classes)
        net.load_state_dict(self.result['model'])
        net = net.to(self.args.device)
        criterion = torch.nn.CrossEntropyLoss().to(self.args.device)
        
        # Step 3: pruning
        mask_file = os.path.join(self.args.save_path, 'mask_values.txt')

        mask_values = read_data(mask_file)
        mask_values = sorted(mask_values, key=lambda x: float(x[2]))
        logging.info('No. \t Layer Name \t Neuron Idx \t Mask \t PoisonLoss \t PoisonACC \t CleanLoss \t CleanACC')
        cl_loss, ori_cl_acc = self.test(model=net, criterion=criterion, data_loader=data_clean_loader)
        po_loss, ori_po_acc = self.test(model=net, criterion=criterion, data_loader=data_bd_loader)
        logging.info('0 \t None     \t None     \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f}'.format(po_loss, ori_po_acc, cl_loss, ori_cl_acc))
        if args.pruning_by == 'threshold':
            results = self.evaluate_by_threshold(
                net, logging, mask_values, pruning_max=args.pruning_max, pruning_step=args.pruning_step,
                criterion=criterion, clean_loader=data_clean_loader, poison_loader=data_bd_loader,
                ori_cl_acc=ori_cl_acc
            )
        else:
            results = self.evaluate_by_number(
                net, logging, mask_values, pruning_max=args.pruning_max, pruning_step=args.pruning_step,
                criterion=criterion, clean_loader=data_clean_loader, poison_loader=data_bd_loader,
                ori_cl_acc=ori_cl_acc
            )
        file_name = os.path.join(self.args.save_path, 'pruning_by_{}.txt'.format(args.pruning_by))
        with open(file_name, "w") as f:
            f.write('No \t Layer Name \t Neuron Idx \t Mask \t PoisonLoss \t PoisonACC \t CleanLoss \t CleanACC\n')
            f.writelines(results)       
        
        # evalution
        test_dataloader_dict = {}
        test_dataloader_dict["clean_test_dataloader"] = data_clean_loader
        test_dataloader_dict["bd_test_dataloader"] = data_bd_loader
        test_dataloader_dict["target_test_dataloader"] = target_test_dataloader
        self.set_trainer(net)
        self.trainer.set_with_dataloader(
            train_dataloader = defense_data_loader,
            test_dataloader_dict = test_dataloader_dict,

            criterion = criterion,
            optimizer = None,
            scheduler = None,
            device = self.args.device,
            amp = self.args.amp,

            frequency_save = self.args.frequency_save,
            save_folder_path = self.args.save_path,
            save_prefix = 'rnp',

            prefetch = self.args.prefetch,
            prefetch_transform_attr_name = "ori_image_transform_in_loading",
            non_blocking = self.args.non_blocking,
        )
        
        clean_test_loss_avg_over_batch, \
                bd_test_loss_avg_over_batch, \
                test_acc, \
                test_asr, \
                test_ra, \
                test_tacc = self.trainer.test_current_model(
            test_dataloader_dict, self.args.device,
        )

        agg = Metric_Aggregator()
        agg({
                "clean_test_loss_avg_over_batch": clean_test_loss_avg_over_batch,
                "bd_test_loss_avg_over_batch": bd_test_loss_avg_over_batch,
                "test_acc": test_acc*100,
                "test_asr": test_asr*100,
                "test_ra": test_ra*100,
                "test_tacc": test_tacc*100
                
            })
        agg.to_dataframe().to_csv(f"{args.save_path}rnp_df_summary.csv")
        
        logging.info(f'clean_loss:{clean_test_loss_avg_over_batch} bd_loss:{bd_test_loss_avg_over_batch} clean_acc:{test_acc} asr:{test_asr} ra:{test_ra}')
        
        result = {}
        result['model'] = net
        save_defense_result(
            model_name=args.model,
            num_classes=args.num_classes,
            model=net.cpu().state_dict(),
            save_path=args.save_path,
        )
        
        logging.info("===End===")
        
        return 0
    
    def defense(self,result_file):
        self.set_result(result_file)
        self.set_logger()
        result = self.mitigation()
        return result

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=sys.argv[0])
    rnp.add_arguments(parser)
    args = parser.parse_args()
    method = rnp(args)
    os.environ["CUDA_VISIBLE_DEVICES"] = "%s" % args.device[-1]
    args.device = 'cuda:0'
    print(method.args)
    result = method.defense(args.result_file)