'''
This file is modified based on the following source:
link : https://github.com/luluppang/BCU/blob/main/defense/ft/distillation_dropout_increase.py
The defense method is called bcu.
'''

import argparse
import logging
import os
import random
import sys

sys.path.append('../')
sys.path.append(os.getcwd())
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from tqdm import tqdm
import numpy as np

# from utils import args
from defense.base import defense
from utils.choose_index import choose_index
from utils.aggregate_block.fix_random import fix_random
from utils.aggregate_block.dataset_and_transform_generate import get_transform
from utils.aggregate_block.model_trainer_generate import generate_cls_model
from utils.bd_dataset import prepro_cls_DatasetBD
from utils.trainer_cls import PureCleanModelTrainer, get_target_dataset_loader, given_dataloader_test, Metric_Aggregator
# from utils.input_aware_utils import progress_bar
from utils.nCHW_nHWC import nCHW_to_nHWC
from utils.save_load_attack import load_attack_result
import yaml
from pprint import pprint, pformat
from utils.bd_dataset_v2 import prepro_cls_DatasetBD_v2
from utils.aggregate_block. dataset_and_transform_generate import get_input_shape, get_num_classes, get_transform

def CrossEntropy(outputs, targets, T=3):
    log_softmax_outputs = F.log_softmax(outputs/T, dim=1)
    softmax_targets = F.softmax(targets/T, dim=1)

    # softmax_targets = torch.eye(softmax_targets.shape[1])[softmax_targets.argmax(-1)].cuda()

    kl_loss = nn.KLDivLoss(reduction="batchmean", log_target=False)
    output = kl_loss(log_softmax_outputs, softmax_targets)
    return output
    #return -(log_softmax_outputs * softmax_targets).sum(dim=1).mean()

class bcu(defense):
    
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
      
    def add_arguments(parser):
        parser.add_argument('--device', type=str, help='cuda, cpu')
        parser.add_argument('--checkpoint_load', type=str)
        parser.add_argument('--checkpoint_save', type=str)
        parser.add_argument('--log', type=str)
        parser.add_argument('--log_file_name', type=str)
        parser.add_argument("--data_root", type=str)

        parser.add_argument('--dataset', type=str, help='mnist, cifar10, gtsrb, celeba, tiny')
        parser.add_argument("--num_classes", type=int)
        parser.add_argument("--input_height", type=int)
        parser.add_argument("--input_width", type=int)
        parser.add_argument("--input_channel", type=int)

        parser.add_argument('--epochs', type=int)
        parser.add_argument('--batch_size', type=int)
        parser.add_argument("--num_workers", type=float)
        parser.add_argument('--lr', type=float)
        parser.add_argument('--lr_scheduler', type=str, help='the scheduler of lr')

        parser.add_argument('--poison_rate', type=float)
        parser.add_argument('--target_type', type=str, help='all2one, all2all, cleanLabel')
        parser.add_argument('--target_label', type=int)

        parser.add_argument('--model', type=str, help='resnet18')
        parser.add_argument('--seed', type=str, help='random seed')
        parser.add_argument('--index', type=str, help='index of clean data')
        parser.add_argument('--result_file', type=str, help='the location of result')

        parser.add_argument('--yaml_path', type=str, default="./config/defense/ft/config.yaml", help='the path of yaml')

        # set the parameter for the ft defense
        parser.add_argument('--ratio', type=float, help='the ratio of clean data loader')

        # set the parameter for cross-entropy with high probability samples
        parser.add_argument('--tau', type=float, default=0.99, help='confidence score of high probability samples')
        parser.add_argument('--layerwise_ratio', type=float, nargs='+')
        # set the parameter for out of distribution
        parser.add_argument('--use_tiny_imagenet', action='store_true')
        parser.add_argument('--use_gtsrb', action='store_true')
        parser.add_argument('--use_cifar10', action='store_true')
        
    def set_result(self, result_file):
        attack_file = 'record/' + result_file
        save_path = 'record/' + result_file + '/defense/bcu/'
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
        
    def test_accuracies(self, args, model_new, data_bd_loader, data_clean_loader):
        with torch.no_grad():
            model_new.eval()
            asr_acc = 0
            for i, (inputs, labels, *_) in enumerate(data_bd_loader):  # type: ignore
                inputs, labels = inputs.to(args.device), labels.to(args.device)
                outputs = model_new(inputs)
                pre_label = torch.max(outputs, dim=1)[1]
                asr_acc += torch.sum(pre_label == labels) / len(data_bd_loader.dataset)
            ASR = (asr_acc.item() * 100)

            clean_acc = 0
            for i, (inputs, labels, *_) in enumerate(data_clean_loader):  # type: ignore
                inputs, labels = inputs.to(args.device), labels.to(args.device)
                outputs = model_new(inputs)
                pre_label = torch.max(outputs, dim=1)[1]
                clean_acc += torch.sum(pre_label == labels) / len(data_clean_loader.dataset)
            ACC = (clean_acc.item() * 100)
        return ASR, ACC
    
    def distillation(self, arg, teacher, student, optimizer, scheduler, criterion, criterion_pseudo_label, epoch, trainloader, testloader_cl=None, testloader_bd=None):
        total_clean, total_clean_correct, train_loss = 0, 0, 0
        batch_loss = []
        for i, (inputs, labels, *_) in enumerate(trainloader):
            student.train()
            inputs, labels = inputs.to(arg.device), labels.to(arg.device)
            with torch.no_grad():
                teacher.eval()
                teacher_outputs = teacher(inputs).detach()
                teacher_targets = torch.softmax(teacher_outputs, dim=1)
                max_p, p_hat = torch.max(teacher_targets, dim=1)
                mask = max_p.ge(args.tau).float()

            outputs = student(inputs)
            # criterion_cls = nn.CrossEntropyLoss(reduction='none')
            loss_cls = (criterion_pseudo_label(outputs, p_hat) * mask).sum(-1).mean()

            loss_KL = CrossEntropy(outputs, teacher_outputs, 1 + (3 / args.epochs) * float(1 + epoch))
            # loss_KL = CrossEntropy(outputs, teacher_outputs, 1)
            # loss = loss_KL + 0.005 * loss_cls
            loss = loss_KL
            # loss = CrossEntropy(outputs, teacher_outputs, 1)
            # loss = criterion(outputs, labels)

            batch_loss.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            total_clean_correct += torch.sum(torch.argmax(outputs[:], dim=1) == labels[:])
            total_clean += inputs.shape[0]
            avg_acc_clean = float(total_clean_correct.item() * 100.0 / total_clean)
            # progress_bar(i, len(trainloader), 'Epoch: %d | Loss: %.3f | Training Acc: %.3f%% (%d/%d)' % (epoch, train_loss / (i + 1), avg_acc_clean, total_clean_correct, total_clean))
        print('Epoch:{} | Loss: {:.3f} | Training Acc: {:.3f}%({}/{})'.format(epoch, train_loss / (i + 1), avg_acc_clean,
                                                                                total_clean_correct, total_clean))
        logging.info('Epoch:{} | Loss: {:.3f} | Training Acc: {:.3f}%({}/{})'.format(epoch, train_loss / (i + 1), avg_acc_clean,
                                                                                total_clean_correct, total_clean))
        student.eval()

        clean_accuracy = 0
        ASR = 0
        if testloader_cl is not None:
            total_clean_test, total_clean_correct_test, test_loss = 0, 0, 0
            for i, (inputs, labels, *_) in enumerate(testloader_cl):
                inputs, labels = inputs.to(arg.device), labels.to(arg.device)
                outputs = student(inputs)
                loss = criterion(outputs, labels)
                test_loss += loss.item()

                total_clean_correct_test += torch.sum(torch.argmax(outputs[:], dim=1) == labels[:])
                total_clean_test += inputs.shape[0]
                avg_acc_clean = float(total_clean_correct_test.item() * 100.0 / total_clean_test)
                clean_accuracy = avg_acc_clean
                # progress_bar(i, len(testloader), 'Test %s ACC: %.3f%% (%d/%d)' % (word, avg_acc_clean, total_clean_correct, total_clean))
            print('Epoch:{} | Test Acc: {:.3f}%({}/{})'.format(epoch, avg_acc_clean, total_clean_correct, total_clean))
            logging.info(
                'Epoch:{} | Test Acc: {:.3f}%({}/{})'.format(epoch, avg_acc_clean, total_clean_correct, total_clean))

        if testloader_bd is not None:
            total_clean_test, total_clean_correct_test, test_loss = 0, 0, 0
            for i, (inputs, labels, *_) in enumerate(testloader_bd):
                inputs, labels = inputs.to(arg.device), labels.to(arg.device)
                outputs = student(inputs)
                loss = criterion(outputs, labels)
                test_loss += loss.item()

                total_clean_correct_test += torch.sum(torch.argmax(outputs[:], dim=1) == labels[:])
                total_clean_test += inputs.shape[0]
                avg_acc_clean = float(total_clean_correct_test.item() * 100.0 / total_clean_test)
                ASR = avg_acc_clean
                # progress_bar(i, len(testloader), 'Test %s ACC: %.3f%% (%d/%d)' % (word, avg_acc_clean, total_clean_correct, total_clean))
            print('Epoch:{} | Test Asr: {:.3f}%({}/{})'.format(epoch, avg_acc_clean, total_clean_correct, total_clean))
            logging.info(
                'Epoch:{} | Test Asr: {:.3f}%({}/{})'.format(epoch, avg_acc_clean, total_clean_correct, total_clean))
        one_epoch_loss = sum(batch_loss) / len(batch_loss)
        if args.lr_scheduler == 'ReduceLROnPlateau':
            scheduler.step(one_epoch_loss)
        elif args.lr_scheduler == 'CosineAnnealingLR':
            scheduler.step()
        return teacher, student, clean_accuracy, ASR
        
    def mitigation(self):
        self.set_devices()
        fix_random(self.args.random_seed)
        
        train_tran = get_transform(self.args.dataset, *([self.args.input_height,self.args.input_width]) , train = True)
        test_tran = get_transform(self.args.dataset, *([self.args.input_height,self.args.input_width]) , train = False)
        
        logging.info('----------- Data Initialization --------------')
        data_set = self.result['clean_train'].wrapped_dataset
        data_set_o = prepro_cls_DatasetBD(
            full_dataset_without_transform=data_set,
            poison_idx=np.zeros(len(data_set)),  # one-hot to determine which image may take bd_transform
            bd_image_pre_transform=None,
            bd_label_pre_transform=None,
            ori_image_transform_in_loading=train_tran,
            ori_label_transform_in_loading=None,
            add_details_in_preprocess=False,
        )
        data_loader = torch.utils.data.DataLoader(data_set_o, batch_size=args.batch_size, num_workers=args.num_workers,
                                                shuffle=True)
        trainloader = data_loader

        clean_dataset = prepro_cls_DatasetBD_v2(self.result['clean_train'].wrapped_dataset)
        data_all_length = len(clean_dataset)
        ran_idx = choose_index(args, data_all_length) 
        log_index = args.log + 'index.txt'
        clean_dataset.subset(ran_idx)
        data_set_without_tran = clean_dataset
        data_set_o = self.result['clean_train']
        data_set_o.wrapped_dataset = data_set_without_tran
        data_set_o.wrap_img_transform = train_tran
        val_loader = torch.utils.data.DataLoader(data_set_o, batch_size=args.batch_size, num_workers=args.num_workers,
                                                shuffle=True)
        
        data_bd_testset = self.result['bd_test']
        data_bd_testset.wrap_img_transform = test_tran
        data_bd_loader = torch.utils.data.DataLoader(data_bd_testset, batch_size=args.batch_size, num_workers=args.num_workers,drop_last=False, shuffle=True,pin_memory=args.pin_memory)

        tran = get_transform(args.dataset, *([args.input_height, args.input_width]), train=False)
        data_clean_test = self.result['clean_test'].wrapped_dataset
        data_clean_testset = prepro_cls_DatasetBD(
            full_dataset_without_transform=data_clean_test,
            poison_idx=np.zeros(len(data_clean_test)),  # one-hot to determine which image may take bd_transform
            bd_image_pre_transform=None,
            bd_label_pre_transform=None,
            ori_image_transform_in_loading=tran,
            ori_label_transform_in_loading=None,
            add_details_in_preprocess=False,
        )
        data_clean_loader = torch.utils.data.DataLoader(data_clean_testset, batch_size=args.batch_size,
                                                        num_workers=args.num_workers, drop_last=False, shuffle=True,
                                                        pin_memory=True)
        
        target_test_dataloader = get_target_dataset_loader(args, data_clean_testset, data_bd_testset[0][1])
        
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
        
        # dropout-based initialization
        model_new = generate_cls_model(model_name=self.args.model, num_classes=self.args.num_classes)
        new_state = model_new.cpu().state_dict()
        old_state = {k: v.cpu() for k, v in copy.deepcopy(self.result['model']).items()}
        num_layers = 0
        for key in self.result['model'].keys():
            if key.find('bn') != -1 or key.find('shortcut.1') != -1:
                continue
            if key.endswith('.weight') or key.endswith('.bias'):
                p = self.args.layerwise_ratio[0]
                if key.startswith('layer1'):
                    p = self.args.layerwise_ratio[1]
                elif key.startswith('layer2'):
                    p = self.args.layerwise_ratio[2]
                elif key.startswith('layer3'):
                    p = self.args.layerwise_ratio[3]
                elif key.startswith('layer4'):
                    p = self.args.layerwise_ratio[4]
                elif key.startswith('fc'):
                    p = self.args.layerwise_ratio[5]

                mask_one = torch.ones(old_state[key].shape) * (1 - p)
                mask = torch.bernoulli(mask_one)
                # masked_weight = old_state[key] * mask * (1/(1-p)) + new_state[key] * (1 - mask)
                masked_weight = old_state[key] * mask + new_state[key] * (1 - mask)     # 1 copy, 0 random
                old_state[key] = masked_weight
        model_new.load_state_dict(old_state, strict=False)
        model_new.to(self.args.device)
        
        optimizer = torch.optim.SGD(model_new.parameters(), lr=self.args.lr, momentum=0.9, weight_decay=5e-4)
        if self.args.lr_scheduler == 'ReduceLROnPlateau':
            scheduler = getattr(torch.optim.lr_scheduler, self.args.lr_scheduler)(optimizer)
        elif self.args.lr_scheduler == 'CosineAnnealingLR':
            scheduler = getattr(torch.optim.lr_scheduler, self.args.lr_scheduler)(optimizer, T_max=100)
        criterion = nn.CrossEntropyLoss()
        criterion_pseudo_label = nn.CrossEntropyLoss(reduction='none')# this is for pseudo-label cross-entropy loss so set reduction=none
        
        original_ASR, original_ACC = self.test_accuracies(self.args, model, data_bd_loader, data_clean_loader)
        logging.info('Original Test Acc: {:.3f}%'.format(original_ACC))
        logging.info('Original Test Asr: {:.3f}%'.format(original_ASR))
        
        clean_accuracies = []
        ASRs = []
        ASRs.append(original_ASR)
        clean_accuracies.append(original_ACC)
        
        best_acc = 0
        best_asr = 0
        for i in range(self.args.epochs):
            teacher, student, clean_accuracy, ASR = self.distillation(self.args, model, model_new, optimizer, scheduler, criterion, criterion_pseudo_label, i, val_loader,
                                                        testloader_cl=data_clean_loader, testloader_bd=data_bd_loader)
            clean_accuracies.append(clean_accuracy)
            ASRs.append(ASR)
            if best_acc < clean_accuracy:
                best_acc = clean_accuracy
                best_asr = ASR

        logging.info('Best Test Acc: {:.3f}%'.format(best_acc))
        logging.info('Best Test Asr: {:.3f}%'.format(best_asr))

        # evalution
        test_dataloader_dict = {}
        test_dataloader_dict["clean_test_dataloader"] = data_clean_loader
        test_dataloader_dict["bd_test_dataloader"] = data_bd_loader
        test_dataloader_dict["target_test_dataloader"] = target_test_dataloader
        self.set_trainer(model_new)
        self.trainer.set_with_dataloader(
            train_dataloader = val_loader,
            test_dataloader_dict = test_dataloader_dict,

            criterion = criterion,
            optimizer = None,
            scheduler = None,
            device = self.args.device,
            amp = self.args.amp,

            frequency_save = self.args.frequency_save,
            save_folder_path = self.args.save_path,
            save_prefix = 'bcu',

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
                "test_acc": test_acc,
                "test_asr": test_asr,
                "test_ra": test_ra,
                "test_tacc": test_tacc
                
            })
        agg.to_dataframe().to_csv(f"{args.save_path}bcu_df_summary.csv")
        
        logging.info(f'clean_loss:{clean_test_loss_avg_over_batch} bd_loss:{bd_test_loss_avg_over_batch} clean_acc:{test_acc} asr:{test_asr} ra:{test_ra}')
        
        save_defense_result(
            model_name=args.model,
            num_classes=args.num_classes,
            model=model_new.cpu().state_dict(),
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
    bcu.add_arguments(parser)
    args = parser.parse_args()
    method = bcu(args)
    os.environ["CUDA_VISIBLE_DEVICES"] = "%s" % args.device[-1]
    args.device = 'cuda:0'
    print(method.args)
    result = method.defense(args.result_file)