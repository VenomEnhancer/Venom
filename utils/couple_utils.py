from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import math

def get_target_class_index(dataset,labels,without_attack_target=False, attack_target=0):
    if dataset=='tiny':
        target_classes = get_target_class(dataset)
        target_indexs = []
        for label_m in target_classes:
            if without_attack_target and label_m == attack_target:
                continue
            target_indexs += torch.where(labels == label_m)[0].tolist()
        total_indexs = set(range(labels.shape[0]))
        other_indexs = list(total_indexs - set(target_indexs))
        return target_indexs, other_indexs
    else:
        return NotImplementedError("Unsupported Dataset")

def get_target_class(dataset):
    if dataset=='tiny':
        return [0, 9, 20, 28, 37, 38, 41, 58, 84, 89, 90, 95, 98, 102, 103, 135, 136, 156, 185, 192]
    else:
        return NotImplementedError("Unsupported Dataset")

def get_target_layer(model_name):
    if model_name == 'vgg19_bn':
        return 'features.49'
    elif model_name == 'preactresnet18':
        return 'layer4.1.conv2'
    elif model_name == 'preactresnet50':
        return 'layer4.2.conv2'
    elif model_name == 'vit_b_16':
        return '1.encoder.layers.encoder_layer_11.dropout'
    elif model_name == 'convnext_tiny':
        return 'features.7.2.block.0'
    else:
        return NotImplementedError("Unsupported Model Structure")

def auto_func(f='func1', batch_num_per_epoch=1):
    def func1(x):
        return torch.exp(-x / 300)
    def func2(x):
        return 1-0.5 * 1.01 ** (x - 1000)
    def cos_func(x):
        return torch.tensor(math.cos(math.pi / 2 * x / (4.0 * batch_num_per_epoch)))
    if f == 'func1':
        return func1
    elif f == 'func2':
        return func2
    elif f == 'cos':
        return cos_func
    return None

def get_single_deep_p1_list():
    return ['single_deep_p1',
            'single_deep_p1_80',
            'single_shallow_conv_5',
            'single_deep_conv_10',
            'single_deep_conv_20',
            'single_deep_conv_30',
            'single_deep_conv_40',
            'single_deep_conv_50',
            'single_deep_conv_60',
            'single_deep_conv_70',
            'single_deep_conv_80',
            'single_deep_conv_90',
            'single_deep_conv_100',
            'single_f40_conv_10',
            'single_f43_conv_10',
            'single_f46_conv_10',
            'cross_2_conv_10',
            'cross_3_conv_10',
            'cross_4_conv_10',
            'common_deep_conv_10',
            'single_deep_conv_110',
            'single_deep_conv_120',
            'single_deep_conv_130',
            'single_deep_conv_140',
            'single_deep_conv_150',
            'random1_deep_conv_10',
            'random2_deep_conv_10',
            'with_common1_10',
            'with_common2_10',
            'new_deep1_conv_10',
            'new_deep2_conv_10',
            'single_10_target2_1',
            'single_10_target2_2'
            ]

def init_layers(model_name, model):
    vgg_layers = {}
    if model_name == 'vgg19_bn':
        target_layer_names = [
            # 'features.36',
            # 'features.40',
            # 'features.43',
            # 'features.46',
            'features.49']
    elif model_name == 'preactresnet18':
        target_layer_names = [
            # 'layer3.0.conv1',
            # 'layer3.0.conv2',
            # 'layer3.1.conv1',
            # 'layer3.1.conv2',
            # 'layer4.0.conv1',
            # 'layer4.0.conv2',
            # 'layer4.1.conv1',
            'layer4.1.conv2',]
    elif model_name == 'convnext_tiny':
        target_layer_names = [
            'features.7.2.block.0',
            'features.7.2.block',]        
    elif model_name == 'InceptionResnetV1':
        target_layer_names = [
            'repeat_3.4.branch0.conv',
            'repeat_3.4.branch1.0.conv',
            'repeat_3.4.branch1.1.conv',
            'repeat_3.4.branch1.2.conv',
            'repeat_3.4.conv2d',
            'block8.branch0.conv',
            'block8.branch1.0.conv',
            'block8.branch1.1.conv',
            'block8.branch1.2.conv',
            'block8.branch1.2.conv',
            'block8.conv2d',]
    else:
        target_layer_names = [get_target_layer(model_name)]
        
    module_dict = dict(model.named_modules())
    for target_layer_name in target_layer_names:
        vgg_layers[target_layer_name] = module_dict[target_layer_name]
    return vgg_layers

def cosine(X,Y):
    # X,Y are 3d tensors
    X = X.view(X.shape[0],-1)
    Y = Y.view(Y.shape[0],-1)
    return F.cosine_similarity(X,Y)

'''
AT with sum of absolute values with power p
Modify the code: https://github.com/AberHu/Knowledge-Distillation-Zoo
'''
class AT(nn.Module):
    '''
	Paying More Attention to Attention: Improving the Performance of Convolutional
	Neural Netkworks wia Attention Transfer
	https://arxiv.org/pdf/1612.03928.pdf
	'''
    def __init__(self, p=2):
        super(AT, self).__init__()
        self.p = p
        
    def forward(self, fm_s, fm_t, mode='clean', sim_lf='at'):
        if sim_lf =='at':
            if mode in get_single_deep_p1_list():
                loss = torch.norm(self.attention_map(fm_s,mode=mode)-self.attention_map(fm_t,mode=mode),self.p,dim=(1,2,3))
                # loss = torch.norm(fm_s-fm_t,self.p,dim=(1,2,3))
            else:
                loss = torch.pow(self.attention_map(fm_s)-self.attention_map(fm_t),2)
                loss = loss.mean(dim=(1,2))
            return loss
        elif sim_lf =='l1':
            loss = torch.mean(torch.abs(fm_s-fm_t),dim=(1,2,3))
            return loss
        elif sim_lf == 'at_l1':
            loss = torch.norm(self.attention_map(fm_s,mode=mode,p=1)-self.attention_map(fm_t,mode=mode,p=1),1,dim=(1,2,3))
            return loss
        return None
    
    def attention_map(self, fm, eps=1e-6, mode='clean',p=0):
        p = self.p if p == 0 else p
        if mode in get_single_deep_p1_list():
            am = torch.pow(torch.abs(fm), p)
            am = torch.sum(am, dim=1, keepdim=True)
            norm = torch.norm(am, dim=(2,3), keepdim=True)
            am = torch.div(am, norm+eps)
        else:
            am = torch.pow(torch.abs(fm), p)
            norm = torch.norm(am, dim=(1,2), keepdim=True)
            am = torch.div(am, norm+eps)
        return am


def get_model_similar_path(model_name,
                        dataset_name,
                        sim_mode='clean',
                        target= 0,
                        args=None,
                        **kwargs,):
    target = args.attack_target
    path = f'../similarity/results/{dataset_name}/{model_name}/json/{sim_mode}.json'
    dic = {}
    with open(path,'r') as f:
        dic = json.load(f)
    layers = dic[str(target)]
    path = f'../similarity/results/{dataset_name}/{model_name}/act/{sim_mode}.pth'
    target_acts = torch.load(path, map_location=args.device)
    res = {}
    if sim_mode in get_single_deep_p1_list():
        for layer_name in layers:
            indexs = layers[layer_name]
            res[layer_name] = {'idx':indexs,'act':target_acts[layer_name]}
            # {'layer1':{'idx':indexs,'act':tensor}}
    else:
        for layer_name in layers:
            indexs = layers[layer_name]
            res[layer_name] = []
            for index in indexs:
                res[layer_name].append({'idx':index,'act':target_acts[layer_name][str(index)]})
        # {'layer1':[{'idx':index,'act':[]}]}
    return res

def get_front_layers(top_sim_channels, model, shared_layer='front'):
    module_dict = dict(model.named_modules())
    front_layers = {}
    for layer_name in top_sim_channels.keys():
        if shared_layer == 'self':
            front_layers[layer_name] = module_dict[layer_name]
        elif shared_layer == 'front':
            if layer_name == 'layer4.1.conv2':
                front_layer_name = 'layer4.1.bn2'
                front_layers[front_layer_name] = module_dict[front_layer_name]
            elif layer_name == 'layer4.1.conv1':
                front_layer_name = 'layer4.1.bn1'
                front_layers[front_layer_name] = module_dict[front_layer_name] 
            elif layer_name == 'features.49':
                front_layer_name = 'features.47'
                front_layers[front_layer_name] = module_dict[front_layer_name] 
            elif layer_name == 'features.46':
                front_layer_name = 'features.44'
                front_layers[front_layer_name] = module_dict[front_layer_name] 
            elif layer_name == 'features.43':
                front_layer_name = 'features.41'
                front_layers[front_layer_name] = module_dict[front_layer_name] 
            elif layer_name == 'features.40':
                front_layer_name = 'features.37'
                front_layers[front_layer_name] = module_dict[front_layer_name] 
    return front_layers 

def get_mtl_opt(weights, args):
    if args.optimizer2 == "sgd":
        optimizer = torch.optim.SGD([weights],
                                    lr=args.lr2,
                                    momentum=args.sgd_momentum,  # 0.9
                                    )
    elif args.optimizer2 == 'adam':
        optimizer = torch.optim.Adam([weights],
                                     lr=args.lr2,
                                     betas=args.adam_betas,
                                     weight_decay=args.wd2,
                                     amsgrad=True)
    else:
        optimizer = None
    return optimizer

def renormalize_weights(weights,T,type='clamp'):
    if type =='add':
        weights = (weights / weights.sum() * T).detach()
    elif type == 'clamp':
        weights = (weights / weights.sum() * T).detach()
        weights = torch.clamp(weights, min=0.01, max=2).detach()
    elif type == 'softmax':
        weights = (torch.softmax(weights, dim=0) * T + 1e-3).detach()
    else:
        pass # do nothing
    return weights
