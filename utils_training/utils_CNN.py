import math
pi=math.pi
import os
import torch
import shutil


def save_checkpoint(flownet_state, posenet_state, is_best, save_path, filename):
    file_prefixes = ['flownet', 'posenet']
    states = [flownet_state, posenet_state]
    #torch.save(state, os.path.join(save_path,filename))
    for (prefix, state) in zip(file_prefixes, states):
        torch.save(state, save_path+'/{}_{}'.format(prefix,filename))
    if is_best:
        for prefix in file_prefixes:
            #shutil.copyfile(os.path.join(save_path,filename), os.path.join(save_path,'model_best.pth'))
            shutil.copyfile(save_path+'/{}_{}'.format(prefix,filename), save_path+'/{}_model_best.pth'.format(prefix))


def load_checkpoint(model, optimizer, scheduler, filename='checkpoint.pth.tar'):
    # Note: Input model & optimizer should be pre-defined.  This routine only updates their states.
    start_epoch = 0
    best_val=-1
    if os.path.isfile(filename):
        print("=> loading checkpoint '{}'".format(filename))
        checkpoint = torch.load(filename)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        #optimizer.load_state_dict(checkpoint['optimizer'])
        #scheduler.load_state_dict(checkpoint['scheduler'])
        try:
            best_val=checkpoint['best_loss']
        except:
            best_val=-1
        print("=> loaded checkpoint '{}' (epoch {})"
                  .format(filename, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(filename))

    return model, optimizer, scheduler, start_epoch, best_val


# Argument parsing
def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'
