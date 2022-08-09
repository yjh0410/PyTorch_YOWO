from torch import optim
import torch


def build_optimizer(model,
                    base_lr=0.0,
                    name='sgd',
                    momentum=0.,
                    weight_decay=0.,
                    resume=None):
    print('==============================')
    print('Optimizer: {}'.format(name))
    print('--momentum: {}'.format(momentum))
    print('--weight_decay: {}'.format(weight_decay))


    if name == 'sgd':
        optimizer = optim.SGD(model.parameters(), 
                                lr=base_lr,
                                momentum=momentum,
                                weight_decay=weight_decay)

    elif name == 'adam':
        optimizer = optim.Adam(model.parameters(), 
                                lr=base_lr,
                                weight_decay=weight_decay)
                                
    elif name == 'adamw':
        optimizer = optim.AdamW(model.parameters(), 
                                lr=base_lr,
                                weight_decay=weight_decay)
    
    start_epoch = 0
    if resume is not None:
        print('keep training: ', resume)
        checkpoint = torch.load(resume)
        # checkpoint state dict
        checkpoint_state_dict = checkpoint.pop("optimizer")
        optimizer.load_state_dict(checkpoint_state_dict)
        start_epoch = checkpoint.pop("epoch")
                                
    return optimizer, start_epoch
