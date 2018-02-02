import sys
from torch import optim

class Optimizer_builder():
    def __init__(self):
        pass
        
    def build(self, cf, net):
        if cf.optimizer.lower() == 'adam':
            return optim.Adam([
                {'params': [param for name, param in net.named_parameters() if name[-4:] == 'bias'],
                     'lr': float(cf.learning_rate_bias)},
                {'params': [param for name, param in net.named_parameters() if name[-4:] != 'bias'],
                     'lr': float(cf.learning_rate), 'weight_decay': float(cf.weight_decay)}], 
                betas=(cf.momentum1, cf.momentum2), eps=1e-08)
        elif cf.optimizer.lower() == 'rmsprop':
            return optim.RMSprop([
                {'params': [param for name, param in net.named_parameters() if name[-4:] == 'bias'],
                     'lr': cf.learning_rate_bias},
                {'params': [param for name, param in net.named_parameters() if name[-4:] != 'bias'],
                     'lr': cf.learning_rate, 'weight_decay': cf.weight_decay}], 
                alpha=0.99, eps=1e-08, momentum=cf.momentum1, centered=False)
        elif cf.optimizer.lower() == 'sgd':
            return optim.SGD([
                {'params': [param for name, param in net.named_parameters() if name[-4:] == 'bias'],
                     'lr': cf.learning_rate_bias},
                {'params': [param for name, param in net.named_parameters() if name[-4:] != 'bias'],
                     'lr': cf.learning_rate, 'weight_decay': cf.weight_decay}], 
                momentum=cf.momentum1, dampening=0, nesterov=False)
        else:
            sys.exit('Optmizer model not defined properly: ' + cf.optimizer)
    
