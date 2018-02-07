import torch.optim.lr_scheduler as lr_scheduler

class scheduler_builder:
    def __init__(self):
        pass
        
    def build(self, cf, optimizer):
    	if cf.scheduler == 'ReduceLROnPlateau':
    		scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=cf.sched_patience, min_lr=1e-10, factor=cf.decay)
    	elif cf.scheduler == 'Step':
    		scheduler = lr_scheduler.StepLR(optimizer, step_size=cf.step_size, gamma=cf.decay, last_epoch=-1)
    	elif cf.scheduler == 'MultiStep':
    		scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=cf.milestone, gamma=cf.decay, last_epoch=-1)
    	elif cf.scheduler == 'Exponential':
    		scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=cf.decay, last_epoch=-1)
        elif cf.scheduler == None or cf.scheduler == 'None':
            scheduler = None
    	else:
    		raise ValueError('Unknown scheduler type')
    	return scheduler