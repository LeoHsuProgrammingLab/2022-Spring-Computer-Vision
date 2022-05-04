

## You could add some configs to perform other training experiments...

LeNet_cfg = {
    'model_type': 'LeNet',
    'data_root' : './p2_data/annotations/train_annos.json',
    
    # ratio of training images and validation images 
    'split_ratio': 0.9,
    # set a random seed to get a fixed initialization 
    'seed': 687,
    
    # training hyperparameters
    'batch_size': 16,
    'lr':0.01,
    'milestones': [15, 25],
    'num_out': 10,
    'num_epoch': 20,
}

myResnet_cfg = {
    'model_type': 'myResnet',
    'data_root' : './p2_data/annotations/train_annos.json',
    
    # ratio of training images and validation images 
    'split_ratio': 0.9,
    # set a random seed to get a fixed initialization 
    'seed': 687,
    
    # training hyperparameters
    'batch_size': 16,
    'lr':0.004,
    'milestones': [15, 25],
    'num_out': 10,
    'num_epoch': 20,
}


Resnet50_cfg = {
    'model_type': 'Resnet50',
    'data_root' : './p2_data/annotations/train_annos.json',
    
    # ratio of training images and validation images 
    'split_ratio': 0.9,
    # set a random seed to get a fixed initialization 
    'seed': 687,
    
    # training hyperparameters
    'batch_size': 16,
    'lr':0.01,
    'milestones': [15, 25],
    'num_out': 10,
    'num_epoch': 20,
}

Resnext50_cfg = {
    'model_type': 'Resnext50',
    'data_root' : './p2_data/annotations/train_annos.json',
    
    # ratio of training images and validation images 
    'split_ratio': 0.9,
    # set a random seed to get a fixed initialization 
    'seed': 687,
    
    # training hyperparameters
    'batch_size': 16,
    'lr':0.01,
    'milestones': [15, 25],
    'num_out': 10,
    'num_epoch': 20,
}

Resnet18_cfg = {
    'model_type': 'Resnet18',
    'data_root' : './p2_data/annotations/train_annos.json',
    
    # ratio of training images and validation images 
    'split_ratio': 0.9,
    # set a random seed to get a fixed initialization 
    'seed': 687,
    
    # training hyperparameters
    'batch_size': 16,
    'lr':0.01,
    'milestones': [15, 25],
    'num_out': 10,
    'num_epoch': 20,
}