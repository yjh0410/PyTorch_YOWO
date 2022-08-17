# Dataset configuration


dataset_config = {
    'ucf24': {
        # dataset
        'data_root': '/mnt/share/ssd2/dataset/STAD/ucf24',
        # 'data_root': 'D:/python_work/spatial-temporal_action_detection/dataset/ucf24',
        # input size
        'train_size': 224,
        'test_size': 224,
        # transform
        'pixel_mean': [0., 0., 0.],
        'pixel_std': [1., 1., 1.],
        'jitter': 0.2,
        'hue': 0.1,
        'saturation': 1.5,
        'exposure': 1.5,
        'sampling_rate': 1,
        'len_clip': 16,
        # freeze backbone
        'freeze_backbone_2d': False,
        'freeze_backbone_3d': False,
        # train config
        'batch_size': 8,
        'test_batch_size': 8,
        'accumulate': 16,
        'optimizer': 'adamw',
        'momentum': 0.9,
        'weight_decay': 5e-4,
        'max_epoch': 5,
        'lr_epoch': [1, 2, 3, 4],
        'base_lr': 1e-4,
        'lr_decay_ratio': 0.5,
        # warmup strategy
        'warmup': 'linear',
        'warmup_factor': 0.00066667,
        'wp_iter': 500,
        # class names
        'label_map': (
                    'Basketball',     'BasketballDunk',    'Biking',            'CliffDiving',
                    'CricketBowling', 'Diving',            'Fencing',           'FloorGymnastics', 
                    'GolfSwing',      'HorseRiding',       'IceDancing',        'LongJump',
                    'PoleVault',      'RopeClimbing',      'SalsaSpin',         'SkateBoarding',
                    'Skiing',         'Skijet',            'SoccerJuggling',    'Surfing',
                    'TennisSwing',    'TrampolineJumping', 'VolleyballSpiking', 'WalkingWithDog'
                ),
    },
    
    'jhmdb21': {
        # dataset
        'data_root': '/mnt/share/ssd2/dataset/STAD/jhmdb21',
        'data_root': 'D:/python_work/spatial-temporal_action_detection/dataset/jhmdb21',
        # input
        'train_size': 224,
        'test_size': 224,
        # transform
        'pixel_mean': [0., 0., 0.],
        'pixel_std': [1., 1., 1.],
        'jitter': 0.2,
        'hue': 0.1,
        'saturation': 1.5,
        'exposure': 1.5,
        'sampling_rate': 1,
        'len_clip': 16,
        # freeze backbone
        'freeze_backbone_2d': True,
        'freeze_backbone_3d': True,
        # train config
        'batch_size': 8,
        'test_batch_size': 8,
        'accumulate': 16,
        'optimizer': 'adamw',
        'momentum': 0.9,
        'weight_decay': 5e-4,
        'max_epoch': 5,
        'lr_epoch': [1, 2, 3, 4],
        'base_lr': 1e-4,
        'lr_decay_ratio': 0.5,
        # warmup strategy
        'warmup': 'linear',
        'warmup_factor': 0.00066667,
        'wp_iter': 500,
        # class names
        'label_map': (
                    'brush_hair',   'catch',          'clap',        'climb_stairs',
                    'golf',         'jump',           'kick_ball',   'pick', 
                    'pour',         'pullup',         'push',        'run',
                    'shoot_ball',   'shoot_bow',      'shoot_gun',   'sit',
                    'stand',        'swing_baseball', 'throw',       'walk',
                    'wave'
                ),
    },
    
    'ava_v2.2':{
        # dataset
        'data_root': '/mnt/share/sda1/dataset/STAD/AVA_Dataset',
        'frames_dir': 'frames/',
        'frame_list': 'frame_lists/',
        'annotation_dir': 'annotations/',
        'train_gt_box_list': 'ava_v2.2/ava_train_v2.2.csv',
        'val_gt_box_list': 'ava_v2.2/ava_val_v2.2.csv',
        'train_exclusion_file': 'ava_v2.2/ava_train_excluded_timestamps_v2.2.csv',
        'val_exclusion_file': 'ava_v2.2/ava_val_excluded_timestamps_v2.2.csv',
        'labelmap_file': 'ava_v2.2/ava_action_list_v2.2.pbtxt',
        # input size
        'train_size': 224,
        'test_size': 224,
        # transform
        'pixel_mean': [0., 0., 0.],
        'pixel_std': [1., 1., 1.],
        'jitter': 0.2,
        'hue': 0.1,
        'saturation': 1.5,
        'exposure': 1.5,
        'sampling_rate': 1,
        'len_clip': 16,
        # freeze backbone
        'freeze_backbone_2d': False,
        'freeze_backbone_3d': False,
        # train config
        'batch_size': 8,
        'test_batch_size': 8,
        'accumulate': 16,
        'optimizer': 'adamw',
        'momentum': 0.9,
        'weight_decay': 5e-4,
        'max_epoch': 5,
        'lr_epoch': [1, 2, 3, 4],
        'base_lr': 1e-4,
        'lr_decay_ratio': 0.5,
        # warmup strategy
        'warmup': 'linear',
        'warmup_factor': 0.00066667,
        'wp_iter': 500,
        # class names
        'label_map': (
                    'Basketball',     'BasketballDunk',    'Biking',            'CliffDiving',
                    'CricketBowling', 'Diving',            'Fencing',           'FloorGymnastics', 
                    'GolfSwing',      'HorseRiding',       'IceDancing',        'LongJump',
                    'PoleVault',      'RopeClimbing',      'SalsaSpin',         'SkateBoarding',
                    'Skiing',         'Skijet',            'SoccerJuggling',    'Surfing',
                    'TennisSwing',    'TrampolineJumping', 'VolleyballSpiking', 'WalkingWithDog'
                ),        
    }
}