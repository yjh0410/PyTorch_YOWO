# Dataset configuration


dataset_config = {
    'ucf24': {
        # dataset
        'data_root': '/mnt/share/ssd2/dataset/STAD/ucf24',
        'data_root': 'D:/python_work/spatial-temporal_action_detection/dataset/ucf24',
        # transform
        'jitter': 0.2,
        'hue': 0.1,
        'saturation': 1.5,
        'exposure': 1.5,
        'sampling_rate': 1,
        'len_clip': 16,
        # freeze backbone
        'freeze_backbone_2d': False,
        'freeze_backbone_3d': False,
        # class names
        'label_map': (
                    'Basketball',     'BasketballDunk',    'Biking',            'CliffDiving',
                    'CricketBowling', 'Diving',            'Fencing',           'FloorGymnastics', 
                    'GolfSwing',      'HorseRiding',       'IceDancing',        'LongJump',
                    'PoleVault',      'RopeClimbing',      'SalsaSpin',         'SkateBoarding',
                    'Skiing',         'Skijet',            'SoccerJuggling',    'Surfing',
                    'TennisSwing',    'TrampolineJumping', 'VolleyballSpiking', 'WalkingWithDog'
                ),
        # anchor size
        'anchor_size': [[22, 38],
                        [40, 81],
                        [51, 130],
                        [73, 158],
                        [112, 189]], # 224
    },
    
    'jhmdb21': {
        # dataset
        'data_root': '/mnt/share/ssd2/dataset/STAD/jhmdb21',
        'data_root': 'D:/python_work/spatial-temporal_action_detection/dataset/jhmdb21',
        'train_split': 1,
        'test_split': 1,
        # input
        'train_size': 224,
        'test_size': 224,
        'jitter': 0.2,
        'hue': 0.1,
        'saturation': 1.5,
        'exposure': 1.5,
        'sampling_rate': 1,
        'len_clip': 16,
        # freeze backbone
        'freeze_backbone_2d': False,
        'freeze_backbone_3d': False,
        # class names
        'label_map': (
                    'brush_hair',   'catch',          'clap',        'climb_stairs',
                    'golf',         'jump',           'kick_ball',   'pick', 
                    'pour',         'pullup',         'push',        'run',
                    'shoot_ball',   'shoot_bow',      'shoot_gun',   'sit',
                    'stand',        'swing_baseball', 'throw',       'walk',
                    'wave'
                ),
        # anchor size
        'anchor_size': [[30,  99],
                        [53, 128],
                        [56, 180],
                        [98, 185],
                        [157, 200]], # 224
    },
    
    'ava':{
        # dataset
        'data_root': '/mnt/share/ssd2/dataset/STAD/UCF101_24',
        'anno_file': 'JHMDB-GT.pkl',
        # loss
    }
}