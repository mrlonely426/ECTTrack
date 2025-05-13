class EnvironmentSettings:
    def __init__(self):
        self.workspace_dir = '/home/xl/codes/ECTtrack'    # Base directory for saving network checkpoints.
        self.tensorboard_dir = '/home/xl/codes/ECTtrack/tensorboard'    # Directory for tensorboard files.
        self.pretrained_networks = '/home/xl/codes/ECTtrack/pretrained_networks'
        self.lasot_dir = '/home1/Dataset/lasot/LaSOTBenchmark'
        self.got10k_dir = '/home1/Dataset/GOT10K/baiduyun/train'
        self.got10k_val_dir = '/home1/Dataset/GOT10K/baiduyun/val'
        self.lasot_lmdb_dir = '/home/xl/codes/ECTtrack/data/lasot_lmdb'
        self.got10k_lmdb_dir = '/home/xl/codes/ECTtrack/data/got10k_lmdb'
        self.trackingnet_dir = '/home1/Dataset/TrackingNet'
        self.trackingnet_lmdb_dir = '/home/xl/codes/ECTtrack/data/trackingnet_lmdb'
        self.coco_dir = '/home1/Dataset/COCO'
        self.coco_lmdb_dir = '/home/xl/codes/ECTtrack/data/coco_lmdb'
        self.lvis_dir = ''
        self.sbd_dir = ''
        self.imagenet_dir = '/home/xl/codes/ECTtrack/data/vid'
        self.imagenet_lmdb_dir = '/home/xl/codes/ECTtrack/data/vid_lmdb'
        self.imagenetdet_dir = ''
        self.ecssd_dir = ''
        self.hkuis_dir = ''
        self.msra10k_dir = ''
        self.davis_dir = ''
        self.youtubevos_dir = ''
