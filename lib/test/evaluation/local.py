from lib.test.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.

    settings.davis_dir = ''
    settings.got10k_lmdb_path = '/home/xl/codes/ECTtrack/data/got10k_lmdb'
    settings.got10k_path = '/home1/Dataset/GOT10K/baiduyun'
    settings.got_packed_results_path = ''
    settings.got_reports_path = ''
    settings.itb_path = '/home/xl/codes/ECTtrack/data/itb'
    settings.lasot_extension_path = '/home1/Dataset/LaSOT_Ext'
    settings.lasot_extension_subset_path_path = '/home/xl/codes/ECTtrack/data/lasot_extension_subset'
    settings.lasot_lmdb_path = '/home/xl/codes/ECTtrack/data/lasot_lmdb'
    settings.lasot_path = '/home1/Dataset/lasot/LaSOTBenchmark'
    settings.network_path = '/home/xl/codes/ECTtrack/output/test/networks'    # Where tracking networks are stored.
    settings.nfs_path = '/home1/Dataset/Nfs'
    settings.otb_path = '/home/xl/codes/ECTtrack/data/otb'
    settings.prj_dir = '/home/xl/codes/ECTtrack'
    settings.result_plot_path = '/home/xl/codes/ECTtrack/output/test/result_plots'
    settings.results_path = '/home/xl/codes/ECTtrack/output/test/tracking_results'    # Where to store tracking results
    settings.save_dir = '/home/xl/codes/ECTtrack/output'
    settings.segmentation_path = '/home/xl/codes/ECTtrack/output/test/segmentation_results'
    settings.tc128_path = '/home/xl/codes/ECTtrack/data/TC128'
    settings.tn_packed_results_path = ''
    settings.tnl2k_path = '/home1/Dataset/TNL2K'
    settings.tpl_path = ''
    settings.trackingnet_path = '/home1/Dataset/TrackingNet'
    settings.uav_path = '/home1/Dataset/UAV123'
    settings.vot18_path = '/home/xl/codes/ECTtrack/data/vot2018'
    settings.vot22_path = '/home/xl/codes/ECTtrack/data/vot2022'
    settings.vot_path = '/home/xl/codes/ECTtrack/data/VOT2019'
    settings.youtubevos_dir = ''

    return settings

