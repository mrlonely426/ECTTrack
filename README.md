# [Efficient Hybrid Linear Self-Attention based Visual Object Tracking with LoRA]


## Installation

Install the dependency packages using the environment file `ecttrack.yml`.

Generate the relevant files:
```
python tracking/create_default_local_file.py --workspace_dir . --data_dir ./data --save_dir ./output
```
After running this command, modify the datasets paths by editing these files
```
lib/train/admin/local.py  # paths about training
lib/test/evaluation/local.py  # paths about testing
```

## Training

* Set the path of training datasets in `lib/train/admin/local.py`
* Place the pretrained backbone model under the `pretrained_models/` folder
* For data preparation, please refer to [this](https://github.com/botaoye/OSTrack/tree/main)

* Run
```
python tracking/train.py --script ecttrack --config ecttrack_256_128x1_ep300 --save_dir ./output --mode single
```
* The training logs will be saved under `output/logs/` folder

## Pretrained tracker model
The pretrained tracker model can be found 

## Tracker Evaluation

* Update the test dataset paths in `lib/test/evaluation/local.py`
* Place the pretrained tracker model under `output/checkpoints/` folder 
* Run
```
python tracking/test.py --tracker_name ecttrack --tracker_param ecttrack_256_128x1_ep300 --dataset got10k_test or trackingnet or lasot
```
* Change the `DEVICE` variable between `cuda` and `cpu` in the `--tracker_param` file for GPU and CPU-based inference, respectively  
* The raw results will be stored under `output/test/` folder





## Acknowledgements
* We use the  Self-Attention Transformer implementation and the pretrained `MobileViTV3` backbone from (https://github.com/micronDLA/MobileViTv3). Thank you!
