Here we provide Jupyter notebook for running Double sparse factorization over Resnet-50 on Imagenet.

We require utilities from [OBC](https://github.com/IST-DASLab/OBC) repository, but we provide small patch for datautils.py. 

To run OBC pipeline do:
```
mkdir models_unstr
mkdir scores
python main_trueobs.py rn50 imagenet unstr --sparse-dir models_unstr
python database.py rn50 imagenet unstr loss
python spdy.py rn50 imagenet 2 unstr --dp
python spdy.py rn50 imagenet 3 unstr --dp
python spdy.py rn50 imagenet 4 unstr --dp
python postproc.py rn50 imagenet rn50_unstr_200x_dp.txt --database unstr --bnt --save pruned_2.pth
python postproc.py rn50 imagenet rn50_unstr_300x_dp.txt --database unstr --bnt --save pruned_3.pth
python postproc.py rn50 imagenet rn50_unstr_400x_dp.txt --database unstr --bnt --save pruned_4.pth
```
