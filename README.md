## Heteroskedastic and Imbalanced Deep Learning with Adaptive Regularization
Kaidi Cao, Yining Chen, Junwei Lu, Nikos Arechiga, Adrien Gaidon, Tengyu Ma
_________________

### Dependency

The code is built with following libraries:

- [PyTorch](https://pytorch.org/) 1.8

### Training 

The whole HAR training pipeline can be done in the following three steps:

- To estimate the statistics through a pretrain step

```bash
python cifar_hetero_est.py --mislabel_type hetero --gpu 0 --split 0
```

- To calculate the weights for regularization

```bash
python weight_est.py --statspath ./log/estimate_cifar10_resnet32_hetero_0.5_0_example/stats0.pkl
```

- Finally train a model from the scratch

```bash
python cifar_train.py --dataset cifar10  --rand-number 0 --mislabel_type hetero --imb_type None --gpu 0 --reg_weight 10 --exp_str example --reg_path ./data/cifar10_example_weights.npy
```

### Reference

If you find our paper and repo useful, please cite as

```
@inproceedings{cao2020heteroskedastic,
  title={Heteroskedastic and imbalanced deep learning with adaptive regularization},
  author={Cao, Kaidi and Chen, Yining and Lu, Junwei and Arechiga, Nikos and Gaidon, Adrien and Ma, Tengyu},
  booktitle={International Conference on Learning Representations}, 
  year={2021} 
}
```