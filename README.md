The repo is the official implementation for the paper: [SDformer:Transformer with Spectral Filter and Dynamic Attention for Multivariate Time Series Long-term Forecasting](https://www.ijcai.org/proceedings/2024/629), which is accepted at the main track of **IJCAI 2024**.

## Usage
All datasets in this paper come from [Time Series Library] (https://github.com/thuml/Time-Series-Library). You can download all datasets from [[Google Drive]](https://drive.google.com/drive/folders/13Cg1KYOlzM5C7K8gK8NfC-F3EYxkM3D2?usp=sharing) or [[Baidu Drive]](https://pan.baidu.com/s/1r3KhGd0Q9PJIUZdfEYoymg?pwd=i9iy), Then place the downloaded data in the folder`./dataset`.

You can reproduce the experiment results as the following examples:

```
bash ./scripts/Traffic.sh
```

## Citation

If you find this repo useful, please cite our paper.

```
@inproceedings{ijcai2024p629,
  title     = {SDformer: Transformer with Spectral Filter and Dynamic Attention for Multivariate Time Series Long-term Forecasting},
  author    = {Zhou, Ziyu and Lyu, Gengyu and Huang, Yiming and Wang, Zihao and Jia, Ziyu and Yang, Zhen},
  booktitle = {Proceedings of the Thirty-Third International Joint Conference on
               Artificial Intelligence, {IJCAI-24}},
  pages     = {5689--5697},
  year      = {2024},
}
```
