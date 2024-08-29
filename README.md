The repo is the official implementation for the **IJCAI'24 Main Track** paper: [SDformer:Transformer with Spectral Filter and Dynamic Attention for Multivariate Time Series Long-term Forecasting](https://www.ijcai.org/proceedings/2024/629). **This paper was selected as the only long oral presentation of the Time Series Session**.

## About Authors
The student authors of this paper, ranked 1st, 3rd, and 4th, all come from HKUST and HKUST(GZ). They completed their undergraduate studies at the School of Computer Science at [Beijing University of Technology](https://www.bjut.edu.cn/). If you are interested in Time Series Analysis and would like further discussion, please feel free to contact Ziyu Zhou (MPhil student) at zzhou651@connect.hkust-gz.edu.cn.

## Usage
All datasets in this paper come from [Time Series Library](https://github.com/thuml/Time-Series-Library). You can download all datasets from [[Google Drive]](https://drive.google.com/drive/folders/13Cg1KYOlzM5C7K8gK8NfC-F3EYxkM3D2?usp=sharing) or [[Baidu Drive]](https://pan.baidu.com/s/1r3KhGd0Q9PJIUZdfEYoymg?pwd=i9iy), Then place the downloaded data in the folder`./dataset`.

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
  year      = {2024}
}
```
