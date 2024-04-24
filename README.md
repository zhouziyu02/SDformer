The repo is the official implementation for the paper: [SDformer:Transformer with Spectral Filter and Dynamic Attention for Multivariate Time Series Long-term Forecasting], which is accepted at the main track of **IJCAI 2024**.

## Usage
 
The project is completely based on the [Time Series Library(TSlib)](https://github.com/thuml/Time-Series-Library).

You can reproduce the training process of SDformer by:

1. adding the model in [SDformer.py](https://github.com/zhouziyu02/SDformer/blob/main/SDformer.py) into the `./models` of TSlib,

2. adding the Dynamic_Directional_Attention in [DDA.py](https://github.com/zhouziyu02/SDformer/blob/main/DDA.py) into the `./layers/SelfAttention_Family`,

3. adding the [SFT.py](https://github.com/zhouziyu02/SDformer/blob/main/SFT.py) into the `./layers`,

4. adding the following lines into `./run.py`. 

```
parser.add_argument('--top_k', type=int, default=4, help='for Filter in SFT')

parser.add_argument('--window_size', type=int, default=12, help='Window Size in SFT')
```
## Initialization

The [results.txt](https://github.com/zhouziyu02/SDformer/blob/main/results.txt) records the complete results of the experiments where SDformer achieved SOTA. 

Please note that the hyperparameter setting for each experiment is set as 'windowsize_topk_long_term_forecast_XXXXX'.
