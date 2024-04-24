The repo is the official implementation for the paper: [SDformer:Transformer with Spectral Filter and Dynamic Attention for Multivariate Time Series Long-term Forecasting], which is accepted at the main track of **IJCAI 2024**.
 
The project is completely based on the [Time Series Library(TSlib)](https://github.com/thuml/Time-Series-Library).

You can reproduce the training process of SDformer by:

adding the model in SDformer.py into the `./models` of TSlib,

adding the Dynamic_Directional_Attention in DDA.py into the `./layers/SelfAttention_Family`,

adding the SFT.py into the `./layers`.

```
parser.add_argument('--top_k', type=int, default=4, help='for Filter in SFT')

parser.add_argument('--window_size', type=int, default=12, help='Window Size in SFT')
```
