# Mn and O
## Dependencies
- Python 3.7 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux))
- [PyTorch = 1.1.0](https://pytorch.org/)

## Preparation
The **pretrained models** can be find in path **"/save_model"**

## Parameters
The configurations in `config.py`can be modified. Here are the details:
- data_info
    - `fea_data`: Mn mineral feature path.
    - `year_data`: age data(Ma) with list form as [50,300].
    
- test_opt
    - `model` and `savename`: The save path and name of pretrained model.
    - `savedir`: The path where the test results will be saved.

## Test Example

Let's create an example. Before running this code, please modify option files to your own configurations including: 
  - proper `fea_data` and `year_data` paths for the data loader. Make sure your Mn mineral characterization 
    table provides a comprehensive coverage of the age to be tested.
  - proper `savedir` for the teat results.
  - whether to use a pretrained model and the path of the model. Remember to change the parameters in 'model_opt' when you replace the model.

To Now, you can implement the atmospheric oxygen concentration prediction with the following code:

```python
python run.py
```
