[切换到中文版本](README_zh.md) 

[Switch to English Version](README.md)

# 2048AI-Mastermind

2048AI-Mastermind is a solution specifically designed to solve the 2048 game. It recommends the best next move direction by analyzing the current state of the 2048 game, helping players achieve better scores.

## Installation

1. Make sure you have Python 3.11 installed.
2. Install the required dependencies using the following command:
   ```
   pip install -r requirements.txt
   ```

## Usage

### Training the Model

To train the model, execute the following command:
```
python train.py
```

### Resuming Training from Checkpoint

If you need to resume training from a checkpoint, add the following code to the `train_model` function in the `train.py` file:
```python
model = load_model(model_checkpoint_path)
```

### Inference

To perform inference, execute the following command:
```
python predict.py
```

By following these steps, you can easily install, train the model, and use this solution to optimize your performance in the 2048 game.
