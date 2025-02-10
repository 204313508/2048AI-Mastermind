[Switch to Chinese Version](README_zh.md)  
[切换到中文版本](README_zh.md)

# 2048AI-Mastermind

2048AI-Mastermind is a solution designed specifically for solving the 2048 game. It analyzes the current state of the 2048 game to recommend the best next move direction, helping players achieve better scores in the game.

## Installation

1. Ensure you have Python version 3.11 installed.
2. Install the required dependencies using the following command:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Running the Interface

To start the `2048AI-Mastermind` interface, execute the following command:
```python
python puzzle.py
```

This will open a window displaying the 2048 game interface. In this interface, you can control the game using the arrow keys on your keyboard or the corresponding shortcuts.

Here is a screenshot example of the game interface:  
![Interface Screenshot](./demo.png)

### Controlling the Game

- **Click Input**: You can input numbers by clicking on the tiles.
- **Arrow Keys**: The `Up, Down, Left, Right` keys correspond to moving up, down, left, and right, respectively.

### Predicting the Next Move

You can also click the "Predict Next Move" button on the interface to let the AI suggest the best move direction. This will display the best move direction recommended by the AI based on the current game state.

---

### Adjusting Parameters

- **Simulation Time**: Adjust the simulation time to optimize the AI's decision-making accuracy. The longer the simulation time, the more simulations the AI performs, leading to more accurate decisions.
- **Exploration Factor**: Adjust the exploration factor to control the AI's level of exploration. A larger exploration factor makes the AI more inclined to explore unknown paths; conversely, a smaller factor makes it lean towards choosing the currently known best path.

#### Checking Parameter Descriptions

- Click on "Help" -> "Parameter Descriptions" in the menu bar to view detailed explanations about simulation time and exploration factors.

### Parameter Effect Explanation

The following are the probabilities of the AI achieving different goals with a simulation time of 0.5 seconds:

| Target Score    | Achievement Probability |
|----------------|-------------------------|
| 512 and above   | 97%                     |
| 1024 and above  | 93%                     |
| 2048           | 60%                     |

#### Exiting the Game

- You can exit the game by pressing the `Esc` key or clicking the close button in the upper right corner of the window.

## Reference Project

This project references the following open-source project: [https://github.com/TwoPoint0200/2048-Python-AI](https://github.com/TwoPoint0200/2048-Python-AI)