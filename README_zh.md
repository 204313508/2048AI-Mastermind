[切换到中文版本](README_zh.md) 

[Switch to English Version](README.md)
# 2048AI-Mastermind

2048AI-Mastermind 是一个专门为解决2048游戏而设计的解决方案。它通过分析当前2048游戏的状态，从而推荐出最佳的下一步移动方向，帮助玩家在游戏中取得更好的成绩。

## 安装

1. 确保您已安装 Python 3.11 版本。
2. 使用以下命令安装所需的依赖：
   ```
   pip install -r requirements.txt
   ```

## 使用方法
### 游玩游戏获取经验
执行以下命令，进行2048游戏的游玩，获取经验：

```
python play_ui.py
```

我们提供图形化游戏界面，点击界面按钮或按键盘上“上下左右”按键即可进行游玩：
![游戏界面示例](play_ui.png)

### 训练模型

要训练模型，请执行以下命令：
```
python train_ui.py
```

我们提供图形化训练界面：  
![训练界面示例](train_ui.png)  
### 从检查点恢复训练

如果需要从检查点恢复训练，请在 `train_ui.py` 文件的 `train_model_from_file` 函数中添加以下代码：
```python
model = load_model(model_checkpoint_path)
```

### 进行推理

要进行推理测试操作，请执行以下命令：
```
python predict.py
```

通过以上步骤，您就可以轻松地安装、训练模型以及使用该方案来优化您在2048游戏中的表现。
