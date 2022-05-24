# Extractors
包含两部分内容，一部分是多智能体强化学习算法部分，一部分是extractors多智能体环境部分。其中前者参考自[MADDPG algorithm](https://github.com/openai/maddpg)，后者参考自[Multi-Agent Particle Environments (MPE)](https://github.com/openai/multiagent-particle-envs).。
## 安装
- 首先使用Anaconda3创建一个虚拟环境，python的版本为3.5.4；切换到此环境
- 运行根目录下的`batchinstall.py`已安装依赖的包
- 进入`./extractors-envs`目录下，键入`pip install -e .`
- 然后进入`./maddpg`目录下，键入`pip install -e .`
## 训练
- 进入`./maddpg/experiments`目录下，键入`python train.py`
### 需要注意的命令行选项
除[此处](https://github.com/openai/maddpg/blob/master/README.md)列举的之外，需要注意的是`--display`由于没有实现渲染而无法使用，`--benchmark`、`--benchmark-iters`和`--benchmark-dir`由于没有benchmark数据而无法使用。
## 回放
- 进入`./maddpg/experiments`目录下，键入`python replay.py`
- 回放记录储存在`./maddpg/experiments/replay_logs`目录下
## 代码结构
- `./maddpg`目录下包含了在extractors环境下的MADDPG算法
- `./extrators-envs`目录下包含了extractors环境