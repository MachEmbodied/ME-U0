"""sim_eval — 独立的仿真测评 client 包。

可直接 rsync 到仿真机器运行，不依赖 leap 包和 torch/CUDA。

依赖: numpy, websockets, msgpack, tqdm, imageio (可选, 用于录制视频)
"""
