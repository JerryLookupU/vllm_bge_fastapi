# vllm_bge_fastapi
# embedding ............................
# 参考 vllm 官方api_server 切换成 AsyncLLMEngine + API
# 并且考虑到 vllm page attention 会产生 显存碎片 定时 清理

# https://github.com/WuNein/vllm4mteb
# https://github.com/vllm-project/vllm/issues/10119


# 开启方法
# CUDA_VISIBLE_DEVICES=4  nohup python vllm_server_fastapi.py > vec_log/vec.log 2>&1 &


# 需要修改模型 architechtures 并注册到模型中
# 注意 具有 long context 模型 需要修改源代码 或者 修改参数 enable_chunked_prefill

# 服务并不复杂 很容易看懂， 能够更便捷支持各种 embedding


# todolist
# pooler 部分优化 目前是 对 fc(hidden_states) ==> pooler 更好得方法是 切片 再进行 fc 再进行 pooler 不过因为内部有token分组 比较麻烦