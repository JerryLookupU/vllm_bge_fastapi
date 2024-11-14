
# code

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from vllm import ModelRegistry
import asyncio
import json
import ssl
from argparse import Namespace
from typing import Any, AsyncGenerator, Optional
import threading
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from typing import Iterable, List, Optional, Tuple
import torch
from torch import nn
from vllm.attention import AttentionMetadata
from vllm.model_executor.layers.pooler import Pooler, PoolingType
from vllm.model_executor.model_loader.weight_utils import default_weight_loader, maybe_remap_kv_scale_name
from vllm.model_executor.models.qwen2 import Qwen2Model
from vllm.model_executor.pooling_metadata import PoolingMetadata
from vllm.sequence import PoolerOutput
from vllm.model_executor.models.utils import is_pp_missing_parameter, make_layers
from vllm.model_executor.models.qwen2 import Qwen2MLP, Qwen2Model

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.entrypoints.launcher import serve_http
from vllm.logger import init_logger
from vllm.sampling_params import SamplingParams
from vllm.usage.usage_lib import UsageContext
from vllm.utils import (FlexibleArgumentParser, iterate_with_cancellation,
                        random_uuid)
from vllm.pooling_params import PoolingParams
from vllm.model_executor.layers.linear import RowParallelLinear

from vllm.version import __version__ as VLLM_VERSION
import os
from functools import wraps


def always_true_is_embedding_model(model_arch: str) -> bool:
    return True


class VllmStellaBGE(nn.Module):
    def __init__(
            self,
            **kwargs,
    ) -> None:
        super().__init__()
        self.model = Qwen2Model(**kwargs)
        vector_dim = 1024
        vector_linear_directory = f"2_Dense_{vector_dim}"
        self.fc = RowParallelLinear(input_size=self.model.config.hidden_size, output_size=vector_dim)
        self._pooler = Pooler(pooling_type=PoolingType.LAST, normalize=True)
        fc_path = os.path.join(self.model.config._name_or_path, vector_linear_directory, "pytorch_model.bin")
        fc_weight = {k.replace("linear.", ""): v for k, v in torch.load(fc_path).items()}
        self.fc.load_state_dict(fc_weight)

    def forward(
            self,
            input_ids: Optional[torch.Tensor],
            positions: torch.Tensor,
            kv_caches: List[torch.Tensor],
            attn_metadata: AttentionMetadata,
            intermediate_tensors: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        hidden_states = self.model.forward(input_ids, positions, kv_caches, attn_metadata, intermediate_tensors)
        return hidden_states

    def pooler(
            self,
            hidden_states: torch.Tensor,
            pooling_metadata: PoolingMetadata,
    ) -> Optional[PoolerOutput]:
        logit, _ = self.fc(hidden_states)
        return self._pooler(logit, pooling_metadata)

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        params_dict = dict(self.named_parameters(remove_duplicate=False))
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            # 这里改一下哦，如果有config
            if self.model.config.tie_word_embeddings and "lm_head.weight" in name:  # model.config 这个要改一下
                continue
            for (param_name, weight_name, shard_id) in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if is_pp_missing_parameter(name, self):
                    continue

                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                # Remapping the name of FP8 kv-scale.
                name = maybe_remap_kv_scale_name(name, params_dict)
                if name is None:
                    continue
                if is_pp_missing_parameter(name, self):
                    continue

                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, loaded_weight)


ModelRegistry.register_model("Stella_bge", VllmStellaBGE)
original_is_embedding_model = ModelRegistry.is_embedding_model
ModelRegistry.is_embedding_model = always_true_is_embedding_model

logger = init_logger("vllm.entrypoints.api_server")
ModelRegistry.register_model("Stella_bge", VllmStellaBGE)

TIMEOUT_KEEP_ALIVE = 5  # seconds.
app = FastAPI()
engine = None

pooling_params = PoolingParams()

visit_count = 0

visit_count_max = 100

lock = threading.Lock()


def visit_counter_cuda_clear(max_count=100):
    global visit_count, visit_count_max, lock

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            global visit_count
            with lock:  # 获取锁
                visit_count += 1
                if visit_count > visit_count_max:
                    visit_count = 0
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
            return func(*args, **kwargs)

        return wrapper

    return decorator


@app.get("/health")
async def health() -> Response:
    """Health check."""
    return Response(status_code=200)


@visit_counter_cuda_clear
@app.post("/encode")
async def encode(request: Request) -> Response:
    """Generate completion for the request.

    The request should be a JSON object with the following fields:
    - prompt: the prompt to use for the generation.
    - stream: whether to stream the results or not.
    - other fields: the sampling parameters (See `SamplingParams` for details).
    """
    request_dict = await request.json()
    # print(request_dict)
    user_query = request_dict.pop("input", "暂无")
    q_type = request_dict.pop("type", "query")
    if q_type == "query":
        query = f"Instruct: Given a web search query, retrieve relevant passages that answer the query. Query: {user_query}"
    else:
        query = f"Instruct: Retrieve semantically similar text. Query: {user_query}"
    request_id = request_dict.pop("request_id", random_uuid())
    # print(request_dict)
    assert engine is not None
    results_generator = engine.encode(query, pooling_params, request_id)
    results_generator = iterate_with_cancellation(
        results_generator, is_cancelled=request.is_disconnected)
    # Non-streaming case
    final_output = None
    try:
        async for request_output in results_generator:
            final_output = request_output
    except asyncio.CancelledError:
        return Response(status_code=499)
    assert final_output is not None
    outputs = final_output.outputs
    assert outputs is not None

    ret = {"embedding": outputs.embedding, "request_id": request_id}
    return JSONResponse(ret)


def build_app(args: Namespace) -> FastAPI:
    global app
    app.root_path = args.root_path
    return app


async def debugfunc(engine):
    query = "hello 你好"
    results_generator = engine.encode(query, pooling_params=PoolingParams(), request_id=1)
    # results_generator = iterate_with_cancellation(results_generator, is_cancelled=request.is_disconnected)
    # Non-streaming case
    final_output = None
    async for request_output in results_generator:
        final_output = request_output
    return final_output


async def init_app(engine_args: AsyncEngineArgs,
                   args: Namespace,
                   llm_engine: Optional[AsyncLLMEngine] = None,
                   ) -> FastAPI:
    app = build_app(args)
    global engine
    engine = (llm_engine
              if llm_engine is not None else AsyncLLMEngine.from_engine_args(
        engine_args, usage_context=UsageContext.API_SERVER))

    return app


async def run_server(vllm_engine_args: AsyncEngineArgs,
                     args: Namespace,
                     llm_engine: Optional[AsyncLLMEngine] = None,
                     **uvicorn_kwargs: Any) -> None:
    logger.info("vLLM API server version %s", VLLM_VERSION)
    logger.info("args: %s", args)

    app = await init_app(vllm_engine_args, args, llm_engine)
    assert engine is not None

    # debug engine
    res = await debugfunc(engine)
    print(res)

    shutdown_task = await serve_http(
        app,
        host=args.host,
        port=args.port,
        log_level=args.log_level,
        timeout_keep_alive=TIMEOUT_KEEP_ALIVE,
        ssl_keyfile=args.ssl_keyfile,
        ssl_certfile=args.ssl_certfile,
        ssl_ca_certs=args.ssl_ca_certs,
        ssl_cert_reqs=args.ssl_cert_reqs,
        **uvicorn_kwargs,
    )

    await shutdown_task


if __name__ == "__main__":
    parser = FlexibleArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9999)
    parser.add_argument("--ssl-keyfile", type=str, default=None)
    parser.add_argument("--ssl-certfile", type=str, default=None)
    parser.add_argument("--ssl-ca-certs",
                        type=str,
                        default=None,
                        help="The CA certificates file")
    parser.add_argument(
        "--ssl-cert-reqs",
        type=int,
        default=int(ssl.CERT_NONE),
        help="Whether client certificate is required (see stdlib ssl module's)"
    )
    parser.add_argument(
        "--root-path",
        type=str,
        default=None,
        help="FastAPI root_path when app is behind a path based routing proxy")
    parser.add_argument("--log-level", type=str, default="debug")
    # server args
    model_path = "../model/stella_en_1.5B_v5_embed"
    vllm_engine_args = AsyncEngineArgs(model_path)
    #########################################
    vllm_engine_args.enable_chunked_prefill = False
    # some model need set chunked_prefill to False
    # in vllm/attention/backends/utils.py  line:170
    # # #####################################
    # block_table = []
    # if inter_data.prefix_cache_hit:
    #     block_table = computed_block_nums
    # elif ((chunked_prefill_enabled or not is_prompt)
    #       and block_tables is not None):
    #     block_table = block_tables[seq_id][-curr_sliding_window_block:]
    # self.block_tables.append(block_table)
    #########################################
    # change to
    #########################################
    # block_table = []
    # if inter_data.prefix_cache_hit:
    #     block_table = computed_block_nums
    # elif ((chunked_prefill_enabled or not is_prompt)
    #       and block_tables is not None and block_tables[seq_id] is not None):
    #     block_table = block_tables[seq_id][-curr_sliding_window_block:]
    # self.block_tables.append(block_table)
    #########################################

    vllm_engine_args.gpu_memory_utilization = 0.8
    args = parser.parse_args()
    asyncio.run(run_server(vllm_engine_args, args))
