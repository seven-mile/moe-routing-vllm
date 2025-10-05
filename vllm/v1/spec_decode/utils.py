# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os
import importlib.util

import torch
import torch.nn.functional as F

from vllm.sampling_params import SamplingParams

_SAMPLING_EPS = 1e-5


def is_spec_decode_unsupported(sampling_params: SamplingParams) -> bool:
    """True if request is incompatible with speculative decoding"""
    return (sampling_params.frequency_penalty != 0.0
            or sampling_params.presence_penalty != 0.0
            or sampling_params.repetition_penalty != 1.0
            or sampling_params.min_p > _SAMPLING_EPS
            or sampling_params.logprobs is not None)


def calc_perplexity(logits, token_ids):
    assert logits.shape[:-1] == token_ids.shape, \
        f"Logits shape {logits.shape} does not match token_ids shape {token_ids.shape}"
    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), token_ids.reshape(-1), reduction='none')
    perplexity = torch.exp(loss)
    return perplexity.view(token_ids.shape)


def load_action_from_config(config_string: str):
    """
    根据 "文件路径:函数名" 格式的字符串，动态加载并返回函数。

    :param config_string: 形如 "/path/to/my_actions.py:test1" 的配置字符串
    :return: 加载到的函数对象，如果失败则返回 None
    """
    try:
        path_str, func_name = config_string.rsplit(':', 1)
    except ValueError:
        print(f"错误: 配置字符串 '{config_string}' 格式不正确。期望格式为 '路径:函数名'。")
        return None

    if not os.path.exists(path_str):
        print(f"错误: 文件路径不存在 '{path_str}'")
        return None

    try:
        # 1. 从文件路径创建模块规范 (Module Spec)
        # 第一个参数是模块名，可以任意取，通常用文件名（不含.py）
        module_name = os.path.splitext(os.path.basename(path_str))[0]
        spec = importlib.util.spec_from_file_location(module_name, path_str)

        if spec is None:
            print(f"错误: 无法从 '{path_str}' 加载模块规范。")
            return None

        # 2. 根据规范创建模块对象
        action_module = importlib.util.module_from_spec(spec)
        
        # 3. 执行模块代码，将其内容加载到模块对象中
        spec.loader.exec_module(action_module)

        # 4. 从加载的模块中获取函数
        action_function = getattr(action_module, func_name)
        
        print(f"成功加载函数 '{func_name}' 从 '{path_str}'")
        return action_function

    except AttributeError:
        print(f"错误: 在文件 '{path_str}' 中找不到函数 '{func_name}'。")
        return None
    except Exception as e:
        print(f"加载时发生未知错误: {e}")
        return None

