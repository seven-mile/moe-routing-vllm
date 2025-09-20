import os
import functools
import importlib.util
import json

from dataclasses import dataclass
from frozendict import deepfreeze

from typing import Any, Callable, Optional

@dataclass(frozen=True)
class UserDefinedFunctionConfig:
    file: str
    function: str
    args: Optional[tuple[Any]] = None
    kwargs: Optional[tuple[tuple[str, Any]]] = None

    def dumps(self) -> str:
        return json.dumps({
            "file": self.file,
            "function": self.function,
            **({"args": self.args} if self.args is not None else {}),
            **({"kwargs": self.kwargs} if self.kwargs is not None else {}),
        }, sort_keys=True)
    
    @functools.cache
    @staticmethod
    def loads(s: str) -> "Optional[UserDefinedFunctionConfig]":
        d = json.loads(s)
        if d is None:
            return None
        assert isinstance(d, dict)
        return UserDefinedFunctionConfig(
            file=d["file"],
            function=d["function"],
            args=deepfreeze(d.get("args", None) or ()),
            kwargs=deepfreeze(sorted(d.get("kwargs", ()) or ())),
        )

@functools.cache
def _import_python_file_function(config_string: str) -> Optional[Callable]:
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

@functools.cache
def load_user_defined_function(config: UserDefinedFunctionConfig) -> Callable:
    func = _import_python_file_function(f"{config.file}:{config.function}")
    if func is None:
        raise ValueError(f"无法从文件 '{config.file}' 加载函数 '{config.function}'")

    args = config.args or ()
    kwargs = config.kwargs or ()

    if not isinstance(args, tuple) or not isinstance(kwargs, tuple):
        raise ValueError(f"错误: 'args' 必须是元组，'kwargs' 必须是键值对的元组。"
                         f"但收到 args={args}, kwargs={kwargs}")
    
    return functools.partial(func, *args, **dict(kwargs))
