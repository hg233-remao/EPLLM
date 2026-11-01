# -*- coding: utf-8 -*-
# type: ignore
"""
providers.py —— 通用模型供应商层（可直接 import 的头文件）

设计只有三条规则：
1. 一个 OpenAICompatProvider 吃下所有「OpenAI 兼容」厂商
   （DeepSeek / Kimi / 智谱 / 通义 / OpenRouter / Ollama / vLLM / 中转站），
   差异全部塞进 VENDORS 这张表和构造参数里，加供应商 = 加一行。
2. 上层只依赖 `chat(system, user) -> ChatResult` 这一个签名（见 ChatProvider）。
3. 只有报文结构真正不同的厂商（如 Anthropic）才写子类，签名保持不变。

典型用法：

    from providers import make_provider, ProviderError, VENDORS

    p = make_provider("deepseek", api_key="sk-xxxx", max_tokens=8192)
    try:
        r = p.chat(PROMPT, 文件原文)
    except ProviderError as e:
        print(e.user_message())          # 能直接给用户看的中文
    else:
        if r.finish_reason == "length":  # 输出被截断，千万别写回文件
            ...
        text = r.text.replace("\\\\n", "\\n")   # 还原 talk 的换行约定

换供应商只改一个字符串：

    p = make_provider("moonshot", api_key=key2)

依赖：openai（可选，只有用 OpenAI 兼容厂商时才需要）、requests（Anthropic 用）。
本模块不读配置文件、不碰文件系统、不起线程，纯供应商层。
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Protocol, runtime_checkable

try:                                    # 没装 openai 也能 import 本模块（Mock 可用）
    import openai as _openai
except ImportError:                     # pragma: no cover
    _openai = None

__all__ = [
    # 结果与异常
    "ChatResult", "Usage", "ProbeResult",
    "ProviderError", "ConfigError", "AuthError", "QuotaError",
    "RateLimitError", "NetworkError", "ServerError",
    "TruncatedOutput", "BadResponse",
    # 接口与实现
    "ChatProvider", "OpenAICompatProvider", "AnthropicProvider", "MockProvider",
    # 目录与工厂
    "VendorSpec", "VENDORS", "VENDOR_IDS", "make_provider", "describe_vendors",
    # 重试
    "RetryPolicy", "chat_with_retry",
]


# ============================================================== 结果对象
@dataclass(frozen=True)
class Usage:
    """token 用量。各家字段名不同（prompt/completion 与 input/output），这里统一。"""
    prompt: int = 0
    completion: int = 0

    @property
    def total(self) -> int:
        return self.prompt + self.completion


@dataclass(frozen=True)
class ChatResult:
    """一次对话的完整结果。永远返回这个对象，不要返回裸字符串。

    - text          模型原文（还没做 talk 的 `\\n` 还原）
    - reasoning     推理内容；非推理模型为 None（绝不裸取 message.reasoning_content）
    - finish_reason "stop" / "length" / ... ；等于 "length" 说明被截断
    """
    text: str
    reasoning: Optional[str] = None
    finish_reason: Optional[str] = None
    model: str = ""
    provider_id: str = ""
    latency_ms: int = 0
    usage: Usage = field(default_factory=Usage)

    @property
    def truncated(self) -> bool:
        return self.finish_reason == "length"


@dataclass(frozen=True)
class ProbeResult:
    """`probe()` 的返回：能不能用、为什么、余额、模型列表。"""
    ok: bool
    message: str                       # 直接显示给用户的中文
    balance: Optional[str] = None      # 例如 "110.00 CNY"，供应商支持才有
    models: tuple[str, ...] = ()       # 能列出就填，可用来做下拉框

    def __bool__(self) -> bool:
        return self.ok

    def has_model(self, model: str) -> bool:
        """模型列表里是否包含该模型。

        列表为空表示「没查出来」而不是「没有」，所以返回 True，
        避免把不支持列模型的网关（或走余额探针的 DeepSeek）误判成不支持。
        """
        if not self.models or not model:
            return True
        return model in self.models


# ============================================================== 异常
class ProviderError(Exception):
    """所有供应商错误的基类。上层只需要 catch 这一个。

    `retryable` 决定要不要重试；`user_message()` 是能直接弹给用户看的中文。
    """
    retryable = False

    def __init__(self, message: str, *, detail: Optional[str] = None,
                 status: Optional[int] = None):
        super().__init__(message)
        self.message = message
        self.detail = detail                 # 原始报错，写日志用
        self.status = status                 # HTTP 状态码，没有就是 None

    def user_message(self) -> str:
        return self.message

    def __str__(self) -> str:
        return self.message


class ConfigError(ProviderError):
    """配置不对（缺 key、模型名空、base_url 空）。重试没意义。"""


class AuthError(ProviderError):
    """401/403：密钥无效或没有权限。"""


class QuotaError(ProviderError):
    """余额不足、额度用尽。"""


class RateLimitError(ProviderError):
    """429：被限流。可重试，且尽量按 retry_after 等。"""
    retryable = True

    def __init__(self, message: str, *, retry_after: Optional[float] = None, **kw):
        super().__init__(message, **kw)
        self.retry_after = retry_after


class NetworkError(ProviderError):
    """超时、连不上。可重试。"""
    retryable = True


class ServerError(ProviderError):
    """5xx。可重试。"""
    retryable = True


class TruncatedOutput(ProviderError):
    """finish_reason == "length"：输出被截断，绝不能写回文件。"""


class BadResponse(ProviderError):
    """返回内容不像一份合法的 talk（缺 %txt 或 %endTxt）。"""


# ============================================================== 接口
@runtime_checkable
class ChatProvider(Protocol):
    """上层代码只认识这 3 行。任何对象只要有这两个方法就能接进来。"""

    def chat(self, system: str, user: str) -> ChatResult:
        ...

    def probe(self) -> ProbeResult:
        ...


# ============================================================== 内部工具
def _retry_after(exc: BaseException) -> Optional[float]:
    resp = getattr(exc, "response", None)
    headers = getattr(resp, "headers", None) or {}
    try:
        return float(headers.get("retry-after") or headers.get("Retry-After"))
    except (TypeError, ValueError):
        return None


def _map_error(exc: BaseException) -> ProviderError:
    """把 openai / requests 的异常翻译成 ProviderError。

    用状态码和类名判断而不是 isinstance 链，这样不同版本的 SDK 都能吃。
    """
    status = getattr(exc, "status_code", None)
    if status is None:
        status = getattr(getattr(exc, "response", None), "status_code", None)
    name = type(exc).__name__
    msg = str(exc)

    if status in (401, 403):
        return AuthError("密钥无效，或这个密钥没有访问该模型的权限。",
                         detail=msg, status=status)
    if status == 402 or "insufficient" in msg.lower() or "quota" in msg.lower():
        return QuotaError("余额不足或额度已用尽。", detail=msg, status=status)
    if status == 429:
        return RateLimitError("被供应商限流了，稍后自动重试。",
                              retry_after=_retry_after(exc), detail=msg, status=status)
    if status == 404:
        return ConfigError("接口地址或模型不存在，请检查 base_url 与模型名。",
                           detail=msg, status=status)
    if status == 400:
        return ConfigError("请求被拒绝：模型名、参数或上下文长度可能不对。",
                           detail=msg, status=status)
    if status is not None and status >= 500:
        return ServerError("供应商服务端出错，稍后自动重试。", detail=msg, status=status)
    if "Timeout" in name or "timeout" in msg.lower():
        return NetworkError("请求超时。", detail=msg, status=status)
    if "Connection" in name or "connect" in msg.lower():
        return NetworkError("无法连接到服务器。", detail=msg, status=status)
    return ProviderError(f"调用失败：{msg}", detail=msg, status=status)


def _content_text(content: Any) -> str:
    """兼容 str 与「分片列表」两种 content 形式。"""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    parts = []
    for part in content:
        text = getattr(part, "text", None)
        if text is None and isinstance(part, dict):
            text = part.get("text")
        if text:
            parts.append(text)
    return "".join(parts)


def _reasoning_of(message: Any, field_name: Optional[str]) -> Optional[str]:
    if not field_name:
        return None
    value = getattr(message, field_name, None)
    if value:
        return value
    extra = getattr(message, "model_extra", None) or {}
    return extra.get(field_name)


def _usage_of(resp: Any) -> Usage:
    usage = getattr(resp, "usage", None)
    if not usage:
        return Usage()
    prompt = getattr(usage, "prompt_tokens", None)
    if prompt is None:
        prompt = getattr(usage, "input_tokens", 0)
    completion = getattr(usage, "completion_tokens", None)
    if completion is None:
        completion = getattr(usage, "output_tokens", 0)
    return Usage(int(prompt or 0), int(completion or 0))


# ============================================================== 通用实现
class OpenAICompatProvider:
    """所有 OpenAI 兼容厂商共用的一个类。

    构造参数分两组：
      连接组：base_url / api_key / model / timeout
      行为组：temperature / max_tokens / send_temperature / reasoning_field / extra_headers

    实例无状态（除底层连接池），可被多个线程共享；一个配置造一个实例即可。
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        *,
        temperature: float = 1.0,
        max_tokens: Optional[int] = None,
        timeout: float = 600.0,
        send_temperature: bool = True,          # 推理模型可能不接受 temperature
        reasoning_field: Optional[str] = None,  # "reasoning_content" / "reasoning" / None
        extra_headers: Optional[dict] = None,   # 例如 OpenRouter 的 HTTP-Referer
        provider_id: str = "openai-compat",
        max_retries: int = 2,                   # 交给 SDK 的传输层重试
        key_check: str = "models",              # "models" | "balance" | "none"
        balance_url: str = "",                  # key_check="balance" 时用
        default_headers: Optional[dict] = None, # 客户端级请求头（UA、会话 ID）
        models_url: str = "",                   # 覆盖 {base_url}/models
    ):
        if _openai is None:
            raise ConfigError("使用该供应商需要先安装 openai：pip install openai")
        if not base_url:
            raise ConfigError("未指定 base_url。")
        if not model:
            raise ConfigError("未指定模型名。")
        self.base_url = base_url
        self.model = model
        self.provider_id = provider_id
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.send_temperature = send_temperature
        self.reasoning_field = reasoning_field
        self.key_check = key_check
        self.balance_url = balance_url
        self.models_url = models_url
        self._api_key = api_key or "not-needed"   # 本地模型不需要 key，但不能传空串
        self._client = _openai.OpenAI(
            api_key=self._api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
            default_headers=default_headers or None,
        )

    # ---------------------------------------------------------- 主方法
    def chat(self, system: str, user: str) -> ChatResult:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        if self.send_temperature:
            kwargs["temperature"] = self.temperature
        if self.max_tokens:
            kwargs["max_tokens"] = self.max_tokens

        started = time.monotonic()
        try:
            resp = self._client.chat.completions.create(**kwargs)
        except Exception as exc:                 # noqa: BLE001 —— 统一翻译
            raise _map_error(exc) from exc

        choice = resp.choices[0]
        message = choice.message
        return ChatResult(
            text=_content_text(message.content),
            reasoning=_reasoning_of(message, self.reasoning_field),
            finish_reason=getattr(choice, "finish_reason", None),
            model=getattr(resp, "model", self.model),
            provider_id=self.provider_id,
            latency_ms=int((time.monotonic() - started) * 1000),
            usage=_usage_of(resp),
        )

    # ---------------------------------------------------------- 探针
    def probe(self) -> ProbeResult:
        """校验地址与密钥。

        - key_check="none"（本地模型）：直接通过，不发请求。
        - key_check="balance"（DeepSeek 等）：查余额，顺带把余额显示出来。
        - 其余：列模型，最便宜的通用探针。
        三种方式都不产生推理费用。
        """
        if self.key_check == "none":
            return ProbeResult(True, "该供应商无需密钥，直接可用。")
        if self.key_check == "balance" and self.balance_url:
            return self._probe_balance()
        try:
            models = self.list_models()
        except ProviderError as exc:
            return ProbeResult(False, exc.user_message())
        except Exception as exc:                 # noqa: BLE001
            return ProbeResult(False, _map_error(exc).user_message())
        if models:
            return ProbeResult(True, f"连接正常，可用模型 {len(models)} 个。",
                               models=tuple(models))
        return ProbeResult(True, "连接正常。")

    def _probe_balance(self) -> ProbeResult:
        """查余额。返回格式兼容 DeepSeek：{"balance_infos":[{"currency","total_balance"}]}"""
        import requests

        try:
            resp = requests.get(
                self.balance_url,
                headers={"Authorization": f"Bearer {self._api_key}",
                         "Accept": "application/json"},
                timeout=15,
            )
        except Exception as exc:                 # noqa: BLE001
            return ProbeResult(False, _map_error(exc).user_message())

        if resp.status_code in (401, 403):
            return ProbeResult(False, "密钥无效，或没有访问权限。")
        if resp.status_code != 200:
            return ProbeResult(True, f"密钥格式通过（余额接口返回 {resp.status_code}）。")

        try:
            infos = resp.json().get("balance_infos") or []
        except Exception:                        # noqa: BLE001
            return ProbeResult(True, "密钥有效。")
        if not infos:
            return ProbeResult(True, "密钥有效。")

        info = infos[0]
        amount = info.get("total_balance") or info.get("topped_up_balance") or "?"
        currency = info.get("currency", "")
        return ProbeResult(True, "密钥有效。", balance=f"{amount} {currency}".strip())

    def list_models(self) -> list[str]:
        if self.models_url:
            return self._list_models_by_url()
        if _openai is None:
            raise ConfigError("尚未安装 openai。")
        try:
            data = self._client.models.list().data
        except Exception as exc:                 # noqa: BLE001
            raise _map_error(exc) from exc
        return [m.id for m in data]

    def _list_models_by_url(self) -> list[str]:
        """有些网关的模型列表不在 base_url 下面（如 OpenCode 的 /go/v1/models）。"""
        import requests

        try:
            resp = requests.get(
                self.models_url,
                headers={"Authorization": f"Bearer {self._api_key}",
                         "Accept": "application/json"},
                timeout=15,
            )
            resp.raise_for_status()
            payload = resp.json()
        except Exception as exc:                 # noqa: BLE001
            raise _map_error(exc) from exc

        items = payload.get("data") if isinstance(payload, dict) else payload
        return [item.get("id", "") for item in (items or []) if isinstance(item, dict)]

    def close(self) -> None:
        try:
            self._client.close()
        except Exception:                        # noqa: BLE001
            pass

    def __repr__(self) -> str:
        return f"<OpenAICompatProvider {self.provider_id} model={self.model}>"


# ============================================================== 非 OpenAI 协议
class AnthropicProvider:
    """Anthropic 原生接口：system 在顶层、x-api-key 头、max_tokens 必填。

    之所以要单独一个类，是因为报文结构不同；但 chat() 的签名完全一致，
    所以上层代码一行都不用改。
    """

    def __init__(self, api_key: str, model: str,
                 max_tokens: int = 8192, timeout: float = 600.0,
                 temperature: Optional[float] = None,
                 base_url: str = "https://api.anthropic.com",
                 provider_id: str = "anthropic"):
        if not api_key:
            raise ConfigError("Anthropic 需要 API Key。")
        if not model:
            raise ConfigError("未指定模型名。")
        self.model = model
        self.provider_id = provider_id
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout
        self.base_url = base_url.rstrip("/")
        self._api_key = api_key

    # ---------------------------------------------------------- 主方法
    def chat(self, system: str, user: str) -> ChatResult:
        import requests                   # 延迟导入：不用 Anthropic 就不必装

        payload: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": system,             # ← 顶层，不在 messages 里
            "messages": [{"role": "user", "content": user}],
        }
        if self.temperature is not None:
            payload["temperature"] = self.temperature

        started = time.monotonic()
        try:
            resp = requests.post(
                f"{self.base_url}/v1/messages",
                headers={
                    "x-api-key": self._api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json=payload,
                timeout=self.timeout,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:          # noqa: BLE001
            raise _map_error(exc) from exc

        text = "".join(
            block.get("text", "") for block in data.get("content", [])
            if block.get("type") == "text"
        )
        usage = data.get("usage") or {}
        return ChatResult(
            text=text,
            reasoning=None,               # Anthropic 不单独返回推理文本
            finish_reason=data.get("stop_reason"),
            model=data.get("model", self.model),
            provider_id=self.provider_id,
            latency_ms=int((time.monotonic() - started) * 1000),
            usage=Usage(int(usage.get("input_tokens", 0)),
                        int(usage.get("output_tokens", 0))),
        )

    def probe(self) -> ProbeResult:
        import requests

        try:
            resp = requests.get(
                f"{self.base_url}/v1/models",
                headers={"x-api-key": self._api_key,
                         "anthropic-version": "2023-06-01"},
                timeout=15,
            )
            resp.raise_for_status()
        except Exception as exc:          # noqa: BLE001
            return ProbeResult(False, _map_error(exc).user_message())
        try:
            models = tuple(m.get("id", "") for m in resp.json().get("data", []))
        except Exception:                 # noqa: BLE001
            models = ()
        return ProbeResult(True, "连接正常。", models=models)

    def list_models(self) -> list[str]:
        return list(self.probe().models)

    def close(self) -> None:
        pass

    def __repr__(self) -> str:
        return f"<AnthropicProvider model={self.model}>"


# ============================================================== 离线实现
class MockProvider:
    """不发网络请求，返回一段固定的合法 talk。用于离线跑界面、写单测。"""

    def __init__(self, model: str = "mock", text: Optional[str] = None,
                 finish_reason: str = "stop"):
        self.model = model
        self.provider_id = "mock"
        self._text = text or (
            "%txtCalm,JP\n"
            "「今天也很有精神呢。」\n"
            "{npc}伸了个懒腰。\n"
            "%endTxt"
        )
        self.finish_reason = finish_reason

    def chat(self, system: str, user: str) -> ChatResult:
        return ChatResult(text=self._text, finish_reason=self.finish_reason,
                          model=self.model, provider_id=self.provider_id)

    def probe(self) -> ProbeResult:
        return ProbeResult(True, "Mock 供应商，不需要联网。")

    def close(self) -> None:
        pass

    def __repr__(self) -> str:
        return "<MockProvider>"


# ============================================================== 供应商目录
@dataclass(frozen=True)
class VendorSpec:
    """一个供应商的静态描述。加供应商就是往 VENDORS 里加一个实例。"""
    id: str
    name: str                              # 下拉框里显示的名字
    protocol: str = "openai"               # "openai" | "anthropic"
    base_url: str = ""
    default_model: str = ""
    models: tuple[str, ...] = ()           # 预设候选，允许用户手填别的
    needs_key: bool = True
    key_check: str = "models"              # "models" | "balance" | "none"
    balance_url: str = ""                  # key_check="balance" 时用
    reasoning_field: Optional[str] = None   # "reasoning_content" / "reasoning" / None
    supports_temperature: bool = True
    extra_headers: dict = field(default_factory=dict)
    models_url: str = ""                   # 显式模型列表地址；空则用 {base_url}/models
    session_header: str = ""               # 需要稳定会话头的网关（如 OpenCode）填头名


VENDORS: dict[str, VendorSpec] = {
    "deepseek": VendorSpec(
        id="deepseek", name="DeepSeek",
        base_url="https://api.deepseek.com",
        default_model="deepseek-chat",
        models=("deepseek-chat", "deepseek-reasoner"),
        key_check="balance",
        balance_url="https://api.deepseek.com/user/balance",
        reasoning_field="reasoning_content",
    ),
    "openai": VendorSpec(
        id="openai", name="OpenAI",
        base_url="https://api.openai.com/v1",
        default_model="gpt-4o-mini",
        models=("gpt-4o-mini", "gpt-4o"),
    ),
    "moonshot": VendorSpec(
        id="moonshot", name="Moonshot / Kimi",
        base_url="https://api.moonshot.cn/v1",
        default_model="moonshot-v1-8k",
        models=("moonshot-v1-8k", "moonshot-v1-32k"),
    ),
    "zhipu": VendorSpec(
        id="zhipu", name="智谱 GLM",
        base_url="https://open.bigmodel.cn/api/paas/v4",
        default_model="glm-4-flash",
        models=("glm-4-flash", "glm-4-plus"),
    ),
    "qwen": VendorSpec(
        id="qwen", name="通义千问（兼容模式）",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        default_model="qwen-plus",
        models=("qwen-plus", "qwen-max"),
    ),
    "opencode-go": VendorSpec(
        id="opencode-go", name="OpenCode Go（订阅制）",
        base_url="https://opencode.ai/inference/go/openai/v1",
        default_model="glm-5.3-flash",
        models=("glm-5.3", "glm-5.3-flash", "deepseek-v4.1-flash",
                "minimax-m3", "qwen3.8-flash", "step-5-preview", "hy3"),
        models_url="https://opencode.ai/inference/go/v1/models",
        session_header="x-opencode-session",
        extra_headers={"User-Agent": "EPLLMv1/1.0"},
    ),
    "openrouter": VendorSpec(
        id="openrouter", name="OpenRouter",
        base_url="https://openrouter.ai/api/v1",
        default_model="anthropic/claude-3.5-sonnet",
        models=(),
        extra_headers={"HTTP-Referer": "https://localhost"},
    ),
    "ollama": VendorSpec(
        id="ollama", name="Ollama（本地）",
        base_url="http://127.0.0.1:11434/v1",
        default_model="qwen2.5:7b",
        models=(),
        needs_key=False,
        key_check="none",
    ),
    "anthropic": VendorSpec(
        id="anthropic", name="Anthropic Claude",
        protocol="anthropic",
        base_url="https://api.anthropic.com",
        default_model="claude-3-5-sonnet-latest",
    ),
    "custom": VendorSpec(
        id="custom", name="自定义（任意兼容端点）",
        base_url="",
        default_model="",
        key_check="models",
    ),
    "mock": VendorSpec(
        id="mock", name="Mock（离线测试）",
        protocol="mock",
        needs_key=False,
        key_check="none",
    ),
}

VENDOR_IDS: tuple[str, ...] = tuple(VENDORS)


def describe_vendors() -> list[tuple[str, str]]:
    """给 GUI 下拉框用：[(id, 显示名), ...]"""
    return [(v.id, v.name) for v in VENDORS.values()]


# ============================================================== 工厂
def make_provider(
    vendor_id: str,
    api_key: str = "",
    model: Optional[str] = None,
    *,
    base_url: Optional[str] = None,
    temperature: float = 1.0,
    max_tokens: Optional[int] = None,
    timeout: float = 600.0,
    extra_headers: Optional[dict] = None,
    spec: Optional[VendorSpec] = None,
) -> ChatProvider:
    """把「一串配置」变成「一个能 chat 的对象」。

    参数：
        vendor_id    VENDORS 里的键，或任意自定义 id（需同时传 spec）
        api_key      密钥；needs_key=False 的供应商可省略
        model        模型名；不传则用该供应商的默认模型
        base_url     覆盖默认地址（中转站、内网代理都靠它）
        max_tokens   输出上限；不传则不发送该参数（除非供应商强制要求）
        extra_headers 额外请求头，与供应商自带的合并

    抛 ConfigError 表示配置问题（缺 key、缺模型名），不必重试。
    """
    if spec is None:
        spec = VENDORS.get(vendor_id)
    if spec is None:
        raise ConfigError(
            f"未知供应商：{vendor_id}，可选：{', '.join(VENDOR_IDS)}"
        )

    if spec.needs_key and not api_key:
        raise ConfigError(f"「{spec.name}」需要 API Key。")

    model_name = (model or spec.default_model).strip()
    url = (base_url or spec.base_url).strip()

    if spec.protocol == "mock":
        return MockProvider(model=model_name or "mock")

    if spec.protocol == "anthropic":
        if not model_name:
            raise ConfigError("未指定模型名。")
        return AnthropicProvider(
            api_key=api_key,
            model=model_name,
            max_tokens=max_tokens or 8192,
            timeout=timeout,
            temperature=temperature,
            base_url=url,
            provider_id=spec.id,
        )

    if spec.protocol == "openai":
        if not url:
            raise ConfigError("未指定 base_url（自定义供应商必须填）。")
        if not model_name:
            raise ConfigError("未指定模型名。")
        headers = dict(spec.extra_headers)
        headers.update(extra_headers or {})
        if spec.session_header:
            # 网关（如 OpenCode）要求每次会话带稳定的 ID，用于路由与提示词缓存
            headers.setdefault(spec.session_header, uuid.uuid4().hex)
        return OpenAICompatProvider(
            base_url=url,
            api_key=api_key,
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            send_temperature=spec.supports_temperature,
            reasoning_field=spec.reasoning_field,
            default_headers=headers or None,
            provider_id=spec.id,
            key_check=spec.key_check,
            balance_url=spec.balance_url,
            models_url=spec.models_url,
        )

    raise ConfigError(f"未知协议：{spec.protocol}")


# ============================================================== 重试
@dataclass
class RetryPolicy:
    """只重试「可重试」的错误（429 / 5xx / 网络），其余立刻抛出。"""
    attempts: int = 3
    base_delay: float = 2.0
    max_delay: float = 30.0

    def call(
        self,
        fn: Callable[[], Any],
        *,
        on_retry: Optional[Callable[[int, ProviderError, float], None]] = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> Any:
        last: Optional[ProviderError] = None
        for attempt in range(1, max(1, self.attempts) + 1):
            try:
                return fn()
            except ProviderError as exc:
                last = exc
                if not exc.retryable or attempt >= self.attempts:
                    raise
                delay = min(self.max_delay, self.base_delay * (2 ** (attempt - 1)))
                retry_after = getattr(exc, "retry_after", None)
                if retry_after:
                    delay = min(self.max_delay, float(retry_after))
                if on_retry:
                    on_retry(attempt, exc, delay)
                sleep(delay)
        raise last  # pragma: no cover


def chat_with_retry(
    provider: ChatProvider,
    system: str,
    user: str,
    policy: Optional[RetryPolicy] = None,
    *,
    on_retry: Optional[Callable[[int, ProviderError, float], None]] = None,
) -> ChatResult:
    """带重试的一次对话，方便直接替换老的 client 调用。"""
    return (policy or RetryPolicy()).call(
        lambda: provider.chat(system, user), on_retry=on_retry
    )


# ============================================================== 自测
if __name__ == "__main__":
    print("== 目录 ==")
    for vid, name in describe_vendors():
        print(f"  {vid:11s} {name}")

    print("\n== Mock ==")
    mock = make_provider("mock")
    result = mock.chat("system", "user")
    print(result.text)
    print("finish_reason =", result.finish_reason,
          "| truncated =", result.truncated)
    print("probe:", mock.probe().message)

    print("\n== 配置校验 ==")
    for bad in (("deepseek", ""), ("custom", "k"), ("nope", "k")):
        try:
            make_provider(bad[0], api_key=bad[1])
        except ProviderError as exc:
            print(f"  {bad[0]:10s} -> {type(exc).__name__}: {exc.user_message()}")
