由于 后两项任务较为工程，因此和 speculative sampling 一起放在本文件中说明。 

# NanoChat Web 采样控制与 top_p 说明

本次改动在 Web UI 中加入了一个可折叠的「Sampling Config」面板，提供 Temperature、Top‑k、Top‑p 三个滑块，同时保留 `/temperature`、`/topk`、`/topp` 等斜杠命令。三者的默认值和取值范围（最小/最大）由后端 `scripts/chat_web.py` 中的 CLI 参数和常量（`MIN/MAX_TEMPERATURE`、`MIN/MAX_TOP_K`、`MIN/MAX_TOP_P`）统一控制，保证前后端行为一致，并避免用户设置过于极端的参数。

在采样实现上，`nanochat/engine.py` 中的 `sample_next_token` 和 `Engine.generate` 新增了 `top_p` 参数：当 `temperature == 0` 时依然是纯 argmax；只设定 `top_p` 时，对完整词表做标准的 nucleus sampling；当同时设定 `top_k` 与 `top_p` 时，会先用 `top_k` 将候选限制在前 k 个 token，再在这 k 个内部按 `top_p` 做二次筛选。先用 `top_k` 再用 `top_p` 的原因主要有两点：一是 `top_k` 能先限制后续 `top_p` 计算所需考虑的词表大小，避免在完整大词表上做排序、累积概率和 softmax，从而减少计算量、提高采样效率；二是在实际应用中，`top_p` 往往会筛出一小撮高概率 token，而 `top_k` 提供一个上限，约束分布不会过于分散，在保证采样质量的前提下更好地控制多样性。

---

# OpenAI 兼容的 `/v1/chat/completions` API 实现

在 `scripts/chat_web.py` 中新增了一个与 OpenAI Chat Completions 兼容的 API：

- 路径与方法：`POST /v1/chat/completions`
- 请求体（主要字段，整体格式与 OpenAI 对齐）：
  - `model: string`（目前仅用于回显，不影响行为）
  - `messages: [{role: "user" | "assistant" | "system", content: string}, ...]`
  - `temperature?: float`
  - `top_p?: float`
  - `top_k?: int`
  - `max_tokens?: int`
  - `stream?: bool`（默认 `false`）
  - `n?: int`（目前只支持 `n=1`）
  - `user?: string`（目前仅透传，不参与逻辑）

这些字段会被转换成内部的 `ChatRequest`，并复用原有的上下文长度校验和采样参数范围校验逻辑；消息内容的 tokenization 仍然采用 `<|user_start|>` / `<|assistant_start|>` 等特殊 token 格式。

1. 非流式请求示例（curl）

```bash
curl http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "nanochat",
    "messages": [
      {"role": "user", "content": "How are you today?"}
    ],
    "temperature": 0.7,
    "max_tokens": 128
  }'
```

```jsonc
{
  "id": "chatcmpl-1765018248-207629700",
  "object": "chat.completion",
  "created": 1765018248,
  "model": "nanochat",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "I'm doing well. I'm an AI assistant specialized in mathematics, capable of addressing a wide range of mathematical topics. I'm here to provide you with clear explanations, concise problem-solving strategies, and insightful mathematical insights. What's on your mind? Do you have a specific topic or problem you'd like to discuss?"
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 9,
    "completion_tokens": 64,
    "total_tokens": 73
  }
}
```

2. 流式请求示例（curl + SSE）

```bash
curl http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "nanochat",
    "messages": [
      {"role": "user", "content": "How are you today?"}
    ],
    "temperature": 0.7,
    "stream": true,
    "max_tokens": 256
  }'
```

```text
data: {
  "id": "chatcmpl-1765018421-1701468774",
  "object": "chat.completion.chunk",
  "created": 1765018421,
  "model": "nanochat",
  "choices": [
    {
      "index": 0,
      "delta": {
        "role": "assistant",
        "content": "I"
      },
      "finish_reason": null
    }
  ]
}

data: {
  "id": "chatcmpl-1765018421-1701468774",
  "object": "chat.completion.chunk",
  "created": 1765018421,
  "model": "nanochat",
  "choices": [
    {
      "index": 0,
      "delta": {
        "content": "'m"
      },
      "finish_reason": null
    }
  ]
}

data: {
  "id": "chatcmpl-1765018421-1701468774",
  "object": "chat.completion.chunk",
  "created": 1765018421,
  "model": "nanochat",
  "choices": [
    {
      "index": 0,
      "delta": {
        "content": " doing"
      },
      "finish_reason": null
    }
  ]
}

data: {
  "id": "chatcmpl-1765018421-1701468774",
  "object": "chat.completion.chunk",
  "created": 1765018421,
  "model": "nanochat",
  "choices": [
    {
      "index": 0,
      "delta": {
        "content": " well"
      },
      "finish_reason": null
    }
  ]
}

...

data: {
  "id": "chatcmpl-1765018421-1701468774",
  "object": "chat.completion.chunk",
  "created": 1765018421,
  "model": "nanochat",
  "choices": [
    {
      "index": 0,
      "delta": {
        "content": "."
      },
      "finish_reason": null
    }
  ]
}

data: {
  "id": "chatcmpl-1765018421-1701468774",
  "object": "chat.completion.chunk",
  "created": 1765018421,
  "model": "nanochat",
  "choices": [
    {
      "index": 0,
      "delta": {},
      "finish_reason": "stop"
    }
  ]
}

data: [DONE]
```

---

## 并发控制、Rate 限制与 API Key 认证

- 并发控制：在 `lifespan` 中创建 `app.state.request_semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)`（默认 16），在 `/chat/completions` 和 `/v1/chat/completions` 入口处 `acquire()`，请求结束或异常时 `release()`，超出并发上限的请求会在队列中等待空闲 slot。
- 全局 Rate 限制：实现了一个简单的滑动窗口计数（`RATE_LIMIT_WINDOW_SECONDS=60`，`RATE_LIMIT_MAX_REQUESTS=120`），保存在 `app.state.rate_limit` 中；每次请求调用 `check_rate_limit()`，超出阈值则直接返回 429。
- API Key 认证：新增 `check_api_key()`，从 `Authorization: Bearer <token>` 中读取 token，并与 `config.json` 中的 `valid_api_keys` 列表匹配（默认包含 `"123456"`）。缺少 Authorization 或 token 不匹配时返回 401。更新 key 只需修改 `config.json` 并重启服务。

### API Key 认证示例

- 正确示例（携带合法 key）：

  ```bash
  curl http://localhost:8000/v1/chat/completions \
    -H "Authorization: Bearer 123456" \
    -H "Content-Type: application/json" \
    -d '{
      "model": "nanochat",
      "messages": [{"role": "user", "content": "Hello"}],
      "temperature": 0.7
    }'
  ```

  输出：

  ```json
  {
    "id": "chatcmpl-1765025597-1326825558",
    "object": "chat.completion",
    "created": 1765025597,
    "model": "nanochat",
    "choices": [
      {
        "index": 0,
        "message": {
          "role": "assistant",
          "content": "Hello! How can I help you today?"
        },
        "finish_reason": "stop"
      }
    ],
    "usage": {
      "prompt_tokens": 5,
      "completion_tokens": 9,
      "total_tokens": 14
    }
  }
  ```

- 缺少 Authorization 头：

  ```bash
  curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model":"nanochat","messages":[{"role":"user","content":"Hello"}]}'
  ```

  输出：

  ```json
  {"detail":"Missing Authorization header"}
  ```

- 错误的 Bearer token：

  ```bash
  curl http://localhost:8000/v1/chat/completions \
    -H "Authorization: Bearer 222456" \
    -H "Content-Type: application/json" \
    -d '{"model":"nanochat","messages":[{"role":"user","content":"Hello"}]}'
  ```

  输出：

  ```json
  {"detail":"Invalid API key"}
  ```

---

# Speculative Sampling

## 目标

在单样本场景下使用一个较小的草稿模型（draft）提前生成多步 token proposal，再由目标模型（target）一次性批量校验这些proposal，希望在保持与常规采样完全一致分布的前提下提升整体吞吐。

## 算法步骤（rejection sampling 视角）

Speculative sampling 可以看作在 draft 分布上做proposal、在 target 分布上做接受/拒绝的一种 rejection sampling 过程，大致可以拆成以下几个步骤：

1. 先用目标模型得到当前 step 的分布 `p_t(·)`，用草稿模型得到对应的分布 `p_d(·)`，要求 draft 在支持上覆盖 target 的高概率区域。
2. 在后续若干步中仅用草稿模型按常规采样（例如温度 + top‑p），连续生成一串候选 token 序列 `y₁, y₂, ..., y_K`，把它们当作一次proposal批。
3. 对这串候选中的每个 token `y_i`，用目标模型重新计算该位置的 `p_t(y_i)`，再采一个均匀随机数 `u ~ U(0, 1)`；如果满足 `u <= p_t(y_i) / p_d(y_i)`，则接受该 token，否则视为拒绝。
4. 一旦遇到第一个被拒绝的 token，就停止沿用剩下的 draft proposal，转而在目标分布上做一次“补偿采样”：在剔除已接受proposal的前提下，对目标分布做重新归一化，从剩余概率质量中重新采一个 token，确保整体输出和直接在 target 上自回归采样的分布完全一致。
5. 在整体生成过程中，最终被接受的 token 序列在分布上等价于纯 target sampling，只是其中相当一部分前向计算可以由草稿模型替代，从而在理想情况下减少目标模型前向次数、提高吞吐。

## 核心流程（nanochat/engine.py）

1. 入口选择  
   当 `speculative_num_tokens > 0` 且 `num_samples == 1` 时，`Engine.generate` 会走 `_generate_speculative` 分支，其余情况则沿用标准自回归路径。

   可以用下面的命令启动带 speculative sampling 的服务：

   ```bash
   CUDA_VISIBLE_DEVICES=0 python -m scripts.chat_web \
     --host 0.0.0.0 \
     --port 8001 \
     --speculative-proposals 3
   ```

   启动后可以在日志中观察 wall-clock 时间和 tok/s，对比开启和关闭 speculative 时的差异。

## 测试

### With Speculative Sampling Enabled (`--speculative-proposals 3`)

Prompt: Write a step-by-step guide to brew pour-over coffee in 10 numbered steps.

```text
[spec] <reject> p_t=0.0000 p_d=0.4807 u=0.7117
[spec] <reject> p_t=0.0051 p_d=0.5801 u=0.4243
[spec] <reject> p_t=0.0000 p_d=0.0492 u=0.4516
[spec] <reject> p_t=0.0000 p_d=0.0138 u=0.2051
[spec] <reject> p_t=0.0000 p_d=0.7380 u=0.6206
[spec] <reject> p_t=0.0000 p_d=0.0147 u=0.9692
[spec] <reject> p_t=0.0025 p_d=0.0416 u=0.6215
[spec] <reject> p_t=0.0177 p_d=0.6607 u=0.5360
[spec] <reject> p_t=0.0000 p_d=0.3912 u=0.7775
[spec] <accept> p_t=0.7844 p_d=0.4626 u=0.5904
[spec] <accept> p_t=0.9968 p_d=0.3168 u=0.1373
[spec] <reject> p_t=0.0000 p_d=0.1147 u=0.9545
[spec] <reject> p_t=0.0000 p_d=0.0134 u=0.7862
[spec] <reject> p_t=0.0000 p_d=0.0041 u=0.8274
[spec] <reject> p_t=0.3505 p_d=0.8074 u=0.4511
[spec] <reject> p_t=0.0000 p_d=0.1367 u=0.5872
[spec] <reject> p_t=0.0000 p_d=0.0042 u=0.1972
[spec] <reject> p_t=0.0000 p_d=0.0538 u=0.2587
[spec] <reject> p_t=0.0000 p_d=0.0736 u=0.1332
[spec] <accept> p_t=0.9969 p_d=0.6326 u=0.4455
[spec] <reject> p_t=0.0000 p_d=0.0330 u=0.4425
[spec] <reject> p_t=0.0000 p_d=0.6415 u=0.6427
[spec] <reject> p_t=0.0000 p_d=0.0230 u=0.5994
[spec] <reject> p_t=0.0000 p_d=0.0555 u=0.8933
[spec] <accept> p_t=1.0000 p_d=0.9329 u=0.5908
[spec] <accept> p_t=1.0000 p_d=0.9238 u=0.5055
[spec] <accept> p_t=1.0000 p_d=0.9950 u=0.0843
[spec] <reject> p_t=0.0000 p_d=0.0283 u=0.6010
[spec] <reject> p_t=0.0000 p_d=0.2682 u=0.3595
[spec] <reject> p_t=0.0000 p_d=0.0106 u=0.5729
[spec] <reject> p_t=0.0000 p_d=0.2050 u=0.2183
[spec] <reject> p_t=0.0273 p_d=0.1509 u=0.8030
[spec] <reject> p_t=0.0000 p_d=0.1225 u=0.5696
[spec] <reject> p_t=0.0000 p_d=0.2258 u=0.3152
[spec] <reject> p_t=0.0000 p_d=0.0082 u=0.9051
[spec] <reject> p_t=0.0000 p_d=0.5610 u=0.3235
[spec] <reject> p_t=0.0000 p_d=0.7742 u=0.7696
[spec] <reject> p_t=0.0000 p_d=0.0019 u=0.7298
[spec] <reject> p_t=0.0000 p_d=0.0095 u=0.1452
[spec] <reject> p_t=0.0000 p_d=0.7820 u=0.9791
[spec] <reject> p_t=0.0002 p_d=0.2217 u=0.2744
[spec] <reject> p_t=0.0000 p_d=0.0025 u=0.7990
[spec] <reject> p_t=0.0000 p_d=0.4220 u=0.1983
[spec] <accept> p_t=0.2667 p_d=0.2205 u=0.4311
[spec] <reject> p_t=0.0000 p_d=0.0024 u=0.3914
[spec] <reject> p_t=0.0000 p_d=0.2110 u=0.9060


[timing] chat_web.stream.gpu0: tokens=277, time=37.026s, throughput=7.48 tok/s, accept_ratio=0.08
```

### Without Speculative Sampling Enabled (`--speculative-proposals 0`)

```text
[timing] chat_web.stream.gpu0: tokens=443, time=24.426s, throughput=18.14 tok/s
```

### 原因分析

在当前这组设置下，开启 speculative sampling 反而会让 tok/s 明显下降，accept_ratio 也很低（约 0.08）。根本原因在于 draft 模型过小（只有 4 层），与目标模型的分布差距较大，导致 rejection 频率偏高：草稿模型提出的很多 token 在目标分布上概率很低甚至接近 0，于是被频繁拒绝，目标模型不得不重新计算并做补采样。这样一来，并没有显著减少目标模型的前向步数，反而增加了草稿模型额外的前向开销和 accept/reject 的逻辑成本。换句话说，在目前这组模型大小和结构的搭配下，speculative sampling 还处在“draft 太弱”的区间，还不足以带来预期的加速效果。
