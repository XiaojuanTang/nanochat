# 批处理推理与 batch generation

本任务的目标是：在不改变模型行为的前提下，为 `nanochat/engine.py` 增加高效的批处理推理（batch generation）能力，并在评测脚本中用上这一能力提升吞吐量。

主要包括四件事：

- 让 `Engine.generate` 能一次处理多条 prompt，并配合 KVCache 做自回归解码。
- 适配目前已有的函数 `Engine.generate_batch` API，保证训练 / 评测直接拿到完整序列，当batch中其中一个碰到结束符时，继续生成，直到所有的序列都结束，但是在读取生成的 tokens 时只读取到第一个结束符处。
- 配合 `nanochat/gpt.py` 中新增的 `attention_mask` 机制，支持不同长度序列的 padding，保证自回归生成的正确性。
- 修改 `scripts/chat_eval.py`，在生成式评测中按批处理多道题，实现动态 batching 提升速度。
- TODO：由于时间关系，没有测试并对⽐不同 batch size 下的吞吐量提升，只对功能正确性进行了验证。

下面按模块说明具体实现方式。

---

## 1. Engine.generate：从单序列到多序列

原始版本的 `Engine.generate` 假设 batch size=1：

- `KVCache` 初始化时写死 `batch_size=1`，`seq_len=len(tokens)`；
- 输入张量是 `[tokens]`，只是在 prompt 外面套了一层 batch 维；
- 不支持显式的 `attention_mask`，也没有处理 padding。

这次改动的第一步，是把 `Engine.generate` 泛化成“真正的 batch 输入”：

这是原来的输入：

```python
@torch.inference_mode()
def generate(self, tokens, attention_mask=None, num_samples=1,
             max_tokens=None, temperature=1.0, top_k=None, seed=42):
    # 支持传入单条或多条序列，内部统一成 List[List[int]]
    assert isinstance(tokens, list) and isinstance(tokens[0], int)
    if isinstance(tokens[0], int):
        tokens = [tokens]
    device = self.model.get_device()
    ...
```

因此需要修改保证：

- 允许调用方传入单条序列（`List[int]`）或多条序列（`List[List[int]]`），内部统一成 `tokens: List[List[int]]`，batch size 就是 `len(tokens)`。

  ```python
      if isinstance(tokens[0], int):
                  tokens = [tokens]
  ```
- KVCache 按 batch 初始化：

  ```python
  m = self.model.config
  kv_model_kwargs = {
      "num_heads": m.n_kv_head,
      "head_dim": m.n_embd // m.n_head,
      "num_layers": m.n_layer,
  }
  kv_cache_prefill = KVCache(
      batch_size=len(tokens),
      seq_len=len(tokens[0]),
      **kv_model_kwargs,
  )
  ids = torch.tensor(tokens, dtype=torch.long, device=device)
  ```

  这样一开始的 prefill 就是一个真正的 batch。
- 接入 attention_mask：由于用一个batch里面的长度不一致，为了保证batch computation 需要对不同长度的序列按照最大长度进行padding。因此调用方需要传入 padding 后的 `attention_mask`，在 prefill 阶段就一起传入 `GPT.forward`：

  ```python
  if attention_mask is not None:
      ids_attention_mask = torch.tensor(attention_mask, dtype=torch.bool, device=device)
  else:
      ids_attention_mask = None

  logits = self.model.forward(
      ids,
      attention_mask=ids_attention_mask,
      kv_cache=kv_cache_prefill,
  )
  logits = logits[:, -1, :]
  ```

  这与 `nanochat/gpt.py` 里新增的 `attention_mask` 逻辑配合：GPT 在内部用 `build_attn_mask` 把 padding 部分屏蔽掉，和原来的causal mask综合处理。
- 兼容目前的 KV Cache方式：在解码阶段，由于保存了 KV Cache，可以每次只输入一个 token，那么对应的 attention mask 也需要在后续动态追加。因此我在解码阶段维护了一个随时间增长的 `ids_attention_mask`：

    ```python
    if attention_mask is not None:
        ids_attention_mask = torch.tensor(attention_mask, dtype=torch.bool, device=device)
    else:
        ids_attention_mask = None
    ...
    while True:
        ...
        if not first_iteration:
            # 每步将新一列 mask 追加到 attention_mask 的时间维
            ids_attention_mask = torch.cat([ids_attention_mask, masks], dim=1)
            logits = self.model.forward(
                ids,
                attention_mask=ids_attention_mask,
                kv_cache=kv_cache_decode,
            )
        ...
        ids = torch.tensor(token_column, dtype=torch.long, device=device).unsqueeze(1)
        masks = torch.tensor(token_masks, dtype=torch.bool, device=device).unsqueeze(1)
    ```

---

## 2. GPT.forward 中的 attention_mask 适配


- `GPT.forward` 会根据 `attention_mask` 重新构造一个 `attn_mask`，传给 Transformer 层：

  ```python
  def build_attn_mask(attn_mask, Tq, Tk, prefix_len):
    base = torch.tril(
        torch.ones((Tq, Tk), dtype=torch.bool, device=x.device),
        diagonal=prefix_len,
    ).unsqueeze(0).unsqueeze(0) # (1, 1, Tq, Tk)
    if attn_mask is None:
        return base
    mask = attn_mask
    
    key_mask = mask[:, None, None, :] # (B, 1, 1, Tk)
   
    return base & key_mask 



    attn_mask = None
    is_causal = True

    if attention_mask is not None or kv_cache is not None:
        attn_mask = build_attn_mask(attention_mask, T, Tk, prefix_len)
        is_causal = False
  ```
- 当 `attention_mask` 或 KVCache 存在时，`is_causal=False`，由显式的 `attn_mask` 控制可见性；
- 事实上为了保证 padding_mask 也可以被 training 模式下支持，`targets` 会根据 `mask_for_loss` 把 padding 位置置成 `-1`（ignore_index），以此来保证不计算那部分的loss
- 在我的实现中，build_attn_mask 放在了 `nanochat/gpt.py` 的 `GPT.forward` 里，和原来在 `CausalSelfAttention` 里实现考虑 KV Cache 下的 attention 计算不一样，因为在目前模型中所有 layer 的 attention mask 都是一样的，所以放在模型前端处理更合适。


---

## 3. batch generation 接口：Engine.generate_batch

- 在原来的已有接口中 `Engine.generate_batch`，需要求返回完整的 token 序列，而不是流式地一边生成一边返回 token。因此需要在额外的维度上维护每条序列的生成状态，直到所有序列都结束为止。

- 另外，由于输入的 prompt 长度不一，因此需要先对输入做左 padding，保证每条序列在 batch 里对齐。
```python
     padded_tokens = []
     masks = []
     for token in tokens:
         padded = [bos] * (max_tokens_len - len(token)) + token
         mask = [0] * (max_tokens_len - len(token)) + [1] * len(token)
         padded_tokens.append(padded)
         masks.append(mask)
```


```python
def generate_batch(self, tokens, num_samples=1, **kwargs):
    """
    Non-streaming batch generation that just returns the final token sequences.
    Returns a list of token sequences (list of lists of ints).
    Terminal tokens (assistant_end, bos) are not included in the results.
    """
    assistant_end = self.tokenizer.encode_special("<|assistant_end|>")
    bos = self.tokenizer.get_bos_token_id()
    max_tokens_len = max(len(token) for token in tokens)
    results = [token.copy() for token in tokens]
    completed = [False] * num_samples
    ...
```




## 4. 更改 `scripts/chat_eval.py` 来进行批处理评测


1. 为生成式评测新增 `--gen-batch-size` 参数：

```bash
python -m scripts.chat_eval \
    -i sft \
    -a GSM8K \
    --gen-batch-size 8 \
    --max-new-tokens 512 \
    --temperature 0.0
```

2. 在 `run_generative_eval` 中按 `gen_batch_size` 分批处理多个问题：

```python
rank_indices = list(range(ddp_rank, num_problems, ddp_world_size))
batch_size = max(1, batch_size)

for i in range(0, len(rank_indices), batch_size):
    batch_indices = rank_indices[i:i + batch_size]
    conversations = [task_object[idx] for idx in batch_indices]
    encoded_prompts = [tokenizer.render_for_completion(conv) for conv in conversations]

    batch_results, _ = engine.generate_batch(
            encoded_prompts,
            num_samples=batch_size,
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
        )
    raw_tokens = [res[len(encoded_prompt):] for res, encoded_prompt in zip(batch_results, encoded_prompts)] 
    completions = [tokenizer.decode(tokens) for tokens in raw_tokens]
    # completions = tokenizer.batch_decode(raw_tokens, skip_special_tokens=True)
    
    outcomes = [task_object.evaluate(conversation, completion) for conversation, completion in zip(conversations, completions)]
    passed = any(outcomes)
```

这样，原来一题一题地做生成，现在一次可以处理多道题，显著减少 Python 层的循环和 `Engine.generate` 调用次数，提高整体吞吐量。

---


## 5. 使用说明

- 在代码中直接使用批处理 API：

  ```python
  engine = Engine(model, tokenizer)
  prompts = [
      tokenizer.encode("The capital of France is", prepend=tokenizer.get_bos_token_id()),
      tokenizer.encode("The chemical symbol of gold is", prepend=tokenizer.get_bos_token_id()),
  ]
  samples, masks = engine.generate_batch(
      prompts,
      num_samples=len(prompts),
      max_tokens=64,
      temperature=0.0,
  )
  ```
- 在评测脚本中打开生成 batch：通过 `scripts/chat_eval.py` 的 `--gen-batch-size` 控制每个 rank 上一次处理多少道题，结合上面的 batch generation 实现，可以在保持输出行为一致的前提下获得更高的 tokens/s 吞吐量。

## 6. 性能验证
采用脚本：
```bash
python -m scripts.chat_eval \
    -i sft \
    -a GSM8K \
    --gen-batch-size 8 \
    --max-new-tokens 512 \
    --temperature 0.0
```
变化 `--gen-batch-size` 参数后，其准确率保持一致，吞吐量有明显提升。