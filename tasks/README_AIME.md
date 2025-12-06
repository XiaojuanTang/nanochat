# AIME 任务说明

AIME（American Invitational Mathematics Examination）是美国高中数学邀请赛，题目为 15 道数值填空，强调多步推理与构造，单题信息密度高，适合评估大模型的数学推理上限。本仓库中实现了 AIME 2024 和 AIME 2025 两套评估任务。

---

## 数据来源

- **AIME 2024**：使用 HuggingFace 上的 `HuggingFaceH4/aime_2024` 数据集：
  - 字段：`id, problem, solution, answer, url, year`
  - 含 2024 年 AIME I / II 共 30 道题。
- **AIME 2025**：使用 OpenCompass 提供的 `opencompass/AIME2025` 数据集：
  - Config：`AIME2025-I`、`AIME2025-II`
  - 字段：`question, answer`
  - 每个 config 各 15 题，总共 30 题。

代码对应关系：

- `tasks/aime_24_25.py` 中：
  - `AIME_24`：封装 `HuggingFaceH4/aime_2024`；
  - `AIME_25`：封装 `opencompass/AIME2025` 两个 config 的合并。
- `scripts/chat_eval.py` 中将其注册为：
  - `AIME-2024` → `AIME_24(split="train")`
  - `AIME-2025` → `AIME_25(split="test")`

---

## 任务格式与实现

两套 AIME 任务都被建模为「开放式数值生成」任务（`eval_type = "generative"`），接口与 GSM8K 类似。

### 对话格式（conversation）

以单题为单位构造一个 `conversation` 字典：

- `messages`：长度至少为 2 的列表，包含：
  - `{"role": "user", "content": user_message}`：用户提示词；
  - `{"role": "assistant", "content": assistant_text}`：理想助手回答（包含标准答案）。


提示词约定（示例）：

- user 侧提示大致为：

  ```text
  Problem:
  <原始题面>

  Given the problem above, reply with the final answer as a LaTeX expression inside \boxed{...}.
  ```

  即只给出题面，并要求助手“直接给出最终答案，写成 LaTeX 的 `\boxed{...}` 形式”。

- assistant 侧理想回答：
  - 对 AIME 2024：可以在内部使用 `solution` 信息进行构造，最终回答形如：
    - `xxxx..., The answer is \boxed{70}.`
  - 对 AIME 2025：数据集没有官方 `solution` 字段，仅构造简短回答，同样以 `\boxed{<answer>}` 形式给出最终答案。

### 答案提取

在这种设定下，评估时只关心 `\boxed{...}` 中的最终数值：

- 在 `evaluate` 中，从模型输出的 `assistant_response` 文本中，用正则或简单字符串处理提取 `\boxed{...}` 内部的内容；
- 将提取出的内容做轻量清洗（去除空格、逗号、前导 0 等），得到预测答案 `pred`；
- 将数据集中提供的 `answer`（必要时做同样的清洗）作为标准答案 `ref`；
- 对比 `ref` 与 `pred` 是否完全一致。

这样，模型可以自由选择是否展示解题过程，而评估逻辑只聚焦于最终数值是否正确。  

---

## 评估指标与计算方式

我们采用**精确匹配准确率（Exact Match Accuracy）**作为主要指标：

- 对每道题：
  - 从 `conversation["answer"]` 或理想 assistant 文本中抽取并归一化标准答案 `ref`；
  - 从模型生成的 `assistant_response` 文本中抽取并归一化预测答案 `pred`；
  - 若 `ref` 与 `pred` 字符串完全一致，则记为 1，否则为 0。
- 在 `scripts/chat_eval.py` 中，AIME 任务走 `run_generative_eval` 流程：
  - 可以配置 `--num-samples`，对同一题采样多次；
  - 只要某次采样 `evaluate` 返回 1，则这道题视为通过，相当于报告简单的 `pass@k`。

整体准确率定义为：

- `accuracy = 通过题数 / 总题数`。

---

## 使用方式（如何运行 AIME 评估）

前提：已按照 `README.md` 中的流程训练出 mid / sft 模型，并准备好虚拟环境。

示例命令：

- 评估 AIME 2024：

```bash
python -m nanochat.scripts.chat_eval -i sft -a AIME-2024
```

- 评估 AIME 2025：

```bash
python -m nanochat.scripts.chat_eval -i sft -a AIME-2025
```

常用参数：

- `-i/--source`：模型来源（`base|mid|sft|rl`）；
- `-a/--task-name`：任务名称（此处为 `AIME-2024` 或 `AIME-2025`）；
- `-t/--temperature`、`-n/--num-samples`：控制采样策略和 pass@k 行为；
- `-x/--max-problems`：可选，仅评估前若干题，便于快速调试。

---

## 设计思路与关键决策

1. **为何将 AIME 视为生成式数值任务而非多选题？**
   - AIME 原题是 0–999 的数值填空，本身不是多选，因此使用生成式数值输出符合原始形式；
   - 生成式设定可以完整考察模型从理解题意到推导、整合的能力，而不仅仅是从四个选项里选一个。

2. **为何要求输出 `\boxed{...}` 形式的答案？**
   - AIME 官方题解与竞赛圈常用记法都是用 `\boxed{...}` 表示最终答案，这样的格式对数学题来说更自然；
   - 相比自定义的标记（如 `#### <answer>`），`\boxed{...}` 更接近真实数学环境，有利于后续与其它 LaTeX 数学任务共享接口；
   - 只要在评估端约定“从 `\boxed{...}` 中抽取出数值”，就能保持提取逻辑简单可靠。

3. **为何将标准答案内嵌在 `conversation` 中而不是单独返回 label？**
   - 与现有 `Task` 抽象保持一致：上层 eval 逻辑只关心 `(conversation, completion)`，不需要关心 label 结构；
   - 同一个 `conversation` 可以同时用于：
     - SFT（理想 assistant message 作为 ground truth）；
     - RL（`evaluate` 和 `reward`）；
     - Chat eval（`run_generative_eval`）。

4. **AIME 2024 与 2025 分别建模为 `AIME_24` / `AIME_25` 的原因**
   - 两个数据集字段结构不同（`problem/solution/answer` vs `question/answer`），分开定义可以在各自类中处理字段和 split。
  
5. **不希望改动破坏现有评估脚本**
    - 直接在 `chat_eval.py` 中注册新任务名，复用已有的 `run_generative_eval` 流程，避免引入新的评估逻辑，保持代码简洁。

---

## 遇到的问题及解决方案

1. **不同年份数据集的字段结构不一致**
   - 问题：一开始假设 AIME 2024 与 AIME 2025 的字段相同（都包含 `problem/solution/answer`），在代码中直接复用同一套字段名会导致读取报错或字段缺失。
   - 解决：为两个年份分别实现 `AIME_24` / `AIME_25`，各自显式声明字段名与 split，并在 `chat_eval.py` 中分别注册 `AIME-2024` / `AIME-2025`。

2. **模型输出格式不稳定、可能不按要求输出 `\boxed{...}`**
   - 问题：实际评估时，模型有时不会严格遵守提示词要求，可能省略 `\boxed{}`、或者在答案前后混入额外文字，直接按整串文本对比会导致误判。
   - 解决：
     - 在提示词中强调“使用 LaTeX 表达式 `\boxed{...}` 给出最终答案”；
     - 部分数值前后允许有空格、逗号等非数字字符，做轻量清洗。
     - 某些ground truth答案中也可能0250等前导0，进一步做清洗处理。


---

## 功能验证与 Demo 说明

为了验证 AIME 任务实现的正确性与稳定性，主要做了以下检查：

1. **最小运行验证**
   - 在已有 SFT 模型上分别运行：
     ```bash
     # AIME 2024
     python -m nanochat.scripts.chat_eval -i sft -a AIME-2024

     # AIME 2025
     python -m nanochat.scripts.chat_eval -i sft -a AIME-2025
     ```
   - 期望结果：
     - 脚本能正常跑完，不出现字段缺失或索引错误；
     - 输出的 accuracy 在 (0%, 100%) 之间，且不同 checkpoint 下有明显变化。

2. **样本级手动检查**
   - 随机选取若干题目，打印对应的 `conversation` 与模型输出：
     - 确认 user 提示中包含完整题面与“用 `\boxed{...}` 输出答案”的要求；
     - 对比模型输出中从 `\boxed{...}` 抽取到的 `pred` 与真实答案 `ref` 是否符合直观判断。



