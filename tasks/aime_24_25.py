"""
AIME 2024 / 2025 evaluation tasks.

- AIME 2024 uses HuggingFaceH4/aime_2024:
  https://huggingface.co/datasets/HuggingFaceH4/aime_2024
  Fields: id, problem, solution, answer, url, year

- AIME 2025 uses opencompass/AIME2025:
  https://huggingface.co/datasets/opencompass/AIME2025
  Configs: AIME2025-I, AIME2025-II
  Fields: question, answer

For both years, we format the conversations so that the assistant is
asked to solve the problem step by step and clearly mark the final answer
wrapped in LaTeX syntax:

    \\boxed{<final answer>}

Evaluation is Exact Match on this final answer string (after extracting
from the \\boxed expression), with only light whitespace normalization.
"""

import re

from datasets import load_dataset

from tasks.common import Task


# We expect the final answer to be given in LaTeX-style \boxed{...}
BOXED_ANSWER_RE = re.compile(r"\\boxed{((?:\\[a-z]+|{[^{}]*}|[^{}])+)}")

def normalize_answer(ans: str):
    """
    Normalize an answer string for comparison.

    - Casts to string
    - Strips leading/trailing whitespace
    - Collapses internal whitespace runs to a single space
    - If the result is a plain integer (optionally signed), normalizes away
      leading zeros and an explicit '+' sign
    """
    if ans is None:
        return None
    ans = str(ans)
    ans = ans.strip()
    if not ans:
        return None
    # collapse internal whitespace so that minor spacing differences don't matter
    ans = re.sub(r"\s+", " ", ans)
    # If this looks like a plain integer, normalize it numerically
    m = re.fullmatch(r"([+-]?)(\d+)", ans)
    m = re.fullmatch(r"([+-]?)(\d+)(\D.*)?", ans)
    if m:
        sign, digits, suffix = m.groups()
        digits = digits.lstrip("0") or "0"
        if sign == "+":
            sign = ""
        ans = sign + digits + (suffix or "")
    return ans
    

def extract_answer(text: str):
    """
    Extract the answer string from a model completion.

    Extracts the contents of a LaTeX \\boxed{...} expression, if present.
    """
    if not isinstance(text, str):
        return None
    match = BOXED_ANSWER_RE.search(text)
    if match:
        return match.group(1).strip()
    return None



def get_reference_answer(conversation):
    """
    Extract the ground-truth answer from a conversation.
    """
    ref = conversation.get("answer")
    return normalize_answer(ref)


def evaluate_numeric_conversation(conversation, assistant_response: str):
    """
    Generic 0/1 evaluation for AIME-style answers based on string match.
    """
    assert isinstance(assistant_response, str), "Assuming simple string response for now"
    # First extract the ground truth answer
    assistant_message = conversation['messages'][-1]
    assert assistant_message['role'] == "assistant", "Last message must be from the Assistant"
    last_text_part = assistant_message['content'] # this contains the final answer in GSM8K
    # Extract both the ground truth answer and the predicted answer
    ref = normalize_answer(extract_answer(last_text_part))
    pred = normalize_answer(extract_answer(assistant_response))
    if ref is None or pred is None:
        return 0
    return ref == pred


class AIME_24(Task):
    """
    AIME 2024 evaluation task (HuggingFaceH4/aime_2024).

    Usage example:
    - AIME_24(split="train")
    """

    def __init__(self, split="train", **kwargs):
        super().__init__(**kwargs)
        assert split in ["train"], "AIME 2024 only defines the 'train' split"
        self.year = 2024
        dataset_name = "HuggingFaceH4/aime_2024"
        # Shuffle once with a fixed seed for deterministic ordering
        self.ds = load_dataset(dataset_name, split=split).shuffle(seed=42)

    @property
    def eval_type(self):
        # Open-ended generative task with exact-match scoring
        return "generative"

    def num_examples(self):
        return len(self.ds)

    def get_example(self, index):
        row = self.ds[index]
        problem = row["problem"]
        solution = row.get("solution", "")
        answer = row["answer"]
        
        # User instruction: solve and end with \boxed{<answer>}
        user_message = (
            f"Problem:\n{problem}\n\n"
            "Given the problem above, reply with the final answer "
            "as a LaTeX expression inside \\boxed{...}."
        )

        # Ideal assistant message: full solution + canonical answer marker
        assistant_text = solution.rstrip()
        if assistant_text:
            assistant_text += "\n\n"
        assistant_text += "\\boxed{" + str(answer) + "}"

        messages = [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": assistant_text},
        ]
        conversation = {
            "messages": messages,
        }
        return conversation

    def evaluate(self, conversation, assistant_response):
        """
        Return 1 if the model's final numeric answer matches the ground truth, else 0.
        """
        return evaluate_numeric_conversation(conversation, assistant_response)

    

class AIME_25(Task):
    """
    AIME 2025 evaluation task (opencompass/AIME2025).

    This task loads both AIME2025-I and AIME2025-II configs and
    concatenates them into a single evaluation set.

    Usage example:
    - AIME_25(split="test")
    """

    def __init__(self, split="test", **kwargs):
        super().__init__(**kwargs)
        assert split in ["test"], "AIME 2025 only defines the 'test' split"
        self.year = 2025
        ds_i = load_dataset("opencompass/AIME2025", "AIME2025-I", split=split)
        ds_ii = load_dataset("opencompass/AIME2025", "AIME2025-II", split=split)
        self.ds = ds_i.concatenate(ds_ii).shuffle(seed=42)

    @property
    def eval_type(self):
        # Open-ended generative task with exact-match scoring
        return "generative"

    def num_examples(self):
        return len(self.ds)

    def get_example(self, index):
        row = self.ds[index]
        problem = row["question"]
        answer = row["answer"]
        
        

        user_message = (
            f"Problem:\n{problem}\n\n"
            "Given the problem above, reply with the final answer "
            "as a LaTeX expression inside \\boxed{...}."
        )

        # AIME2025 dataset does not provide official solutions, so we only
        # include the final answer pattern in the ideal assistant message.
        assistant_text = "\\boxed{" + str(answer) + "}"
        
        messages = [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": assistant_text},
        ]
        conversation = {
            "messages": messages,
        }
        return conversation

    def evaluate(self, conversation, assistant_response):
        """
        Return 1 if the model's final numeric answer matches the ground truth, else 0.
        """
        return evaluate_numeric_conversation(conversation, assistant_response)

    