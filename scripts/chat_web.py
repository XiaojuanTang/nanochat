#!/usr/bin/env python3
"""
Unified web chat server - serves both UI and API from a single FastAPI instance.

Uses data parallelism to distribute requests across multiple GPUs. Each GPU loads
a full copy of the model, and incoming requests are distributed to available workers.

Launch examples:

- single available GPU (default)
python -m scripts.chat_web

- 4 GPUs
python -m scripts.chat_web --num-gpus 4

To chat, open the URL printed in the console. (If on cloud box, make sure to use public IP)

Endpoints:
  GET  /           - Chat UI
  POST /chat/completions - Chat API (streaming only)
  GET  /health     - Health check with worker pool status
  GET  /stats      - Worker pool statistics and GPU utilization

Abuse Prevention:
  - Maximum 500 messages per request
  - Maximum 8000 characters per message
  - Maximum 32000 characters total conversation length
  - Temperature clamped to 0.0-2.0
  - Top-k clamped to 1-200
  - Top-p clamped to 0.0-1.0
  - Max tokens clamped to 1-4096
"""

import argparse
import json
import os
import torch
import asyncio
import logging
import random
import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, HTMLResponse, FileResponse
from pydantic import BaseModel
from typing import List, Optional, AsyncGenerator
from dataclasses import dataclass
from contextlib import nullcontext
from nanochat.common import compute_init, autodetect_device_type
from nanochat.checkpoint_manager import load_model
from nanochat.engine import Engine

# Abuse prevention limits
MAX_MESSAGES_PER_REQUEST = 500
MAX_MESSAGE_LENGTH = 8000
MAX_TOTAL_CONVERSATION_LENGTH = 32000
MIN_TEMPERATURE = 0.0
MAX_TEMPERATURE = 2.0
MIN_TOP_K = 1
MAX_TOP_K = 200
MIN_TOP_P = 0.0
MAX_TOP_P = 1.0
MIN_MAX_TOKENS = 1
MAX_MAX_TOKENS = 4096

# Basic concurrency & rate limiting
MAX_CONCURRENT_REQUESTS = 16
RATE_LIMIT_WINDOW_SECONDS = 60
RATE_LIMIT_MAX_REQUESTS = 120

parser = argparse.ArgumentParser(description='NanoChat Web Server')
parser.add_argument('-n', '--num-gpus', type=int, default=1, help='Number of GPUs to use (default: 1)')
parser.add_argument('-i', '--source', type=str, default="sft", help="Source of the model: sft|mid|rl")
parser.add_argument('-t', '--temperature', type=float, default=0.8, help='Default temperature for generation')
parser.add_argument('-k', '--top-k', type=int, default=50, help='Default top-k sampling parameter')
parser.add_argument('--top-p', type=float, default=1.0, help='Default top-p sampling parameter (0.0-1.0, 1.0 = disabled)')
parser.add_argument('-m', '--max-tokens', type=int, default=512, help='Default max tokens for generation')
parser.add_argument('-g', '--model-tag', type=str, default=None, help='Model tag to load')
parser.add_argument('-s', '--step', type=int, default=None, help='Step to load')
parser.add_argument('-p', '--port', type=int, default=8000, help='Port to run the server on')
parser.add_argument('-d', '--dtype', type=str, default='bfloat16', choices=['float32', 'bfloat16'])
parser.add_argument('--device-type', type=str, default='', choices=['cuda', 'cpu', 'mps'], help='Device type for evaluation: cuda|cpu|mps. empty => autodetect')
parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to bind the server to')
parser.add_argument('--speculative-proposals', type=int, default=0, help='Number of draft tokens to propose (0 disables speculative decoding)')
parser.add_argument('--draft-source', type=str, default="draft", help='Source of the draft model: sft|mid|rl (optional)')
parser.add_argument('--draft-model-tag', type=str, default="d4", help='Model tag to load for draft model')
parser.add_argument('--draft-step', type=int, default=None, help='Step to load for draft model')
parser.add_argument('--verbose-speculative', action='store_true', help='Log per-token speculative accept/reject info')
parser.add_argument('--log-timing', action='store_true', help='Log wall-clock and throughput stats for generation')
args = parser.parse_args()

# Configure logging for conversation traffic
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

device_type = autodetect_device_type() if args.device_type == "" else args.device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
ptdtype = torch.float32 if args.dtype == 'float32' else torch.bfloat16

@dataclass
class Worker:
    """A worker with a model loaded on a specific GPU."""
    gpu_id: int
    device: torch.device
    engine: Engine
    tokenizer: object
    autocast_ctx: torch.amp.autocast
    draft: Optional[object] = None

class WorkerPool:
    """Pool of workers, each with a model replica on a different GPU."""

    def __init__(self, num_gpus: Optional[int] = None):
        if num_gpus is None:
            if device_type == "cuda":
                num_gpus = torch.cuda.device_count()
            else:
                num_gpus = 1 # e.g. cpu|mps
        self.num_gpus = num_gpus
        self.workers: List[Worker] = []
        self.available_workers: asyncio.Queue = asyncio.Queue()

    async def initialize(self, source: str, model_tag: Optional[str] = None, step: Optional[int] = None):
        """Load model on each GPU."""
        print(f"Initializing worker pool with {self.num_gpus} GPUs...")
        if self.num_gpus > 1:
            assert device_type == "cuda", "Only CUDA supports multiple workers/GPUs. cpu|mps does not."

        for gpu_id in range(self.num_gpus):

            if device_type == "cuda":
                device = torch.device(f"cuda:{gpu_id}")
                print(f"Loading model on GPU {gpu_id}...")
            else:
                device = torch.device(device_type) # e.g. cpu|mps
                print(f"Loading model on {device_type}...")

            model, tokenizer, _ = load_model(source, device, phase="eval", model_tag=model_tag, step=step)
            draft_model = None
            draft_tokenizer = None
            if args.speculative_proposals > 0 and args.draft_source is not None:
                try:
                    draft_model, draft_tokenizer, _ = load_model(
                        args.draft_source,
                        device,
                        phase="eval",
                        model_tag=args.draft_model_tag,
                        step=args.draft_step
                    )
                    print(f"Loaded draft model for speculative decoding on {device}")
                except Exception as e:
                    print(f"Warning: failed to load draft model, speculative decoding disabled: {e}")
                    draft_model = None
                    draft_tokenizer = None

            engine = Engine(model, tokenizer, draft_model=draft_model, draft_tokenizer=draft_tokenizer)
            engine.verbose_speculative = args.verbose_speculative
            autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype) if device_type == "cuda" else nullcontext()

            worker = Worker(
                gpu_id=gpu_id,
                device=device,
                engine=engine,
                tokenizer=tokenizer,
                autocast_ctx=autocast_ctx,
                draft=draft_model
            )
            self.workers.append(worker)
            await self.available_workers.put(worker)

        print(f"All {self.num_gpus} workers initialized!")

    async def acquire_worker(self) -> Worker:
        """Get an available worker from the pool."""
        return await self.available_workers.get()

    async def release_worker(self, worker: Worker):
        """Return a worker to the pool."""
        await self.available_workers.put(worker)

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    top_k: Optional[int] = None
    top_p: Optional[float] = None

class OpenAIChatMessage(BaseModel):
    role: str
    content: str

class OpenAIChatCompletionRequest(BaseModel):
    model: Optional[str] = None
    messages: List[OpenAIChatMessage]
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    max_tokens: Optional[int] = None
    stream: Optional[bool] = False
    n: Optional[int] = 1
    user: Optional[str] = None

def validate_chat_request(request: ChatRequest):
    """Validate chat request to prevent abuse."""
    # Check number of messages
    if len(request.messages) == 0:
        raise HTTPException(status_code=400, detail="At least one message is required")
    if len(request.messages) > MAX_MESSAGES_PER_REQUEST:
        raise HTTPException(
            status_code=400,
            detail=f"Too many messages. Maximum {MAX_MESSAGES_PER_REQUEST} messages allowed per request"
        )

    # Check individual message lengths and total conversation length
    total_length = 0
    for i, message in enumerate(request.messages):
        if not message.content:
            raise HTTPException(status_code=400, detail=f"Message {i} has empty content")

        msg_length = len(message.content)
        if msg_length > MAX_MESSAGE_LENGTH:
            raise HTTPException(
                status_code=400,
                detail=f"Message {i} is too long. Maximum {MAX_MESSAGE_LENGTH} characters allowed per message"
            )
        total_length += msg_length

    if total_length > MAX_TOTAL_CONVERSATION_LENGTH:
        raise HTTPException(
            status_code=400,
            detail=f"Total conversation is too long. Maximum {MAX_TOTAL_CONVERSATION_LENGTH} characters allowed"
        )

    # Validate role values (system messages are allowed but currently ignored in tokenization)
    for i, message in enumerate(request.messages):
        if message.role not in ["user", "assistant", "system"]:
            raise HTTPException(
                status_code=400,
                detail=f"Message {i} has invalid role. Must be 'user', 'assistant', or 'system'"
            )

    # Validate temperature
    if request.temperature is not None:
        if not (MIN_TEMPERATURE <= request.temperature <= MAX_TEMPERATURE):
            raise HTTPException(
                status_code=400,
                detail=f"Temperature must be between {MIN_TEMPERATURE} and {MAX_TEMPERATURE}"
            )

    # Validate top_k
    if request.top_k is not None:
        if not (MIN_TOP_K <= request.top_k <= MAX_TOP_K):
            raise HTTPException(
                status_code=400,
                detail=f"top_k must be between {MIN_TOP_K} and {MAX_TOP_K}"
            )

    # Validate top_p
    if request.top_p is not None:
        if not (MIN_TOP_P <= request.top_p <= MAX_TOP_P):
            raise HTTPException(
                status_code=400,
                detail=f"top_p must be between {MIN_TOP_P} and {MAX_TOP_P}"
            )

    # Validate max_tokens
    if request.max_tokens is not None:
        if not (MIN_MAX_TOKENS <= request.max_tokens <= MAX_MAX_TOKENS):
            raise HTTPException(
                status_code=400,
                detail=f"max_tokens must be between {MIN_MAX_TOKENS} and {MAX_MAX_TOKENS}"
            )

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on all GPUs on startup."""
    print("Loading nanochat models across GPUs...")
    app.state.worker_pool = WorkerPool(num_gpus=args.num_gpus)
    await app.state.worker_pool.initialize(args.source, model_tag=args.model_tag, step=args.step)
    app.state.request_semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    app.state.rate_limit = {"window_start": time.time(), "count": 0}
    # Load API keys from config.json (optional)
    config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "config.json"))
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
            api_keys = {str(k) for k in config["valid_api_keys"] if k}
    except Exception as e:
        logging.warning(f"Warning: failed to load config.json: {e}")
    app.state.api_keys = api_keys
    print(f"Server ready at http://localhost:{args.port}")
    yield

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def check_rate_limit():
    """Global, very simple rate limiter."""
    rl = getattr(app.state, "rate_limit", None)
    if rl is None:
        return
    now = time.time()
    if now - rl["window_start"] > RATE_LIMIT_WINDOW_SECONDS:
        rl["window_start"] = now
        rl["count"] = 0
    if rl["count"] >= RATE_LIMIT_MAX_REQUESTS:
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    rl["count"] += 1


def check_api_key(authorization_header: Optional[str]):
    """Basic Bearer token check."""
    valid_keys = app.state.api_keys
    if not authorization_header:
        raise HTTPException(status_code=401, detail="Missing Authorization header")
    if not authorization_header.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Invalid Authorization header")
    token = authorization_header.split(" ", 1)[1].strip()
    if token not in valid_keys:
        raise HTTPException(status_code=401, detail="Invalid API key")

@app.get("/")
async def root():
    """Serve the chat UI."""
    ui_html_path = os.path.join("nanochat", "ui.html")
    with open(ui_html_path, "r", encoding="utf-8") as f:
        html_content = f.read()
    # Replace the API_URL to use the same origin
    html_content = html_content.replace(
        "const API_URL = `http://${window.location.hostname}:8000`;",
        "const API_URL = '';"
    )
    # Inject the default temperature/top-k and their limits from server args into the UI
    html_content = html_content.replace(
        "__DEFAULT_TEMPERATURE__",
        str(args.temperature)
    )
    html_content = html_content.replace(
        "__DEFAULT_TOP_K__",
        str(args.top_k)
    )
    html_content = html_content.replace(
        "__DEFAULT_TOP_P__",
        str(args.top_p)
    )
    html_content = html_content.replace(
        "__MIN_TEMPERATURE__",
        str(MIN_TEMPERATURE)
    )
    html_content = html_content.replace(
        "__MAX_TEMPERATURE__",
        str(MAX_TEMPERATURE)
    )
    html_content = html_content.replace(
        "__MIN_TOP_K__",
        str(MIN_TOP_K)
    )
    html_content = html_content.replace(
        "__MAX_TOP_K__",
        str(MAX_TOP_K)
    )
    html_content = html_content.replace(
        "__MIN_TOP_P__",
        str(MIN_TOP_P)
    )
    html_content = html_content.replace(
        "__MAX_TOP_P__",
        str(MAX_TOP_P)
    )
    return HTMLResponse(content=html_content)


@app.get("/logo.svg")
async def logo():
    """Serve the NanoChat logo for favicon and header."""
    logo_path = os.path.join("nanochat", "logo.svg")
    return FileResponse(logo_path, media_type="image/svg+xml")

async def generate_stream(
    worker: Worker,
    tokens,
    temperature=None,
    max_new_tokens=None,
    top_k=None,
    top_p=None,
    speculative_num_tokens: int = 0
) -> AsyncGenerator[str, None]:
    """Generate assistant response with streaming."""
    temperature = temperature if temperature is not None else args.temperature
    max_new_tokens = max_new_tokens if max_new_tokens is not None else args.max_tokens
    top_k = top_k if top_k is not None else args.top_k
    # Treat top_p >= 1.0 as disabled to preserve previous behavior
    top_p = top_p if (top_p is not None and top_p < 1.0) else args.top_p

    assistant_end = worker.tokenizer.encode_special("<|assistant_end|>")
    bos = worker.tokenizer.get_bos_token_id()

    # Accumulate tokens to properly handle multi-byte UTF-8 characters (like emojis)
    accumulated_tokens = []
    # Track the last complete UTF-8 string (without replacement characters)
    last_clean_text = ""

    with worker.autocast_ctx:
        for token_column, token_masks in worker.engine.generate(
            tokens,
            num_samples=1,
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            seed=random.randint(0, 2**31 - 1),
            speculative_num_tokens=speculative_num_tokens,
            log_timing=args.log_timing,
            timing_label=f"chat_web.stream.gpu{worker.gpu_id}"
        ):
            token = token_column[0]

            # Stopping criteria
            if token == assistant_end or token == bos:
                break

            # Append the token to sequence
            accumulated_tokens.append(token)
            # Decode all accumulated tokens to get proper UTF-8 handling
            # Note that decode is a quite efficient operation, basically table lookup and string concat
            current_text = worker.tokenizer.decode(accumulated_tokens)
            # Only emit text if it doesn't end with a replacement character
            # This ensures we don't emit incomplete UTF-8 sequences
            if not current_text.endswith('�'):
                # Extract only the new text since last clean decode
                new_text = current_text[len(last_clean_text):]
                if new_text:  # Only yield if there's new content
                    yield f"data: {json.dumps({'token': new_text, 'gpu': worker.gpu_id}, ensure_ascii=False)}\n\n"
                    last_clean_text = current_text

    yield f"data: {json.dumps({'done': True})}\n\n"

@app.post("/chat/completions")
async def chat_completions(request: ChatRequest):
    """Chat completion endpoint (streaming only) - uses worker pool for multi-GPU."""

    # Basic validation to prevent abuse
    validate_chat_request(request)

    # Log incoming conversation to console
    logger.info("="*20)
    for i, message in enumerate(request.messages):
        logger.info(f"[{message.role.upper()}]: {message.content}")
    logger.info("-"*20)

    worker_pool = app.state.worker_pool
    worker = await worker_pool.acquire_worker()

    try:
        # Build conversation tokens
        bos = worker.tokenizer.get_bos_token_id()
        user_start = worker.tokenizer.encode_special("<|user_start|>")
        user_end = worker.tokenizer.encode_special("<|user_end|>")
        assistant_start = worker.tokenizer.encode_special("<|assistant_start|>")
        assistant_end = worker.tokenizer.encode_special("<|assistant_end|>")

        conversation_tokens = [bos]
        for message in request.messages:
            if message.role == "user":
                conversation_tokens.append(user_start)
                conversation_tokens.extend(worker.tokenizer.encode(message.content))
                conversation_tokens.append(user_end)
            elif message.role == "assistant":
                conversation_tokens.append(assistant_start)
                conversation_tokens.extend(worker.tokenizer.encode(message.content))
                conversation_tokens.append(assistant_end)

        conversation_tokens.append(assistant_start)

        # Streaming response with worker release after completion
        response_tokens = []
        async def stream_and_release():
            try:
                async for chunk in generate_stream(
                    worker,
                    conversation_tokens,
                    temperature=request.temperature,
                    max_new_tokens=request.max_tokens,
                    top_k=request.top_k,
                    top_p=request.top_p,
                    speculative_num_tokens=args.speculative_proposals
                ):
                    # Accumulate response for logging
                    chunk_data = json.loads(chunk.replace("data: ", "").strip())
                    if "token" in chunk_data:
                        response_tokens.append(chunk_data["token"])
                    yield chunk
            finally:
                # Log the assistant response to console
                full_response = "".join(response_tokens)
                logger.info(f"[ASSISTANT] (GPU {worker.gpu_id}): {full_response}")
                logger.info("="*20)
                # Release worker back to pool after streaming is done
                await worker_pool.release_worker(worker)

        return StreamingResponse(
            stream_and_release(),
            media_type="text/event-stream"
        )
    except Exception as e:
        # Make sure to release worker even on error
        await worker_pool.release_worker(worker)
        raise e

@app.post("/v1/chat/completions")
async def openai_chat_completions(request: OpenAIChatCompletionRequest, authorization: Optional[str] = Header(None)):
    """
    OpenAI-compatible chat completions endpoint.
    Supports both streaming and non-streaming responses.
    """
    if request.n is not None and request.n != 1:
        raise HTTPException(status_code=400, detail="Only n=1 is supported at the moment")

    internal_request = ChatRequest(
        messages=[ChatMessage(role=m.role, content=m.content) for m in request.messages],
        temperature=request.temperature,
        max_tokens=request.max_tokens,
        top_k=request.top_k,
        top_p=request.top_p,
    )

    validate_chat_request(internal_request)
    check_rate_limit()
    check_api_key(authorization)

    concurrency_semaphore = getattr(app.state, "request_semaphore", None)
    if concurrency_semaphore is not None:
        await concurrency_semaphore.acquire()

    worker_pool = app.state.worker_pool
    worker = await worker_pool.acquire_worker()

    created = int(time.time())
    completion_id = f"chatcmpl-{created}-{random.randint(0, 2**31 - 1)}"
    model_name = request.model or args.model_tag or "nanochat"

    # Build conversation tokens (reuse the same scheme as /chat/completions)
    bos = worker.tokenizer.get_bos_token_id()
    user_start = worker.tokenizer.encode_special("<|user_start|>")
    user_end = worker.tokenizer.encode_special("<|user_end|>")
    assistant_start = worker.tokenizer.encode_special("<|assistant_start|>")
    assistant_end = worker.tokenizer.encode_special("<|assistant_end|>")

    conversation_tokens = [bos]
    for message in internal_request.messages:
        if message.role == "user":
            conversation_tokens.append(user_start)
            conversation_tokens.extend(worker.tokenizer.encode(message.content))
            conversation_tokens.append(user_end)
        elif message.role == "assistant":
            conversation_tokens.append(assistant_start)
            conversation_tokens.extend(worker.tokenizer.encode(message.content))
            conversation_tokens.append(assistant_end)
        # system messages are currently ignored in tokenization

    conversation_tokens.append(assistant_start)
    prompt_tokens = len(conversation_tokens)

    max_new_tokens = internal_request.max_tokens or args.max_tokens
    top_k = internal_request.top_k or args.top_k
    temperature = internal_request.temperature if internal_request.temperature is not None else args.temperature
    top_p = internal_request.top_p if internal_request.top_p is not None else args.top_p

    async def stream_response():
        accumulated_tokens = []
        last_clean_text = ""
        full_text = ""
        first_chunk = True

        try:
            with worker.autocast_ctx:
                for token_column, token_masks in worker.engine.generate(
                    conversation_tokens,
                    num_samples=1,
                    max_tokens=max_new_tokens,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    seed=random.randint(0, 2**31 - 1),
                    speculative_num_tokens=args.speculative_proposals
                ):
                    token = token_column[0]

                    if token == assistant_end or token == bos:
                        break

                    accumulated_tokens.append(token)
                    current_text = worker.tokenizer.decode(accumulated_tokens)
                    if not current_text.endswith("�"):
                        new_text = current_text[len(last_clean_text):]
                        if new_text:
                            full_text += new_text
                            last_clean_text = current_text

                            delta = {"content": new_text}
                            if first_chunk:
                                delta["role"] = "assistant"
                                first_chunk = False

                            chunk = {
                                "id": completion_id,
                                "object": "chat.completion.chunk",
                                "created": created,
                                "model": model_name,
                                "choices": [
                                    {
                                        "index": 0,
                                        "delta": delta,
                                        "finish_reason": None,
                                    }
                                ],
                            }
                            yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"

            # Final chunk with finish_reason
            final_chunk = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model_name,
                "choices": [
                    {
                        "index": 0,
                        "delta": {},
                        "finish_reason": "stop",
                    }
                ],
            }
            yield f"data: {json.dumps(final_chunk, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"
        finally:
            logger.info(f"[OPENAI STREAM] (GPU {worker.gpu_id}): {full_text}")
            logger.info("="*20)
            await worker_pool.release_worker(worker)
            if concurrency_semaphore is not None:
                concurrency_semaphore.release()

    try:
        if request.stream:
            return StreamingResponse(
                stream_response(),
                media_type="text/event-stream"
            )

        # Non-streaming path: accumulate full response and return once
        accumulated_tokens = []
        last_clean_text = ""
        full_text = ""

        with worker.autocast_ctx:
            for token_column, token_masks in worker.engine.generate(
                conversation_tokens,
                num_samples=1,
                max_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                seed=random.randint(0, 2**31 - 1),
                speculative_num_tokens=args.speculative_proposals
            ):
                token = token_column[0]

                if token == assistant_end or token == bos:
                    break

                accumulated_tokens.append(token)
                current_text = worker.tokenizer.decode(accumulated_tokens)
                if not current_text.endswith("�"):
                    new_text = current_text[len(last_clean_text):]
                    if new_text:
                        full_text += new_text
                        last_clean_text = current_text

        completion_tokens = len(accumulated_tokens)
        total_tokens = prompt_tokens + completion_tokens

        logger.info(f"[OPENAI] (GPU {worker.gpu_id}): {full_text}")
        logger.info("="*20)

        await worker_pool.release_worker(worker)
        if concurrency_semaphore is not None:
            concurrency_semaphore.release()

        return {
            "id": completion_id,
            "object": "chat.completion",
            "created": created,
            "model": model_name,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": full_text,
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
            },
        }
    except Exception as e:
        await worker_pool.release_worker(worker)
        if concurrency_semaphore is not None:
            concurrency_semaphore.release()
        raise e

@app.get("/health")
async def health():
    """Health check endpoint."""
    worker_pool = getattr(app.state, 'worker_pool', None)
    return {
        "status": "ok",
        "ready": worker_pool is not None and len(worker_pool.workers) > 0,
        "num_gpus": worker_pool.num_gpus if worker_pool else 0,
        "available_workers": worker_pool.available_workers.qsize() if worker_pool else 0
    }

@app.get("/stats")
async def stats():
    """Get worker pool statistics."""
    worker_pool = app.state.worker_pool
    return {
        "total_workers": len(worker_pool.workers),
        "available_workers": worker_pool.available_workers.qsize(),
        "busy_workers": len(worker_pool.workers) - worker_pool.available_workers.qsize(),
        "workers": [
            {
                "gpu_id": w.gpu_id,
                "device": str(w.device)
            } for w in worker_pool.workers
        ]
    }

if __name__ == "__main__":
    import uvicorn
    print(f"Starting NanoChat Web Server")
    print(f"Temperature: {args.temperature}, Top-k: {args.top_k}, Top-p: {args.top_p}, Max tokens: {args.max_tokens}")
    uvicorn.run(app, host=args.host, port=args.port)
