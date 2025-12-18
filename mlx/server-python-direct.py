#!/usr/bin/env python3
"""
MLX Python 기반 HTTP 서버 (직접 구현)
FastAPI를 사용하여 SSE 스트리밍을 직접 구현합니다.
"""
import os
import sys
import json
import asyncio
import time
import psutil
import re
from pathlib import Path
from typing import Optional, List
from contextlib import asynccontextmanager

try:
    from fastapi import FastAPI, Request, HTTPException, WebSocket, WebSocketDisconnect
    from fastapi.responses import StreamingResponse, JSONResponse
    from fastapi.middleware.cors import CORSMiddleware
    import uvicorn
except ImportError:
    print("ERROR: FastAPI와 uvicorn이 설치되지 않았습니다.", file=sys.stderr)
    print("설치 명령: pip install fastapi uvicorn websockets", file=sys.stderr)
    sys.exit(1)

try:
    import mlx.core as mx
    from mlx_lm import load
    try:
        from mlx_lm.utils import generate_step
    except ImportError:
        # mlx_lm 0.29+ 버전에서는 generate_step이 generate 모듈에 있음
        from mlx_lm.generate import generate_step
    from mlx_lm.sample_utils import make_sampler, make_repetition_penalty
except ImportError as e:
    print(f"ERROR: MLX 라이브러리 미설치: {e}. 'pip install mlx-lm' 실행 필요", file=sys.stderr)
    sys.exit(1)

# 모델 경로
MODEL_PATH = os.getenv("MLX_MODEL_PATH", "./models/deepseek-moe-16b-chat-mlx-q4_0")
PORT = int(os.getenv("PORT", "8081"))

# 전역 변수
model = None
tokenizer = None
ready = False
processing = False
request_queue = []
log_websockets = []  # WebSocket 연결 리스트
metrics_websockets = []  # WebSocket 연결 리스트

# 메트릭 추적 변수
tokens_generated = 0
tokens_generated_total = 0
generation_start_time = None
last_token_time = None

# 로그 브로드캐스트 (전역 로그 리스트에 추가)
log_messages = []

def broadcast_log(message: str):
    """로그를 전역 리스트에 추가하고 콘솔에 출력"""
    log_messages.append(message)
    print(f"[LOG] {message}", flush=True)
    # 최근 1000개만 유지
    if len(log_messages) > 1000:
        log_messages.pop(0)
    
    # WebSocket 브로드캐스트는 각 WebSocket 핸들러에서 처리
    # (비동기 컨텍스트에서만 가능하므로 여기서는 리스트에만 추가)

async def broadcast_log_async(message: str):
    """비동기 컨텍스트에서 로그 브로드캐스트"""
    broadcast_log(message)
    # WebSocket으로 즉시 브로드캐스트
    disconnected = []
    for ws in log_websockets:
        try:
            await ws.send_json({"type": "log", "text": message})
        except (WebSocketDisconnect, ConnectionResetError, BrokenPipeError):
            disconnected.append(ws)
        except Exception as e:
            print(f"[ERROR] Failed to broadcast log to WebSocket: {e}", flush=True)
            disconnected.append(ws)
    
    # 연결이 끊어진 WebSocket 제거
    for ws in disconnected:
        if ws in log_websockets:
            log_websockets.remove(ws)

def broadcast_metrics():
    """메트릭 브로드캐스트 (WebSocket은 각 핸들러에서 처리)"""
    # 메트릭은 각 WebSocket 핸들러에서 주기적으로 전송하므로
    # 여기서는 별도 처리 불필요
    pass

@asynccontextmanager
async def lifespan(app: FastAPI):
    """애플리케이션 생명주기 관리"""
    global model, tokenizer, ready
    
    # 시작 시 모델 로드 (비동기로 실행하여 서버가 먼저 시작되도록)
    async def load_model_async():
        global model, tokenizer, ready
        await broadcast_log_async(f"Loading model from {MODEL_PATH}...")
        try:
            # GPU 사용 보장
            if 'MLX_DEFAULT_DEVICE' in os.environ and os.environ['MLX_DEFAULT_DEVICE'] == 'cpu':
                del os.environ['MLX_DEFAULT_DEVICE']
                await broadcast_log_async("Ensuring GPU (Metal) usage...")
            
            # 모델 디렉토리 크기 확인 (진행률 추정용)
            model_path_obj = Path(MODEL_PATH)
            total_size = 0
            if model_path_obj.exists():
                if model_path_obj.is_dir():
                    # 디렉토리인 경우 모든 파일 크기 합산
                    for file_path in model_path_obj.rglob('*'):
                        if file_path.is_file():
                            total_size += file_path.stat().st_size
                else:
                    # 파일인 경우
                    total_size = model_path_obj.stat().st_size
            
            # 모델 로딩 시작 시간
            load_start_time = time.time()
            last_progress_time = load_start_time
            
            # 모델 로딩을 별도 태스크로 실행 (진행률 모니터링 포함)
            async def load_with_progress():
                global model, tokenizer
                loop = asyncio.get_event_loop()
                
                # 백그라운드에서 모델 로딩 시작
                load_task = loop.run_in_executor(None, load, MODEL_PATH)
                
                # 진행률 모니터링 (0.5초마다)
                process = psutil.Process()
                initial_memory = process.memory_info().rss
                last_loaded_size = 0
                progress_steps = 0
                
                while not load_task.done():
                    await asyncio.sleep(0.5)
                    current_memory = process.memory_info().rss
                    loaded_size = current_memory - initial_memory
                    
                    # 메모리 증가량을 기반으로 진행률 추정
                    if total_size > 0:
                        # 메모리 증가량이 전체 크기의 일부라고 가정 (압축 해제 등 고려)
                        estimated_progress = min(95, (loaded_size / total_size) * 100)
                        if estimated_progress > last_loaded_size:
                            progress_steps += 1
                            if progress_steps % 4 == 0:  # 2초마다 업데이트
                                bar_length = 30
                                filled = int(bar_length * estimated_progress / 100)
                                bar = '█' * filled + '░' * (bar_length - filled)
                                await broadcast_log_async(f"Loading progress: [{bar}] {estimated_progress:.1f}% ({loaded_size / 1024 / 1024:.1f} MB loaded)")
                                last_loaded_size = estimated_progress
                    else:
                        # 크기를 알 수 없는 경우 메모리 사용량만 표시
                        if loaded_size > last_loaded_size:
                            progress_steps += 1
                            if progress_steps % 4 == 0:  # 2초마다 업데이트
                                await broadcast_log_async(f"Loading... ({loaded_size / 1024 / 1024:.1f} MB loaded)")
                                last_loaded_size = loaded_size
                
                # 모델 로딩 완료 대기
                model, tokenizer = await load_task
            
            await load_with_progress()
            
            load_time = time.time() - load_start_time
            ready = True
            await broadcast_log_async("✅ Model loaded successfully")
            await broadcast_log_async(f"⏱️  Loading time: {load_time:.2f} seconds")
            broadcast_metrics()
        except Exception as e:
            await broadcast_log_async(f"❌ Model loading failed: {str(e)}")
            import traceback
            await broadcast_log_async(traceback.format_exc())
            # 모델 로딩 실패해도 서버는 계속 실행 (ready=False 상태)
            ready = False
    
    # 모델 로딩을 백그라운드에서 시작
    asyncio.create_task(load_model_async())
    
    yield
    
    # 종료 시 정리
    broadcast_log("Shutting down...")

# FastAPI 앱 생성
app = FastAPI(lifespan=lifespan)

# CORS 미들웨어 추가
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Health check
@app.get("/health")
async def health():
    return {"status": "ready" if ready else "loading", "engine": "python-mlx-lm-direct"}

# Metrics endpoint
@app.get("/metrics")
async def metrics():
    system_metrics = get_system_metrics()
    return {
        "ready": ready,
        "processing": processing,
        "queueLength": len(request_queue),
        "engine": "python-mlx-lm-direct",
        **system_metrics
    }

def get_system_metrics():
    """시스템 메트릭 수집"""
    try:
        process = psutil.Process()
        
        # VRAM 정보 (macOS Metal)
        vram_total = 0
        vram_used = 0
        try:
            # macOS에서 Metal VRAM 정보 가져오기 시도
            import subprocess
            result = subprocess.run(['system_profiler', 'SPDisplaysDataType'], 
                                  capture_output=True, text=True, timeout=2)
            # 프로세스 메모리 사용량을 VRAM으로 간주
            process_memory = process.memory_info().rss
            vram_used = process_memory
            
            # 시스템 전체 메모리에서 GPU 메모리 추정 (일반적으로 시스템 메모리의 일부가 GPU에 할당됨)
            sys_mem = psutil.virtual_memory()
            # macOS에서 일반적으로 시스템 메모리의 일부가 GPU에 할당되므로 추정값 사용
            # 실제 GPU 메모리는 시스템 프로파일러에서 가져올 수 있지만 복잡함
            # 여기서는 프로세스 메모리를 기반으로 추정
            if process_memory > 0:
                # 프로세스가 사용하는 메모리를 기반으로 전체 GPU 메모리 추정 (보수적으로 2배)
                vram_total = max(process_memory * 2, sys_mem.total * 0.1)  # 최소 시스템 메모리의 10%
        except Exception as e:
            # 실패 시 프로세스 메모리만 사용
            process_memory = process.memory_info().rss
            vram_used = process_memory
            sys_mem = psutil.virtual_memory()
            vram_total = max(process_memory * 2, sys_mem.total * 0.1)
        
        # 시스템 메모리
        sys_mem = psutil.virtual_memory()
        sys_mem_total = sys_mem.total
        sys_mem_used = sys_mem.used
        
        # CPU 정보
        cpu_cores = psutil.cpu_count()
        cpu_times = process.cpu_times()
        proc_cpu_sec = cpu_times.user + cpu_times.system
        
        # 토큰 생성 속도 계산
        tps = 0.0
        global tokens_generated, generation_start_time, last_token_time
        if tokens_generated > 0 and generation_start_time:
            elapsed = time.time() - generation_start_time
            if elapsed > 0:
                tps = tokens_generated / elapsed
        
        return {
            "vramTotal": vram_total,
            "vramUsed": vram_used,
            "sysMemTotal": sys_mem_total,
            "sysMemUsed": sys_mem_used,
            "cpuCores": cpu_cores,
            "procCpuSec": proc_cpu_sec,
            "tps": tps,
            "predictedTotal": tokens_generated_total
        }
    except Exception as e:
        print(f"[ERROR] Failed to get system metrics: {e}", flush=True)
        return {
            "vramTotal": 0,
            "vramUsed": 0,
            "sysMemTotal": 0,
            "sysMemUsed": 0,
            "cpuCores": 1,
            "procCpuSec": 0.0,
            "tps": 0.0,
            "predictedTotal": 0
        }

# Metrics WebSocket endpoint
@app.websocket("/metrics/stream")
async def metrics_stream(websocket: WebSocket):
    """WebSocket으로 메트릭 스트리밍"""
    try:
        # print(f"[DEBUG] WebSocket connection attempt from {websocket.client}", flush=True)
        await websocket.accept()
        # print(f"[DEBUG] WebSocket accepted: {websocket.client}", flush=True)
        metrics_websockets.append(websocket)
        
        # 초기 메트릭 전송
        system_metrics = get_system_metrics()
        metrics = {
            "type": "metrics",
            "ready": ready,
            "processing": processing,
            "queueLength": len(request_queue),
            "engine": "python-mlx-lm-direct",
            **system_metrics
        }
        try:
            await websocket.send_json(metrics)
        except Exception as e:
            print(f"[ERROR] Failed to send initial metrics: {e}", flush=True)
        
        # 주기적으로 메트릭 전송 (0.5초마다 - 더 빠른 업데이트)
        while True:
            await asyncio.sleep(0.5)
            system_metrics = get_system_metrics()
            metrics = {
                "type": "metrics",
                "ready": ready,
                "processing": processing,
                "queueLength": len(request_queue),
                "engine": "python-mlx-lm-direct",
                **system_metrics
            }
            try:
                await websocket.send_json(metrics)
            except (WebSocketDisconnect, ConnectionResetError, BrokenPipeError):
                break
            except Exception as e:
                print(f"[ERROR] Metrics WebSocket send error: {e}", flush=True)
                break
    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"[ERROR] Metrics WebSocket error: {e}", flush=True)
    finally:
        if websocket in metrics_websockets:
            metrics_websockets.remove(websocket)

# Logs WebSocket endpoint
@app.websocket("/logs/stream")
async def logs_stream(websocket: WebSocket):
    """WebSocket으로 로그 스트리밍"""
    try:
        await websocket.accept()
        log_websockets.append(websocket)
        
        # 초기 로그 전송
        await websocket.send_json({"type": "log", "text": "[Server] Connected to log stream"})
        
        # 기존 로그 전송 (최근 100개)
        for msg in log_messages[-100:]:
            try:
                await websocket.send_json({"type": "log", "text": msg})
            except (WebSocketDisconnect, ConnectionResetError, BrokenPipeError):
                break
            except Exception as e:
                print(f"[ERROR] Logs WebSocket send error: {e}", flush=True)
                break
        
        # 새로운 로그 대기
        last_index = len(log_messages)
        while True:
            await asyncio.sleep(0.5)  # 0.5초마다 체크
            
            # 새로운 로그 전송
            if last_index < len(log_messages):
                for msg in log_messages[last_index:]:
                    try:
                        await websocket.send_json({"type": "log", "text": msg})
                    except (WebSocketDisconnect, ConnectionResetError, BrokenPipeError):
                        break
                    except Exception as e:
                        print(f"[ERROR] Logs WebSocket send error: {e}", flush=True)
                        break
                last_index = len(log_messages)
    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"[ERROR] Logs WebSocket error: {e}", flush=True)
    finally:
        if websocket in log_websockets:
            log_websockets.remove(websocket)

# Chat endpoint
@app.post("/chat")
async def chat(request: Request):
    """채팅 요청 처리 (SSE 스트리밍)"""
    global processing
    
    if not ready:
        raise HTTPException(status_code=503, detail="Model is loading...")
    
    if processing:
        raise HTTPException(status_code=503, detail="Server is busy")
    
    try:
        body = await request.json()
        prompt = body.get("prompt", "")
        max_tokens = body.get("max_tokens", body.get("n_predict", 512))
        temperature = body.get("temperature", 0.7)
        top_p = body.get("top_p", 0.95)
        min_p = body.get("min_p", 0.0)
        repeat_penalty = body.get("repeat_penalty", body.get("repeat_penalty", 1.1))
        repeat_last_n = body.get("repeat_last_n", body.get("repetition_context_size", 64))
        
        if not prompt:
            raise HTTPException(status_code=400, detail="Prompt is required")
        
        processing = True
        broadcast_metrics()
        
        async def generate():
            global processing, tokens_generated, tokens_generated_total, generation_start_time, last_token_time
            try:
                # 채팅 템플릿 적용
                try:
                    messages = [{"role": "user", "content": prompt}]
                    prompt_formatted = tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                except:
                    prompt_formatted = prompt
                
                # 프롬프트 토큰화
                prompt_tokens = tokenizer.encode(prompt_formatted)
                prompt_tokens_list = prompt_tokens.tolist() if hasattr(prompt_tokens, 'tolist') else list(prompt_tokens)
                prompt_array = mx.array(prompt_tokens)
                
                # 샘플러 생성 (mlx_lm 0.29+ 버전용)
                sampler = make_sampler(
                    temp=temperature,
                    top_p=top_p,
                    min_p=min_p
                )
                
                # Repetition penalty를 logits processor로 생성
                logits_processors = []
                if repeat_penalty != 1.0:
                    logits_processors.append(make_repetition_penalty(penalty=repeat_penalty, context_size=repeat_last_n))
                
                # generate_step으로 토큰 단위 생성
                token_count = 0
                generation_start_time = time.time()
                tokens_generated = 0
                eos_token_id = getattr(tokenizer, 'eos_token_id', None)
                
                # 토큰 누적을 위한 리스트 (생성된 토큰만 포함)
                accumulated_tokens = []
                previous_full_text = ""
                
                step_generator = generate_step(
                    prompt_array,
                    model,
                    sampler=sampler,
                    logits_processors=logits_processors if logits_processors else None,
                    max_tokens=max_tokens
                )
                
                # 토큰 단위로 생성 및 스트리밍
                while token_count < max_tokens:
                    try:
                        token_array, logits = next(step_generator)
                    except StopIteration:
                        break
                    
                    # 토큰 ID 추출
                    if isinstance(token_array, mx.array):
                        token_id = int(token_array.item())
                    elif isinstance(token_array, (int, float)):
                        token_id = int(token_array)
                    else:
                        try:
                            token_id = int(token_array)
                        except:
                            if hasattr(token_array, 'item'):
                                token_id = int(token_array.item())
                            else:
                                token_id = int(token_array)
                    
                    # EOS 토큰 체크
                    if eos_token_id is not None and token_id == eos_token_id:
                        break
                    
                    # 토큰을 누적 리스트에 추가
                    accumulated_tokens.append(token_id)
                    
                    # 누적된 토큰들을 디코딩 (멀티바이트 문자 올바른 처리)
                    try:
                        # 생성된 토큰만 디코딩 (프롬프트 제외, 스페셜 토큰 제거)
                        current_full_text = tokenizer.decode(accumulated_tokens, skip_special_tokens=True)
                        
                        # 스페셜 토큰 패턴 제거 (혹시 모를 경우 대비)
                        current_full_text = re.sub(r'<\|[^>]*\|>', '', current_full_text)
                        
                        # 이전 전체 텍스트와 비교하여 새로운 부분만 추출
                        if previous_full_text:
                            if current_full_text.startswith(previous_full_text):
                                token_text = current_full_text[len(previous_full_text):]
                            else:
                                # 시작 부분이 다르면 빈 문자열 (누적 방지)
                                token_text = ""
                        else:
                            # 첫 토큰인 경우
                            token_text = current_full_text
                        
                        # 추출된 텍스트에서도 스페셜 토큰 제거
                        if token_text:
                            token_text = re.sub(r'<\|[^>]*\|>', '', token_text)
                        
                        # UTF-8 인코딩 보장
                        if isinstance(token_text, bytes):
                            token_text = token_text.decode('utf-8', errors='replace')
                        
                        # previous_full_text 업데이트 (다음 비교를 위해)
                        if token_text:
                            previous_full_text = current_full_text
                    except Exception as e:
                        # 디코딩 실패 시 빈 문자열 사용
                        token_text = ""
                        if not previous_full_text:
                            previous_full_text = ""
                    
                    # SSE 형식으로 전송 (ensure_ascii=False로 한글 등 유니코드 문자 보존)
                    if token_text:  # 빈 문자열이 아닐 때만 전송
                        data = json.dumps({"content": token_text}, ensure_ascii=False)
                        yield f"data: {data}\n\n"
                    
                    token_count += 1
                    tokens_generated += 1
                    tokens_generated_total += 1
                    last_token_time = time.time()
                    
                    # 메트릭 업데이트 (5토큰마다)
                    if token_count % 5 == 0:
                        broadcast_metrics()
                    
                    # 토큰 생성 로그
                    if token_count % 10 == 0:  # 10토큰마다 로그
                        broadcast_log(f"Generated {token_count} tokens...")
                
                # 완료 신호
                yield f"data: {json.dumps({'stop': True})}\n\n"
                broadcast_log(f"Generation completed: {token_count} tokens")
                
            except Exception as e:
                import traceback
                error_msg = f"Generation failed: {str(e)}"
                broadcast_log(error_msg)
                broadcast_log(traceback.format_exc())
                yield f"data: {json.dumps({'error': error_msg})}\n\n"
            finally:
                processing = False
                tokens_generated = 0
                generation_start_time = None
                broadcast_metrics()
        
        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            }
        )
        
    except Exception as e:
        processing = False
        broadcast_metrics()
        raise HTTPException(status_code=500, detail=str(e))

# Chat WebSocket endpoint
@app.websocket("/chat/ws")
async def chat_websocket(websocket: WebSocket):
    """WebSocket으로 채팅 요청 처리 (실시간 스트리밍)"""
    global processing
    
    await websocket.accept()
    
    if not ready:
        await websocket.send_json({"type": "error", "message": "Model is loading..."})
        await websocket.close()
        return
    
    if processing:
        await websocket.send_json({"type": "error", "message": "Server is busy"})
        await websocket.close()
        return
    
    try:
        # 요청 수신
        data = await websocket.receive_json()
        prompt = data.get("prompt", "")
        max_tokens = data.get("max_tokens", data.get("n_predict", 512))
        temperature = data.get("temperature", 0.7)
        top_p = data.get("top_p", 0.95)
        min_p = data.get("min_p", 0.0)
        repeat_penalty = data.get("repeat_penalty", data.get("repeat_penalty", 1.1))
        repeat_last_n = data.get("repeat_last_n", data.get("repetition_context_size", 64))
        
        if not prompt:
            await websocket.send_json({"type": "error", "message": "Prompt is required"})
            await websocket.close()
            return
        
        processing = True
        broadcast_metrics()
        
        try:
            # 채팅 템플릿 적용
            try:
                messages = [{"role": "user", "content": prompt}]
                prompt_formatted = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            except:
                prompt_formatted = prompt
            
            # 프롬프트 토큰화
            prompt_tokens = tokenizer.encode(prompt_formatted)
            prompt_tokens_list = prompt_tokens.tolist() if hasattr(prompt_tokens, 'tolist') else list(prompt_tokens)
            prompt_array = mx.array(prompt_tokens)
            
            # 샘플러 생성 (mlx_lm 0.29+ 버전용)
            sampler = make_sampler(
                temp=temperature,
                top_p=top_p,
                min_p=min_p
            )
            
            # Repetition penalty를 logits processor로 생성
            logits_processors = []
            if repeat_penalty != 1.0:
                logits_processors.append(make_repetition_penalty(penalty=repeat_penalty, context_size=repeat_last_n))
            
            # generate_step으로 토큰 단위 생성
            token_count = 0
            eos_token_id = getattr(tokenizer, 'eos_token_id', None)
            
            # 토큰 누적을 위한 리스트 (생성된 토큰만 포함)
            accumulated_tokens = []
            previous_full_text = ""
            
            step_generator = generate_step(
                prompt_array,
                model,
                sampler=sampler,
                logits_processors=logits_processors if logits_processors else None,
                max_tokens=max_tokens
            )
            
            # 토큰 단위로 생성 및 WebSocket으로 실시간 전송
            while token_count < max_tokens:
                try:
                    token_array, logits = next(step_generator)
                except StopIteration:
                    break
                
                # 토큰 ID 추출
                if isinstance(token_array, mx.array):
                    token_id = int(token_array.item())
                elif isinstance(token_array, (int, float)):
                    token_id = int(token_array)
                else:
                    try:
                        token_id = int(token_array)
                    except:
                        if hasattr(token_array, 'item'):
                            token_id = int(token_array.item())
                        else:
                            token_id = int(token_array)
                
                # EOS 토큰 체크
                if eos_token_id is not None and token_id == eos_token_id:
                    break
                
                # 토큰을 누적 리스트에 추가
                accumulated_tokens.append(token_id)
                
                # 누적된 토큰들을 디코딩 (멀티바이트 문자 올바른 처리)
                try:
                    # 생성된 토큰만 디코딩 (프롬프트 제외, 스페셜 토큰 제거)
                    current_full_text = tokenizer.decode(accumulated_tokens, skip_special_tokens=True)
                    
                    # 스페셜 토큰 패턴 제거 (혹시 모를 경우 대비)
                    current_full_text = re.sub(r'<\|[^>]*\|>', '', current_full_text)
                    
                    # 이전 전체 텍스트와 비교하여 새로운 부분만 추출
                    if previous_full_text:
                        if current_full_text.startswith(previous_full_text):
                            token_text = current_full_text[len(previous_full_text):]
                        else:
                            # 시작 부분이 다르면 빈 문자열 (누적 방지)
                            token_text = ""
                    else:
                        # 첫 토큰인 경우
                        token_text = current_full_text
                    
                    # 추출된 텍스트에서도 스페셜 토큰 제거
                    if token_text:
                        token_text = re.sub(r'<\|[^>]*\|>', '', token_text)
                    
                    # UTF-8 인코딩 보장
                    if isinstance(token_text, bytes):
                        token_text = token_text.decode('utf-8', errors='replace')
                    
                    # previous_full_text 업데이트 (다음 비교를 위해)
                    if token_text:
                        previous_full_text = current_full_text
                except Exception as e:
                    # 디코딩 실패 시 빈 문자열 사용
                    token_text = ""
                    if not previous_full_text:
                        previous_full_text = ""
                
                # WebSocket으로 실시간 전송 (ensure_ascii=False로 한글 등 유니코드 문자 보존)
                if token_text:  # 빈 문자열이 아닐 때만 전송
                    await websocket.send_json({"type": "token", "content": token_text})
                
                token_count += 1
                
                # 메트릭 업데이트 (5토큰마다)
                if token_count % 5 == 0:
                    broadcast_metrics()
                
                # EOS 토큰 체크
                if eos_token_id is not None and token_id == eos_token_id:
                    break
                
                # 토큰 생성 로그
                if token_count % 10 == 0:
                    broadcast_log(f"Generated {token_count} tokens...")
            
            # 완료 신호
            await websocket.send_json({"type": "done", "stop": True})
            broadcast_log(f"Generation completed: {token_count} tokens")
            
        except Exception as e:
            import traceback
            error_msg = f"Generation failed: {str(e)}"
            broadcast_log(error_msg)
            broadcast_log(traceback.format_exc())
            await websocket.send_json({"type": "error", "message": error_msg})
        finally:
            global tokens_generated, generation_start_time
            processing = False
            tokens_generated = 0
            generation_start_time = None
            broadcast_metrics()
            
    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"[ERROR] Chat WebSocket error: {e}", flush=True)
        processing = False
        broadcast_metrics()
    finally:
        try:
            await websocket.close()
        except:
            pass

# Completion endpoint (llama.cpp 호환)
@app.post("/completion")
async def completion(request: Request):
    """Completion 요청 처리 (SSE 스트리밍, llama.cpp 호환)"""
    global processing
    
    if not ready:
        raise HTTPException(status_code=503, detail="Model is loading...")
    
    if processing:
        raise HTTPException(status_code=503, detail="Server is busy")
    
    try:
        body = await request.json()
        prompt = body.get("prompt", "")
        stream = body.get("stream", True)
        n_predict = body.get("n_predict", body.get("max_tokens", 512))
        temperature = body.get("temperature", 0.7)
        top_p = body.get("top_p", 0.95)
        top_k = body.get("top_k", 40)  # MLX는 top_k를 직접 지원하지 않지만 무시
        min_p = body.get("min_p", 0.0)
        repeat_penalty = body.get("repeat_penalty", 1.1)
        repeat_last_n = body.get("repeat_last_n", body.get("repetition_context_size", 64))
        stop = body.get("stop", [])
        
        if not prompt:
            raise HTTPException(status_code=400, detail="Prompt is required")
        
        if not stream:
            # 비스트리밍 응답 (구현 필요)
            raise HTTPException(status_code=501, detail="Non-streaming completion not implemented")
        
        processing = True
        broadcast_metrics()
        
        async def generate():
            try:
                # 프롬프트를 그대로 사용 (이미 포맷팅되어 있을 수 있음)
                prompt_formatted = prompt
                
                # 프롬프트 토큰화
                prompt_tokens = tokenizer.encode(prompt_formatted)
                prompt_tokens_list = prompt_tokens.tolist() if hasattr(prompt_tokens, 'tolist') else list(prompt_tokens)
                prompt_array = mx.array(prompt_tokens)
                
                # 샘플러 생성 (mlx_lm 0.29+ 버전용)
                sampler = make_sampler(
                    temp=temperature,
                    top_p=top_p,
                    min_p=min_p
                )
                
                # Repetition penalty를 logits processor로 생성
                logits_processors = []
                if repeat_penalty != 1.0:
                    logits_processors.append(make_repetition_penalty(penalty=repeat_penalty, context_size=repeat_last_n))
                
                # generate_step으로 토큰 단위 생성
                global tokens_generated, tokens_generated_total, generation_start_time, last_token_time
                token_count = 0
                generation_start_time = time.time()
                tokens_generated = 0
                eos_token_id = getattr(tokenizer, 'eos_token_id', None)
                stop_tokens = []
                if stop:
                    try:
                        stop_tokens = [tokenizer.encode(s)[0] for s in stop if s]
                    except:
                        pass
                
                # 토큰 누적을 위한 리스트 (생성된 토큰만 포함)
                accumulated_tokens = []
                previous_full_text = ""
                
                step_generator = generate_step(
                    prompt_array,
                    model,
                    sampler=sampler,
                    logits_processors=logits_processors if logits_processors else None,
                    max_tokens=n_predict if n_predict > 0 else 256
                )
                
                # 토큰 단위로 생성 및 SSE 스트리밍
                while token_count < n_predict or n_predict < 0:
                    try:
                        token_array, logits = next(step_generator)
                    except StopIteration:
                        break
                    
                    # 토큰 ID 추출
                    if isinstance(token_array, mx.array):
                        token_id = int(token_array.item())
                    elif isinstance(token_array, (int, float)):
                        token_id = int(token_array)
                    else:
                        try:
                            token_id = int(token_array)
                        except:
                            if hasattr(token_array, 'item'):
                                token_id = int(token_array.item())
                            else:
                                token_id = int(token_array)
                    
                    # Stop 토큰 체크
                    if stop_tokens and token_id in stop_tokens:
                        yield f"data: {json.dumps({'stop': True, 'stop_reason': 'stop'})}\n\n"
                        break
                    
                    # EOS 토큰 체크
                    if eos_token_id is not None and token_id == eos_token_id:
                        yield f"data: {json.dumps({'stop': True, 'stop_reason': 'eos'})}\n\n"
                        break
                    
                    # 토큰을 누적 리스트에 추가
                    accumulated_tokens.append(token_id)
                    
                    # 누적된 토큰들을 디코딩 (멀티바이트 문자 올바른 처리)
                    try:
                        # 생성된 토큰만 디코딩 (프롬프트 제외, 스페셜 토큰 제거)
                        current_full_text = tokenizer.decode(accumulated_tokens, skip_special_tokens=True)
                        
                        # 스페셜 토큰 패턴 제거 (혹시 모를 경우 대비)
                        current_full_text = re.sub(r'<\|[^>]*\|>', '', current_full_text)
                        
                        # 이전 전체 텍스트와 비교하여 새로운 부분만 추출
                        if previous_full_text:
                            if current_full_text.startswith(previous_full_text):
                                token_text = current_full_text[len(previous_full_text):]
                            else:
                                # 시작 부분이 다르면 빈 문자열 (누적 방지)
                                token_text = ""
                        else:
                            # 첫 토큰인 경우
                            token_text = current_full_text
                        
                        # 추출된 텍스트에서도 스페셜 토큰 제거
                        if token_text:
                            token_text = re.sub(r'<\|[^>]*\|>', '', token_text)
                        
                        # UTF-8 인코딩 보장
                        if isinstance(token_text, bytes):
                            token_text = token_text.decode('utf-8', errors='replace')
                        
                        # previous_full_text 업데이트 (다음 비교를 위해)
                        if token_text:
                            previous_full_text = current_full_text
                    except Exception as e:
                        # 디코딩 실패 시 빈 문자열 사용
                        token_text = ""
                        if not previous_full_text:
                            previous_full_text = ""
                    
                    # llama.cpp 형식으로 SSE 전송 (ensure_ascii=False로 한글 등 유니코드 문자 보존)
                    if token_text:  # 빈 문자열이 아닐 때만 전송
                        data = json.dumps({"content": token_text}, ensure_ascii=False)
                        yield f"data: {data}\n\n"
                    
                    token_count += 1
                    tokens_generated += 1
                    tokens_generated_total += 1
                    last_token_time = time.time()
                    
                    # 메트릭 업데이트 (5토큰마다)
                    if token_count % 5 == 0:
                        broadcast_metrics()
                    
                    # 토큰 생성 로그
                    if token_count % 10 == 0:
                        broadcast_log(f"Generated {token_count} tokens...")
                
                # 완료 신호
                yield f"data: {json.dumps({'stop': True})}\n\n"
                broadcast_log(f"Generation completed: {token_count} tokens")
                
            except Exception as e:
                import traceback
                error_msg = f"Generation failed: {str(e)}"
                broadcast_log(error_msg)
                broadcast_log(traceback.format_exc())
                yield f"data: {json.dumps({'error': error_msg})}\n\n"
            finally:
                global processing
                processing = False
                broadcast_metrics()
        
        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            }
        )
        
    except Exception as e:
        processing = False
        broadcast_metrics()
        raise HTTPException(status_code=500, detail=str(e))

# Tokenize endpoint
@app.post("/tokenize")
async def tokenize(request: Request):
    """토큰화 요청 처리"""
    body = await request.json()
    content = body.get("content", "")
    with_pieces = body.get("with_pieces", False)
    add_special = body.get("add_special", True)
    parse_special = body.get("parse_special", True)
    
    if not content:
        raise HTTPException(status_code=400, detail="Content is required")
    
    if not ready:
        raise HTTPException(status_code=503, detail="Model is loading...")
    
    try:
        # 토큰화 (스페셜 토큰 포함)
        # mlx_lm tokenizer는 add_special_tokens 파라미터를 지원하지 않을 수 있으므로
        # 직접 encode를 호출하고, 필요시 스페셜 토큰을 수동으로 추가
        try:
            tokens = tokenizer.encode(content, add_special_tokens=add_special)
        except TypeError:
            # add_special_tokens 파라미터를 지원하지 않는 경우
            tokens = tokenizer.encode(content)
        
        token_list = tokens.tolist() if hasattr(tokens, 'tolist') else list(tokens)
        
        if with_pieces:
            # 각 토큰의 piece(텍스트) 정보 포함
            token_data = []
            # 전체 시퀀스를 먼저 디코딩하여 올바른 UTF-8 처리를 보장
            try:
                full_decoded = tokenizer.decode(token_list, skip_special_tokens=False)
            except Exception:
                full_decoded = ""
            
            # 각 토큰을 개별적으로 디코딩 시도
            for idx, token_id in enumerate(token_list):
                try:
                    # 개별 토큰 디코딩 (스페셜 토큰 포함)
                    piece = tokenizer.decode([token_id], skip_special_tokens=False)
                    
                    # UTF-8 인코딩 보장 (재인코딩 제거 - 이미 올바른 UTF-8 문자열)
                    if isinstance(piece, bytes):
                        piece = piece.decode('utf-8', errors='replace')
                    # str 타입이면 그대로 사용 (불필요한 재인코딩 제거)
                    
                    token_data.append({
                        "id": token_id,
                        "piece": piece
                    })
                except Exception as e:
                    # 개별 토큰 디코딩 실패 시, 전체 시퀀스에서 해당 위치 추출 시도
                    try:
                        # 이전 토큰까지 디코딩
                        prev_tokens = token_list[:idx]
                        # 현재 토큰 포함 디코딩
                        curr_tokens = token_list[:idx+1]
                        if prev_tokens:
                            prev_text = tokenizer.decode(prev_tokens, skip_special_tokens=False)
                            curr_text = tokenizer.decode(curr_tokens, skip_special_tokens=False)
                            piece = curr_text[len(prev_text):]
                        else:
                            piece = tokenizer.decode([token_id], skip_special_tokens=False)
                        
                        # UTF-8 인코딩 보장
                        if isinstance(piece, bytes):
                            piece = piece.decode('utf-8', errors='replace')
                        elif isinstance(piece, str):
                            piece = piece.encode('utf-8', errors='replace').decode('utf-8', errors='replace')
                        
                        token_data.append({
                            "id": token_id,
                            "piece": piece
                        })
                    except Exception:
                        # 모든 디코딩 실패 시 토큰 ID만 포함
                        token_data.append({
                            "id": token_id,
                            "piece": f"<token_{token_id}>"
                        })
            
            return {
                "tokens": token_data,
                "count": len(token_list)
            }
        else:
            return {
                "tokens": token_list,
                "count": len(token_list)
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    broadcast_log(f"Starting MLX Python HTTP server on port {PORT}...")
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=PORT, 
        log_level="info",
        ws="websockets",  # websockets 라이브러리 사용 (명시적)
        limit_concurrency=100,  # 동시 연결 수 제한
        limit_max_requests=10000,  # 최대 요청 수
        timeout_keep_alive=30,  # Keep-alive 타임아웃
    )

