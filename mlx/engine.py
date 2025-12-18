#!/usr/bin/env python3
"""
MLX 추론 엔진 (Python mlx_lm 기반)
Node.js 서버와 IPC로 통신하여 모델 추론을 수행합니다.
"""
import sys
import json
import os
import signal
from pathlib import Path

# 타임아웃 방지를 위한 시그널 핸들러
def timeout_handler(signum, frame):
    print(json.dumps({"status": "error", "message": "Model loading timeout"}), flush=True)
    sys.exit(1)

# 시그널 핸들러 설정 (선택적, 필요시 활성화)
# signal.signal(signal.SIGALRM, timeout_handler)

try:
    import mlx.core as mx
    from mlx_lm import load, generate, stream_generate
    from mlx_lm.utils import generate_step
    # generate_stepwise는 mlx_lm에서 직접 import 가능한지 확인
    try:
        from mlx_lm import generate_stepwise
    except ImportError:
        try:
            from mlx_lm.generate_stepwise import generate_stepwise
        except ImportError:
            generate_stepwise = None
except ImportError as e:
    print(json.dumps({"status": "error", "message": f"MLX 라이브러리 미설치: {e}. 'pip install mlx-lm' 실행 필요"}), flush=True)
    sys.exit(1)

# 모델 경로 (환경변수 또는 기본값)
MODEL_PATH = os.getenv("MLX_MODEL_PATH", "./models/deepseek-moe-16b-chat-mlx-q4_0")

def main():
    model = None
    tokenizer = None
    
    # 1. 모델 로드 (mlx_lm이 샤딩, 구조, 가중치 병합을 자동으로 처리함)
    try:
        # Node.js로 상태 전송
        print(json.dumps({"status": "loading", "message": f"Loading model from {MODEL_PATH}..."}), flush=True)
        
        # 모델 경로 확인
        if not os.path.exists(MODEL_PATH):
            error_msg = f"Model path not found: {MODEL_PATH}"
            print(json.dumps({"status": "error", "message": error_msg}), flush=True)
            print(f"[ERROR] {error_msg}", file=sys.stderr, flush=True)
            sys.exit(1)
        
        print(json.dumps({"status": "loading", "message": "Starting model load (this may take a while for large models)..."}), flush=True)
        
        # 모델과 토크나이저 로드 (이 과정이 오래 걸릴 수 있음)
        # Metal GPU를 사용하여 로드 (명시적으로 GPU 사용 보장)
        import time
        
        # GPU 사용을 명시적으로 보장 (MLX는 기본적으로 Metal GPU 사용)
        # CPU 폴백을 방지하기 위해 환경 변수 확인
        if 'MLX_DEFAULT_DEVICE' in os.environ and os.environ['MLX_DEFAULT_DEVICE'] == 'cpu':
            del os.environ['MLX_DEFAULT_DEVICE']
            print(json.dumps({"status": "loading", "message": "Ensuring GPU (Metal) usage..."}), flush=True)
        
        start_time = time.time()
        
        # Metal GPU를 사용하여 모델 로드
        model, tokenizer = load(MODEL_PATH)
        
        load_time = time.time() - start_time
        
        # 로드된 모델의 디바이스 확인
        try:
            # 모델 파라미터의 디바이스 확인
            sample_param = next(iter(model.parameters().values())) if hasattr(model, 'parameters') else None
            device_info = "GPU (Metal)" if sample_param is None or str(sample_param.device) != 'cpu' else "CPU"
            print(json.dumps({"status": "loading", "message": f"Model loaded in {load_time:.2f} seconds on {device_info}"}), flush=True)
        except:
            print(json.dumps({"status": "loading", "message": f"Model loaded in {load_time:.2f} seconds"}), flush=True)
        
        print(json.dumps({"status": "ready", "message": "Model loaded successfully"}), flush=True)
        print("[INFO] Model loaded successfully", file=sys.stderr, flush=True)
    except KeyboardInterrupt:
        print(json.dumps({"status": "error", "message": "Model loading interrupted"}), flush=True)
        sys.exit(1)
    except Exception as e:
        import traceback
        error_msg = f"Model loading failed: {str(e)}"
        print(json.dumps({"status": "error", "message": error_msg}), flush=True)
        print(f"[ERROR] {error_msg}", file=sys.stderr, flush=True)
        print(f"[ERROR] Traceback: {traceback.format_exc()}", file=sys.stderr, flush=True)
        sys.exit(1)

    # 2. 요청 대기 루프 (Stdin)
    for line in sys.stdin:
        if not line.strip():
            continue
        
        try:
            request = json.loads(line.strip())
            prompt = request.get("prompt", "")
            max_tokens = request.get("max_tokens", 512)
            temperature = request.get("temperature", 0.7)
            top_p = request.get("top_p", 0.95)
            min_p = request.get("min_p", 0.0)
            repeat_penalty = request.get("repeat_penalty", 1.1)
            repeat_last_n = request.get("repeat_last_n", 64)
            
            # mlx_lm에서 지원하지 않는 파라미터 확인 및 경고
            unsupported_params = []
            if "top_k" in request and request.get("top_k") is not None:
                unsupported_params.append("top_k")
            if request.get("mirostat") and request.get("mirostat") != 0:
                unsupported_params.append("mirostat")
            if request.get("tfs_z") is not None:
                unsupported_params.append("tfs_z")
            if request.get("typical_p") is not None:
                unsupported_params.append("typical_p")
            if request.get("penalize_nl") is not None:
                unsupported_params.append("penalize_nl")
            if request.get("dry_multiplier") is not None:
                unsupported_params.append("dry_multiplier")
            if request.get("presence_penalty") is not None:
                unsupported_params.append("presence_penalty")
            if request.get("frequency_penalty") is not None:
                unsupported_params.append("frequency_penalty")
            
            if unsupported_params:
                print(json.dumps({
                    "status": "warning",
                    "message": f"MLX does not support these parameters (ignored): {', '.join(unsupported_params)}"
                }), flush=True)
            
            if not prompt:
                print(json.dumps({"status": "error", "message": "Empty prompt"}), flush=True)
                continue

            # DeepSeek 채팅 템플릿 적용
            try:
                messages = [{"role": "user", "content": prompt}]
                prompt_formatted = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            except Exception as e:
                # 템플릿 적용 실패 시 프롬프트 그대로 사용
                prompt_formatted = prompt

            # 3. 생성 및 스트리밍
            # mlx_lm의 generate_step을 사용하여 실제 토큰 단위 스트리밍 구현
            # generate_step은 제너레이터를 반환하며, 각 yield는 (token, logits) 튜플
            # max_tokens 제한을 수동으로 제어하여 무한 루프 방지
            try:
                # 프롬프트를 토큰화
                prompt_tokens = tokenizer.encode(prompt_formatted)
                
                # 생성 파라미터 설정
                generate_kwargs = {
                    "temp": temperature,
                    "top_p": top_p,
                    "min_p": min_p,
                    "repetition_penalty": repeat_penalty,
                    "repetition_context_size": repeat_last_n
                }
                
                # 프롬프트를 mx.array로 변환
                prompt_array = mx.array(prompt_tokens)
                
                # generate_step은 제너레이터를 반환하며, 각 yield는 (token, logits) 튜플
                # 실제 토큰 단위 스트리밍을 제공
                token_count = 0
                eos_token_id = getattr(tokenizer, 'eos_token_id', None)
                
                # generate_step 제너레이터 생성
                step_generator = generate_step(
                    prompt_array,
                    model,
                    **generate_kwargs
                )
                
                # 토큰 단위로 생성 및 스트리밍
                while token_count < max_tokens:
                    try:
                        # 제너레이터에서 다음 토큰 가져오기
                        token_array, logits = next(step_generator)
                    except StopIteration:
                        # 제너레이터가 종료됨
                        break
                    
                    # 토큰 ID 추출
                    # token_array가 mx.array인지 int인지 확인
                    if isinstance(token_array, mx.array):
                        token_id = int(token_array.item())
                    elif isinstance(token_array, (int, float)):
                        token_id = int(token_array)
                    else:
                        # numpy array 등 다른 타입 처리
                        try:
                            token_id = int(token_array)
                        except (TypeError, ValueError):
                            # mx.array로 변환 시도
                            if hasattr(token_array, 'item'):
                                token_id = int(token_array.item())
                            else:
                                token_id = int(token_array)
                    
                    # 토큰을 디코딩하여 전송
                    token_text = tokenizer.decode([token_id])
                    
                    # 즉시 Node.js에 전송 (flush=True로 즉시 출력)
                    print(json.dumps({"status": "token", "content": token_text}), flush=True)
                    
                    token_count += 1
                    
                    # EOS 토큰 체크
                    if eos_token_id is not None and token_id == eos_token_id:
                        break
                
                # 생성 완료
                print(json.dumps({"status": "done"}), flush=True)
                
            except ImportError:
                # generate_step을 사용할 수 없는 경우, stream_generate 사용 (비스트리밍)
                try:
                    # stream_generate는 제너레이터를 반환하지만, 실제로는 전체 텍스트를 반환할 수 있음
                    # 따라서 accumulate 방식으로 처리
                    accumulated_text = ""
                    for response in stream_generate(
                        model,
                        tokenizer,
                        prompt_formatted,
                        max_tokens=max_tokens,
                        temp=temperature,
                        top_p=top_p,
                        min_p=min_p,
                        repetition_penalty=repeat_penalty,
                        repetition_context_size=repeat_last_n
                    ):
                        # response는 문자열일 수 있으므로 처리
                        if isinstance(response, str):
                            # 새로운 부분만 추출
                            new_text = response[len(accumulated_text):]
                            if new_text:
                                accumulated_text = response
                                print(json.dumps({"status": "token", "content": new_text}), flush=True)
                        else:
                            # 토큰 ID인 경우
                            token_text = tokenizer.decode([response]) if isinstance(response, int) else str(response)
                            accumulated_text += token_text
                            print(json.dumps({"status": "token", "content": token_text}), flush=True)
                    
                    # 생성 완료
                    print(json.dumps({"status": "done"}), flush=True)
                except Exception as e:
                    print(json.dumps({"status": "error", "message": f"Generation failed: {str(e)}"}), flush=True)
            except Exception as e:
                print(json.dumps({"status": "error", "message": f"Generation failed: {str(e)}"}), flush=True)

        except json.JSONDecodeError as e:
            print(json.dumps({"status": "error", "message": f"Invalid JSON: {str(e)}"}), flush=True)
        except Exception as e:
            print(json.dumps({"status": "error", "message": f"Request processing failed: {str(e)}"}), flush=True)

if __name__ == "__main__":
    main()

