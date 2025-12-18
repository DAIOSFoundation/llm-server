#!/usr/bin/env python3
"""
MLX 추론 엔진 (Python mlx_lm 기반)
Node.js 서버와 IPC로 통신하여 모델 추론을 수행합니다.
"""
import sys
import json
import os
from pathlib import Path

try:
    import mlx.core as mx
    from mlx_lm import load, generate, stream_generate
    from mlx_lm.utils import generate_step
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
        print(json.dumps({"status": "loading", "message": "Loading model..."}), flush=True)
        
        # 모델 경로 확인
        if not os.path.exists(MODEL_PATH):
            print(json.dumps({"status": "error", "message": f"Model path not found: {MODEL_PATH}"}), flush=True)
            sys.exit(1)
        
        # 모델과 토크나이저 로드
        model, tokenizer = load(MODEL_PATH)
        
        print(json.dumps({"status": "ready", "message": "Model loaded successfully"}), flush=True)
    except Exception as e:
        print(json.dumps({"status": "error", "message": f"Model loading failed: {str(e)}"}), flush=True)
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
            # mlx_lm의 stream_generate는 제너레이터를 반환하지만, 실제로는 전체 텍스트를 한 번에 반환할 수 있음
            # 따라서 generate_step을 직접 사용하여 토큰 단위 스트리밍 구현
            try:
                from mlx_lm.utils import generate_step
                
                # 프롬프트를 토큰화
                prompt_tokens = tokenizer.encode(prompt_formatted)
                
                # 생성 파라미터 설정 (mlx_lm의 generate_step이 지원하는 파라미터만 사용)
                generate_kwargs = {
                    "temp": temperature,
                    "top_p": top_p,
                    "min_p": min_p,
                    "repetition_penalty": repeat_penalty,
                    "repetition_context_size": repeat_last_n
                }
                
                # 토큰 단위로 생성 및 스트리밍
                # mlx_lm의 모델은 forward pass를 통해 logits를 생성
                tokens = prompt_tokens.copy()
                
                for _ in range(max_tokens):
                    # 현재 토큰 시퀀스를 mx.array로 변환
                    tokens_array = mx.array([tokens])
                    
                    # 모델 forward pass (전체 시퀀스 처리)
                    logits = model(tokens_array)
                    
                    # 마지막 토큰의 logits만 사용
                    next_token_logits = logits[0, -1, :]
                    
                    # generate_step으로 다음 토큰 생성
                    next_token = generate_step(
                        next_token_logits,
                        **generate_kwargs
                    )
                    
                    # 토큰 ID 추출
                    token_id = int(next_token.item())
                    
                    # 토큰을 디코딩하여 전송
                    token_text = tokenizer.decode([token_id])
                    
                    # 생성된 토큰을 시퀀스에 추가
                    tokens.append(token_id)
                    
                    # 즉시 Node.js에 전송 (flush=True로 즉시 출력)
                    print(json.dumps({"status": "token", "content": token_text}), flush=True)
                    
                    # EOS 토큰 체크
                    eos_token_id = getattr(tokenizer, 'eos_token_id', None)
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

