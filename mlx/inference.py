#!/usr/bin/env python3
import json
import sys
import argparse
import mlx.core as mx
from mlx_lm import load, generate_stepwise

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--prompt', required=True)
    parser.add_argument('--options', required=True)
    
    args = parser.parse_args()
    
    try:
        options = json.loads(args.options)
        
        # 모델 로드
        print(json.dumps({'status': 'loading_model'}), flush=True, file=sys.stderr)
        model, tokenizer = load(args.model)
        print(json.dumps({'status': 'model_loaded'}), flush=True, file=sys.stderr)
        
        # 생성 파라미터 설정 (MLX가 지원하는 파라미터만 사용)
        # MLX_LM은 일부 고급 샘플링 옵션을 지원하지 않을 수 있음
        generate_kwargs = {
            'temp': options.get('temperature', 0.7),
            'top_k': options.get('top_k', 40),
            'top_p': options.get('top_p', 0.95),
            'min_p': options.get('min_p', 0.05),
            'repetition_penalty': options.get('repeat_penalty', 1.2),
            'repetition_context_size': options.get('repeat_last_n', 128),
            'max_tokens': options.get('max_tokens', 600),
        }
        
        # 지원되지 않는 옵션에 대한 경고 (로그만 출력)
        unsupported = []
        if options.get('typical_p', 1.0) != 1.0:
            unsupported.append('typical_p')
        if options.get('tfs_z', 1.0) != 1.0:
            unsupported.append('tfs_z')
        if options.get('presence_penalty', 0.0) != 0.0:
            unsupported.append('presence_penalty')
        if options.get('frequency_penalty', 0.0) != 0.0:
            unsupported.append('frequency_penalty')
        if options.get('dry_multiplier', 0.0) != 0.0:
            unsupported.append('dry_multiplier')
        if options.get('mirostat', 0) != 0:
            unsupported.append('mirostat')
        
        if unsupported:
            print(json.dumps({'warning': f'Unsupported options (using defaults): {", ".join(unsupported)}'}), flush=True, file=sys.stderr)
        
        # Stop 토큰 설정
        stop_tokens = options.get('stop', [])
        
        # 스트리밍 생성
        print(json.dumps({'status': 'generating'}), flush=True, file=sys.stderr)
        generated_text = ''
        for token, prob in generate_stepwise(
            model, tokenizer, 
            prompt=args.prompt,
            **generate_kwargs
        ):
            if token:
                token_str = tokenizer.decode([token]) if isinstance(token, int) else str(token)
                generated_text += token_str
                print(json.dumps({'token': token_str}), flush=True)
                
                # Stop 토큰 체크
                if stop_tokens:
                    for stop_token in stop_tokens:
                        if stop_token in generated_text:
                            print(json.dumps({'stop': True}), flush=True)
                            return
        
        print(json.dumps({'stop': True}), flush=True)
        
    except Exception as e:
        import traceback
        error_msg = f"{str(e)}\n{traceback.format_exc()}"
        print(json.dumps({'error': error_msg}), flush=True)
        sys.exit(1)

if __name__ == '__main__':
    main()
