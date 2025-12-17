import mlx.core as mx
from mlx_lm import load, stream_generate

def main():
    # 1. 변환된 모델 경로 (쉘 스크립트에서 설정한 OUTPUT_DIR)
    model_path = "./deepseek-16b-mlx-q4"
    
    print(f"Loading model from {model_path}...")
    
    # 모델과 토크나이저 로드
    # tokenizer_config에는 eos_token, chat_template 등이 포함됨
    model, tokenizer = load(model_path)
    
    print("Model loaded. Start chatting! (Type 'quit' to exit)")
    print("-" * 50)

    # 대화 히스토리 관리
    messages = []

    while True:
        user_input = input("\nUser: ")
        if user_input.lower() in ["quit", "exit"]:
            break

        # 메시지 추가
        messages.append({"role": "user", "content": user_input})

        # 채팅 템플릿 적용 (DeepSeek 포맷 자동 적용)
        prompt = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )

        print("Assistant: ", end="", flush=True)

        # 스트리밍 생성
        response_content = ""
        for response in stream_generate(model, tokenizer, prompt, max_tokens=1024, temp=0.7):
            print(response, end="", flush=True)
            response_content += response

        print() # 줄바꿈
        
        # 어시스턴트 응답 저장
        messages.append({"role": "assistant", "content": response_content})

if __name__ == "__main__":
    main()

