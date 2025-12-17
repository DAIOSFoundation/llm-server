#!/usr/bin/env python3
import json
import sys
import argparse
from jinja2 import Template

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--template', required=True)
    parser.add_argument('--messages', required=True)
    
    args = parser.parse_args()
    
    try:
        # 템플릿 파일 읽기
        with open(args.template, 'r') as f:
            template_str = f.read()
        
        # 메시지 파싱
        messages = json.loads(args.messages)
        
        # Jinja 템플릿 적용
        template = Template(template_str)
        prompt = template.render(
            messages=messages,
            add_generation_prompt=True,
            bos_token='<|begin_of_text|>',
            eos_token='<|end_of_text|>',
        )
        
        print(json.dumps({'prompt': prompt}), flush=True)
    except Exception as e:
        print(json.dumps({'error': str(e)}), flush=True)
        sys.exit(1)

if __name__ == '__main__':
    main()
