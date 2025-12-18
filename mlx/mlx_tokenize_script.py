#!/usr/bin/env python3
import json
import sys
import argparse
from mlx_lm import load

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--text', required=True)
    
    args = parser.parse_args()
    
    try:
        model, tokenizer = load(args.model)
        tokens = tokenizer.encode(args.text)
        print(json.dumps({'tokens': tokens}), flush=True)
    except Exception as e:
        print(json.dumps({'error': str(e)}), flush=True)
        sys.exit(1)

if __name__ == '__main__':
    main()
