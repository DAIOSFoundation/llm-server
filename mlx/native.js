const path = require('path');
const fs = require('fs');

let mlxServerModule = null;

try {
  // mlx 디렉토리의 build 폴더에서 모듈 로드
  mlxServerModule = require('./build/Release/mlx_server.node');
} catch (error) {
  console.warn('[MLX Server] Native module not found, using fallback:', error.message);
}

class MlxServerNative {
  constructor(modelDir) {
    if (!mlxServerModule) {
      throw new Error('MLX server native module not available');
    }
    this.inference = new mlxServerModule.MlxInference(modelDir);
  }

  async generateStream(prompt, options, onToken, onError, onComplete) {
    return new Promise((resolve, reject) => {
      const tokenCallback = (data) => {
        if (onToken) {
          onToken(data);
        }
        if (data.stop) {
          if (onComplete) {
            onComplete();
          }
          resolve();
        }
      };

      const errorCallback = (error) => {
        if (onError) {
          onError(error);
        }
        reject(new Error(error));
      };

      const completeCallback = () => {
        if (onComplete) {
          onComplete();
        }
        resolve();
      };

      // 옵션을 C++ 모듈이 기대하는 형식으로 변환
      const cppOptions = {
        temperature: options.temperature,
        topK: options.topK,
        topP: options.topP,
        minP: options.minP,
        typicalP: options.typicalP,
        tfsZ: options.tfsZ,
        repeatPenalty: options.repeatPenalty,
        repeatLastN: options.repeatLastN,
        presencePenalty: options.presencePenalty,
        frequencyPenalty: options.frequencyPenalty,
        dryMultiplier: options.dryMultiplier,
        dryBase: options.dryBase,
        dryAllowedLength: options.dryAllowedLength,
        dryPenaltyLastN: options.dryPenaltyLastN,
        mirostat: options.mirostat,
        mirostatTau: options.mirostatTau,
        mirostatEta: options.mirostatEta,
        maxTokens: options.maxTokens,
        stop: options.stop || [],
        seed: options.seed,
      };

      this.inference.generateStream(
        prompt,
        cppOptions,
        tokenCallback,
        errorCallback,
        completeCallback
      );
    });
  }

  createInferenceScript() {
    if (this.inference.createInferenceScript) {
      this.inference.createInferenceScript();
    }
  }
}

module.exports = MlxServerNative;

