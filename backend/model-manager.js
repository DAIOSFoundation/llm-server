import { readFileSync, writeFileSync, existsSync } from 'fs';
import { randomUUID } from 'crypto';

export class ModelManager {
  constructor(modelsPath) {
    this.modelsPath = modelsPath;
    this.models = this.loadModels();
  }

  loadModels() {
    if (existsSync(this.modelsPath)) {
      try {
        const data = readFileSync(this.modelsPath, 'utf-8');
        return JSON.parse(data);
      } catch (error) {
        console.error('모델 로드 실패:', error);
        return [];
      }
    }
    return [];
  }

  saveModels() {
    try {
      writeFileSync(this.modelsPath, JSON.stringify(this.models, null, 2), 'utf-8');
      return true;
    } catch (error) {
      console.error('모델 저장 실패:', error);
      return false;
    }
  }

  getModels() {
    return this.models;
  }

  getModel(id) {
    return this.models.find(m => m.id === id);
  }

  addModel(modelData) {
    const model = {
      id: randomUUID(),
      name: modelData.name,
      path: modelData.path,
      quantization: modelData.quantization || 'Q4_0',
      quantizationLevel: modelData.quantizationLevel,
      device: modelData.device || 'CPU',
      gpuBackend: modelData.gpuBackend || null,
      inferenceConfig: modelData.inferenceConfig || this.getDefaultInferenceConfig(),
      createdAt: new Date().toISOString(),
      updatedAt: new Date().toISOString()
    };

    this.models.push(model);
    this.saveModels();
    return model;
  }

  updateModel(id, updates) {
    const model = this.models.find(m => m.id === id);
    if (!model) {
      return null;
    }

    Object.assign(model, updates, {
      updatedAt: new Date().toISOString()
    });

    this.saveModels();
    return model;
  }

  deleteModel(id) {
    const index = this.models.findIndex(m => m.id === id);
    if (index === -1) {
      return false;
    }

    this.models.splice(index, 1);
    this.saveModels();
    return true;
  }

  getDefaultInferenceConfig() {
    return {
      contextSize: 2048,
      batchSize: 512,
      maxTokens: 512,
      temperature: 0.7,
      topK: 40,
      topP: 0.9,
      repeatPenalty: 1.1,
      repeatLastN: 64,
      threads: 4,
      threadsBatch: 4,
      seed: -1,
      stopSequences: []
    };
  }
}

