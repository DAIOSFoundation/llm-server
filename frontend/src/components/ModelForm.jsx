import React, { useState, useEffect, useMemo } from 'react';
import { useLanguage } from '../contexts/LanguageContext';
import './ModelForm.css';

// (Fallback) GGUF 모델 경로(파일명)를 기반으로 양자화 정보를 추정하는 헬퍼
// NOTE: 기본은 GGUF 내부 메타데이터(헤더/KV/텐서 타입)를 직접 읽어 표시합니다.
// 다만 파일 읽기 실패/권한 문제 등으로 메타데이터를 읽지 못할 때만 파일명 기반 추정을 사용합니다.
const inferQuantizationFromPath = (modelPath) => {
  if (!modelPath) {
    return { type: '-', detail: '-' };
  }

  const fileName = modelPath.split(/[\\/]/).pop() || '';
  const upper = fileName.toUpperCase();

  // FP16 / F16
  if (upper.includes('FP16') || upper.includes('F16')) {
    return { type: 'FP16', detail: 'FP16' };
  }

  // 8-bit
  if (upper.includes('Q8_0')) {
    return { type: '8-bit', detail: 'Q8_0' };
  }

  // 4-bit 계열 (Q4_0, Q4_1, Q4_K_M, Q4_K_S 등)
  const q4Match = upper.match(/Q4_([A-Z0-9_]+)/);
  if (q4Match) {
    const variant = q4Match[1]; // 예: "0", "K_M", "K_S"
    return {
      type: '4-bit',
      detail: `Q4_${variant}`,
    };
  }

  // 기타 Qx_ 계열 (Q5_0, Q5_1 등)
  const qMatch = upper.match(/Q([0-9])_([A-Z0-9_]+)/);
  if (qMatch) {
    const bits = qMatch[1];
    const variant = qMatch[2];
    return {
      type: `${bits}-bit`,
      detail: `Q${bits}_${variant}`,
    };
  }

  // 알 수 없는 경우
  return { type: 'Unknown', detail: fileName };
};

const buildQuantizationGuide = (language, quantDetail, quantType, ggufInfo) => {
  const detail = (quantDetail || '').toUpperCase();

  const isKo = language === 'ko';

  const base = {
    title: isKo ? '옵션 설명' : 'Option Guide',
    paragraphs: [],
  };

  // GGUF 메타데이터를 읽지 못한 경우(에러) 안내
  if (ggufInfo && ggufInfo.ok === false) {
    base.paragraphs.push(
      isKo
        ? `GGUF 메타데이터를 읽지 못했습니다: ${ggufInfo.error || 'Unknown error'}`
        : `Failed to read GGUF metadata: ${ggufInfo.error || 'Unknown error'}`,
    );
    base.paragraphs.push(
      isKo
        ? '현재 표시는 파일명 기반 추정(Fallback)입니다.'
        : 'Current display is filename-based inference (fallback).',
    );
  }

  if (!detail || detail === '-' || quantType === '-') {
    base.paragraphs = [
      isKo
        ? '모델을 선택하면 GGUF 내부 메타데이터(헤더/KV/텐서 타입)를 읽어 양자화 정보를 표시합니다.'
        : 'When you select a model, we read GGUF metadata (header/KV/tensor types) to display quantization.',
      isKo
        ? '파일 접근이 불가한 경우에만 파일명 기반 추정으로 대체합니다.'
        : 'If the file cannot be read, we fall back to filename-based inference.',
    ];
    return base;
  }

  // 공통 설명
  base.paragraphs.push(
    isKo
      ? '표시는 GGUF 내부 메타데이터의 general.file_type(있다면) 및 텐서 타입 통계를 기반으로 합니다.'
      : 'Displayed values are based on GGUF metadata (general.file_type when present) and tensor type stats.',
  );

  // 메타데이터 기반의 "구체적인 내용" 출력
  if (ggufInfo && ggufInfo.ok) {
    if (ggufInfo.ggufVersion != null) {
      base.paragraphs.push(
        isKo
          ? `GGUF 버전: v${ggufInfo.ggufVersion}`
          : `GGUF version: v${ggufInfo.ggufVersion}`,
      );
    }

    if (ggufInfo.fileTypeName) {
      base.paragraphs.push(
        isKo
          ? `general.file_type: ${ggufInfo.fileTypeName}`
          : `general.file_type: ${ggufInfo.fileTypeName}`,
      );
    }

    // 텐서 타입 통계(상위 5개) 표시
    if (ggufInfo.tensorTypes && typeof ggufInfo.tensorTypes === 'object') {
      const entries = Object.entries(ggufInfo.tensorTypes)
        .sort((a, b) => (b[1] || 0) - (a[1] || 0))
        .slice(0, 5)
        .map(([k, v]) => `${k}: ${v}`);

      if (entries.length > 0) {
        base.paragraphs.push(
          isKo
            ? `텐서 타입 분포(상위 5): ${entries.join(', ')}`
            : `Tensor type stats (top 5): ${entries.join(', ')}`,
        );
      }
    }
  }

  if (ggufInfo && ggufInfo.qkv && (ggufInfo.qkv.q || ggufInfo.qkv.k || ggufInfo.qkv.v)) {
    const q = ggufInfo.qkv.q || '-';
    const k = ggufInfo.qkv.k || '-';
    const v = ggufInfo.qkv.v || '-';
    base.paragraphs.push(
      isKo
        ? `어텐션 Q/K/V 텐서 타입: Q=${q}, K=${k}, V=${v} (모델에 따라 텐서별로 혼합될 수 있습니다).`
        : `Attention Q/K/V tensor types: Q=${q}, K=${k}, V=${v} (models may mix per-tensor types).`,
    );
  }

  // FP16
  if (detail.includes('FP16') || detail.includes('F16')) {
    base.paragraphs.push(
      isKo
        ? 'FP16은 16-bit 부동소수(half) 정밀도 가중치로, 품질이 좋은 대신 파일/메모리 사용량이 큽니다.'
        : 'FP16 uses 16-bit floating-point weights: best quality, larger size/VRAM.',
    );
    return base;
  }

  // 8-bit
  if (detail.includes('Q8_0')) {
    base.paragraphs.push(
      isKo
        ? 'Q8_0은 8-bit 양자화로, 품질 손실이 비교적 적고(대체로 안정적) 4-bit 대비 메모리 사용량은 더 큽니다.'
        : 'Q8_0 is 8-bit quantization: usually stable quality, more VRAM than 4-bit.',
    );
  }

  // Legacy 4-bit (Q4_0, Q4_1)
  if (detail === 'Q4_0' || detail === 'Q4_1') {
    base.paragraphs.push(
      isKo
        ? 'Q4_0/Q4_1의 “0/1”은 비교적 오래된(legacy) 4-bit 양자화 변형을 의미합니다. 용량은 작지만 최신 K-quant 계열보다 품질이 떨어질 수 있습니다.'
        : '“0/1” in Q4_0/Q4_1 refers to legacy 4-bit variants. Small size, but may be worse than newer K-quants.',
    );
  }

  // K-quants
  if (detail.includes('_K_')) {
    base.paragraphs.push(
      isKo
        ? '“K”는 K-quant 계열(개선된 양자화)을 의미하며, 같은 비트수에서 품질/속도 균형이 더 좋은 편입니다.'
        : '“K” indicates the K-quant family (improved quantization), often better quality/throughput balance.',
    );
    if (detail.endsWith('_S')) {
      base.paragraphs.push(
        isKo
          ? '“S”는 보통 더 작은 용량(더 공격적인 압축) 쪽으로, 품질 손실이 더 클 수 있습니다.'
          : '“S” is usually smaller/more aggressive, potentially more quality loss.',
      );
    }
    if (detail.endsWith('_M')) {
      base.paragraphs.push(
        isKo
          ? '“M”은 보통 품질을 더 우선하는 변형으로, 같은 4-bit라도 S보다 품질이 나은 경우가 많습니다.'
          : '“M” usually prioritizes quality (often better than S at the same bit-width).',
      );
    }
  }

  // q/k/v 용어 설명 (사용자 요청)
  base.paragraphs.push(
    isKo
      ? '참고: “q/k/v”는 어텐션의 Query/Key/Value 가중치 텐서를 의미합니다. 일부 GGUF는 텐서별로 서로 다른 양자화를 혼합할 수 있어, 이를 표시하려면 gguf 내부 메타데이터(텐서 타입)를 직접 읽어야 합니다.'
      : 'Note: “q/k/v” refer to attention Query/Key/Value tensors. Some GGUFs can mix quantization per tensor; showing that requires parsing GGUF tensor metadata.',
  );

  return base;
};

const ModelForm = ({ config, onChange }) => {
  const { t, language } = useLanguage();
  const [formData, setFormData] = useState({});
  const [ggufInfo, setGgufInfo] = useState(null);

  useEffect(() => {
    const defaults = {
      name: 'New Model', modelPath: '', accelerator: 'auto', gpuLayers: 0,
      contextSize: 2048, maxTokens: 600, temperature: 0.7, topK: 40, topP: 0.95,
      minP: 0.05, tfsZ: 1.0, typicalP: 1.0, repeatPenalty: 1.15, repeatLastN: 128,
      penalizeNL: true, presencePenalty: 0.0, frequencyPenalty: 0.0,
      dryMultiplier: 0.5, dryBase: 1.75, dryAllowedLength: 3, dryPenaltyLastN: -1,
      mirostatMode: 0, mirostatTau: 5.0, mirostatEta: 0.1,
      showSpecialTokens: false,
    };
    setFormData({ ...defaults, ...config });
  }, [config]);

  // GGUF 메타데이터 읽기 (modelPath 변경 시)
  useEffect(() => {
    let cancelled = false;

    const run = async () => {
      const modelPath = formData.modelPath;
      if (!modelPath || !window.electronAPI || !window.electronAPI.getGgufInfo) {
        setGgufInfo(null);
        return;
      }

      try {
        const info = await window.electronAPI.getGgufInfo(modelPath);
        if (!cancelled) {
          setGgufInfo(info && info.ok ? info : info);
        }
      } catch (_e) {
        if (!cancelled) {
          setGgufInfo({ ok: false, error: 'Failed to read GGUF metadata' });
        }
      }
    };

    run();
    return () => { cancelled = true; };
  }, [formData.modelPath]);

  const handleChange = (e) => {
    const { name, value, type, checked } = e.target;
    const newFormData = {
      ...formData,
      [name]: type === 'checkbox' ? checked : (type === 'number' ? parseFloat(value) : value),
    };
    setFormData(newFormData);
    onChange(newFormData);
  };

  const handleFindModel = async () => {
    if (window.electronAPI) {
      const filePath = await window.electronAPI.selectFile();
      if (filePath) {
        const newFormData = { ...formData, modelPath: filePath };
        setFormData(newFormData);
        onChange(newFormData);
      }
    }
  };
  
  const handleNameChange = (e) => {
    const newFormData = { ...formData, name: e.target.value };
    setFormData(newFormData);
    onChange(newFormData);
  }

  // 양자화 정보: 1) GGUF 메타데이터 기반 (우선) 2) 파일명 추정 (fallback)
  const quantInfo = useMemo(() => {
    if (ggufInfo && ggufInfo.ok) {
      // prefer general.file_type if present
      if (ggufInfo.fileTypeName) {
        const name = String(ggufInfo.fileTypeName);
        const upper = name.toUpperCase();
        if (upper.includes('F16')) return { type: 'FP16', detail: 'FP16' };
        if (upper.includes('BF16')) return { type: 'BF16', detail: 'BF16' };
        const qMatch = upper.match(/Q([0-9])_([A-Z0-9_]+)/);
        if (qMatch) return { type: `${qMatch[1]}-bit`, detail: `Q${qMatch[1]}_${qMatch[2]}` };
        // Fallback: show raw ftype name
        return { type: 'Mixed', detail: name.replace(/^MOSTLY_/, '') };
      }

      // fallback to tensor-type stats if file_type missing
      const types = ggufInfo.tensorTypes || {};
      const entries = Object.entries(types);
      if (entries.length > 0) {
        // pick most common non-F32 for a more meaningful label
        const sorted = entries
          .filter(([k]) => k !== 'F32')
          .sort((a, b) => (b[1] || 0) - (a[1] || 0));
        const top = sorted[0] || entries[0];
        const topType = top ? top[0] : '-';
        const m = String(topType).match(/^Q([0-9])_/);
        if (m) return { type: `${m[1]}-bit`, detail: topType };
        if (topType === 'F16') return { type: 'FP16', detail: 'FP16' };
        if (topType === 'BF16') return { type: 'BF16', detail: 'BF16' };
        return { type: 'Mixed', detail: topType };
      }
    }

    return inferQuantizationFromPath(formData.modelPath);
  }, [formData.modelPath, ggufInfo]);

  const quantGuide = useMemo(
    () => buildQuantizationGuide(language, quantInfo.detail, quantInfo.type, ggufInfo && ggufInfo.ok ? ggufInfo : null),
    [language, quantInfo.detail, quantInfo.type, ggufInfo],
  );

  return (
    <div className="model-form">
      {/* Acceleration Section */}
      <div className="form-section">
        <h3>{t('settings.acceleration')}</h3>
        <div className="form-grid">
          <div className="form-group">
            <label>{t('settings.accelerator')}</label>
            <select name="accelerator" value={formData.accelerator || 'auto'} onChange={handleChange}>
              <option value="auto">{t('accelerator.auto')}</option>
              <option value="mps">{t('accelerator.mps')}</option>
              <option value="cuda">{t('accelerator.cuda')}</option>
              <option value="opencl">{t('accelerator.opencl')}</option>
            </select>
          </div>
          <div className="form-group">
            <label>{t('settings.gpuLayers')}</label>
            <input type="number" name="gpuLayers" value={formData.gpuLayers} onChange={handleChange} min="-1" />
          </div>
        </div>
      </div>
      
      {/* Model Name and Path */}
      <div className="form-section">
        <h3>{t('settings.modelName')}</h3>
        <div className="form-group">
           <input type="text" name="name" value={formData.name || ''} onChange={handleNameChange} className="model-name-input" autoComplete="off" />
        </div>
      </div>
      <div className="form-section">
        <h3>{t('settings.modelPath')}</h3>
        <div className="form-group model-path-group">
          <input
            type="text"
            name="modelPath"
            value={formData.modelPath || ''}
            readOnly
            placeholder="No model selected"
            autoComplete="off"
          />
          <button
            type="button"
            onClick={handleFindModel}
            className="find-button"
          >
            {t('settings.findModel')}
          </button>
        </div>
      </div>

      {/* Quantization Info Section */}
      <div className="form-section">
        <h3>{t('settings.quantizationInfo')}</h3>
        <div className="quantization-grid">
          <div className="quantization-left">
            <div className="form-group">
              <label>{t('settings.quantizationType')}</label>
              <div className="static-value">
                {quantInfo.type || '-'}
              </div>
            </div>
            <div className="form-group">
              <label>{t('settings.quantizationDetail')}</label>
              <div className="static-value">
                {quantInfo.detail || '-'}
              </div>
            </div>
          </div>

          <div className="quantization-description">
            <div className="quantization-description-title">
              {t('settings.quantizationGuide')}
            </div>
            {quantGuide.paragraphs.map((p, idx) => (
              <p key={idx} className="quantization-description-paragraph">{p}</p>
            ))}
          </div>
        </div>
      </div>
      
      {/* Inference Section */}
      <div className="form-section">
        <h3>{t('settings.inference')}</h3>
        <div className="form-grid">
          <div className="form-group">
            <label>{t('settings.contextSize')}</label>
            <input type="number" name="contextSize" value={formData.contextSize} onChange={handleChange} min="1" />
          </div>
          <div className="form-group">
            <label>{t('settings.maxTokens')}</label>
            <input type="number" name="maxTokens" value={formData.maxTokens} onChange={handleChange} min="-1" />
          </div>
        </div>
      </div>

      {/* Sampling Section */}
      <div className="form-section">
        <h3>{t('settings.sampling')}</h3>
        <div className="form-grid">
           <div className="form-group">
            <label>{t('settings.temperature')}</label>
            <input type="number" name="temperature" value={formData.temperature} onChange={handleChange} step="0.01" min="0" />
          </div>
          <div className="form-group">
            <label>{t('settings.topK')}</label>
            <input type="number" name="topK" value={formData.topK} onChange={handleChange} min="1" />
          </div>
          <div className="form-group">
            <label>{t('settings.topP')}</label>
            <input type="number" name="topP" value={formData.topP} onChange={handleChange} step="0.01" min="0" max="1" />
          </div>
          <div className="form-group">
            <label>{t('settings.minP')}</label>
            <input type="number" name="minP" value={formData.minP} onChange={handleChange} step="0.01" min="0" max="1" />
          </div>
          <div className="form-group">
            <label>{t('settings.tfsZ')}</label>
            <input type="number" name="tfsZ" value={formData.tfsZ} onChange={handleChange} step="0.01" min="0" />
          </div>
          <div className="form-group">
            <label>{t('settings.typicalP')}</label>
            <input type="number" name="typicalP" value={formData.typicalP} onChange={handleChange} step="0.01" min="0" max="1" />
          </div>
        </div>
      </div>
      
      {/* Penalties Section */}
      <div className="form-section">
        <h3>{t('settings.penalties')}</h3>
        <div className="form-grid">
          <div className="form-group">
            <label>{t('settings.repeatPenalty')}</label>
            <input type="number" name="repeatPenalty" value={formData.repeatPenalty} onChange={handleChange} step="0.01" min="0" />
          </div>
          <div className="form-group">
            <label>{t('settings.repeatLastN')}</label>
            <input type="number" name="repeatLastN" value={formData.repeatLastN} onChange={handleChange} min="-1" />
          </div>
          <div className="form-group">
            <label>{t('settings.frequencyPenalty')}</label>
            <input type="number" name="frequencyPenalty" value={formData.frequencyPenalty} onChange={handleChange} step="0.01" min="0" />
          </div>
          <div className="form-group">
            <label>{t('settings.presencePenalty')}</label>
            <input type="number" name="presencePenalty" value={formData.presencePenalty} onChange={handleChange} step="0.01" min="0" />
          </div>
          
          {/* DRY Sampling Settings */}
          <div className="form-group">
            <label>{t('settings.dryMultiplier')}</label>
            <input type="number" name="dryMultiplier" value={formData.dryMultiplier} onChange={handleChange} step="0.01" min="0" />
          </div>
          <div className="form-group">
            <label>{t('settings.dryBase')}</label>
            <input type="number" name="dryBase" value={formData.dryBase} onChange={handleChange} step="0.01" min="0" />
          </div>
          <div className="form-group">
            <label>{t('settings.dryAllowedLength')}</label>
            <input type="number" name="dryAllowedLength" value={formData.dryAllowedLength} onChange={handleChange} min="0" />
          </div>
          <div className="form-group">
            <label>{t('settings.dryPenaltyLastN')}</label>
            <input type="number" name="dryPenaltyLastN" value={formData.dryPenaltyLastN} onChange={handleChange} min="-1" />
          </div>

          <div className="form-group checkbox-group" style={{ gridColumn: '1 / -1' }}>
            <label htmlFor="penalizeNL">{t('settings.penalizeNL')}</label>
            <input id="penalizeNL" type="checkbox" name="penalizeNL" checked={!!formData.penalizeNL} onChange={handleChange} />
          </div>
        </div>
      </div>
      
      {/* Mirostat Section */}
      <div className="form-section">
        <h3>{t('settings.mirostat')}</h3>
        <div className="form-grid">
          <div className="form-group">
            <label>{t('settings.mirostatMode')}</label>
             <select name="mirostatMode" value={formData.mirostatMode} onChange={handleChange}>
              <option value="0">Disabled</option>
              <option value="1">Mirostat v1</option>
              <option value="2">Mirostat v2</option>
            </select>
          </div>
          <div className="form-group">
            <label>{t('settings.mirostatTau')}</label>
            <input type="number" name="mirostatTau" value={formData.mirostatTau} onChange={handleChange} step="0.1" min="0" />
          </div>
          <div className="form-group">
            <label>{t('settings.mirostatEta')}</label>
            <input type="number" name="mirostatEta" value={formData.mirostatEta} onChange={handleChange} step="0.01" min="0" />
          </div>
        </div>
      </div>

      {/* Debug Section */}
      <div className="form-section">
        <h3>{t('settings.debug')}</h3>
        <div className="form-grid">
          <div className="form-group toggle-group">
            <label htmlFor="showSpecialTokens">{t('settings.showSpecialTokens')}</label>
            <button
              type="button"
              className={`toggle-button ${formData.showSpecialTokens ? 'active' : ''}`}
              onClick={() => {
                const newFormData = { ...formData, showSpecialTokens: !formData.showSpecialTokens };
                setFormData(newFormData);
                onChange(newFormData);
              }}
            >
              <span className="toggle-slider"></span>
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ModelForm;
