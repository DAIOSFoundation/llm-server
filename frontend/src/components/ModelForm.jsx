import React, { useState, useEffect, useMemo } from 'react';
import { useLanguage } from '../contexts/LanguageContext';
import './ModelForm.css';
import { LLAMA_BASE_URL } from '../services/api';

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

const buildQuantizationSummaryLines = (language, ggufInfo) => {
  const isKo = language === 'ko';

  // 읽기 실패/미지원: 간단히만 표시
  if (!ggufInfo) return [];
  if (ggufInfo.ok === false) {
    return [];
  }

  const lines = [];

  if (ggufInfo.ggufVersion != null) {
    lines.push(isKo ? `GGUF v${ggufInfo.ggufVersion}` : `GGUF v${ggufInfo.ggufVersion}`);
  }

  if (ggufInfo.fileTypeName) {
    lines.push(`general.file_type: ${ggufInfo.fileTypeName}`);
  }

  if (ggufInfo.qkv && (ggufInfo.qkv.q || ggufInfo.qkv.k || ggufInfo.qkv.v)) {
    const q = ggufInfo.qkv.q || '-';
    const k = ggufInfo.qkv.k || '-';
    const v = ggufInfo.qkv.v || '-';
    lines.push(`Q/K/V: ${q} / ${k} / ${v}`);
  }

  if (ggufInfo.tensorTypes && typeof ggufInfo.tensorTypes === 'object') {
    const entries = Object.entries(ggufInfo.tensorTypes)
      .sort((a, b) => (b[1] || 0) - (a[1] || 0))
      .slice(0, 3)
      .map(([k, v]) => `${k}:${v}`);
    if (entries.length > 0) {
      lines.push((isKo ? '텐서 타입: ' : 'Tensor types: ') + entries.join(', '));
    }
  }

  return lines;
};

const ModelForm = ({ config, onChange }) => {
  const { t, language } = useLanguage();
  const [formData, setFormData] = useState({});
  const [ggufInfo, setGgufInfo] = useState(null);
  const [pathCheck, setPathCheck] = useState({ status: 'idle', message: '' }); // idle | checking | ok | error

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

  // modelPath 변경 시: 기존 검증/메타데이터 상태 초기화
  useEffect(() => {
    setGgufInfo(null);
    setPathCheck({ status: 'idle', message: '' });
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

  const handleVerifyPath = async () => {
    const modelPath = (formData.modelPath || '').trim();
    if (!modelPath) {
      setPathCheck({ status: 'error', message: t('settings.verifyPathEmpty') });
      setGgufInfo(null);
      return;
    }

    setPathCheck({ status: 'checking', message: '' });
    try {
      const res = await fetch(`${LLAMA_BASE_URL}/gguf-info`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ path: modelPath }),
        signal: AbortSignal.timeout(3000),
      });

      const info = await res.json().catch(() => null);

      if (res.ok && info && info.ok) {
        setGgufInfo(info);
        setPathCheck({ status: 'ok', message: t('settings.verifyPathOk') });
      } else {
        setGgufInfo(null);
        setPathCheck({
          status: 'error',
          message: (info && info.error) ? info.error : t('settings.verifyPathFail'),
        });
      }
    } catch (_e) {
      setGgufInfo(null);
      setPathCheck({ status: 'error', message: t('settings.verifyPathFail') });
    }
  };
  
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

  const quantSummaryLines = useMemo(
    () => buildQuantizationSummaryLines(language, ggufInfo),
    [language, ggufInfo],
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
      
      {/* Model Path */}
      <div className="form-section">
        <h3>{t('settings.modelPath')}</h3>
        <div className="form-group">
          <div className="model-path-row">
            <input
              type="text"
              name="modelPath"
              value={formData.modelPath || ''}
              onChange={handleChange}
              placeholder={t('settings.modelPathPlaceholder')}
              autoComplete="off"
            />
            <button
              type="button"
              className="verify-button"
              onClick={handleVerifyPath}
              disabled={pathCheck.status === 'checking'}
              title={t('settings.verifyPath')}
            >
              {pathCheck.status === 'checking' ? t('settings.verifying') : t('settings.verifyPath')}
            </button>
          </div>
          {pathCheck.status === 'error' && (
            <div className="path-status error">{pathCheck.message}</div>
          )}
          {pathCheck.status === 'ok' && (
            <div className="path-status ok">{pathCheck.message}</div>
          )}
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
            {quantSummaryLines.length > 0 ? (
              <ul className="quantization-summary">
                {quantSummaryLines.map((line, idx) => (
                  <li key={idx} className="quantization-summary-item">{line}</li>
                ))}
              </ul>
            ) : (
              <div className="quantization-summary-empty">-</div>
            )}
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
