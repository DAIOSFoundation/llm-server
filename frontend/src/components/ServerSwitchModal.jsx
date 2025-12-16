import React from 'react';
import './ServerSwitchModal.css';

const ServerSwitchModal = ({ isOpen, onConfirm, onCancel, currentServerType, newServerType, isLoading, progressMessage }) => {
  if (!isOpen) return null;

  const getServerTypeName = (type) => {
    if (type === 'mlx') return 'MLX';
    if (type === 'gguf') return 'GGUF (llama.cpp)';
    return type || 'Unknown';
  };

  return (
    <div className="server-switch-modal-overlay" onClick={onCancel}>
      <div className="server-switch-modal" onClick={(e) => e.stopPropagation()}>
        <div className="server-switch-modal-header">
          <h3>서버 전환 필요</h3>
        </div>
        <div className="server-switch-modal-body">
          {isLoading ? (
            <div className="server-switch-loading">
              <div className="loading-spinner"></div>
              <p className="loading-message">{progressMessage || '서버를 전환하는 중...'}</p>
              <p className="loading-submessage">잠시만 기다려주세요.</p>
            </div>
          ) : (
            <>
              <p>선택한 모델은 다른 서버 타입을 사용합니다.</p>
              <div className="server-switch-info">
                <div className="server-info-item">
                  <span className="server-label">현재 서버:</span>
                  <span className="server-value">{getServerTypeName(currentServerType)}</span>
                </div>
                <div className="server-info-arrow">→</div>
                <div className="server-info-item">
                  <span className="server-label">새 서버:</span>
                  <span className="server-value">{getServerTypeName(newServerType)}</span>
                </div>
              </div>
              <p className="server-switch-warning">
                기존 서버를 종료하고 새 서버를 시작합니다. 이 작업은 몇 초 정도 소요될 수 있습니다.
              </p>
            </>
          )}
        </div>
        {!isLoading && (
          <div className="server-switch-modal-footer">
            <button className="server-switch-button cancel-button" onClick={onCancel}>
              취소
            </button>
            <button className="server-switch-button confirm-button" onClick={onConfirm}>
              확인
            </button>
          </div>
        )}
      </div>
    </div>
  );
};

export default ServerSwitchModal;

