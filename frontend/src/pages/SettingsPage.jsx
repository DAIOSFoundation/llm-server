import React, { useState, useEffect } from 'react';
import { useLanguage } from '../contexts/LanguageContext';
import { getModels, addModel, updateModel, deleteModel } from '../services/api';
import ModelForm from '../components/ModelForm';
import './SettingsPage.css';

const SettingsPage = () => {
  const { t } = useLanguage();
  const [models, setModels] = useState([]);
  const [editingModel, setEditingModel] = useState(null);
  const [showAddForm, setShowAddForm] = useState(false);

  useEffect(() => {
    loadModels();
  }, []);

  const loadModels = async () => {
    try {
      const data = await getModels();
      if (data.success) {
        setModels(data.models);
      }
    } catch (error) {
      console.error('모델 로드 실패:', error);
      alert('모델을 불러오는데 실패했습니다.');
    }
  };

  const handleAddModel = async (modelData) => {
    try {
      const response = await addModel(modelData);
      if (response.success) {
        await loadModels();
        setShowAddForm(false);
      } else {
        throw new Error(response.error || '모델 추가 실패');
      }
    } catch (error) {
      console.error('모델 추가 실패:', error);
      alert(`모델 추가 실패: ${error.message}`);
    }
  };

  const handleUpdateModel = async (id, updates) => {
    try {
      const response = await updateModel(id, updates);
      if (response.success) {
        await loadModels();
        setEditingModel(null);
      } else {
        throw new Error(response.error || '모델 업데이트 실패');
      }
    } catch (error) {
      console.error('모델 업데이트 실패:', error);
      alert(`모델 업데이트 실패: ${error.message}`);
    }
  };

  const handleDeleteModel = async (id) => {
    if (!confirm('정말 이 모델을 삭제하시겠습니까?')) {
      return;
    }

    try {
      const response = await deleteModel(id);
      if (response.success) {
        await loadModels();
      } else {
        throw new Error(response.error || '모델 삭제 실패');
      }
    } catch (error) {
      console.error('모델 삭제 실패:', error);
      alert(`모델 삭제 실패: ${error.message}`);
    }
  };

  return (
    <div className="settings-page">
      <div className="settings-container">
        <div className="settings-header">
          <h1>{t('settings.title')}</h1>
          <button
            className="add-model-button"
            onClick={() => setShowAddForm(true)}
          >
            {t('settings.addModel')}
          </button>
        </div>

        {showAddForm && (
          <div className="model-form-container">
            <ModelForm
              onSubmit={handleAddModel}
              onCancel={() => setShowAddForm(false)}
            />
          </div>
        )}

        <div className="models-list">
          {models.length === 0 ? (
            <div className="empty-models">
              <p>추가된 모델이 없습니다. 모델을 추가해주세요.</p>
            </div>
          ) : (
            models.map((model) => (
              <div key={model.id} className="model-card">
                {editingModel?.id === model.id ? (
                  <ModelForm
                    model={model}
                    onSubmit={(updates) => handleUpdateModel(model.id, updates)}
                    onCancel={() => setEditingModel(null)}
                  />
                ) : (
                  <>
                    <div className="model-card-header">
                      <h3>{model.name}</h3>
                      <div className="model-card-actions">
                        <button
                          className="edit-button"
                          onClick={() => setEditingModel(model)}
                        >
                          {t('settings.edit')}
                        </button>
                        <button
                          className="delete-button"
                          onClick={() => handleDeleteModel(model.id)}
                        >
                          {t('settings.delete')}
                        </button>
                      </div>
                    </div>
                    <div className="model-card-info">
                      <div className="info-row">
                        <span className="info-label">경로:</span>
                        <span className="info-value">{model.path}</span>
                      </div>
                      <div className="info-row">
                        <span className="info-label">{t('settings.quantization')}:</span>
                        <span className="info-value">{model.quantization}</span>
                      </div>
                      <div className="info-row">
                        <span className="info-label">{t('settings.device')}:</span>
                        <span className="info-value">{model.device}</span>
                      </div>
                      {model.gpuBackend && (
                        <div className="info-row">
                          <span className="info-label">{t('settings.gpuBackend')}:</span>
                          <span className="info-value">{model.gpuBackend}</span>
                        </div>
                      )}
                    </div>
                  </>
                )}
              </div>
            ))
          )}
        </div>
      </div>
    </div>
  );
};

export default SettingsPage;

