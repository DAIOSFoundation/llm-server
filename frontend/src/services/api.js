import axios from 'axios';

const API_BASE_URL = '/api';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json'
  }
});

export const getModels = async () => {
  const response = await api.get('/models');
  return response.data;
};

export const addModel = async (modelData) => {
  const response = await api.post('/models', modelData);
  return response.data;
};

export const updateModel = async (id, updates) => {
  const response = await api.put(`/models/${id}`, updates);
  return response.data;
};

export const deleteModel = async (id) => {
  const response = await api.delete(`/models/${id}`);
  return response.data;
};

export const sendChatMessage = async (modelId, messages, stream = false) => {
  if (stream) {
    // 스트리밍은 EventSource 또는 fetch 사용
    const response = await fetch(`${API_BASE_URL}/chat`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({ modelId, messages, stream: true })
    });

    if (!response.ok) {
      throw new Error('채팅 요청 실패');
    }

    // 스트리밍 응답 처리
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    return new Promise((resolve, reject) => {
      const processStream = async () => {
        try {
          while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n');
            buffer = lines.pop() || '';

            for (const line of lines) {
              if (line.startsWith('data: ')) {
                const data = line.slice(6);
                if (data === '[DONE]') {
                  resolve({ success: true, content: '' });
                  return;
                }
                try {
                  const parsed = JSON.parse(data);
                  // 스트리밍 청크 처리
                  if (parsed.content) {
                    // 스트리밍 콜백 호출
                  }
                } catch (e) {
                  // JSON 파싱 오류 무시
                }
              }
            }
          }
        } catch (error) {
          reject(error);
        }
      };

      processStream();
    });
  } else {
    const response = await api.post('/chat', { modelId, messages, stream: false });
    return response.data;
  }
};

export const getSystemInfo = async () => {
  const response = await api.get('/system');
  return response.data;
};

