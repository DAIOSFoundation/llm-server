import axios from 'axios';

const API_BASE_URL = 'http://localhost:3001';

const api = axios.create({
  baseURL: API_BASE_URL,
});

const buildLlama3Prompt = (messages) => {
  const systemPrompt = "You are a helpful AI assistant. Please respond in the same language as the user's question.\n당신은 유용한 AI 어시스턴트입니다. 사용자의 질문과 같은 언어로 답변해주세요.";
  let prompt = `<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n${systemPrompt}<|eot_id|>`;

  messages.forEach(message => {
    const role = message.role === 'assistant' ? 'assistant' : 'user';
    prompt += `<|start_header_id|>${role}<|end_header_id|>\n\n${message.content}<|eot_id|>`;
  });

  prompt += `<|start_header_id|>assistant<|end_header_id|>\n\n`;
  return prompt;
};

export const sendChatMessage = async (messages, onToken) => {
  try {
    const prompt = buildLlama3Prompt(messages);
    const config = JSON.parse(localStorage.getItem('modelConfig')) || {};
    
    // llama.cpp server uses snake_case for parameters
    const payload = {
      prompt,
      stream: true,
      n_predict: config.maxTokens || 1024,
      temperature: config.temperature ?? 0.7,
      top_k: config.topK || 40,
      top_p: config.topP || 0.95,
      min_p: config.minP || 0.05,
      tfs_z: config.tfsZ || 1.0,
      typical_p: config.typicalP || 1.0,
      repeat_penalty: config.repeatPenalty || 1.1,
      repeat_last_n: config.repeatLastN || 64,
      penalize_nl: config.penalizeNL || false,
      mirostat: config.mirostatMode || 0,
      mirostat_tau: config.mirostatTau || 5.0,
      mirostat_eta: config.mirostatEta || 0.1,
      stop: ["<|eot_id|>", "<|end_of_text|>"]
    };

    const response = await fetch('http://localhost:8080/completion', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(payload),
    });

    if (!response.ok) {
      const errorText = await response.text();
      console.error("Error from server:", errorText);
      throw new Error(`Server responded with status: ${response.status}`);
    }
    
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let partialData = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) {
        break;
      }
      
      partialData += decoder.decode(value, { stream: true });

      // The server sends data in "data: { ...json... }" format.
      // We need to parse these chunks.
      const lines = partialData.split('\n');
      partialData = lines.pop(); // Keep the last, possibly incomplete, line

      for (const line of lines) {
        if (line.startsWith('data: ')) {
          const jsonString = line.substring(6);
          if (jsonString) {
            try {
              const parsed = JSON.parse(jsonString);
              if (parsed.content) {
                onToken(parsed.content);
              }
              if (parsed.stop) {
                return; // Stop processing further tokens
              }
            } catch (e) {
              console.error('Failed to parse stream chunk:', jsonString, e);
            }
          }
        }
      }
    }

  } catch (error) {
    console.error('채팅 스트림 오류:', error);
    throw error;
  }
};

export const getSystemInfo = async () => {
  const response = await api.get('/system');
  return response.data;
};