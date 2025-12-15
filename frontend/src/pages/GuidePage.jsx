import React, { useMemo, useState } from 'react';
import { useLanguage } from '../contexts/LanguageContext';
import { LLAMA_BASE_URL } from '../services/api';
import './GuidePage.css';

const getExampleModelId = () => {
  try {
    const raw = localStorage.getItem('llmServerClientConfig');
    if (raw) {
      const cfg = JSON.parse(raw);
      const models = cfg?.models || [];
      const active = models.find((m) => m?.id === cfg?.activeModelId);
      const modelPath = String(active?.modelPath || '').trim();
      if (modelPath) return modelPath;
    }
  } catch (_e) {}
  try {
    const config = JSON.parse(localStorage.getItem('modelConfig')) || {};
    const modelPath = String(config.modelPath || '').trim();
    if (modelPath) return modelPath;
  } catch (_e) {}
  return 'llama31-banyaa-q4_k_m';
};

const CodeCard = ({ title, description, code, isOpen, onToggle }) => {
  return (
    <section className="guide-card">
      <div className="guide-card-header">
        <div className="guide-card-title-row">
          <h3>{title}</h3>
          <button
            type="button"
            className="guide-toggle-button"
            onClick={onToggle}
            aria-expanded={isOpen}
            aria-label={isOpen ? 'collapse' : 'expand'}
            title={isOpen ? '접기' : '펼치기'}
          >
            <span className={`guide-chevron ${isOpen ? 'open' : ''}`}>▾</span>
          </button>
        </div>
        {description ? <p>{description}</p> : null}
      </div>
      {isOpen ? (
        <pre className="guide-code">
          <code>{code}</code>
        </pre>
      ) : null}
    </section>
  );
};

const GuidePage = () => {
  const { t } = useLanguage();

  const modelId = useMemo(() => getExampleModelId(), []);

  const cards = useMemo(() => {
    const base = LLAMA_BASE_URL;
    const sampleModel = modelId;

    return [
      {
        key: 'curl',
        title: 'Curl',
        description: t('guide.desc.curl'),
        code: `# Health
curl -sS "${base}/health"

# Models (router mode)
curl -sS "${base}/models"

# Chat Completions (OpenAI-compatible)
curl -sS "${base}/v1/chat/completions" \\
  -H "Content-Type: application/json" \\
  -d '{
    "model": "${sampleModel}",
    "messages": [{"role":"user","content":"안녕하세요. 간단히 자기소개해 주세요."}],
    "stream": false,
    "max_tokens": 128,
    "temperature": 0.7
  }'`,
      },
      {
        key: 'js',
        title: 'JavaScript',
        description: t('guide.desc.javascript'),
        code: `// Node.js 18+ (fetch 내장)
const BASE_URL = "${base}";
const MODEL = "${sampleModel}";

async function main() {
  const health = await fetch(\`\${BASE_URL}/health\`).then((r) => r.json());
  console.log("health:", health);

  const res = await fetch(\`\${BASE_URL}/v1/chat/completions\`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      model: MODEL,
      messages: [{ role: "user", content: "안녕하세요. 간단히 자기소개해 주세요." }],
      stream: false,
      max_tokens: 128,
      temperature: 0.7,
    }),
  });

  const json = await res.json();
  console.log(json.choices?.[0]?.message?.content ?? json);
}

main().catch(console.error);`,
      },
      {
        key: 'react',
        title: 'React',
        description: t('guide.desc.react'),
        code: `import React, { useState } from "react";

const BASE_URL = "${base}";
const MODEL = "${sampleModel}";

export default function LlmGuideExample() {
  const [text, setText] = useState("");
  const [loading, setLoading] = useState(false);

  const ask = async () => {
    setLoading(true);
    try {
      const res = await fetch(\`\${BASE_URL}/v1/chat/completions\`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          model: MODEL,
          messages: [{ role: "user", content: "안녕하세요. 간단히 자기소개해 주세요." }],
          stream: false,
          max_tokens: 128,
          temperature: 0.7,
        }),
      });
      const json = await res.json();
      setText(json.choices?.[0]?.message?.content ?? JSON.stringify(json, null, 2));
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <button onClick={ask} disabled={loading}>
        {loading ? "요청 중..." : "요청 보내기"}
      </button>
      <pre style={{ whiteSpace: "pre-wrap" }}>{text}</pre>
    </div>
  );
}`,
      },
      {
        key: 'python',
        title: 'Python',
        description: t('guide.desc.python'),
        code: `import requests

BASE_URL = "${base}"
MODEL = "${sampleModel}"

r = requests.post(
    f"{BASE_URL}/v1/chat/completions",
    json={
        "model": MODEL,
        "messages": [{"role": "user", "content": "안녕하세요. 간단히 자기소개해 주세요."}],
        "stream": False,
        "max_tokens": 128,
        "temperature": 0.7,
    },
    timeout=30,
)
r.raise_for_status()
data = r.json()
print(data["choices"][0]["message"]["content"])`,
      },
      {
        key: 'java',
        title: 'Java',
        description: t('guide.desc.java'),
        code: `// Java 11+
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;

public class LlmGuideExample {
  public static void main(String[] args) throws Exception {
    String baseUrl = "${base}";
    String model = "${sampleModel}";

    String body = """
      {
        "model": "%s",
        "messages": [{"role":"user","content":"안녕하세요. 간단히 자기소개해 주세요."}],
        "stream": false,
        "max_tokens": 128,
        "temperature": 0.7
      }
      """.formatted(model);

    HttpClient client = HttpClient.newHttpClient();
    HttpRequest req = HttpRequest.newBuilder()
      .uri(URI.create(baseUrl + "/v1/chat/completions"))
      .header("Content-Type", "application/json")
      .POST(HttpRequest.BodyPublishers.ofString(body))
      .build();

    HttpResponse<String> res = client.send(req, HttpResponse.BodyHandlers.ofString());
    System.out.println(res.body());
  }
}`,
      },
      {
        key: 'csharp',
        title: 'C#',
        description: t('guide.desc.csharp'),
        code: `using System;
using System.Net.Http;
using System.Text;
using System.Text.Json;
using System.Threading.Tasks;

class LlmGuideExample {
  static async Task Main() {
    var baseUrl = "${base}";
    var model = "${sampleModel}";

    using var http = new HttpClient();
    var payload = new {
      model = model,
      messages = new[] { new { role = "user", content = "안녕하세요. 간단히 자기소개해 주세요." } },
      stream = false,
      max_tokens = 128,
      temperature = 0.7
    };

    var json = JsonSerializer.Serialize(payload);
    var res = await http.PostAsync(baseUrl + "/v1/chat/completions",
      new StringContent(json, Encoding.UTF8, "application/json"));

    var text = await res.Content.ReadAsStringAsync();
    Console.WriteLine(text);
  }
}`,
      },
      {
        key: 'cpp',
        title: 'C++',
        description: t('guide.desc.cpp'),
        code: `// libcurl 예제
// g++ main.cpp -lcurl -std=c++17
#include <curl/curl.h>
#include <string>
#include <iostream>

static size_t write_cb(char *ptr, size_t size, size_t nmemb, void *userdata) {
  auto *out = static_cast<std::string *>(userdata);
  out->append(ptr, size * nmemb);
  return size * nmemb;
}

int main() {
  const std::string baseUrl = "${base}";
  const std::string model = "${sampleModel}";
  const std::string url = baseUrl + "/v1/chat/completions";
  const std::string body =
    "{"
    "\\"model\\": \\"" + model + "\\","
    "\\"messages\\":[{\\"role\\":\\"user\\",\\"content\\":\\"안녕하세요. 간단히 자기소개해 주세요.\\"}],"
    "\\"stream\\":false,"
    "\\"max_tokens\\":128,"
    "\\"temperature\\":0.7"
    "}";

  curl_global_init(CURL_GLOBAL_DEFAULT);
  CURL *curl = curl_easy_init();
  if (!curl) return 1;

  struct curl_slist *headers = nullptr;
  headers = curl_slist_append(headers, "Content-Type: application/json");

  std::string resp;
  curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
  curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
  curl_easy_setopt(curl, CURLOPT_POSTFIELDS, body.c_str());
  curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_cb);
  curl_easy_setopt(curl, CURLOPT_WRITEDATA, &resp);

  CURLcode rc = curl_easy_perform(curl);
  if (rc != CURLE_OK) {
    std::cerr << "curl error: " << curl_easy_strerror(rc) << "\\n";
  } else {
    std::cout << resp << "\\n";
  }

  curl_slist_free_all(headers);
  curl_easy_cleanup(curl);
  curl_global_cleanup();
  return 0;
}`,
      },
    ];
  }, [t, modelId]);

  // default: collapsed
  const [openCards, setOpenCards] = useState(() => ({}));

  const isCardOpen = (key) => Boolean(openCards[key]);
  const toggleCard = (key) => {
    setOpenCards((prev) => ({ ...(prev || {}), [key]: !prev?.[key] }));
  };

  return (
    <div className="guide-page">
      <div className="guide-header">
        <h2>{t('guide.title')}</h2>
        <p>{t('guide.subtitle')}</p>
        <div className="guide-meta">
          <div>
            <span className="guide-meta-label">{t('guide.baseUrlLabel')}</span>
            <code className="guide-inline-code">{LLAMA_BASE_URL}</code>
          </div>
          <div>
            <span className="guide-meta-label">{t('guide.exampleModelId')}</span>
            <code className="guide-inline-code">{modelId}</code>
          </div>
        </div>
        <div className="guide-note">{t('guide.routerNote')}</div>
      </div>

      <div className="guide-grid">
        {cards.map((c) => (
          <CodeCard
            key={c.key}
            title={c.title}
            description={c.description}
            code={c.code}
            isOpen={isCardOpen(c.key)}
            onToggle={() => toggleCard(c.key)}
          />
        ))}
      </div>
    </div>
  );
};

export default GuidePage;

