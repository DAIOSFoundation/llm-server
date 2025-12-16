#include <napi.h>
#include <string>
#include <vector>
#include <memory>
#include <thread>
#include <mutex>
#include <atomic>
#include <functional>
#include <map>
#include <fstream>
#include <sstream>
#include <dirent.h>
#include <sys/stat.h>
#include <algorithm>
#include <unordered_map>
#include <unordered_set>
#include <regex>
#include <iomanip>

// MLX C++ API 헤더 파일들
#include <mlx/mlx.h>
#include <mlx/array.h>
#include <mlx/io.h>
#include <mlx/ops.h>
#include <mlx/random.h>
#include <mlx/stream.h>
#include <mlx/device.h>
#include <mlx/fast.h>

namespace mx = mlx::core;

// BPE pair hash 함수 (llama.cpp 스타일)
struct PairHash {
    size_t operator()(const std::pair<std::string, std::string>& p) const {
        return std::hash<std::string>{}(p.first) ^ (std::hash<std::string>{}(p.second) << 1);
    }
};

// MLX 모델 구조체 (ggml-metal 스타일)
struct MlxModel {
    std::unordered_map<std::string, mx::array> weights;
    std::unordered_map<std::string, std::string> metadata;
    std::string modelPath;
    mx::Device device;
    mx::Stream stream;
    bool loaded;
    
    // 모델 하이퍼파라미터
    int vocabSize;
    int hiddenSize;
    int numLayers;
    int numHeads;
    int numKeyValueHeads;
    int intermediateSize;
    int maxContextLength;
    
    // 토큰화 관련 데이터 (llama.cpp 스타일)
    std::unordered_map<std::string, int> vocab; // token -> id
    std::unordered_map<int, std::string> idToToken; // id -> token
    std::unordered_map<std::pair<std::string, std::string>, int, PairHash> bpeRanks; // BPE merges
    std::unordered_set<int> specialTokens; // special token IDs
    int bosTokenId;
    int eosTokenId;
    int unkTokenId;
    bool addBos;
    bool addEos;
    
    MlxModel() : device(mx::Device::gpu), stream(mx::new_stream(mx::Device::gpu)), 
                 loaded(false), vocabSize(0), hiddenSize(0), numLayers(0), 
                 numHeads(0), numKeyValueHeads(0), intermediateSize(0), maxContextLength(0),
                 bosTokenId(-1), eosTokenId(-1), unkTokenId(-1), addBos(false), addEos(false) {}
};

class MlxInference : public Napi::ObjectWrap<MlxInference> {
public:
    static Napi::Object Init(Napi::Env env, Napi::Object exports);
    MlxInference(const Napi::CallbackInfo& info);
    ~MlxInference();

private:
    static Napi::FunctionReference constructor;
    
    std::string modelDir_;
    std::unique_ptr<MlxModel> model_;
    std::mutex mutex_;
    std::atomic<bool> isRunning_;
    
    Napi::ThreadSafeFunction onTokenCallback_;
    Napi::ThreadSafeFunction onErrorCallback_;
    Napi::ThreadSafeFunction onCompleteCallback_;
    bool hasTokenCallback_;
    bool hasErrorCallback_;
    bool hasCompleteCallback_;
    
    Napi::Value GenerateStream(const Napi::CallbackInfo& info);
    Napi::Value LoadModel(const Napi::CallbackInfo& info);
    Napi::Value Tokenize(const Napi::CallbackInfo& info);
    Napi::Value Decode(const Napi::CallbackInfo& info);
    
    void RunGeneration(const std::string& prompt, const std::map<std::string, double>& options);
    bool LoadModelFromPath(const std::string& modelPath);
    std::map<std::string, double> ParseOptions(const std::string& optionsJson);
    
    // MLX를 사용한 모델 로딩 및 추론 함수들
    bool LoadSafetensors(const std::string& modelDir);
    bool LoadGGUF(const std::string& filePath);
    bool LoadTokenizer(const std::string& modelPath);
    std::vector<int> Tokenize(const std::string& text);
    std::string Decode(const std::vector<int>& tokens);
    
    // Transformer forward pass (ggml-metal 스타일)
    mx::array ForwardPass(const std::vector<int>& tokens, int pos);
    mx::array AttentionLayer(const mx::array& x, int layerIdx);
    mx::array FeedForwardLayer(const mx::array& x, int layerIdx);
    mx::array LayerNorm(const mx::array& x, const std::string& weightKey);
    
    // 샘플링 함수들
    mx::array GenerateNextToken(const mx::array& logits, double temperature, int topK, double topP, double minP);
    mx::array ApplyTopK(const mx::array& probs, int k);
    mx::array ApplyTopP(const mx::array& probs, double p);
    mx::array ApplyMinP(const mx::array& probs, double minP);
    int SampleToken(const mx::array& probs);
    
    // 토큰화 헬퍼 함수들 (llama.cpp 스타일)
    std::vector<std::string> BPEWordTokenize(const std::string& text);
    std::vector<int> BPETokenizeWord(const std::string& word);
    std::pair<std::string, std::string> GetBestBpePair(const std::vector<std::string>& word);
    std::unordered_map<uint8_t, std::string> BuildByteToUnicode();
    std::string UnicodeToBytes(const std::string& text);
    std::string SimpleJsonParse(const std::string& jsonStr, const std::string& key);
    
    // 유틸리티 함수들
    mx::array GetWeight(const std::string& key);
    bool HasWeight(const std::string& key);
    void EvalArray(mx::array& arr);
};

Napi::FunctionReference MlxInference::constructor;

MlxInference::MlxInference(const Napi::CallbackInfo& info) 
    : Napi::ObjectWrap<MlxInference>(info), 
      isRunning_(false),
      hasTokenCallback_(false), hasErrorCallback_(false), hasCompleteCallback_(false) {
    Napi::Env env = info.Env();
    
    if (info.Length() < 1 || !info[0].IsString()) {
        Napi::TypeError::New(env, "Expected modelDir string").ThrowAsJavaScriptException();
        return;
    }
    
    modelDir_ = info[0].As<Napi::String>().Utf8Value();
    
    // 모델 로드
    if (!LoadModelFromPath(modelDir_)) {
        Napi::Error::New(env, "Failed to load model from: " + modelDir_).ThrowAsJavaScriptException();
        return;
    }
}

MlxInference::~MlxInference() {
    std::lock_guard<std::mutex> lock(mutex_);
    isRunning_ = false;
    
    if (hasTokenCallback_) {
        onTokenCallback_.Release();
    }
    if (hasErrorCallback_) {
        onErrorCallback_.Release();
    }
    if (hasCompleteCallback_) {
        onCompleteCallback_.Release();
    }
    
    model_.reset();
}

Napi::Object MlxInference::Init(Napi::Env env, Napi::Object exports) {
    Napi::Function func = DefineClass(env, "MlxInference", {
        InstanceMethod("generateStream", &MlxInference::GenerateStream),
        InstanceMethod("loadModel", &MlxInference::LoadModel),
        InstanceMethod("tokenize", &MlxInference::Tokenize),
        InstanceMethod("decode", &MlxInference::Decode),
    });
    
    constructor = Napi::Persistent(func);
    constructor.SuppressDestruct();
    
    exports.Set("MlxInference", func);
    return exports;
}

Napi::Value MlxInference::LoadModel(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    MlxInference* self = Napi::ObjectWrap<MlxInference>::Unwrap(info.This().As<Napi::Object>());
    
    if (info.Length() < 1 || !info[0].IsString()) {
        Napi::TypeError::New(env, "Expected modelDir string").ThrowAsJavaScriptException();
        return env.Null();
    }
    
    std::string modelPath = info[0].As<Napi::String>().Utf8Value();
    bool success = self->LoadModelFromPath(modelPath);
    
    return Napi::Boolean::New(env, success);
}

bool MlxInference::LoadModelFromPath(const std::string& modelPath) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    try {
        model_ = std::make_unique<MlxModel>();
        model_->modelPath = modelPath;
        model_->device = mx::Device::gpu;
        model_->stream = mx::new_stream(model_->device);
        
        // MLX 모델 디렉토리 확인
        struct stat st;
        if (stat(modelPath.c_str(), &st) != 0 || !S_ISDIR(st.st_mode)) {
            return false;
        }
        
        // config.json 확인
        std::string configPath = modelPath + "/config.json";
        if (stat(configPath.c_str(), &st) != 0) {
            return false;
        }
        
        // safetensors 또는 GGUF 파일 찾기
        bool loaded = false;
        
        // 먼저 model.safetensors.index.json 확인 (여러 파일로 나뉜 모델)
        std::string indexPath = modelPath + "/model.safetensors.index.json";
        if (stat(indexPath.c_str(), &st) == 0) {
            // 여러 safetensors 파일로 나뉜 모델
            loaded = LoadSafetensors(modelPath);
        } else {
            // 단일 파일 모델 찾기
            DIR* dir = opendir(modelPath.c_str());
            if (dir != nullptr) {
                struct dirent* entry;
                std::vector<std::string> safetensorsFiles;
                std::vector<std::string> ggufFiles;
                
                while ((entry = readdir(dir)) != nullptr) {
                    std::string filename = entry->d_name;
                    
                    // safetensors 파일 찾기
                    if (filename.size() > 11 && filename.substr(filename.size() - 11) == ".safetensors") {
                        safetensorsFiles.push_back(modelPath + "/" + filename);
                    }
                    
                    // GGUF 파일 찾기
                    if (filename.size() > 5 && filename.substr(filename.size() - 5) == ".gguf") {
                        ggufFiles.push_back(modelPath + "/" + filename);
                    }
                }
                closedir(dir);
                
                // safetensors 파일이 있으면 로드
                if (!safetensorsFiles.empty()) {
                    loaded = LoadSafetensors(modelPath);
                } else if (!ggufFiles.empty()) {
                    // 첫 번째 GGUF 파일 로드
                    loaded = LoadGGUF(ggufFiles[0]);
                }
            }
        }
        
        if (!loaded) {
            // 모델 가중치 파일을 찾을 수 없음
            return false;
        }
        
        // 토큰화 로드
        if (!LoadTokenizer(modelPath)) {
            // 토큰화 로드 실패는 치명적이지 않을 수 있음
            // 하지만 실제 사용 시에는 필요함
        }
        
        model_->loaded = true;
        modelDir_ = modelPath;
        
        return true;
    } catch (const std::exception& e) {
        model_->loaded = false;
        return false;
    }
}

bool MlxInference::LoadSafetensors(const std::string& modelDir) {
    try {
        // model.safetensors.index.json 확인
        std::string indexPath = modelDir + "/model.safetensors.index.json";
        struct stat indexStat;
        
        if (stat(indexPath.c_str(), &indexStat) == 0) {
            // 여러 파일로 나뉜 모델: index.json을 파싱하여 모든 파일 로드
            // 간단한 구현: 모든 .safetensors 파일을 찾아서 로드
            DIR* dir = opendir(modelDir.c_str());
            if (dir == nullptr) {
                return false;
            }
            
            std::vector<std::string> safetensorsFiles;
            struct dirent* entry;
            while ((entry = readdir(dir)) != nullptr) {
                std::string filename = entry->d_name;
                if (filename.size() > 11 && filename.substr(filename.size() - 11) == ".safetensors") {
                    safetensorsFiles.push_back(modelDir + "/" + filename);
                }
            }
            closedir(dir);
            
            // 모든 safetensors 파일 로드
            for (const auto& filePath : safetensorsFiles) {
                try {
                    auto result = mx::load_safetensors(filePath, model_->stream);
                    // 가중치 병합
                    for (const auto& [key, value] : result.first) {
                        model_->weights.insert({key, value});
                    }
                    // 메타데이터 병합
                    for (const auto& [key, value] : result.second) {
                        model_->metadata[key] = value;
                    }
                } catch (const std::exception& e) {
                    // 개별 파일 로드 실패는 무시하고 계속
                    continue;
                }
            }
            
            return !model_->weights.empty();
        } else {
            // 단일 safetensors 파일 로드
            std::string filePath = modelDir + "/model.safetensors";
            struct stat fileStat;
            if (stat(filePath.c_str(), &fileStat) != 0) {
                return false;
            }
            
            auto result = mx::load_safetensors(filePath, model_->stream);
            model_->weights = result.first;
            model_->metadata = result.second;
            
            return !model_->weights.empty();
        }
    } catch (const std::exception& e) {
        return false;
    }
}

bool MlxInference::LoadGGUF(const std::string& filePath) {
    try {
        // MLX C++ API를 사용하여 GGUF 로드
        auto result = mx::load_gguf(filePath, model_->stream);
        model_->weights = result.first;
        
        // GGUF 메타데이터 처리
        for (const auto& [key, value] : result.second) {
            if (std::holds_alternative<std::string>(value)) {
                model_->metadata[key] = std::get<std::string>(value);
            }
        }
        
        return !model_->weights.empty();
    } catch (const std::exception& e) {
        return false;
    }
}

bool MlxInference::LoadTokenizer(const std::string& modelPath) {
    try {
        std::string tokenizerPath = modelPath + "/tokenizer.json";
        std::ifstream file(tokenizerPath);
        if (!file.is_open()) {
            return false;
        }
        
        std::string jsonContent((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
        file.close();
        
        // 간단한 JSON 파싱 (실제로는 JSON 라이브러리 사용 권장)
        // 여기서는 기본적인 구조만 파싱
        
        // vocab 로드 (간단한 구현)
        // 실제로는 tokenizer.json의 전체 구조를 파싱해야 함
        
        // BPE merges 로드
        // 실제로는 tokenizer.json에서 merges 배열을 읽어야 함
        
        // Special tokens 로드
        std::string configPath = modelPath + "/tokenizer_config.json";
        std::ifstream configFile(configPath);
        if (configFile.is_open()) {
            std::string configContent((std::istreambuf_iterator<char>(configFile)), std::istreambuf_iterator<char>());
            configFile.close();
            
            // 간단한 파싱으로 bos_token, eos_token 등 추출
            // 실제로는 JSON 파서 사용 필요
        }
        
        return true;
    } catch (const std::exception& e) {
        return false;
    }
}

std::unordered_map<uint8_t, std::string> MlxInference::BuildByteToUnicode() {
    // Byte-to-Unicode 매핑 생성 (llama.cpp 스타일)
    std::unordered_map<uint8_t, std::string> byteToUnicode;
    
    // 기본 ASCII 문자
    for (int i = 0; i < 256; ++i) {
        uint8_t byte = static_cast<uint8_t>(i);
        if ((byte >= 33 && byte <= 126) || (byte >= 161 && byte <= 172) || (byte >= 174 && byte <= 255)) {
            byteToUnicode[byte] = std::string(1, static_cast<char>(byte));
        } else {
            // Unicode 변환
            std::ostringstream oss;
            oss << "\\u" << std::hex << std::setw(4) << std::setfill('0') << (256 + byte);
            byteToUnicode[byte] = oss.str();
        }
    }
    
    return byteToUnicode;
}

std::vector<std::string> MlxInference::BPEWordTokenize(const std::string& text) {
    // 단어 단위로 토큰화 (llama.cpp 스타일)
    std::vector<std::string> words;
    std::regex wordRegex(R"((\S+))");
    std::sregex_iterator iter(text.begin(), text.end(), wordRegex);
    std::sregex_iterator end;
    
    for (; iter != end; ++iter) {
        words.push_back(iter->str());
    }
    
    return words;
}

std::pair<std::string, std::string> MlxInference::GetBestBpePair(const std::vector<std::string>& word) {
    // BPE merge에서 가장 높은 우선순위의 pair 찾기 (llama.cpp 스타일)
    std::pair<std::string, std::string> bestPair;
    int bestRank = std::numeric_limits<int>::max();
    
    for (size_t i = 0; i < word.size() - 1; ++i) {
        auto pair = std::make_pair(word[i], word[i + 1]);
        auto it = model_->bpeRanks.find(pair);
        if (it != model_->bpeRanks.end() && it->second < bestRank) {
            bestRank = it->second;
            bestPair = pair;
        }
    }
    
    return bestPair;
}

std::vector<int> MlxInference::BPETokenizeWord(const std::string& word) {
    // BPE 토큰화 (llama.cpp 스타일)
    std::vector<std::string> chars;
    for (char c : word) {
        chars.push_back(std::string(1, c));
    }
    
    // BPE merges 적용
    while (chars.size() > 1) {
        auto pair = GetBestBpePair(chars);
        if (pair.first.empty() || pair.second.empty()) {
            break; // 더 이상 merge할 수 없음
        }
        
        // Merge 수행
        std::vector<std::string> newChars;
        size_t i = 0;
        while (i < chars.size()) {
            if (i < chars.size() - 1 && chars[i] == pair.first && chars[i + 1] == pair.second) {
                newChars.push_back(pair.first + pair.second);
                i += 2;
            } else {
                newChars.push_back(chars[i]);
                i += 1;
            }
        }
        chars = newChars;
    }
    
    // 토큰을 ID로 변환
    std::vector<int> tokenIds;
    for (const auto& token : chars) {
        auto it = model_->vocab.find(token);
        if (it != model_->vocab.end()) {
            tokenIds.push_back(it->second);
        } else if (model_->unkTokenId >= 0) {
            tokenIds.push_back(model_->unkTokenId);
        }
    }
    
    return tokenIds;
}

std::vector<int> MlxInference::Tokenize(const std::string& text) {
    // llama.cpp 스타일의 토큰화 구현
    std::vector<int> tokens;
    
    if (text.empty()) {
        if (model_->addBos && model_->bosTokenId >= 0) {
            tokens.push_back(model_->bosTokenId);
        }
        return tokens;
    }
    
    // BOS 토큰 추가
    if (model_->addBos && model_->bosTokenId >= 0) {
        tokens.push_back(model_->bosTokenId);
    }
    
    // 단어 단위로 토큰화
    std::vector<std::string> words = BPEWordTokenize(text);
    
    for (const auto& word : words) {
        std::vector<int> wordTokens = BPETokenizeWord(word);
        tokens.insert(tokens.end(), wordTokens.begin(), wordTokens.end());
    }
    
    // EOS 토큰 추가
    if (model_->addEos && model_->eosTokenId >= 0) {
        tokens.push_back(model_->eosTokenId);
    }
    
    return tokens;
}

std::string MlxInference::Decode(const std::vector<int>& tokens) {
    // llama.cpp 스타일의 디토큰화 구현
    std::string text;
    
    for (int tokenId : tokens) {
        // Special token 체크
        if (model_->specialTokens.find(tokenId) != model_->specialTokens.end()) {
            // Special token은 건너뛰거나 특별 처리
            if (tokenId == model_->bosTokenId || tokenId == model_->eosTokenId) {
                continue;
            }
        }
        
        // 토큰 ID를 텍스트로 변환
        auto it = model_->idToToken.find(tokenId);
        if (it != model_->idToToken.end()) {
            text += it->second;
        } else if (model_->unkTokenId >= 0) {
            // Unknown token 처리
            text += "<unk>";
        }
    }
    
    return text;
}

mx::array MlxInference::GetWeight(const std::string& key) {
    auto it = model_->weights.find(key);
    if (it != model_->weights.end()) {
        return it->second;
    }
    throw std::runtime_error("Weight not found: " + key);
}

bool MlxInference::HasWeight(const std::string& key) {
    return model_->weights.find(key) != model_->weights.end();
}

void MlxInference::EvalArray(mx::array& arr) {
    mx::eval(arr);
}

mx::array MlxInference::LayerNorm(const mx::array& x, const std::string& weightKey) {
    // Layer Normalization 구현 (ggml-metal 스타일)
    mx::array weight = GetWeight(weightKey + ".weight");
    mx::array bias = GetWeight(weightKey + ".bias");
    
    mx::array mean = mx::mean(x, -1, true);
    mx::array var = mx::mean(mx::square(x - mean), -1, true);
    mx::array std_dev = mx::sqrt(var + mx::array(1e-5f));
    
    mx::array normalized = (x - mean) / std_dev;
    return normalized * weight + bias;
}

mx::array MlxInference::AttentionLayer(const mx::array& x, int layerIdx) {
    // Multi-Head Attention 구현 (ggml-metal 스타일)
    std::string prefix = "layers." + std::to_string(layerIdx) + ".attention.";
    
    mx::array q_proj = GetWeight(prefix + "q_proj.weight");
    mx::array k_proj = GetWeight(prefix + "k_proj.weight");
    mx::array v_proj = GetWeight(prefix + "v_proj.weight");
    mx::array o_proj = GetWeight(prefix + "o_proj.weight");
    
    // Query, Key, Value 계산
    mx::array q = mx::matmul(x, q_proj);
    mx::array k = mx::matmul(x, k_proj);
    mx::array v = mx::matmul(x, v_proj);
    
    // Reshape for multi-head attention
    int numHeads = model_->numHeads;
    int headDim = model_->hiddenSize / numHeads;
    
    // Scaled dot-product attention
    mx::array scale = mx::array(1.0f / std::sqrt(static_cast<float>(headDim)));
    mx::array scores = mx::matmul(q, mx::transpose(k, {0, 2, 1})) * scale;
    
    // Causal mask 적용
    int seqLen = x.shape(1);
    mx::array mask = mx::triu(mx::ones({seqLen, seqLen}), 1) * mx::array(-1e9f);
    scores = scores + mask;
    
    mx::array attn = mx::softmax(scores, -1);
    mx::array out = mx::matmul(attn, v);
    
    // Output projection
    out = mx::matmul(out, o_proj);
    
    return out;
}

mx::array MlxInference::FeedForwardLayer(const mx::array& x, int layerIdx) {
    // Feed Forward Network 구현 (ggml-metal 스타일)
    std::string prefix = "layers." + std::to_string(layerIdx) + ".feed_forward.";
    
    mx::array gate_proj = GetWeight(prefix + "gate_proj.weight");
    mx::array up_proj = GetWeight(prefix + "up_proj.weight");
    mx::array down_proj = GetWeight(prefix + "down_proj.weight");
    
    mx::array gate = mx::matmul(x, gate_proj);
    // SiLU activation: x * sigmoid(x)
    gate = gate * mx::sigmoid(gate);
    
    mx::array up = mx::matmul(x, up_proj);
    
    mx::array hidden = gate * up;
    mx::array out = mx::matmul(hidden, down_proj);
    
    return out;
}

mx::array MlxInference::ForwardPass(const std::vector<int>& tokens, int pos) {
    // Transformer forward pass 구현 (ggml-metal 스타일)
    
    // Token embeddings
    mx::array embed = GetWeight("embed_tokens.weight");
    
    // tokens를 array로 변환
    std::vector<int32_t> tokens32(tokens.begin(), tokens.end());
    mx::array tokenArray(tokens32.data(), {static_cast<int>(tokens32.size())}, mx::int32);
    mx::array x = mx::take(embed, tokenArray, 0);
    
    // Position embeddings (RoPE는 별도로 처리)
    
    // Transformer layers
    for (int i = 0; i < model_->numLayers; ++i) {
        mx::array residual = x;
        
        // Pre-norm
        x = LayerNorm(x, "layers." + std::to_string(i) + ".input_layernorm");
        
        // Self-attention
        mx::array attn_out = AttentionLayer(x, i);
        x = residual + attn_out;
        
        residual = x;
        
        // Post-norm
        x = LayerNorm(x, "layers." + std::to_string(i) + ".post_attention_layernorm");
        
        // Feed-forward
        mx::array ff_out = FeedForwardLayer(x, i);
        x = residual + ff_out;
    }
    
    // Final layer norm
    x = LayerNorm(x, "norm");
    
    // Output projection
    mx::array lm_head = GetWeight("lm_head.weight");
    mx::array logits = mx::matmul(x, mx::transpose(lm_head, {1, 0}));
    
    // 마지막 토큰의 logits만 반환
    mx::array lastLogits = mx::take(logits, mx::array({static_cast<int>(tokens.size() - 1)}), 1);
    EvalArray(lastLogits);
    
    return lastLogits;
}

mx::array MlxInference::ApplyTopK(const mx::array& probs, int k) {
    // Top-K 샘플링 구현
    if (k <= 0 || k >= probs.shape(-1)) return probs;
    
    // Top-K 값 가져오기 (MLX의 topk는 단일 배열 반환)
    mx::array topKValues = mx::topk(probs, k, -1);
    
    // Top-K 값만 유지하고 나머지는 0으로 설정
    // 실제 구현에서는 scatter를 사용하여 mask 생성해야 함
    // 여기서는 간단하게 probs 반환 (실제 구현 필요)
    
    return probs;
}

mx::array MlxInference::ApplyTopP(const mx::array& probs, double p) {
    // Top-P (nucleus) 샘플링 구현
    if (p >= 1.0) return probs;
    
    // 정렬된 확률의 누적합 계산
    mx::array sortedProbs = mx::sort(probs, -1);
    mx::array cumsum = mx::cumsum(sortedProbs, -1);
    
    // 누적합이 p를 초과하는 지점 찾기
    mx::array mask = cumsum <= mx::array(static_cast<float>(p));
    
    return probs * mask;
}

mx::array MlxInference::ApplyMinP(const mx::array& probs, double minP) {
    // Min-P 샘플링 구현
    if (minP <= 0.0) return probs;
    
    mx::array maxProb = mx::max(probs, -1, true);
    mx::array threshold = maxProb * mx::array(static_cast<float>(minP));
    mx::array mask = probs >= threshold;
    
    return probs * mask;
}

int MlxInference::SampleToken(const mx::array& probs) {
    // Multinomial 샘플링
    // 간단하게 argmax 사용 (실제로는 누적 확률로 샘플링해야 함)
    mx::array sampledIdx = mx::argmax(probs, -1, false);
    EvalArray(sampledIdx);
    
    // sampledIdx는 스칼라 배열이므로 item<float>()로 읽기
    return static_cast<int>(sampledIdx.item<float>());
}

mx::array MlxInference::GenerateNextToken(const mx::array& logits, double temperature, int topK, double topP, double minP) {
    // MLX를 사용한 다음 토큰 생성 (ggml-metal 스타일)
    
    // Temperature scaling
    mx::array scaled_logits = mx::divide(logits, mx::array(static_cast<float>(temperature)));
    EvalArray(scaled_logits);
    
    // Softmax
    mx::array probs = mx::softmax(scaled_logits, -1);
    EvalArray(probs);
    
    // Min-P 적용
    if (minP > 0.0) {
        probs = ApplyMinP(probs, minP);
        // 정규화
        mx::array sumArr = mx::sum(probs);
        EvalArray(sumArr);
        float sum = sumArr.item<float>();
        probs = mx::divide(probs, mx::array(sum));
        EvalArray(probs);
    }
    
    // Top-K 적용
    if (topK > 0 && topK < probs.shape(-1)) {
        probs = ApplyTopK(probs, topK);
        // 정규화
        mx::array sumArr = mx::sum(probs);
        EvalArray(sumArr);
        float sum = sumArr.item<float>();
        probs = mx::divide(probs, mx::array(sum));
        EvalArray(probs);
    }
    
    // Top-P 적용
    if (topP > 0.0 && topP < 1.0) {
        probs = ApplyTopP(probs, topP);
        // 정규화
        mx::array sumArr = mx::sum(probs);
        EvalArray(sumArr);
        float sum = sumArr.item<float>();
        probs = mx::divide(probs, mx::array(sum));
        EvalArray(probs);
    }
    
    // 샘플링
    int tokenId = SampleToken(probs);
    
    return mx::array(static_cast<int32_t>(tokenId), mx::int32);
}

std::map<std::string, double> MlxInference::ParseOptions(const std::string& optionsJson) {
    std::map<std::string, double> options;
    
    // 기본값 설정
    options["temperature"] = 0.7;
    options["top_k"] = 40;
    options["top_p"] = 0.95;
    options["min_p"] = 0.05;
    options["repeat_penalty"] = 1.2;
    options["repeat_last_n"] = 128;
    options["max_tokens"] = 600;
    
    // 실제 JSON 파싱은 Node.js에서 수행되므로 여기서는 기본값만 사용
    // 실제 구현에서는 JSON 라이브러리를 사용하여 파싱
    
    return options;
}

Napi::Value MlxInference::GenerateStream(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    MlxInference* self = Napi::ObjectWrap<MlxInference>::Unwrap(info.This().As<Napi::Object>());
    
    if (info.Length() < 3) {
        Napi::TypeError::New(env, "Expected (prompt, options, callbacks)").ThrowAsJavaScriptException();
        return env.Null();
    }
    
    if (!info[0].IsString() || !info[1].IsObject()) {
        Napi::TypeError::New(env, "Expected prompt string and options object").ThrowAsJavaScriptException();
        return env.Null();
    }
    
    std::string prompt = info[0].As<Napi::String>().Utf8Value();
    Napi::Object options = info[1].As<Napi::Object>();
    
    // 옵션을 JSON 문자열로 변환
    Napi::Object jsonObj = Napi::Object::New(env);
    if (options.Has("temperature")) {
        jsonObj.Set("temperature", options.Get("temperature"));
    }
    if (options.Has("topK")) {
        jsonObj.Set("top_k", options.Get("topK"));
    }
    if (options.Has("topP")) {
        jsonObj.Set("top_p", options.Get("topP"));
    }
    if (options.Has("minP")) {
        jsonObj.Set("min_p", options.Get("minP"));
    }
    if (options.Has("repeatPenalty")) {
        jsonObj.Set("repeat_penalty", options.Get("repeatPenalty"));
    }
    if (options.Has("repeatLastN")) {
        jsonObj.Set("repeat_last_n", options.Get("repeatLastN"));
    }
    if (options.Has("maxTokens")) {
        jsonObj.Set("max_tokens", options.Get("maxTokens"));
    }
    if (options.Has("stop")) {
        jsonObj.Set("stop", options.Get("stop"));
    }
    if (options.Has("seed")) {
        jsonObj.Set("seed", options.Get("seed"));
    }
    
    // JSON 문자열화
    Napi::Object global = env.Global();
    Napi::Object JSON = global.Get("JSON").As<Napi::Object>();
    Napi::Function stringify = JSON.Get("stringify").As<Napi::Function>();
    Napi::String optionsJsonStr = stringify.Call({jsonObj}).As<Napi::String>();
    std::string optionsJson = optionsJsonStr.Utf8Value();
    
    // 옵션 파싱
    std::map<std::string, double> parsedOptions = self->ParseOptions(optionsJson);
    
    // 콜백 저장
    if (info.Length() >= 3 && info[2].IsFunction()) {
        if (self->hasTokenCallback_) {
            self->onTokenCallback_.Release();
        }
        self->onTokenCallback_ = Napi::ThreadSafeFunction::New(
            env,
            info[2].As<Napi::Function>(),
            "onToken",
            0,
            1
        );
        self->hasTokenCallback_ = true;
    }
    if (info.Length() >= 4 && info[3].IsFunction()) {
        if (self->hasErrorCallback_) {
            self->onErrorCallback_.Release();
        }
        self->onErrorCallback_ = Napi::ThreadSafeFunction::New(
            env,
            info[3].As<Napi::Function>(),
            "onError",
            0,
            1
        );
        self->hasErrorCallback_ = true;
    }
    if (info.Length() >= 5 && info[4].IsFunction()) {
        if (self->hasCompleteCallback_) {
            self->onCompleteCallback_.Release();
        }
        self->onCompleteCallback_ = Napi::ThreadSafeFunction::New(
            env,
            info[4].As<Napi::Function>(),
            "onComplete",
            0,
            1
        );
        self->hasCompleteCallback_ = true;
    }
    
    // 비동기로 생성 실행
    std::thread([self, prompt, parsedOptions]() {
        self->RunGeneration(prompt, parsedOptions);
    }).detach();
    
    return env.Undefined();
}

void MlxInference::RunGeneration(const std::string& prompt, const std::map<std::string, double>& options) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (isRunning_ || !model_ || !model_->loaded) {
        if (hasErrorCallback_) {
            auto callback = [](Napi::Env env, Napi::Function jsCallback, std::string* errorMsg) {
                jsCallback.Call({Napi::String::New(env, *errorMsg)});
                delete errorMsg;
            };
            onErrorCallback_.BlockingCall(new std::string("Model not loaded or already running"), callback);
        }
        return;
    }
    
    isRunning_ = true;
    
    try {
        // 토큰화
        std::vector<int> tokens = Tokenize(prompt);
        
        if (tokens.empty()) {
            throw std::runtime_error("Failed to tokenize prompt");
        }
        
        // 생성 파라미터
        double temperature = options.at("temperature");
        int topK = static_cast<int>(options.at("top_k"));
        double topP = options.at("top_p");
        double minP = options.at("min_p");
        double repeatPenalty = options.at("repeat_penalty");
        int repeatLastN = static_cast<int>(options.at("repeat_last_n"));
        int maxTokens = static_cast<int>(options.at("max_tokens"));
        
        std::vector<int> generatedTokens;
        std::vector<int> lastNTokens(tokens.begin(), tokens.end());
        
        // MLX를 사용한 토큰 생성 (ggml-metal 스타일)
        for (int i = 0; i < maxTokens; ++i) {
            // Forward pass
            mx::array logits = ForwardPass(lastNTokens, i);
            
            // Repeat penalty 적용
            if (repeatPenalty != 1.0 && !generatedTokens.empty()) {
                int startIdx = std::max(0, static_cast<int>(generatedTokens.size()) - repeatLastN);
                for (int j = startIdx; j < static_cast<int>(generatedTokens.size()); ++j) {
                    int tokenId = generatedTokens[j];
                    if (tokenId >= 0 && tokenId < logits.shape(-1)) {
                        // logits의 특정 인덱스에 penalty 적용
                        // 실제 구현에서는 logits[tokenId]에 접근하여 업데이트해야 함
                        // 여기서는 간단하게 스킵 (실제 구현 필요)
                    }
                }
            }
            
            // 다음 토큰 생성
            mx::array nextTokenArr = GenerateNextToken(logits, temperature, topK, topP, minP);
            EvalArray(nextTokenArr);
            // nextTokenArr는 int32 배열이므로 item<int32_t>()로 읽기
            int nextToken = static_cast<int>(nextTokenArr.item<int32_t>());
            
            generatedTokens.push_back(nextToken);
            lastNTokens.push_back(nextToken);
            
            // Context window 유지
            if (static_cast<int>(lastNTokens.size()) > model_->maxContextLength) {
                lastNTokens.erase(lastNTokens.begin());
            }
            
            // 토큰 디코딩
            std::string tokenStr = Decode({nextToken});
            
            if (hasTokenCallback_) {
                std::string tokenStrCopy = tokenStr;
                auto callback = [](Napi::Env env, Napi::Function jsCallback, std::string* tokenStr) {
                    Napi::Object tokenObj = Napi::Object::New(env);
                    tokenObj.Set("token", Napi::String::New(env, *tokenStr));
                    jsCallback.Call({tokenObj});
                    delete tokenStr;
                };
                onTokenCallback_.BlockingCall(new std::string(tokenStrCopy), callback);
            }
            
            // Stop 토큰 체크
            if (nextToken == model_->eosTokenId) {
                break;
            }
        }
        
        if (hasCompleteCallback_) {
            auto callback = [](Napi::Env env, Napi::Function jsCallback) {
                jsCallback.Call({});
            };
            onCompleteCallback_.BlockingCall(callback);
        }
        
    } catch (const std::exception& e) {
        if (hasErrorCallback_) {
            auto callback = [](Napi::Env env, Napi::Function jsCallback, std::string* errorMsg) {
                jsCallback.Call({Napi::String::New(env, *errorMsg)});
                delete errorMsg;
            };
            onErrorCallback_.BlockingCall(new std::string(e.what()), callback);
        }
    }
    
    isRunning_ = false;
}

Napi::Value MlxInference::Tokenize(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    MlxInference* self = Napi::ObjectWrap<MlxInference>::Unwrap(info.This().As<Napi::Object>());
    
    if (info.Length() < 1 || !info[0].IsString()) {
        Napi::TypeError::New(env, "Expected text string").ThrowAsJavaScriptException();
        return env.Null();
    }
    
    std::string text = info[0].As<Napi::String>().Utf8Value();
    
    std::lock_guard<std::mutex> lock(self->mutex_);
    
    if (!self->model_ || !self->model_->loaded) {
        Napi::Error::New(env, "Model not loaded").ThrowAsJavaScriptException();
        return env.Null();
    }
    
    try {
        std::vector<int> tokens = self->Tokenize(text);
        
        Napi::Array tokenArray = Napi::Array::New(env, tokens.size());
        for (size_t i = 0; i < tokens.size(); ++i) {
            tokenArray[i] = Napi::Number::New(env, tokens[i]);
        }
        
        return tokenArray;
    } catch (const std::exception& e) {
        Napi::Error::New(env, e.what()).ThrowAsJavaScriptException();
        return env.Null();
    }
}

Napi::Value MlxInference::Decode(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    MlxInference* self = Napi::ObjectWrap<MlxInference>::Unwrap(info.This().As<Napi::Object>());
    
    if (info.Length() < 1 || !info[0].IsArray()) {
        Napi::TypeError::New(env, "Expected tokens array").ThrowAsJavaScriptException();
        return env.Null();
    }
    
    Napi::Array tokenArray = info[0].As<Napi::Array>();
    std::vector<int> tokens;
    
    for (uint32_t i = 0; i < tokenArray.Length(); ++i) {
        Napi::Value val = tokenArray[i];
        if (val.IsNumber()) {
            tokens.push_back(val.As<Napi::Number>().Int32Value());
        }
    }
    
    std::lock_guard<std::mutex> lock(self->mutex_);
    
    if (!self->model_ || !self->model_->loaded) {
        Napi::Error::New(env, "Model not loaded").ThrowAsJavaScriptException();
        return env.Null();
    }
    
    try {
        std::string text = self->Decode(tokens);
        return Napi::String::New(env, text);
    } catch (const std::exception& e) {
        Napi::Error::New(env, e.what()).ThrowAsJavaScriptException();
        return env.Null();
    }
}

Napi::Object Init(Napi::Env env, Napi::Object exports) {
    MlxInference::Init(env, exports);
    return exports;
}

NODE_API_MODULE(mlx_server, Init)
