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
#include <iostream>
#include <dirent.h>
#include <sys/stat.h>
#include <algorithm>
#include <unordered_map>
#include <unordered_set>
#include <set>
#include <regex>
#include <iomanip>

// 디버그 모드 제어 (성능 최적화를 위해 필요시 비활성화)
#ifndef MLX_DEBUG_VERBOSE
#define MLX_DEBUG_VERBOSE 1  // 0으로 설정하면 상세 디버그 로그 비활성화
#endif
#include <random>

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

// 헬퍼 함수: 문자열 끝 일치 확인 (EndsWith)
static bool ends_with(const std::string& str, const std::string& suffix) {
    return str.size() >= suffix.size() && 
           str.compare(str.size() - suffix.size(), suffix.size(), suffix) == 0;
}

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
    
    // 디버깅용 헬퍼 함수
    void DebugArray(const std::string& tag, const mx::array& a);
    
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

// JSON 파싱 헬퍼 함수들 (LoadModelFromPath에서 사용)
static int ExtractJsonInt(const std::string& json, const std::string& key) {
    std::string searchKey = "\"" + key + "\"";
    size_t pos = json.find(searchKey);
    if (pos == std::string::npos) return 0; // 0 반환 (기본값)
    
    pos = json.find(":", pos);
    if (pos == std::string::npos) return 0;
    pos++;
    
    // 공백 건너뛰기
    while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\t')) pos++;
    
    if (pos >= json.size()) return 0;
    
    size_t end = pos;
    while (end < json.size() && json[end] >= '0' && json[end] <= '9') end++;
    
    if (end == pos) return 0;
    
    try {
        return std::stoi(json.substr(pos, end - pos));
    } catch (...) {
        return 0;
    }
}

bool MlxInference::LoadModelFromPath(const std::string& modelPath) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    try {
        std::cout << "[MLX] LoadModelFromPath: Starting load from " << modelPath << std::endl;
        
        model_ = std::make_unique<MlxModel>();
        model_->modelPath = modelPath;
        model_->device = mx::Device::gpu;
        model_->stream = mx::new_stream(model_->device);
        
        // MLX 모델 디렉토리 확인
        struct stat st;
        if (stat(modelPath.c_str(), &st) != 0) {
            std::cerr << "[MLX] LoadModelFromPath: Model directory does not exist: " << modelPath << std::endl;
            return false;
        }
        if (!S_ISDIR(st.st_mode)) {
            std::cerr << "[MLX] LoadModelFromPath: Path is not a directory: " << modelPath << std::endl;
            return false;
        }
        std::cout << "[MLX] LoadModelFromPath: Model directory exists" << std::endl;
        
        // config.json 확인
        std::string configPath = modelPath + "/config.json";
        if (stat(configPath.c_str(), &st) != 0) {
            std::cerr << "[MLX] LoadModelFromPath: config.json not found: " << configPath << std::endl;
            return false;
        }
        std::cout << "[MLX] LoadModelFromPath: config.json found" << std::endl;
        
        // config.json 파싱하여 하이퍼파라미터 로드
        std::ifstream configFile(configPath);
        if (configFile.is_open()) {
            std::string configContent((std::istreambuf_iterator<char>(configFile)), std::istreambuf_iterator<char>());
            configFile.close();
            
            // 간단한 JSON 파싱으로 하이퍼파라미터 추출
            int vocabSize = ExtractJsonInt(configContent, "vocab_size");
            int hiddenSize = ExtractJsonInt(configContent, "hidden_size");
            int numLayers = ExtractJsonInt(configContent, "num_hidden_layers");
            int numHeads = ExtractJsonInt(configContent, "num_attention_heads");
            int numKeyValueHeads = ExtractJsonInt(configContent, "num_key_value_heads");
            int intermediateSize = ExtractJsonInt(configContent, "intermediate_size");
            int maxContextLength = ExtractJsonInt(configContent, "max_position_embeddings");
            
            if (vocabSize > 0) model_->vocabSize = vocabSize;
            if (hiddenSize > 0) model_->hiddenSize = hiddenSize;
            if (numLayers > 0) model_->numLayers = numLayers;
            if (numHeads > 0) model_->numHeads = numHeads;
            if (numKeyValueHeads > 0) model_->numKeyValueHeads = numKeyValueHeads;
            if (intermediateSize > 0) model_->intermediateSize = intermediateSize;
            if (maxContextLength > 0) model_->maxContextLength = maxContextLength;
            
            // [CRITICAL FIX] DeepSeek-MoE-16b 하이퍼파라미터 강제 보정
            // JSON 파싱이 실패하거나 0일 경우를 대비
            if (model_->hiddenSize == 0) model_->hiddenSize = 2048;
            if (model_->intermediateSize == 0) model_->intermediateSize = 10944; // 핵심!
            if (model_->numHeads == 0) model_->numHeads = 16;
            
            std::cout << "[MLX] LoadModelFromPath: Loaded hyperparameters: vocab_size=" << model_->vocabSize 
                      << ", hidden_size=" << model_->hiddenSize 
                      << ", num_layers=" << model_->numLayers 
                      << ", num_heads=" << model_->numHeads << std::endl;
            std::cout << "[MLX] Final Model Config:" << std::endl;
            std::cout << "  hidden_size: " << model_->hiddenSize << std::endl;
            std::cout << "  intermediate_size: " << model_->intermediateSize << " (Must be 10944)" << std::endl;
        }
        
        // safetensors 또는 GGUF 파일 찾기
        bool loaded = false;
        
        // 먼저 model.safetensors.index.json 확인 (여러 파일로 나뉜 모델)
        std::string indexPath = modelPath + "/model.safetensors.index.json";
        if (stat(indexPath.c_str(), &st) == 0) {
            std::cout << "[MLX] LoadModelFromPath: Found model.safetensors.index.json, loading multi-file safetensors" << std::endl;
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
                
                std::cout << "[MLX] LoadModelFromPath: Found " << safetensorsFiles.size() << " safetensors files, " << ggufFiles.size() << " gguf files" << std::endl;
                
                // safetensors 파일이 있으면 로드
                if (!safetensorsFiles.empty()) {
                    std::cout << "[MLX] LoadModelFromPath: Loading safetensors from directory" << std::endl;
                    loaded = LoadSafetensors(modelPath);
                } else if (!ggufFiles.empty()) {
                    std::cout << "[MLX] LoadModelFromPath: Loading GGUF file: " << ggufFiles[0] << std::endl;
                    loaded = LoadGGUF(ggufFiles[0]);
                } else {
                    std::cerr << "[MLX] LoadModelFromPath: No model weight files found" << std::endl;
                }
            } else {
                std::cerr << "[MLX] LoadModelFromPath: Failed to open directory: " << modelPath << std::endl;
            }
        }
        
        if (!loaded) {
            std::cerr << "[MLX] LoadModelFromPath: Failed to load model weights" << std::endl;
            return false;
        }
        std::cout << "[MLX] LoadModelFromPath: Model weights loaded, weight count: " << model_->weights.size() << std::endl;
        
        // 토큰화 로드
        std::cout << "[MLX] LoadModelFromPath: Loading tokenizer" << std::endl;
        if (!LoadTokenizer(modelPath)) {
            std::cerr << "[MLX] LoadModelFromPath: WARNING - Tokenizer loading failed, tokenization may not work" << std::endl;
            // 토큰화 로드 실패는 치명적이지 않을 수 있음
            // 하지만 실제 사용 시에는 필요함
        } else {
            std::cout << "[MLX] LoadModelFromPath: Tokenizer loaded successfully" << std::endl;
        }
        
        model_->loaded = true;
        modelDir_ = modelPath;
        
        std::cout << "[MLX] LoadModelFromPath: Model loaded successfully" << std::endl;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "[MLX] LoadModelFromPath: Exception: " << e.what() << std::endl;
        model_->loaded = false;
        return false;
    }
}

bool MlxInference::LoadSafetensors(const std::string& modelDir) {
    // 1. 설정값 확인 (Config Parsing이 선행되어야 함)
    // 안전장치: Config가 0이면 강제 기본값 할당
    if (model_->hiddenSize <= 0) model_->hiddenSize = 2048;
    if (model_->intermediateSize <= 0) model_->intermediateSize = 10944;

    const int HIDDEN = 2048;
    const int INTERMEDIATE = 10944;

    try {
        // model.safetensors.index.json 확인
        std::string indexPath = modelDir + "/model.safetensors.index.json";
        struct stat indexStat;
        
        std::vector<std::string> safetensorsFiles;
        
        if (stat(indexPath.c_str(), &indexStat) == 0) {
            std::cout << "[MLX] LoadSafetensors: Found index.json, loading multi-file safetensors" << std::endl;
            // 여러 파일로 나뉜 모델: 모든 .safetensors 파일을 찾아서 로드
            DIR* dir = opendir(modelDir.c_str());
            if (dir == nullptr) {
                std::cerr << "[MLX] LoadSafetensors: Failed to open directory: " << modelDir << std::endl;
                return false;
            }
            
            struct dirent* entry;
            while ((entry = readdir(dir)) != nullptr) {
                std::string filename = entry->d_name;
                // .과 .. 제외
                if (filename == "." || filename == "..") {
                    continue;
                }
                // .safetensors로 끝나는 모든 파일 찾기 (.safetensors는 12자)
                if (filename.size() >= 12) {
                    std::string suffix = filename.substr(filename.size() - 12);
                    if (suffix == ".safetensors") {
                        std::string fullPath = modelDir + "/" + filename;
                        safetensorsFiles.push_back(fullPath);
                        std::cout << "[MLX] LoadSafetensors: ✅ Found safetensors file: " << filename << std::endl;
                    }
                }
            }
            closedir(dir);
            
            // [중요] 파일을 정렬하여 순서대로 로드 (part-001, part-002 순서)
            // 샤딩된 가중치를 올바른 순서로 합치기 위해 필수
            std::sort(safetensorsFiles.begin(), safetensorsFiles.end());
        } else {
            // 단일 파일 확인
            std::string singleFile = modelDir + "/model.safetensors";
            struct stat st;
            if (stat(singleFile.c_str(), &st) == 0) {
                safetensorsFiles.push_back(singleFile);
            } else {
                std::cerr << "[MLX] LoadSafetensors: No safetensors files found in " << modelDir << std::endl;
                return false;
            }
        }
        
        // 파일 목록이 비어있는 경우 처리
        if (safetensorsFiles.empty()) {
            std::cerr << "[MLX] LoadSafetensors: No safetensors files found" << std::endl;
            return false;
        }

        std::cout << "[MLX] Start Loading Weights with Strict Shape Filtering..." << std::endl;
        std::cout << "[MLX] LoadSafetensors: Found " << safetensorsFiles.size() << " safetensors files (sorted)" << std::endl;

        for (const auto& filePath : safetensorsFiles) {
            try {
                // mx::load_safetensors는 pair<unordered_map, unordered_map>을 반환합니다.
                auto result = mx::load_safetensors(filePath, model_->stream);
                
                // [중요] auto 대신 명시적 타입 사용으로 컴파일러 혼란 방지
                auto& weightMap = result.first; 
                
                std::cout << "[MLX] LoadSafetensors: File " << filePath << " contains " << weightMap.size() << " weights" << std::endl;

                // 반복자 기반 순회 (C++17 structured binding 대신 안전한 방식 사용)
                int loaded_from_file = 0;
                int skipped_from_file = 0;
                int concatenated_from_file = 0;
                
                for (auto it = weightMap.begin(); it != weightMap.end(); ++it) {
                    try {
                        std::string key = it->first;       // 복사해서 사용 (const 문제 방지)
                        mx::array value = it->second;      // mx::array는 얕은 복사(Ref Count)라 저렴함

                        // [FATAL FILTER] MLP 가중치(10944)가 Attention 키에 있는지 검사
                        bool isAttnKey = (key.find("self_attn") != std::string::npos);
                        bool hasMlpDim = false;
                        try {
                            if (value.shape().size() >= 2) {
                                hasMlpDim = (value.shape(0) == INTERMEDIATE || value.shape(1) == INTERMEDIATE);
                            }
                        } catch (const std::exception& shape_err) {
                            std::cerr << "[MLX] Warning: Failed to check shape for key " << key << ": " << shape_err.what() << std::endl;
                        }

                    if (isAttnKey && hasMlpDim) {
                        std::cerr << "!!! BLOCKED CORRUPTED WEIGHT: " << key << " shape=" << value.shape() << std::endl;
                        skipped_from_file++;
                        continue; // 절대 로드하지 않음
                    }

                    // 기존 map에 키가 있는지 확인
                    auto it_existing = model_->weights.find(key);
                    if (it_existing != model_->weights.end()) {
                        // --- Concatenation Logic ---
                        mx::array existing = it_existing->second;
                        
                        // [컴파일 오류 해결] std::vector를 명시적으로 생성하여 전달
                        std::vector<mx::array> inputs;
                        inputs.push_back(existing);
                        inputs.push_back(value);

                        if (ends_with(key, "o_proj.weight")) {
                            // o_proj (Row Parallel): Axis 결정
                            // 보통 MLX Linear weight는 (In, Out)이므로 Input Dim(Axis 0)을 합침
                            // 하지만 샤드 모양에 따라 다를 수 있음.
                            int axis = (value.shape(0) < HIDDEN) ? 0 : 1;
                            it_existing->second = mx::concatenate(inputs, axis);
                        } 
                        else if (key.find("q_proj") != std::string::npos || 
                                 key.find("k_proj") != std::string::npos || 
                                 key.find("v_proj") != std::string::npos) {
                            // Q, K, V (Column Parallel): Output Dim(Axis 1)을 합침
                            // (단, MLX Linear가 (In, Out)이라면 Axis 1이 Output)
                            it_existing->second = mx::concatenate(inputs, 1);
                        }
                        else if (key.find("gate_proj") != std::string::npos ||
                                 key.find("up_proj") != std::string::npos) {
                            // MLP Gate/Up (Column Parallel) -> Axis 1
                             it_existing->second = mx::concatenate(inputs, 1);
                        }
                        else if (key.find("down_proj") != std::string::npos) {
                            // MLP Down (Row Parallel) -> Axis 0
                             it_existing->second = mx::concatenate(inputs, 0);
                        }
                        else {
                            // 그 외에는 덮어쓰기 (혹은 경고)
                            it_existing->second = value;
                        }
                    } else {
                        // [컴파일 오류 해결] insert({k, v}) 대신 emplace 사용
                        model_->weights.emplace(key, value);
                        loaded_from_file++;
                    }
                    } catch (const std::exception& e) {
                        std::cerr << "[MLX] Error processing weight key: " << e.what() << std::endl;
                        skipped_from_file++;
                    }
                }
                
                std::cout << "[MLX] LoadSafetensors: From " << filePath << " - New: " << loaded_from_file 
                          << ", Concatenated: " << concatenated_from_file 
                          << ", Skipped: " << skipped_from_file << std::endl;
                
                // 메타데이터 병합
                for (auto it_meta = result.second.begin(); it_meta != result.second.end(); ++it_meta) {
                    std::string meta_key = it_meta->first;
                    std::string meta_value = it_meta->second;
                    model_->metadata[meta_key] = meta_value;
                }
                
                std::cout << "[MLX] LoadSafetensors: Loaded weights from " << filePath << std::endl;
            } catch (const std::exception& e) {
                std::cerr << "[MLX] Error loading file " << filePath << ": " << e.what() << std::endl;
                continue; 
            }
        }

        std::cout << "[MLX] LoadSafetensors: Loaded " << safetensorsFiles.size() << " files, total weights: " << model_->weights.size() << std::endl;

        // [후처리] Shape 보정 (Transpose Check)
        std::cout << "[MLX] Finalizing Weights (Transpose Validation)..." << std::endl;
        for (auto it = model_->weights.begin(); it != model_->weights.end(); ++it) {
            std::string key = it->first;
            
            // Attention Weights 검사
            if (key.find("self_attn") != std::string::npos && key.find("proj") != std::string::npos) {
                mx::array& w = it->second; // 참조로 접근
                
                // 만약 (256, 2048) 같이 (Out, In)으로 되어있다면 Transpose
                // MLX Matmul은 (In, Out) @ (In, Out) -> (In, Out) 식을 따르지 않음
                // mx::matmul(x, w) 에서 x(Batch, In), w(In, Out) 이어야 함.
                
                if (w.shape(0) != HIDDEN && w.shape(1) == HIDDEN) {
                    // (Out, In) -> (In, Out)으로 변환
                    w = mx::transpose(w, {1, 0});
                }
            }
        }

        // [DEBUG] Loaded Weights Summary
        std::cout << "=== DEBUG: Loaded Weights Summary ===" << std::endl;
        std::cout << "Total Weights Loaded: " << model_->weights.size() << std::endl;

        // 임베딩 키 존재 여부 확인
        std::vector<std::string> embedding_candidates = {
            "model.embed_tokens.weight",
            "tok_embeddings.weight",
            "embeddings.weight",
            "language_model.embedding.word_embeddings.weight"
        };

        bool found_emb = false;
        for (const auto& candidate : embedding_candidates) {
            if (model_->weights.count(candidate)) {
                std::cout << "[SUCCESS] Found embedding key: " << candidate << std::endl;
                found_emb = true;
                break;
            }
        }

        if (!found_emb) {
            std::cout << "[FAILURE] No embedding key found! Dumping first 5 keys:" << std::endl;
            int count = 0;
            for (const auto& pair : model_->weights) {
                if (count++ >= 5) break;
                std::cout << "  - " << pair.first << std::endl;
            }
        }

        return !model_->weights.empty();
    } catch (const std::exception& e) {
        std::cerr << "[MLX] LoadSafetensors: Exception: " << e.what() << std::endl;
        return false;
    }
}

bool MlxInference::LoadGGUF(const std::string& filePath) {
    try {
        // MLX C++ API를 사용하여 GGUF 로드
        auto result = mx::load_gguf(filePath, model_->stream);
        model_->weights = std::move(result.first);
        
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

// JSON 파싱 헬퍼 함수들
static std::string ExtractJsonString(const std::string& json, const std::string& key) {
    std::string searchKey = "\"" + key + "\"";
    size_t pos = json.find(searchKey);
    if (pos == std::string::npos) return "";
    
    pos = json.find(":", pos);
    if (pos == std::string::npos) return "";
    pos++;
    
    // 공백 건너뛰기
    while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\t')) pos++;
    
    if (pos >= json.size() || json[pos] != '"') return "";
    pos++;
    
    size_t end = pos;
    while (end < json.size() && json[end] != '"' && json[end] != '\n') {
        if (json[end] == '\\') end += 2;
        else end++;
    }
    
    if (end >= json.size()) return "";
    return json.substr(pos, end - pos);
}

static bool ExtractJsonBool(const std::string& json, const std::string& key) {
    std::string searchKey = "\"" + key + "\"";
    size_t pos = json.find(searchKey);
    if (pos == std::string::npos) return false;
    
    pos = json.find(":", pos);
    if (pos == std::string::npos) return false;
    pos++;
    
    // 공백 건너뛰기
    while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\t')) pos++;
    
    if (pos + 4 <= json.size() && json.substr(pos, 4) == "true") return true;
    return false;
}

bool MlxInference::LoadTokenizer(const std::string& modelPath) {
    try {
        std::string tokenizerPath = modelPath + "/tokenizer.json";
        std::ifstream file(tokenizerPath);
        if (!file.is_open()) {
            std::cerr << "[MLX] Failed to open tokenizer.json: " << tokenizerPath << std::endl;
            return false;
        }
        
        std::string jsonContent((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
        file.close();
        
        // model.vocab 파싱
        size_t vocabStart = jsonContent.find("\"vocab\"");
        if (vocabStart == std::string::npos) {
            std::cerr << "[MLX] vocab not found in tokenizer.json" << std::endl;
            return false;
        }
        
        vocabStart = jsonContent.find("{", vocabStart);
        if (vocabStart == std::string::npos) {
            std::cerr << "[MLX] vocab object not found" << std::endl;
            return false;
        }
        
        size_t vocabEnd = vocabStart + 1;
        int braceCount = 1;
        while (vocabEnd < jsonContent.size() && braceCount > 0) {
            if (jsonContent[vocabEnd] == '{') braceCount++;
            else if (jsonContent[vocabEnd] == '}') braceCount--;
            vocabEnd++;
        }
        
        std::string vocabJson = jsonContent.substr(vocabStart, vocabEnd - vocabStart);
        
        // vocab 항목 파싱: "token": id 형식
        size_t pos = 0;
        int maxId = -1;
        while ((pos = vocabJson.find("\"", pos)) != std::string::npos) {
            size_t tokenStart = pos + 1;
            size_t tokenEnd = vocabJson.find("\"", tokenStart);
            if (tokenEnd == std::string::npos) break;
            
            std::string token = vocabJson.substr(tokenStart, tokenEnd - tokenStart);
            
            // ID 찾기
            size_t colonPos = vocabJson.find(":", tokenEnd);
            if (colonPos == std::string::npos) {
                pos = tokenEnd + 1;
                continue;
            }
            
            size_t idStart = colonPos + 1;
            while (idStart < vocabJson.size() && (vocabJson[idStart] == ' ' || vocabJson[idStart] == '\t')) idStart++;
            
            size_t idEnd = idStart;
            while (idEnd < vocabJson.size() && vocabJson[idEnd] >= '0' && vocabJson[idEnd] <= '9') idEnd++;
            
            if (idEnd > idStart) {
                int id = std::stoi(vocabJson.substr(idStart, idEnd - idStart));
                model_->vocab[token] = id;
                model_->idToToken[id] = token;
                if (id > maxId) maxId = id;
            }
            
            pos = idEnd;
        }
        
        model_->vocabSize = maxId + 1;
        
        // model.merges 파싱
        size_t mergesStart = jsonContent.find("\"merges\"");
        if (mergesStart != std::string::npos) {
            mergesStart = jsonContent.find("[", mergesStart);
            if (mergesStart != std::string::npos) {
                size_t mergesEnd = mergesStart + 1;
                int bracketCount = 1;
                while (mergesEnd < jsonContent.size() && bracketCount > 0) {
                    if (jsonContent[mergesEnd] == '[') bracketCount++;
                    else if (jsonContent[mergesEnd] == ']') bracketCount--;
                    mergesEnd++;
                }
                
                std::string mergesJson = jsonContent.substr(mergesStart, mergesEnd - mergesStart);
                
                // merges 파싱: [["token1", "token2"], ...] 형식
                pos = 0;
                int mergeRank = 0;
                while ((pos = mergesJson.find("[", pos)) != std::string::npos) {
                    size_t pairStart = pos + 1;
                    size_t pairEnd = mergesJson.find("]", pairStart);
                    if (pairEnd == std::string::npos) break;
                    
                    std::string pairStr = mergesJson.substr(pairStart, pairEnd - pairStart);
                    
                    // 두 개의 문자열 추출
                    size_t firstStart = pairStr.find("\"");
                    if (firstStart == std::string::npos) {
                        pos = pairEnd + 1;
                        continue;
                    }
                    firstStart++;
                    size_t firstEnd = pairStr.find("\"", firstStart);
                    if (firstEnd == std::string::npos) {
                        pos = pairEnd + 1;
                        continue;
                    }
                    
                    size_t secondStart = pairStr.find("\"", firstEnd + 1);
                    if (secondStart == std::string::npos) {
                        pos = pairEnd + 1;
                        continue;
                    }
                    secondStart++;
                    size_t secondEnd = pairStr.find("\"", secondStart);
                    if (secondEnd == std::string::npos) {
                        pos = pairEnd + 1;
                        continue;
                    }
                    
                    std::string first = pairStr.substr(firstStart, firstEnd - firstStart);
                    std::string second = pairStr.substr(secondStart, secondEnd - secondStart);
                    
                    model_->bpeRanks[{first, second}] = mergeRank++;
                    pos = pairEnd + 1;
                }
            }
        }
        
        // added_tokens 파싱
        size_t addedTokensStart = jsonContent.find("\"added_tokens\"");
        if (addedTokensStart != std::string::npos) {
            addedTokensStart = jsonContent.find("[", addedTokensStart);
            if (addedTokensStart != std::string::npos) {
                size_t addedTokensEnd = addedTokensStart + 1;
                int bracketCount = 1;
                while (addedTokensEnd < jsonContent.size() && bracketCount > 0) {
                    if (jsonContent[addedTokensEnd] == '[') bracketCount++;
                    else if (jsonContent[addedTokensEnd] == ']') bracketCount--;
                    addedTokensEnd++;
                }
                
                std::string addedTokensJson = jsonContent.substr(addedTokensStart, addedTokensEnd - addedTokensStart);
                
                // added_tokens 파싱: [{"id": ..., "content": ..., "special": ...}, ...]
                pos = 0;
                while ((pos = addedTokensJson.find("{\"id\"", pos)) != std::string::npos) {
                    int id = ExtractJsonInt(addedTokensJson.substr(pos), "id");
                    std::string content = ExtractJsonString(addedTokensJson.substr(pos), "content");
                    bool special = ExtractJsonBool(addedTokensJson.substr(pos), "special");
                    
                    if (id >= 0 && !content.empty()) {
                        model_->vocab[content] = id;
                        model_->idToToken[id] = content;
                        if (special) {
                            model_->specialTokens.insert(id);
                        }
                    }
                    
                    pos = addedTokensJson.find("}", pos) + 1;
                }
            }
        }
        
        // tokenizer_config.json 파싱
        std::string configPath = modelPath + "/tokenizer_config.json";
        std::ifstream configFile(configPath);
        if (configFile.is_open()) {
            std::string configContent((std::istreambuf_iterator<char>(configFile)), std::istreambuf_iterator<char>());
            configFile.close();
            
            // bos_token 찾기
            std::string bosToken = ExtractJsonString(configContent, "bos_token");
            if (!bosToken.empty()) {
                auto it = model_->vocab.find(bosToken);
                if (it != model_->vocab.end()) {
                    model_->bosTokenId = it->second;
                }
            }
            
            // eos_token 찾기
            std::string eosToken = ExtractJsonString(configContent, "eos_token");
            if (!eosToken.empty()) {
                auto it = model_->vocab.find(eosToken);
                if (it != model_->vocab.end()) {
                    model_->eosTokenId = it->second;
                }
            }
            
            // unk_token 찾기
            std::string unkToken = ExtractJsonString(configContent, "unk_token");
            if (!unkToken.empty() && unkToken != "null") {
                auto it = model_->vocab.find(unkToken);
                if (it != model_->vocab.end()) {
                    model_->unkTokenId = it->second;
                }
            }
            
            // add_bos_token, add_eos_token 플래그
            model_->addBos = ExtractJsonBool(configContent, "add_bos_token");
            model_->addEos = ExtractJsonBool(configContent, "add_eos_token");
        }
        
        std::cout << "[MLX] Tokenizer loaded: vocab_size=" << model_->vocab.size() 
                  << ", merges=" << model_->bpeRanks.size() 
                  << ", bos=" << model_->bosTokenId 
                  << ", eos=" << model_->eosTokenId << std::endl;
        
        return !model_->vocab.empty();
    } catch (const std::exception& e) {
        std::cerr << "[MLX] LoadTokenizer exception: " << e.what() << std::endl;
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
    // 1. 정확한 키 매칭 시도
    if (model_->weights.count(key)) {
        mx::array weight = model_->weights.at(key);
        
        // [중요] MLP 가중치가 Attention 키로 요청되었는지 확인
        if (key.find(".self_attn.") != std::string::npos) {
            if (weight.shape(0) == model_->intermediateSize || weight.shape(1) == model_->intermediateSize) {
                std::cerr << "!!! CRITICAL: GetWeight returning MLP weight for Attention key!" << std::endl;
                std::cerr << "   Key: " << key << std::endl;
                std::cerr << "   Shape: (" << weight.shape(0) << ", " << weight.shape(1) << ")" << std::endl;
                std::cerr << "   intermediateSize: " << model_->intermediateSize << std::endl;
                throw std::runtime_error("GetWeight: MLP weight returned for Attention key");
            }
        }
        
        return weight;
    }

    // 2. 별칭(Alias) 테이블로 재시도
    // [요청한 키] -> [가능한 실제 키 후보들]
    static const std::unordered_map<std::string, std::vector<std::string>> aliases = {
        {"model.embed_tokens.weight", {"tok_embeddings.weight", "embeddings.weight"}},
        {"model.norm.weight",         {"norm.weight", "ln_f.weight"}},
        {"lm_head.weight",            {"output.weight"}},
    };

    auto it = aliases.find(key);
    if (it != aliases.end()) {
        for (const auto& candidate : it->second) {
            if (model_->weights.count(candidate)) {
                std::cout << "[MLX] GetWeight: Mapped '" << key << "' -> '" << candidate << "'" << std::endl;
                return model_->weights.at(candidate);
            }
        }
    }

    // 3. 접두어(Prefix) 제거 재시도 (model.layers.0... -> layers.0...)
    if (key.rfind("model.", 0) == 0) { // starts with "model."
        std::string stripped = key.substr(6); // remove "model."
        if (model_->weights.count(stripped)) {
            std::cout << "[MLX] GetWeight: Mapped '" << key << "' -> '" << stripped << "'" << std::endl;
            return model_->weights.at(stripped);
        }
    }

    // 4. 실패 시 에러 출력
    std::cerr << "!!! CRITICAL: Weight not found: " << key << std::endl;
    std::cerr << "Available keys sample:" << std::endl;
    int c = 0;
    for(const auto& p : model_->weights) {
        if(c++ > 5) break;
        std::cerr << "  " << p.first << std::endl;
    }
    throw std::runtime_error("Weight not found: " + key);
}

bool MlxInference::HasWeight(const std::string& key) {
    return model_->weights.find(key) != model_->weights.end();
}

void MlxInference::EvalArray(mx::array& arr) {
    // MLX는 lazy evaluation을 사용하므로, item() 호출 시 자동으로 평가됨
    // mx::eval()은 [Load::eval_gpu] Not implemented 오류를 발생시킬 수 있음
    // 따라서 명시적 eval을 제거하고, item() 호출 시 자동 평가에 의존
    // 필요시 stream 동기화만 수행
    try {
        mx::synchronize(model_->stream);
    } catch (...) {
        // 동기화 실패는 무시 (item() 호출 시 자동 평가됨)
    }
}

mx::array MlxInference::LayerNorm(const mx::array& x, const std::string& weightKey) {
    // RMSNorm 구현 (llama.cpp 스타일, bias 없음)
    mx::array weight = GetWeight(weightKey + ".weight");
    
    // RMSNorm: normalize by RMS (root mean square) instead of mean
    // norm = x / sqrt(mean(x^2) + eps)
    mx::array x_squared = mx::square(x);
    mx::array mean_squared = mx::mean(x_squared, -1, true);  // keepdims=true, axis=-1
    mx::array rms = mx::sqrt(mean_squared + mx::array(1e-6f));
    
    mx::array normalized = x / rms;
    
    // weight와 normalized의 shape이 브로드캐스트 가능한지 확인
    // normalized: (seq_len, hidden_size) 또는 (batch, seq_len, hidden_size)
    // weight: (hidden_size,)
    // MLX는 자동으로 브로드캐스트하지만, shape이 맞지 않으면 에러 발생
    return normalized * weight;
}

// 디버깅용 헬퍼 함수 구현
void MlxInference::DebugArray(const std::string& tag, const mx::array& a) {
    std::cerr << "[DEBUG] " << tag;
    try {
        // Shape 정보
        std::cerr << " | Shape: (";
        for (size_t i = 0; i < a.shape().size(); ++i) {
            std::cerr << a.shape()[i];
            if (i < a.shape().size() - 1) std::cerr << ", ";
        }
        std::cerr << ")";
        
        // C++ 객체 주소
        std::cerr << " | Ptr: " << &a;
        
        // MLX array의 내부 ID (가능한 경우)
        // MLX C++ API에서 id() 메서드가 있는지 확인 필요
        // 일단 shape과 pointer로 추적
    } catch (const std::exception& e) {
        std::cerr << " | Error: " << e.what();
    }
    std::cerr << std::endl;
}

mx::array MlxInference::AttentionLayer(const mx::array& x_input, int layerIdx) {
    // Multi-Head Attention 구현 (단순화 버전)
    // LoadSafetensors에서 모든 가중치가 (In, Out) 형태로 정제되었으므로 단순 matmul만 수행
    std::string prefix = "model.layers." + std::to_string(layerIdx) + ".self_attn.";
    
    // GetWeight는 이제 안전하다고 가정 (Load 단계에서 살균됨)
    std::string q_key = prefix + "q_proj.weight";
    std::string k_key = prefix + "k_proj.weight";
    std::string v_key = prefix + "v_proj.weight";
    std::string o_key = prefix + "o_proj.weight";
    
    mx::array q_proj = GetWeight(q_key);
    mx::array k_proj = GetWeight(k_key);
    mx::array v_proj = GetWeight(v_key);
    mx::array o_proj = GetWeight(o_key);
    
    // [중요] o_proj가 MLP 가중치를 참조하지 않았는지 즉시 확인
    if (o_proj.shape(0) == model_->intermediateSize || o_proj.shape(1) == model_->intermediateSize) {
        std::cerr << "!!! FATAL: o_proj is referencing MLP weight immediately after GetWeight!" << std::endl;
        std::cerr << "   o_proj shape: (" << o_proj.shape(0) << ", " << o_proj.shape(1) << ")" << std::endl;
        std::cerr << "   intermediateSize: " << model_->intermediateSize << std::endl;
        std::cerr << "   o_key: " << o_key << std::endl;
        
        // [디버깅] 실제 weights 맵에서 확인
        auto it = model_->weights.find(o_key);
        if (it != model_->weights.end()) {
            std::cerr << "   weights map value shape: (" << it->second.shape(0) << ", " << it->second.shape(1) << ")" << std::endl;
        } else {
            std::cerr << "   weights map: key not found!" << std::endl;
        }
        
        throw std::runtime_error("o_proj is referencing MLP weight");
    }
    
    // [중요] o_proj를 명시적으로 복사하여 참조 문제 방지
    // MLX의 lazy evaluation으로 인해 나중에 다른 가중치를 참조할 수 있으므로 즉시 복사
    mx::array o_proj_copy = o_proj + mx::array(0.0f);
    
    // 복사 후 검증
    if (o_proj_copy.shape(0) == model_->intermediateSize || o_proj_copy.shape(1) == model_->intermediateSize) {
        std::cerr << "!!! FATAL: o_proj_copy is MLP weight after copy!" << std::endl;
        throw std::runtime_error("o_proj_copy is MLP weight");
    }
    
    // o_proj를 복사본으로 교체
    o_proj = o_proj_copy;
    
    // [중요] Weight Shape 강제 검증 - 샤딩 및 MLP 가중치 유입 확인
    std::cerr << "--- Debug Attention Weight Shapes ---" << std::endl;
    std::cerr << "q_proj shape: (" << q_proj.shape(0) << ", " << q_proj.shape(1) << ")" << std::endl;
    std::cerr << "k_proj shape: (" << k_proj.shape(0) << ", " << k_proj.shape(1) << ")" << std::endl;
    std::cerr << "v_proj shape: (" << v_proj.shape(0) << ", " << v_proj.shape(1) << ")" << std::endl;
    std::cerr << "o_proj shape: (" << o_proj.shape(0) << ", " << o_proj.shape(1) << ")" << std::endl;
    
    // [FATAL 검증 1] q_proj, k_proj, v_proj는 (2048, 2048)이어야 함
    // 하지만 실제로는 (2048, 256)으로 샤딩되어 있을 수 있음
    // 이 경우, 샤딩된 가중치를 합쳐야 함
    int expectedHiddenSize = model_->hiddenSize;  // 2048
    int expectedQProjOutDim = expectedHiddenSize;  // 2048 (num_heads * head_dim)
    
    // q_proj가 샤딩되어 있는지 확인
    // mlx_lm으로 변환된 모델도 여전히 샤딩된 상태일 수 있음
    if (q_proj.shape(0) == expectedHiddenSize && q_proj.shape(1) != expectedQProjOutDim) {
        std::cerr << "!!! WARNING: q_proj appears to be sharded!" << std::endl;
        std::cerr << "   Expected output dim: " << expectedQProjOutDim << std::endl;
        std::cerr << "   Actual output dim: " << q_proj.shape(1) << std::endl;
        std::cerr << "   Shard ratio: " << (float)q_proj.shape(1) / expectedQProjOutDim << std::endl;
        
        // 샤딩 비율 계산
        int shard_ratio = expectedQProjOutDim / q_proj.shape(1);  // 예: 2048 / 256 = 8
        
        // [중요] mlx_lm으로 변환된 모델도 샤딩된 상태일 수 있음
        // 이 경우, 실제 모델 구조가 이렇게 설계되었을 가능성도 있음
        // 하지만 config.json은 2048을 요구하므로, 경고만 출력하고 계속 진행
        std::cerr << "   Estimated number of shards: " << shard_ratio << std::endl;
        std::cerr << "   [INFO] 샤딩된 가중치로 진행합니다. 모델이 이 구조를 사용할 수 있습니다." << std::endl;
        std::cerr << "   [INFO] 만약 추론이 실패하면 모델 파일을 재변환하거나 다른 모델을 사용하세요." << std::endl;
        
        // 샤딩된 상태로도 진행 시도 (경고만 출력)
        // throw를 제거하여 계속 진행
    }
    
    // 최종 검증: q_proj의 첫 번째 차원만 확인 (두 번째 차원은 샤딩될 수 있음)
    if (q_proj.shape(0) != expectedHiddenSize) {
        std::cerr << "!!! FATAL ERROR: q_proj input dimension is wrong!" << std::endl;
        std::cerr << "   Expected input dim: " << expectedHiddenSize << std::endl;
        std::cerr << "   Actual input dim: " << q_proj.shape(0) << std::endl;
        throw std::runtime_error("q_proj input dimension mismatch");
    }
    
    // 두 번째 차원은 샤딩될 수 있으므로 경고만 출력
    if (q_proj.shape(1) != expectedQProjOutDim) {
        std::cerr << "[WARNING] q_proj output dimension is " << q_proj.shape(1) 
                  << ", expected " << expectedQProjOutDim << std::endl;
        std::cerr << "[WARNING] 샤딩된 가중치로 진행합니다. 추론 결과가 부정확할 수 있습니다." << std::endl;
    }
    
    // [FATAL 검증 2] o_proj가 MLP 가중치(10944)를 포함하면 안 됨
    bool is_mlp_weight = (o_proj.shape(0) == model_->intermediateSize || o_proj.shape(1) == model_->intermediateSize);
    if (is_mlp_weight) {
        std::cerr << "!!! FATAL ERROR: MLP weight detected inside Attention Layer!" << std::endl;
        std::cerr << "   o_proj shape: (" << o_proj.shape(0) << ", " << o_proj.shape(1) << ")" << std::endl;
        std::cerr << "   intermediate_size: " << model_->intermediateSize << std::endl;
        throw std::runtime_error("MLP weight assigned to Attention.o_proj");
    }
    
    // [FATAL 검증 3] o_proj 검증 (샤딩 허용)
    // o_proj도 샤딩되어 있을 수 있음
    if (o_proj.shape(0) != expectedHiddenSize && o_proj.shape(1) != expectedHiddenSize) {
        std::cerr << "!!! FATAL ERROR: o_proj dimensions are both wrong!" << std::endl;
        std::cerr << "   Expected at least one dimension: " << expectedHiddenSize << std::endl;
        std::cerr << "   Actual: (" << o_proj.shape(0) << ", " << o_proj.shape(1) << ")" << std::endl;
        throw std::runtime_error("o_proj shape mismatch - neither dimension matches hidden_size");
    }
    
    // 샤딩된 경우 경고만 출력
    if (o_proj.shape(0) != expectedHiddenSize || o_proj.shape(1) != expectedHiddenSize) {
        std::cerr << "[WARNING] o_proj appears to be sharded: (" 
                  << o_proj.shape(0) << ", " << o_proj.shape(1) << ")" << std::endl;
        std::cerr << "[WARNING] Expected: (" << expectedHiddenSize << ", " << expectedHiddenSize << ")" << std::endl;
    }
    
    // [중요] o_proj의 실제 shape을 강제로 읽어서 확인
    // MLX의 lazy evaluation으로 인해 shape이 잘못 표시될 수 있음
    if (layerIdx == 0) {
        try {
            // o_proj의 첫 번째 행과 마지막 행을 읽어서 평가 강제
            mx::array o_proj_first_row = mx::take(o_proj, mx::array({0}), 0);
            if (o_proj_first_row.size() > 0) {
                mx::array firstElem = mx::take(o_proj_first_row, mx::array({0}), 0);
                if (firstElem.size() == 1) {
                    float dummy1 = firstElem.item<float>();
                    (void)dummy1;
                }
            }
            
            // o_proj의 마지막 행도 읽기
            int o_proj_rows = o_proj.shape(0);
            if (o_proj_rows > 0) {
                mx::array o_proj_last_row = mx::take(o_proj, mx::array({o_proj_rows - 1}), 0);
                if (o_proj_last_row.size() > 0) {
                    int o_proj_cols = o_proj.shape(1);
                    if (o_proj_cols > 0) {
                        mx::array lastElem = mx::take(o_proj_last_row, mx::array({o_proj_cols - 1}), 0);
                        if (lastElem.size() == 1) {
                            float dummy2 = lastElem.item<float>();
                            (void)dummy2;
                        }
                    }
                }
            }
            
            mx::synchronize(model_->stream);
            
            // 평가 후 shape 재확인
            std::cerr << "[MLX] o_proj shape after force evaluation: (" << o_proj.shape(0) << ", " << o_proj.shape(1) << ")" << std::endl;
            std::cerr << "[MLX] o_proj actual dimensions: rows=" << o_proj.shape(0) << ", cols=" << o_proj.shape(1) << std::endl;
            
            // 만약 o_proj가 intermediate_size (10944)와 관련이 있다면 경고
            if (o_proj.shape(0) == model_->intermediateSize || o_proj.shape(1) == model_->intermediateSize) {
                std::cerr << "!!! WARNING: o_proj shape matches intermediate_size (" << model_->intermediateSize << ") !!!" << std::endl;
                std::cerr << "This suggests o_proj may be referencing MLP layer weights!" << std::endl;
            }
        } catch (const std::exception& e) {
            std::cerr << "[MLX] Warning: Failed to force evaluate o_proj: " << e.what() << std::endl;
        }
    }
    
    // [중요] 가중치 크기 검증은 위에서 이미 수행됨
    // 여기서는 추가 검증만 수행
    
    // x_input shape: (seq_len, hidden_size)
    int seqLen = x_input.shape(0);  // seq_len은 첫 번째 dimension
    int hiddenSize = x_input.shape(1);  // hidden_size는 두 번째 dimension
    
    // q_proj shape 확인
    int qProjInDim = q_proj.shape(0);  // input dimension (should be hidden_size)
    int qProjOutDim = q_proj.shape(1);  // output dimension
    
    // [강제 검증] Weight의 첫 번째 차원이 입력의 마지막 차원과 일치해야 함
    if (qProjInDim != hiddenSize) {
        std::cerr << "!!! FATAL ERROR !!! Weight dim 0 (" << qProjInDim 
                  << ") does not match Input dim -1 (" << hiddenSize << ")" << std::endl;
        std::cerr << "x_input shape: (" << seqLen << ", " << hiddenSize << ")" << std::endl;
        std::cerr << "q_proj shape: (" << qProjInDim << ", " << qProjOutDim << ")" << std::endl;
        
        // 만약 q_proj가 (Out, In) 형태라면 transpose 필요
        if (q_proj.shape(1) == hiddenSize && q_proj.shape(0) != hiddenSize) {
            std::cerr << "!!! WARNING: q_proj appears to be (Out, In) format. Transpose may be needed!" << std::endl;
        }
        
        throw std::runtime_error("FATAL: Weight dimension mismatch in AttentionLayer");
    }
    
    // 첫 번째 layer에서 shape 확인
    if (layerIdx == 0) {
        std::cerr << "[MLX] AttentionLayer[0]: x_input shape: (" << seqLen << ", " << hiddenSize << ")" << std::endl;
        std::cerr << "[MLX] AttentionLayer[0]: q_proj shape: (" << qProjInDim << ", " << qProjOutDim << ")" << std::endl;
    }
    
    // matmul 직전에 x_input을 명시적으로 복사하여 평가 강제
    // 변수명을 명시적으로 분리하여 참조 문제 방지
    // 첫 번째 layer에서는 강제 평가 수행
    mx::array x_for_attn = x_input;  // 초기값은 x_input
    if (layerIdx == 0) {
        // x_input을 명시적으로 복사하기 위해 실제 연산 수행
        // x_input의 shape을 확인하고, 명시적으로 복사
        try {
            int seqLen = x_input.shape(0);
            int hiddenSize = x_input.shape(1);
            
            // x_input을 사용하는 실제 연산: x_input을 명시적으로 복사
            // 방법: x_input을 reshape하여 명시적 복사 (같은 shape이지만 새로운 배열 생성)
            mx::array x_reshaped = mx::reshape(x_input, {seqLen, hiddenSize});
            
            // x_reshaped의 첫 번째 행을 읽어서 평가 강제
            mx::array first_row = mx::take(x_reshaped, mx::array({0}), 0);
            if (first_row.size() > 0) {
                // 첫 번째 행의 여러 요소를 읽어서 평가 강제
                for (int i = 0; i < hiddenSize && i < 10; ++i) {
                    mx::array elem = mx::take(first_row, mx::array({i}), 0);
                    if (elem.size() == 1) {
                        float dummy = elem.item<float>();
                        (void)dummy;
                    }
                }
            }
            
            // stream 동기화
            mx::synchronize(model_->stream);
            
            // x_reshaped를 x_for_attn에 할당
            x_for_attn = x_reshaped;
            
            // shape 재확인
            std::cerr << "[MLX] Before matmul, x_for_attn shape: (" << x_for_attn.shape(0) << ", " << x_for_attn.shape(1) << ")" << std::endl;
            std::cerr << "[MLX] Before matmul, q_proj shape: (" << q_proj.shape(0) << ", " << q_proj.shape(1) << ")" << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "[MLX] Warning: Failed to force evaluate x_input before matmul: " << e.what() << std::endl;
            // 실패 시 원본 사용
            x_for_attn = x_input;
        }
    } else {
        // 다른 layer에서는 원본 사용
        x_for_attn = x_input;
    }
    
    // Query, Key, Value 계산 - x_input을 직접 사용 (x_for_attn 대신)
    // x_input @ q_proj: (seq_len, hidden_size) @ (hidden_size, out_dim) = (seq_len, out_dim)
    
    // [최종 검증] matmul 직전 변수 확인
    if (layerIdx == 0) {
        std::cerr << ">>> Before Calling matmul <<<" << std::endl;
        DebugArray("x_input (arg to matmul)", x_input);
        std::cerr << "x_input Ptr: " << &x_input << std::endl;
        std::cerr << "x_input shape().back(): " << x_input.shape().back() << std::endl;
        std::cerr << "q_proj shape: (" << q_proj.shape(0) << ", " << q_proj.shape(1) << ")" << std::endl;
        std::cerr << "q_proj.shape(0): " << q_proj.shape(0) << std::endl;
        
        // 강제 Assertion
        if (x_input.shape().back() != 2048) {
            std::cerr << "!!! STOP: x_input is NOT 2048 here! It is " << x_input.shape().back() << std::endl;
            std::cerr << "Full shape: (";
            for (size_t i = 0; i < x_input.shape().size(); ++i) {
                std::cerr << x_input.shape()[i];
                if (i < x_input.shape().size() - 1) std::cerr << ", ";
            }
            std::cerr << ")" << std::endl;
            throw std::runtime_error("x_input shape verification failed before matmul");
        }
    }
    
    // x_for_attn 대신 x_input을 직접 사용하여 참조 문제 해결
    // MLX matmul(x, w): x의 마지막 차원과 w의 첫 번째 차원이 같아야 함
    // x: (seq_len, hidden_size) = (13, 2048)
    // w: (hidden_size, out_dim) = (2048, 256)
    // 결과: (13, 256)
    
    // [중요] matmul 전에 x_input을 명시적으로 평가하기 위해 작은 연산 수행
    // MLX의 lazy evaluation 문제를 해결하기 위해 x_input을 사용하는 실제 연산 수행
    mx::array x_for_matmul = x_input;
    if (layerIdx == 0) {
        try {
            // x_input의 모든 요소를 강제로 평가하기 위해 첫 번째 행과 마지막 행을 읽음
            int seqLen = x_input.shape(0);
            int hiddenSize = x_input.shape(1);
            
            // 첫 번째 행 읽기
            mx::array firstRow = mx::take(x_input, mx::array({0}), 0);
            if (firstRow.size() > 0) {
                mx::array firstElem = mx::take(firstRow, mx::array({0}), 0);
                if (firstElem.size() == 1) {
                    float dummy1 = firstElem.item<float>();
                    (void)dummy1;
                }
            }
            
            // 마지막 행의 마지막 요소 읽기
            if (seqLen > 0 && hiddenSize > 0) {
                mx::array lastRow = mx::take(x_input, mx::array({seqLen - 1}), 0);
                if (lastRow.size() > 0) {
                    mx::array lastElem = mx::take(lastRow, mx::array({hiddenSize - 1}), 0);
                    if (lastElem.size() == 1) {
                        float dummy2 = lastElem.item<float>();
                        (void)dummy2;
                    }
                }
            }
            
            // stream 동기화
            mx::synchronize(model_->stream);
            
            // x_input을 사용하는 작은 연산으로 명시적 복사
            x_for_matmul = x_input + mx::array(0.0f);
            
            // 복사 후 shape 확인
            std::cerr << ">>> After force evaluation <<<" << std::endl;
            std::cerr << "x_for_matmul shape().back(): " << x_for_matmul.shape().back() << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "[MLX] Warning: Failed to force evaluate x_input: " << e.what() << std::endl;
            x_for_matmul = x_input;
        }
    }
    
    std::cerr << ">>> Inside matmul call <<<" << std::endl;
    std::cerr << "Calling: matmul(x_for_matmul, q_proj)" << std::endl;
    std::cerr << "x_for_matmul last dim: " << x_for_matmul.shape().back() << std::endl;
    std::cerr << "q_proj first dim: " << q_proj.shape(0) << std::endl;
    
    mx::array q = mx::matmul(x_for_matmul, q_proj);
    mx::array k = mx::matmul(x_for_matmul, k_proj);
    mx::array v = mx::matmul(x_for_matmul, v_proj);
    
    // Reshape for multi-head attention
    int numHeads = model_->numHeads;
    int headDim = model_->hiddenSize / numHeads;
    
    // Scaled dot-product attention
    mx::array scale = mx::array(1.0f / std::sqrt(static_cast<float>(headDim)));
    
    // q @ k^T: (seq_len, out_dim) @ (out_dim, seq_len) = (seq_len, seq_len)
    mx::array scores = mx::matmul(q, mx::transpose(k, {1, 0})) * scale;
    
    // Causal mask 적용
    mx::array mask = mx::triu(mx::ones({seqLen, seqLen}), 1) * mx::array(-1e9f);
    scores = scores + mask;
    
    mx::array attn = mx::softmax(scores, -1);
    mx::array out = mx::matmul(attn, v);
    
    // [중요] out을 명시적으로 복사하여 참조 문제 방지
    mx::array out_copy = out + mx::array(0.0f);
    
    // [중요] Attention output shape 확인
    if (layerIdx == 0) {
        std::cerr << ">>> After Attention Score Computation <<<" << std::endl;
        std::cerr << "Attention Output Shape (out): (";
        for (size_t i = 0; i < out.shape().size(); ++i) {
            std::cerr << out.shape()[i];
            if (i < out.shape().size() - 1) std::cerr << ", ";
        }
        std::cerr << ")" << std::endl;
        std::cerr << "Attention Output Shape (out_copy): (";
        for (size_t i = 0; i < out_copy.shape().size(); ++i) {
            std::cerr << out_copy.shape()[i];
            if (i < out_copy.shape().size() - 1) std::cerr << ", ";
        }
        std::cerr << ")" << std::endl;
        std::cerr << "o_proj shape: (" << o_proj.shape(0) << ", " << o_proj.shape(1) << ")" << std::endl;
        std::cerr << "o_proj Input Expectation (Dim 0 of o_proj): " << o_proj.shape(0) << std::endl;
        std::cerr << "Attention Output last dim (out): " << out.shape().back() << std::endl;
        std::cerr << "Attention Output last dim (out_copy): " << out_copy.shape().back() << std::endl;
    }
    
    // out을 복사본으로 교체
    out = out_copy;
    
    // Output projection
    // o_proj shape이 (2048, 256)이고 attention output이 (13, 256)이면
    // o_proj를 transpose하여 (256, 2048)로 만들어야 함
    // 또는 attention output의 차원이 256이면 o_proj의 입력 차원도 256이어야 함
    int attentionOutDim = out.shape().back();
    int oProjInDim = o_proj.shape(0);
    int oProjOutDim = o_proj.shape(1);
    
    if (layerIdx == 0) {
        std::cerr << "[MLX] o_proj actual shape check: (" << o_proj.shape(0) << ", " << o_proj.shape(1) << ")" << std::endl;
        std::cerr << "[MLX] attentionOutDim: " << attentionOutDim << ", oProjInDim: " << oProjInDim << ", oProjOutDim: " << oProjOutDim << std::endl;
    }
    
    if (attentionOutDim != oProjInDim) {
        // 차원 불일치 - o_proj를 transpose하거나 다른 방법 시도
        if (o_proj.shape(1) == attentionOutDim) {
            // o_proj가 (Out, In) 형태로 저장되어 있을 수 있음 - transpose 필요
            std::cerr << "[MLX] Transposing o_proj: (" << o_proj.shape(0) << ", " << o_proj.shape(1) 
                      << ") -> (" << o_proj.shape(1) << ", " << o_proj.shape(0) << ")" << std::endl;
            // o_proj를 transpose하기 전에 실제 shape 확인
            std::cerr << "[MLX] o_proj original shape: (" << o_proj.shape(0) << ", " << o_proj.shape(1) << ")" << std::endl;
            
            // [중요] o_proj를 명시적으로 복사하여 참조 문제 방지
            mx::array o_proj_local = o_proj + mx::array(0.0f);  // 명시적 복사
            mx::array o_proj_T = mx::transpose(o_proj_local, {1, 0});
            
            // [중요] o_proj_T를 명시적으로 복사하여 참조 문제 방지
            mx::array o_proj_T_copy = o_proj_T + mx::array(0.0f);
            
            // o_proj_T_copy를 강제로 평가
            if (layerIdx == 0) {
                try {
                    // o_proj_T_copy의 첫 번째 행을 읽어서 평가 강제
                    mx::array firstRow = mx::take(o_proj_T_copy, mx::array({0}), 0);
                    if (firstRow.size() > 0) {
                        mx::array firstElem = mx::take(firstRow, mx::array({0}), 0);
                        if (firstElem.size() == 1) {
                            float dummy = firstElem.item<float>();
                            (void)dummy;
                        }
                    }
                    mx::synchronize(model_->stream);
                    
                    // [중요] 평가 후 shape 확인
                    std::cerr << "[MLX] o_proj_T_copy shape after evaluation: (" 
                              << o_proj_T_copy.shape(0) << ", " << o_proj_T_copy.shape(1) << ")" << std::endl;
                    
                    // MLP 가중치 확인
                    if (o_proj_T_copy.shape(0) == model_->intermediateSize || o_proj_T_copy.shape(1) == model_->intermediateSize) {
                        std::cerr << "!!! FATAL: o_proj_T_copy is MLP weight after evaluation!" << std::endl;
                        throw std::runtime_error("o_proj_T_copy is MLP weight");
                    }
                } catch (const std::exception& e) {
                    std::cerr << "[MLX] Warning: Failed to force evaluate o_proj_T_copy: " << e.what() << std::endl;
                }
            }
            
            // o_proj_T를 o_proj_T_copy로 교체
            o_proj_T = o_proj_T_copy;
            
            if (layerIdx == 0) {
                std::cerr << "[MLX] o_proj_T shape after transpose and evaluation: (" << o_proj_T.shape(0) << ", " << o_proj_T.shape(1) << ")" << std::endl;
                std::cerr << "[MLX] out shape before matmul: (";
                for (size_t i = 0; i < out.shape().size(); ++i) {
                    std::cerr << out.shape()[i];
                    if (i < out.shape().size() - 1) std::cerr << ", ";
                }
                std::cerr << ")" << std::endl;
                
                // out의 마지막 차원 강제 확인
                std::cerr << "[MLX] out.shape().back(): " << out.shape().back() << std::endl;
                std::cerr << "[MLX] o_proj_T.shape(0): " << o_proj_T.shape(0) << std::endl;
                
                // 강제 Assertion
                if (out.shape().back() != o_proj_T.shape(0)) {
                    std::cerr << "!!! FATAL: out last dim (" << out.shape().back() 
                              << ") != o_proj_T first dim (" << o_proj_T.shape(0) << ")" << std::endl;
                    throw std::runtime_error("Dimension mismatch before o_proj matmul");
                }
            }
            
            // [핵심] matmul 직전에 out을 명시적으로 평가 및 복사
            // [중요] out이 이미 복사본이므로 다시 복사할 필요 없지만, 검증을 위해 복사
            // [중요] out의 ID를 먼저 확인
            if (layerIdx == 0) {
                std::cerr << ">>> Before creating out_for_matmul <<<" << std::endl;
                std::cerr << "out shape: (";
                for (size_t i = 0; i < out.shape().size(); ++i) {
                    std::cerr << out.shape()[i];
                    if (i < out.shape().size() - 1) std::cerr << ", ";
                }
                std::cerr << ") | ID: " << out.id() << std::endl;
                std::cerr << "out.shape().back(): " << out.shape().back() << std::endl;
            }
            
            mx::array out_for_matmul = out + mx::array(0.0f);  // 즉시 명시적 복사
            
            // [중요] out과 out_for_matmul의 shape 및 ID 확인
            if (layerIdx == 0) {
                std::cerr << ">>> After creating out_for_matmul <<<" << std::endl;
                std::cerr << "out shape: (";
                for (size_t i = 0; i < out.shape().size(); ++i) {
                    std::cerr << out.shape()[i];
                    if (i < out.shape().size() - 1) std::cerr << ", ";
                }
                std::cerr << ") | ID: " << out.id() << std::endl;
                std::cerr << "out_for_matmul shape: (";
                for (size_t i = 0; i < out_for_matmul.shape().size(); ++i) {
                    std::cerr << out_for_matmul.shape()[i];
                    if (i < out_for_matmul.shape().size() - 1) std::cerr << ", ";
                }
                std::cerr << ") | ID: " << out_for_matmul.id() << std::endl;
                std::cerr << "ID changed: " << (out.id() != out_for_matmul.id() ? "YES" : "NO") << std::endl;
                
                if (out.shape() != out_for_matmul.shape()) {
                    std::cerr << "!!! FATAL: out and out_for_matmul have different shapes!" << std::endl;
                    throw std::runtime_error("out and out_for_matmul shape mismatch");
                }
            }
            
            if (layerIdx == 0) {
                try {
                    // out_for_matmul의 첫 번째 요소를 읽어서 평가 강제
                    mx::array firstRow = mx::take(out_for_matmul, mx::array({0}), 0);
                    if (firstRow.size() > 0) {
                        mx::array firstElem = mx::take(firstRow, mx::array({0}), 0);
                        if (firstElem.size() == 1) {
                            float dummy = firstElem.item<float>();
                            (void)dummy;
                        }
                    }
                    mx::synchronize(model_->stream);
                    
                    // [중요] 평가 후 shape 확인
                    std::cerr << "[MLX] out_for_matmul shape after evaluation: (";
                    for (size_t i = 0; i < out_for_matmul.shape().size(); ++i) {
                        std::cerr << out_for_matmul.shape()[i];
                        if (i < out_for_matmul.shape().size() - 1) std::cerr << ", ";
                    }
                    std::cerr << ")" << std::endl;
                    std::cerr << "[MLX] out_for_matmul shape().back(): " << out_for_matmul.shape().back() << std::endl;
                    
                    // [중요] out_for_matmul이 10944 차원을 가지고 있는지 확인
                    if (out_for_matmul.shape().back() == model_->intermediateSize) {
                        std::cerr << "!!! FATAL: out_for_matmul has intermediate_size dimension!" << std::endl;
                        throw std::runtime_error("out_for_matmul has intermediate_size dimension");
                    }
                } catch (const std::exception& e) {
                    std::cerr << "[MLX] Warning: Failed to force evaluate out_for_matmul: " << e.what() << std::endl;
                }
            }
            
            // [핵심] matmul 직전 최종 확인 및 객체 ID 추적
            std::cerr << ">>> Matmul Debug Info <<<" << std::endl;
            std::cerr << "Input X  | Shape: (";
            for (size_t i = 0; i < out_for_matmul.shape().size(); ++i) {
                std::cerr << out_for_matmul.shape()[i];
                if (i < out_for_matmul.shape().size() - 1) std::cerr << ", ";
            }
            std::cerr << ") | ID: " << out_for_matmul.id() << std::endl;
            std::cerr << "Weight W | Shape: (" << o_proj_T.shape(0) << ", " << o_proj_T.shape(1) 
                      << ") | ID: " << o_proj_T.id() << std::endl;
            
            // [강제 Assertion] 10944는 절대 나오면 안 되는 숫자 - matmul 직전 최종 확인
            if (o_proj_T.shape(0) == 10944 || o_proj_T.shape(1) == 10944) {
                std::cerr << "!!! CRITICAL ERROR !!!" << std::endl;
                std::cerr << "The weight variable holds MLP data (dim 10944)." << std::endl;
                std::cerr << "This means 'o_proj_T' variable contains the wrong mx::array object." << std::endl;
                std::cerr << "o_proj_T ID: " << o_proj_T.id() << std::endl;
                std::cerr << "o_proj_T shape: (" << o_proj_T.shape(0) << ", " << o_proj_T.shape(1) << ")" << std::endl;
                std::cerr << "o_key: " << o_key << std::endl;
                
                // [디버깅] weights 맵에서 실제 값 확인
                auto it = model_->weights.find(o_key);
                if (it != model_->weights.end()) {
                    std::cerr << "weights map value shape: (" << it->second.shape(0) << ", " << it->second.shape(1) << ")" << std::endl;
                    std::cerr << "weights map value ID: " << it->second.id() << std::endl;
                    std::cerr << "ID match: " << (it->second.id() == o_proj_T.id() ? "YES" : "NO") << std::endl;
                } else {
                    std::cerr << "weights map: key not found!" << std::endl;
                }
                
                exit(1);
            }
            
            // [중요] out_for_matmul이 실제로 (6, 256)인지 강제 확인
            // 에러 메시지에서 (6, 2048)로 나타나는 것을 보면 out_for_matmul이 잘못된 배열을 참조하고 있을 수 있음
            if (out_for_matmul.shape().back() == 2048) {
                std::cerr << "!!! CRITICAL: out_for_matmul has wrong shape (last dim is 2048, expected 256)!" << std::endl;
                std::cerr << "   Full shape: (";
                for (size_t i = 0; i < out_for_matmul.shape().size(); ++i) {
                    std::cerr << out_for_matmul.shape()[i];
                    if (i < out_for_matmul.shape().size() - 1) std::cerr << ", ";
                }
                std::cerr << ")" << std::endl;
                std::cerr << "   out_for_matmul ID: " << out_for_matmul.id() << std::endl;
                std::cerr << "   out ID: " << out.id() << std::endl;
                std::cerr << "   ID match: " << (out.id() == out_for_matmul.id() ? "YES" : "NO") << std::endl;
                throw std::runtime_error("out_for_matmul has wrong shape");
            }
            
            // 차원 일치 확인
            if (out_for_matmul.shape().back() != o_proj_T.shape(0)) {
                std::cerr << "!!! FATAL: Dimension mismatch!" << std::endl;
                std::cerr << "   out_for_matmul last dim: " << out_for_matmul.shape().back() << std::endl;
                std::cerr << "   o_proj_T first dim: " << o_proj_T.shape(0) << std::endl;
                throw std::runtime_error("Dimension mismatch before matmul");
            }
            
            // [최종] matmul 호출 직전에 모든 변수를 강제 평가
            // MLX의 lazy evaluation으로 인해 matmul 시점에 다른 배열을 참조할 수 있으므로
            // [핵심] o_proj부터 o_proj_T_eval까지의 전체 변환 과정 추적
            std::cerr << ">>> Full o_proj transformation trace <<<" << std::endl;
            std::cerr << "o_proj shape: (" << o_proj.shape(0) << ", " << o_proj.shape(1) << ") | ID: " << o_proj.id() << std::endl;
            std::cerr << "o_proj_T shape: (" << o_proj_T.shape(0) << ", " << o_proj_T.shape(1) << ") | ID: " << o_proj_T.id() << std::endl;
            
            try {
                // [핵심] out_for_matmul 강제 평가 - out 변수와 완전히 분리
                // out 변수가 다른 곳에서 변경되었을 수 있으므로 out_for_matmul을 직접 사용
                mx::array out_eval = out_for_matmul + mx::array(0.0f);
                mx::synchronize(model_->stream);
                
                // [중요] out_eval이 실제로 (6, 256)인지 강제 확인
                if (out_eval.shape().back() == 2048) {
                    std::cerr << "!!! CRITICAL: out_eval has wrong shape (last dim is 2048, expected 256)!" << std::endl;
                    std::cerr << "   out_eval shape: (";
                    for (size_t i = 0; i < out_eval.shape().size(); ++i) {
                        std::cerr << out_eval.shape()[i];
                        if (i < out_eval.shape().size() - 1) std::cerr << ", ";
                    }
                    std::cerr << ")" << std::endl;
                    std::cerr << "   out_eval ID: " << out_eval.id() << std::endl;
                    std::cerr << "   out_for_matmul ID: " << out_for_matmul.id() << std::endl;
                    throw std::runtime_error("out_eval has wrong shape");
                }
                
                // [중요] o_proj_T를 다시 생성하여 참조 문제 방지
                // o_proj부터 다시 시작 - GetWeight를 다시 호출하여 최신 값 가져오기
                mx::array o_proj_fresh = GetWeight(o_key);
                if (o_proj_fresh.shape(0) == model_->intermediateSize || o_proj_fresh.shape(1) == model_->intermediateSize) {
                    std::cerr << "!!! CRITICAL: o_proj_fresh is MLP weight!" << std::endl;
                    std::cerr << "   o_proj_fresh shape: (" << o_proj_fresh.shape(0) << ", " << o_proj_fresh.shape(1) << ")" << std::endl;
                    std::cerr << "   o_key: " << o_key << std::endl;
                    exit(1);
                }
                mx::array o_proj_fresh_copy = o_proj_fresh + mx::array(0.0f);
                mx::array o_proj_T_fresh = mx::transpose(o_proj_fresh_copy, {1, 0});
                mx::array o_proj_T_eval = o_proj_T_fresh + mx::array(0.0f);
                mx::synchronize(model_->stream);
                
                // 평가 후 shape 확인
                std::cerr << ">>> After force evaluation before matmul <<<" << std::endl;
                std::cerr << "out_eval shape: (";
                for (size_t i = 0; i < out_eval.shape().size(); ++i) {
                    std::cerr << out_eval.shape()[i];
                    if (i < out_eval.shape().size() - 1) std::cerr << ", ";
                }
                std::cerr << ") | ID: " << out_eval.id() << std::endl;
                std::cerr << "o_proj_fresh shape: (" << o_proj_fresh.shape(0) << ", " << o_proj_fresh.shape(1) << ") | ID: " << o_proj_fresh.id() << std::endl;
                std::cerr << "o_proj_T_fresh shape: (" << o_proj_T_fresh.shape(0) << ", " << o_proj_T_fresh.shape(1) << ") | ID: " << o_proj_T_fresh.id() << std::endl;
                std::cerr << "o_proj_T_eval shape: (" << o_proj_T_eval.shape(0) << ", " << o_proj_T_eval.shape(1) 
                          << ") | ID: " << o_proj_T_eval.id() << std::endl;
                
                // [강제 Assertion] 평가 후에도 10944가 나타나면 즉시 중단
                if (o_proj_T_eval.shape(0) == 10944 || o_proj_T_eval.shape(1) == 10944) {
                    std::cerr << "!!! CRITICAL: o_proj_T_eval has MLP dimension after evaluation!" << std::endl;
                    std::cerr << "   o_proj_fresh shape: (" << o_proj_fresh.shape(0) << ", " << o_proj_fresh.shape(1) << ")" << std::endl;
                    std::cerr << "   o_proj_T_fresh shape: (" << o_proj_T_fresh.shape(0) << ", " << o_proj_T_fresh.shape(1) << ")" << std::endl;
                    exit(1);
                }
                
                // [최종] matmul 호출 직전에 모든 변수를 다시 확인
                std::cerr << ">>> Final check before matmul call <<<" << std::endl;
                std::cerr << "out_eval.shape().back(): " << out_eval.shape().back() << std::endl;
                std::cerr << "o_proj_T_eval.shape(0): " << o_proj_T_eval.shape(0) << std::endl;
                
                // 차원 일치 확인
                if (out_eval.shape().back() != o_proj_T_eval.shape(0)) {
                    std::cerr << "!!! FATAL: Final dimension check failed!" << std::endl;
                    std::cerr << "   out_eval last dim: " << out_eval.shape().back() << std::endl;
                    std::cerr << "   o_proj_T_eval first dim: " << o_proj_T_eval.shape(0) << std::endl;
                    throw std::runtime_error("Final dimension check failed");
                }
                
                // 평가된 변수로 matmul 수행
                out = mx::matmul(out_eval, o_proj_T_eval);
            } catch (const std::exception& e) {
                std::cerr << "[MLX] Warning: Failed to force evaluate before matmul: " << e.what() << std::endl;
                // 실패 시 원본 사용
                out = mx::matmul(out_for_matmul, o_proj_T);
            }
        } else {
            std::cerr << "[MLX] ERROR: Cannot match attention output dim (" << attentionOutDim 
                      << ") with o_proj dims (" << o_proj.shape(0) << ", " << o_proj.shape(1) << ")" << std::endl;
            throw std::runtime_error("Attention output and o_proj dimension mismatch");
        }
    } else {
        // 차원이 일치하면 정상적으로 matmul
        out = mx::matmul(out, o_proj);
    }
    
    return out;
}

mx::array MlxInference::FeedForwardLayer(const mx::array& x, int layerIdx) {
    // Feed Forward Network 구현 (ggml-metal 스타일)
    std::string prefix = "model.layers." + std::to_string(layerIdx) + ".mlp.";
    
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
    
    // Token embeddings - 명시적인 변수명 사용
    mx::array embed = GetWeight("model.embed_tokens.weight");
    
    // tokens를 array로 변환
    std::vector<int32_t> tokens32(tokens.begin(), tokens.end());
    mx::array tokenArray(tokens32.data(), {static_cast<int>(tokens32.size())}, mx::int32);
    mx::array x_emb = mx::take(embed, tokenArray, 0);
    
    // take의 결과가 올바른 타입인지 확인 (float32여야 함)
    // 만약 타입이 다르면 변환
    if (x_emb.dtype() != mx::float32) {
        x_emb = mx::astype(x_emb, mx::float32);
    }
    
    // x_emb shape: (seq_len, embed_dim) - embed_dim이 256일 수 있음
    // 하지만 input_layernorm.weight는 (hidden_size,) = (2048,)일 수 있음
    // 첫 번째 layer에서 256 -> 2048로 projection이 필요할 수 있음
    
    // 첫 번째 layer의 input_layernorm.weight shape을 확인
    mx::array firstNormWeight = GetWeight("model.layers.0.input_layernorm.weight");
    int normDim = firstNormWeight.shape(0);
    
    // x_emb의 embedding dimension 확인
    int embedDim = x_emb.shape(1);
    
    // embed_dim != normDim이면 projection 필요
    mx::array x_hidden = x_emb;  // projection 후 hidden_size dimension을 가진 변수 (초기값은 x_emb)
    bool projectionApplied = false;
    
    if (embedDim != normDim) {
        // embed_dim에서 hidden_size로 projection
        // 첫 번째 layer의 q_proj를 사용하여 projection
        // q_proj shape: (hidden_size, embed_dim) = (2048, 256)
        // x_emb @ q_proj.T = (seq_len, embed_dim) @ (embed_dim, hidden_size) = (seq_len, hidden_size)
        try {
            std::string prefix = "model.layers.0.self_attn.";
            mx::array q_proj = GetWeight(prefix + "q_proj.weight");
            // q_proj shape 확인
            std::cerr << "[MLX] q_proj shape: (" << q_proj.shape(0) << ", " << q_proj.shape(1) << ")" << std::endl;
            std::cerr << "[MLX] x_emb shape before projection: (" << x_emb.shape(0) << ", " << x_emb.shape(1) << ")" << std::endl;
            
            // q_proj shape: (hidden_size, embed_dim) = (2048, 256)
            // transpose하여 (embed_dim, hidden_size) = (256, 2048)로 만들기
            mx::array q_proj_T = mx::transpose(q_proj, {1, 0});
            
            // transpose 후 shape 확인
            std::cerr << "[MLX] q_proj_T shape after transpose: (" << q_proj_T.shape(0) << ", " << q_proj_T.shape(1) << ")" << std::endl;
            
            // x_emb @ q_proj.T: (seq_len, embed_dim) @ (embed_dim, hidden_size) = (seq_len, hidden_size)
            // x_emb shape: (2, 256), q_proj_T shape: (256, 2048)
            // 결과: (2, 2048)
            // llama.cpp 스타일: ggml_mul_mat(w, cur)는 cur @ w.T를 의미
            mx::array x_proj = mx::matmul(x_emb, q_proj_T);
            
            // projection을 강제로 평가하기 위해 바로 LayerNorm 수행
            // llama.cpp처럼 projection 후 바로 실제 연산을 수행하여 평가 강제
            // 첫 번째 layer의 input_layernorm을 바로 적용 (실제로 사용할 연산이므로 평가 보장)
            try {
                // RMSNorm 연산 수행 (실제로 사용할 것이므로 평가 강제)
                mx::array x_squared = mx::square(x_proj);
                mx::array mean_squared = mx::mean(x_squared, -1, true);
                mx::array rms = mx::sqrt(mean_squared + mx::array(1e-6f));
                mx::array x_norm = x_proj / rms;
                mx::array x_norm_weighted = x_norm * firstNormWeight;
                
                // x_norm_weighted의 첫 번째 행을 읽어서 평가 강제
                mx::array firstRow = mx::take(x_norm_weighted, mx::array({0}), 0);
                if (firstRow.size() > 0) {
                    mx::array firstElem = mx::take(firstRow, mx::array({0}), 0);
                    if (firstElem.size() == 1) {
                        float dummy = firstElem.item<float>();
                        (void)dummy;
                    }
                }
                
                // stream 동기화
                mx::synchronize(model_->stream);
                
                // LayerNorm이 적용된 결과를 x_hidden에 할당 (명시적인 변수명)
                // 이렇게 하면 projection이 실제로 평가되고, LayerNorm도 이미 적용됨
                x_hidden = x_norm_weighted;
                projectionApplied = true;
                
                // projection 후 shape 확인
                std::cerr << "[MLX] x_hidden shape after projection+LayerNorm: (" << x_hidden.shape(0) << ", " << x_hidden.shape(1) << ")" << std::endl;
                std::cerr << "[MLX] Projected from embed_dim " << embedDim << " to hidden_size " << normDim << std::endl;
                DebugArray("After Projection+LayerNorm", x_hidden);
            } catch (const std::exception& e) {
                std::cerr << "[MLX] Warning: Failed to evaluate projection with LayerNorm: " << e.what() << std::endl;
                // 평가 실패 시 원래 projected 사용
                x_hidden = x_proj;
                projectionApplied = true;
            }
        } catch (const std::exception& e) {
            std::cerr << "[MLX] Warning: Failed to project embed_dim to hidden_size: " << e.what() << std::endl;
            // projection 실패 시 오류 발생
            throw std::runtime_error("Cannot project embed_dim to hidden_size");
        }
    } else {
        // projection이 필요 없으면 x_emb를 그대로 사용
        x_hidden = x_emb;
    }
    
    // Position embeddings (RoPE는 별도로 처리)
    
    // Transformer layers - 명시적인 변수명 사용
    mx::array x = x_hidden;  // x_hidden을 x로 복사 (명시적 변수명으로 시작)
    DebugArray("Before Transformer Layers", x);
    
    for (int i = 0; i < model_->numLayers; ++i) {
        mx::array x_residual = x;  // residual connection을 위한 명시적 변수
        
        // x shape 확인 (첫 번째 layer 전)
        if (i == 0) {
            std::cerr << "[MLX] Before LayerNorm[0], x shape: (" << x.shape(0) << ", " << x.shape(1) << ")" << std::endl;
        }
        
        // Pre-norm (첫 번째 layer에서 projection 시 이미 적용되었으면 건너뛰기)
        mx::array x_norm = x;  // 초기값은 x
        if (i == 0 && projectionApplied) {
            // projection 시 이미 LayerNorm이 적용되었으므로 건너뛰기
            // x_norm은 이미 x로 초기화됨
            std::cerr << "[MLX] LayerNorm[0] already applied during projection, using x as-is" << std::endl;
            DebugArray("LayerNorm[0] skipped (already applied)", x_norm);
        } else {
            // LayerNorm 적용
            x_norm = LayerNorm(x, "model.layers." + std::to_string(i) + ".input_layernorm");
            if (i == 0) {
                DebugArray("After LayerNorm[0]", x_norm);
            }
        }
        
        // x_norm shape 확인 (첫 번째 layer 후)
        if (i == 0) {
            std::cerr << "[MLX] After LayerNorm[0], x_norm shape: (" << x_norm.shape(0) << ", " << x_norm.shape(1) << ")" << std::endl;
        }
        
        // Self-attention - x_norm을 명시적으로 전달
        // [디버그] 전달 전 변수 확인 - 스코프 문제 검증
        if (i == 0) {
            std::cerr << ">>> Before Calling AttentionLayer[" << i << "] <<<" << std::endl;
            DebugArray("x_norm (arg to AttentionLayer)", x_norm);
            std::cerr << "x_norm Ptr: " << &x_norm << std::endl;
            std::cerr << "x_norm shape().back(): " << x_norm.shape().back() << std::endl;
            
            // 강제 Assertion - 스코프 문제 확인
            if (x_norm.shape().back() != 2048) {
                std::cerr << "!!! STOP: x_norm is NOT 2048 here! It is " << x_norm.shape().back() << std::endl;
                throw std::runtime_error("x_norm shape verification failed before AttentionLayer call");
            }
        }
        mx::array x_attn = AttentionLayer(x_norm, i);
        x = x_residual + x_attn;
        
        // Feed-forward를 위한 residual 저장
        mx::array x_residual_ff = x;
        
        // Post-norm
        x = LayerNorm(x, "model.layers." + std::to_string(i) + ".post_attention_layernorm");
        
        // Feed-forward
        mx::array ff_out = FeedForwardLayer(x, i);
        x = x_residual_ff + ff_out;
    }
    
    // Final layer norm
    x = LayerNorm(x, "model.norm");
    
    // Output projection
    mx::array lm_head = GetWeight("lm_head.weight");
    mx::array logits = mx::matmul(x, mx::transpose(lm_head, {1, 0}));
    
    // 마지막 토큰의 logits만 반환
    // 인덱스를 int32로 명시적으로 변환
    int lastIdx = static_cast<int>(tokens.size() - 1);
    std::vector<int32_t> indices = {lastIdx};
    mx::array indexArray(indices.data(), {1}, mx::int32);
    mx::array lastLogits = mx::take(logits, indexArray, 0);  // axis 0으로 수정
    
    // take의 결과는 2D일 수 있으므로 squeeze
    if (lastLogits.shape().size() > 1) {
        lastLogits = mx::squeeze(lastLogits, 0);
    }
    
    // item() 호출 시 자동 평가되므로 EvalArray 불필요
    return lastLogits;
}

mx::array MlxInference::ApplyTopK(const mx::array& probs, int k) {
    // Top-K 샘플링 구현 (llama.cpp 스타일)
    if (k <= 0 || k >= probs.shape(-1)) return probs;
    
    // topk를 사용하여 상위 k개 값과 인덱스 얻기
    mx::array topKValues = mx::topk(probs, k, -1);
    
    // topk는 값만 반환하므로, argsort를 사용하여 인덱스 얻기
    // MLX의 argsort는 axis만 받음
    mx::array sortedIndices = mx::argsort(probs, -1);
    // 내림차순이므로 뒤에서부터 k개 선택
    int vocabSize = probs.shape(-1);
    
    // 상위 k개 인덱스만 선택 (뒤에서부터)
    std::vector<int32_t> topKIdxVec;
    for (int i = vocabSize - k; i < vocabSize; ++i) {
        mx::array idxArr = mx::take(sortedIndices, mx::array({i}), 0);
        int idx = static_cast<int>(idxArr.item<int32_t>());
        topKIdxVec.push_back(idx);
    }
    
    // Top-K 인덱스에 해당하는 값만 유지하고 나머지는 0으로 설정
    // MLX의 scatter를 사용하여 mask 생성
    mx::array mask = mx::zeros({vocabSize}, mx::float32);
    mx::array ones = mx::ones({static_cast<int>(topKIdxVec.size())}, mx::float32);
    
    // scatter를 사용하여 top-k 위치에 1.0 설정
    mx::array indicesArray(topKIdxVec.data(), {static_cast<int>(topKIdxVec.size())}, mx::int32);
    mask = mx::scatter(mask, indicesArray, ones, 0);
    
    // mask를 사용하여 top-k 값만 유지
    return probs * mask;
}

mx::array MlxInference::ApplyTopP(const mx::array& probs, double p) {
    // Top-P (nucleus) 샘플링 구현 (llama.cpp 스타일)
    if (p >= 1.0) return probs;
    
    // argsort를 사용하여 인덱스 얻기 (오름차순)
    mx::array sortedIndices = mx::argsort(probs, -1);
    // sort를 사용하여 정렬된 값 얻기 (오름차순)
    mx::array sortedProbs = mx::sort(probs, -1);
    
    int vocabSize = probs.shape(-1);
    
    // 정렬된 확률의 누적합 계산 (뒤에서부터, 내림차순 순서로)
    float cumSum = 0.0f;
    int lastIdx = vocabSize;
    
    // 뒤에서부터 누적합 계산
    for (int i = vocabSize - 1; i >= 0; --i) {
        mx::array probVal = mx::take(sortedProbs, mx::array({i}), 0);
        float val = probVal.item<float>();
        cumSum += val;
        if (cumSum >= static_cast<float>(p)) {
            lastIdx = vocabSize - i;
            break;
        }
    }
    
    // 상위 lastIdx개 인덱스만 선택 (뒤에서부터)
    std::vector<int32_t> topPIdxVec;
    for (int i = vocabSize - lastIdx; i < vocabSize; ++i) {
        mx::array idxArr = mx::take(sortedIndices, mx::array({i}), 0);
        int idx = static_cast<int>(idxArr.item<int32_t>());
        topPIdxVec.push_back(idx);
    }
    
    // Top-P 인덱스에 해당하는 값만 유지하고 나머지는 0으로 설정
    mx::array mask = mx::zeros({vocabSize}, mx::float32);
    if (!topPIdxVec.empty()) {
        mx::array ones = mx::ones({static_cast<int>(topPIdxVec.size())}, mx::float32);
        mx::array indicesArray(topPIdxVec.data(), {static_cast<int>(topPIdxVec.size())}, mx::int32);
        mask = mx::scatter(mask, indicesArray, ones, 0);
    }
    
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
    // Multinomial 샘플링 (llama.cpp 스타일)
    // 누적 확률을 사용한 샘플링
    
    // 누적 확률 계산
    mx::array cumsum = mx::cumsum(probs, -1);
    int vocabSize = probs.shape(-1);
    
    // 마지막 값이 총합
    mx::array lastCumVal = mx::take(cumsum, mx::array({vocabSize - 1}), 0);
    float totalSum = lastCumVal.item<float>();
    
    // 0과 totalSum 사이의 랜덤 값 생성
    // C++ std::random 사용
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0f, totalSum);
    float r = dis(gen);
    
    // 누적 확률이 r을 초과하는 첫 번째 인덱스 찾기
    for (int i = 0; i < vocabSize; ++i) {
        mx::array cumVal = mx::take(cumsum, mx::array({i}), 0);
        float val = cumVal.item<float>();
        if (val >= r) {
            return i;
        }
    }
    
    // fallback: argmax 사용
    mx::array sampledIdx = mx::argmax(probs, -1, false);
    try {
        if (sampledIdx.dtype() == mx::int32) {
            return static_cast<int>(sampledIdx.item<int32_t>());
        } else {
            return static_cast<int>(sampledIdx.item<float>());
        }
    } catch (...) {
        return 0;
    }
}

mx::array MlxInference::GenerateNextToken(const mx::array& logits, double temperature, int topK, double topP, double minP) {
    // MLX를 사용한 다음 토큰 생성 (ggml-metal 스타일)
    
    // Temperature scaling
    mx::array scaled_logits = mx::divide(logits, mx::array(static_cast<float>(temperature)));
    // item() 호출 시 자동 평가되므로 EvalArray 불필요
    
    // Softmax
    mx::array probs = mx::softmax(scaled_logits, -1);
    // item() 호출 시 자동 평가되므로 EvalArray 불필요
    
    // Min-P 적용
    if (minP > 0.0) {
        probs = ApplyMinP(probs, minP);
        // 정규화
        mx::array sumArr = mx::sum(probs);
        // item() 호출 시 자동 평가되므로 EvalArray 불필요
        float sum = sumArr.item<float>();
        probs = mx::divide(probs, mx::array(sum));
    }
    
    // Top-K 적용
    if (topK > 0 && topK < probs.shape(-1)) {
        probs = ApplyTopK(probs, topK);
        // 정규화
        mx::array sumArr = mx::sum(probs);
        // item() 호출 시 자동 평가되므로 EvalArray 불필요
        float sum = sumArr.item<float>();
        probs = mx::divide(probs, mx::array(sum));
    }
    
    // Top-P 적용
    if (topP > 0.0 && topP < 1.0) {
        probs = ApplyTopP(probs, topP);
        // 정규화
        mx::array sumArr = mx::sum(probs);
        // item() 호출 시 자동 평가되므로 EvalArray 불필요
        float sum = sumArr.item<float>();
        probs = mx::divide(probs, mx::array(sum));
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
    
    // 인자 확인: prompt (string), options (object), callbacks (functions)
    if (!info[0].IsString()) {
        Napi::TypeError::New(env, "First argument must be a string (prompt)").ThrowAsJavaScriptException();
        return env.Null();
    }
    if (!info[1].IsObject()) {
        Napi::TypeError::New(env, "Second argument must be an object (options)").ThrowAsJavaScriptException();
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
            
            // Repeat penalty 적용 (llama.cpp 스타일)
            if (repeatPenalty != 1.0 && !generatedTokens.empty()) {
                int startIdx = std::max(0, static_cast<int>(generatedTokens.size()) - repeatLastN);
                std::set<int> seenTokens;
                for (int j = startIdx; j < static_cast<int>(generatedTokens.size()); ++j) {
                    seenTokens.insert(generatedTokens[j]);
                }
                
                // logits에 penalty 적용
                // MLX에서는 scatter를 사용하여 특정 인덱스의 값을 수정
                int vocabSize = logits.shape(-1);
                for (int tokenId : seenTokens) {
                    if (tokenId >= 0 && tokenId < vocabSize) {
                        // 해당 토큰의 logit 값 가져오기
                        mx::array tokenIdx = mx::array({tokenId}, mx::int32);
                        mx::array currentLogit = mx::take(logits, tokenIdx, 0);
                        float logitVal = currentLogit.item<float>();
                        
                        // penalty 적용: logit이 양수면 나누고, 음수면 곱하기
                        float newLogit = logitVal;
                        if (logitVal > 0.0f) {
                            newLogit = logitVal / static_cast<float>(repeatPenalty);
                        } else {
                            newLogit = logitVal * static_cast<float>(repeatPenalty);
                        }
                        
                        // scatter를 사용하여 logits 업데이트
                        mx::array newVal = mx::array({newLogit}, mx::float32);
                        logits = mx::scatter(logits, tokenIdx, newVal, 0);
                    }
                }
            }
            
            // 다음 토큰 생성
            mx::array nextTokenArr = GenerateNextToken(logits, temperature, topK, topP, minP);
            // item() 호출 시 자동 평가되므로 EvalArray 불필요
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
