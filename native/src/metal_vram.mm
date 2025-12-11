#include <napi.h>
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

Napi::Object GetVRAMInfo(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  Napi::Object result = Napi::Object::New(env);
  
  @autoreleasepool {
    // Metal 디바이스 가져오기
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    
    if (device == nil) {
      result.Set("error", Napi::String::New(env, "No Metal device found"));
      result.Set("total", Napi::Number::New(env, 0));
      result.Set("used", Napi::Number::New(env, 0));
      return result;
    }
    
    // 권장 최대 작업 세트 크기 (대략적인 VRAM 총량)
    uint64_t recommendedSize = [device recommendedMaxWorkingSetSize];
    
    // 현재 할당된 메모리 크기
    uint64_t allocatedSize = [device currentAllocatedSize];
    
    result.Set("total", Napi::Number::New(env, static_cast<double>(recommendedSize)));
    result.Set("used", Napi::Number::New(env, static_cast<double>(allocatedSize)));
    result.Set("error", env.Null());
  }
  
  return result;
}

Napi::Object Init(Napi::Env env, Napi::Object exports) {
  exports.Set(
    Napi::String::New(env, "getVRAMInfo"),
    Napi::Function::New(env, GetVRAMInfo)
  );
  return exports;
}

NODE_API_MODULE(metal_vram, Init)
