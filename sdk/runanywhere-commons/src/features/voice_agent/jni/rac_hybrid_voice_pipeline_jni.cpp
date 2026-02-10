/**
 * @file rac_hybrid_voice_pipeline_jni.cpp
 * @brief JNI Bridge for HybridVoicePipeline (Android)
 */

#include <jni.h>
#include <string>

#include "rac/features/voice_agent/hybrid_voice_pipeline.h"

#ifdef __ANDROID__
#include <android/log.h>
#define LOG_TAG "HybridPipelineJNI"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)
#endif

extern "C" {

// =============================================================================
// Lifecycle
// =============================================================================

JNIEXPORT jlong JNICALL
Java_com_runanywhere_voice_HybridVoicePipeline_nativeCreate(
    JNIEnv* env, jclass clazz) {

    LOGI("Creating HybridVoicePipeline");
    return reinterpret_cast<jlong>(new rac::voice::HybridVoicePipeline());
}

JNIEXPORT void JNICALL
Java_com_runanywhere_voice_HybridVoicePipeline_nativeDestroy(
    JNIEnv* env, jclass clazz, jlong handle) {

    auto* pipeline = reinterpret_cast<rac::voice::HybridVoicePipeline*>(handle);
    if (pipeline) {
        pipeline->cleanup();
        delete pipeline;
    }
}

JNIEXPORT jboolean JNICALL
Java_com_runanywhere_voice_HybridVoicePipeline_nativeInitialize(
    JNIEnv* env, jclass clazz, jlong handle,
    // Memory config
    jlong llmKvCacheMb, jlong asrBufferMb, jlong ttsBufferMb, jlong scratchMb,
    // Model paths
    jstring asrModelPath, jstring asrLanguage,
    jstring llmModelPath, jstring systemPrompt, jint llmMaxTokens, jfloat llmTemperature,
    jstring ttsModelPath, jstring ttsVoiceId, jint ttsSampleRate,
    // Options
    jint numThreads) {

    auto* pipeline = reinterpret_cast<rac::voice::HybridVoicePipeline*>(handle);
    if (!pipeline) {
        return JNI_FALSE;
    }

    rac::voice::HybridPipelineConfig config;

    // Memory config
    config.memory.llm_kv_cache_mb = static_cast<size_t>(llmKvCacheMb);
    config.memory.asr_buffer_mb = static_cast<size_t>(asrBufferMb);
    config.memory.tts_buffer_mb = static_cast<size_t>(ttsBufferMb);
    config.memory.scratch_mb = static_cast<size_t>(scratchMb);

    // Model paths
    if (asrModelPath) {
        const char* path = env->GetStringUTFChars(asrModelPath, nullptr);
        config.models.asr_model_path = path;
        env->ReleaseStringUTFChars(asrModelPath, path);
    }
    if (asrLanguage) {
        const char* lang = env->GetStringUTFChars(asrLanguage, nullptr);
        config.models.asr_language = lang;
        env->ReleaseStringUTFChars(asrLanguage, lang);
    }
    if (llmModelPath) {
        const char* path = env->GetStringUTFChars(llmModelPath, nullptr);
        config.models.llm_model_path = path;
        env->ReleaseStringUTFChars(llmModelPath, path);
    }
    if (systemPrompt) {
        const char* prompt = env->GetStringUTFChars(systemPrompt, nullptr);
        config.models.system_prompt = prompt;
        env->ReleaseStringUTFChars(systemPrompt, prompt);
    }
    config.models.llm_max_tokens = llmMaxTokens > 0 ? llmMaxTokens : 256;
    config.models.llm_temperature = llmTemperature > 0 ? llmTemperature : 0.7f;

    if (ttsModelPath) {
        const char* path = env->GetStringUTFChars(ttsModelPath, nullptr);
        config.models.tts_model_path = path;
        env->ReleaseStringUTFChars(ttsModelPath, path);
    }
    if (ttsVoiceId) {
        const char* voice = env->GetStringUTFChars(ttsVoiceId, nullptr);
        config.models.tts_voice_id = voice;
        env->ReleaseStringUTFChars(ttsVoiceId, voice);
    }
    config.models.tts_sample_rate = ttsSampleRate > 0 ? ttsSampleRate : 22050;

    config.num_threads = numThreads;

    return pipeline->initialize(config) ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT jboolean JNICALL
Java_com_runanywhere_voice_HybridVoicePipeline_nativeIsReady(
    JNIEnv* env, jclass clazz, jlong handle) {

    auto* pipeline = reinterpret_cast<rac::voice::HybridVoicePipeline*>(handle);
    return pipeline && pipeline->is_ready() ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT void JNICALL
Java_com_runanywhere_voice_HybridVoicePipeline_nativeCleanup(
    JNIEnv* env, jclass clazz, jlong handle) {

    auto* pipeline = reinterpret_cast<rac::voice::HybridVoicePipeline*>(handle);
    if (pipeline) {
        pipeline->cleanup();
    }
}

// =============================================================================
// Individual Stage Processing
// =============================================================================

JNIEXPORT jstring JNICALL
Java_com_runanywhere_voice_HybridVoicePipeline_nativeTranscribe(
    JNIEnv* env, jclass clazz, jlong handle, jfloatArray audioData) {

    auto* pipeline = reinterpret_cast<rac::voice::HybridVoicePipeline*>(handle);
    if (!pipeline) {
        return env->NewStringUTF("");
    }

    jsize length = env->GetArrayLength(audioData);
    jfloat* audioPtr = env->GetFloatArrayElements(audioData, nullptr);

    auto result = pipeline->transcribe(audioPtr, static_cast<size_t>(length));

    env->ReleaseFloatArrayElements(audioData, audioPtr, JNI_ABORT);

    return env->NewStringUTF(result.text.c_str());
}

JNIEXPORT jstring JNICALL
Java_com_runanywhere_voice_HybridVoicePipeline_nativeGenerate(
    JNIEnv* env, jclass clazz, jlong handle, jstring input) {

    auto* pipeline = reinterpret_cast<rac::voice::HybridVoicePipeline*>(handle);
    if (!pipeline || !input) {
        return env->NewStringUTF("");
    }

    const char* inputChars = env->GetStringUTFChars(input, nullptr);
    auto result = pipeline->generate_response(inputChars);
    env->ReleaseStringUTFChars(input, inputChars);

    return env->NewStringUTF(result.text.c_str());
}

JNIEXPORT jfloatArray JNICALL
Java_com_runanywhere_voice_HybridVoicePipeline_nativeSynthesize(
    JNIEnv* env, jclass clazz, jlong handle, jstring text) {

    auto* pipeline = reinterpret_cast<rac::voice::HybridVoicePipeline*>(handle);
    if (!pipeline || !text) {
        return nullptr;
    }

    const char* textChars = env->GetStringUTFChars(text, nullptr);
    auto result = pipeline->synthesize(textChars);
    env->ReleaseStringUTFChars(text, textChars);

    if (result.audio.empty()) {
        return nullptr;
    }

    jfloatArray output = env->NewFloatArray(static_cast<jsize>(result.audio.size()));
    env->SetFloatArrayRegion(output, 0,
        static_cast<jsize>(result.audio.size()), result.audio.data());

    return output;
}

// =============================================================================
// Full Pipeline with Result Object
// =============================================================================

JNIEXPORT jobject JNICALL
Java_com_runanywhere_voice_HybridVoicePipeline_nativeProcess(
    JNIEnv* env, jclass clazz, jlong handle, jfloatArray audioData) {

    auto* pipeline = reinterpret_cast<rac::voice::HybridVoicePipeline*>(handle);
    if (!pipeline) {
        return nullptr;
    }

    jsize length = env->GetArrayLength(audioData);
    jfloat* audioPtr = env->GetFloatArrayElements(audioData, nullptr);

    auto result = pipeline->process(audioPtr, static_cast<size_t>(length));

    env->ReleaseFloatArrayElements(audioData, audioPtr, JNI_ABORT);

    // Find the PipelineResult class
    jclass resultClass = env->FindClass("com/runanywhere/voice/PipelineResult");
    if (!resultClass) {
        LOGE("PipelineResult class not found");
        return nullptr;
    }

    // Get constructor
    jmethodID constructor = env->GetMethodID(resultClass, "<init>",
        "(Ljava/lang/String;FDLjava/lang/String;IDLjava/lang/String;[FIDZZ)V");
    if (!constructor) {
        LOGE("PipelineResult constructor not found");
        return nullptr;
    }

    // Create audio array
    jfloatArray ttsAudio = nullptr;
    if (!result.tts.audio.empty()) {
        ttsAudio = env->NewFloatArray(static_cast<jsize>(result.tts.audio.size()));
        env->SetFloatArrayRegion(ttsAudio, 0,
            static_cast<jsize>(result.tts.audio.size()), result.tts.audio.data());
    }

    // Create result object
    return env->NewObject(resultClass, constructor,
        env->NewStringUTF(result.asr.text.c_str()),
        result.asr.confidence,
        result.asr.latency_ms,
        env->NewStringUTF(result.llm.text.c_str()),
        result.llm.tokens_generated,
        result.llm.latency_ms,
        ttsAudio ? env->NewStringUTF("") : env->NewStringUTF("No audio"),
        ttsAudio,
        result.tts.sample_rate,
        result.tts.latency_ms,
        result.total_latency_ms,
        result.success ? JNI_TRUE : JNI_FALSE);
}

// =============================================================================
// Memory Management
// =============================================================================

JNIEXPORT void JNICALL
Java_com_runanywhere_voice_HybridVoicePipeline_nativeResetForNextUtterance(
    JNIEnv* env, jclass clazz, jlong handle) {

    auto* pipeline = reinterpret_cast<rac::voice::HybridVoicePipeline*>(handle);
    if (pipeline) {
        pipeline->reset_for_next_utterance();
    }
}

JNIEXPORT jlong JNICALL
Java_com_runanywhere_voice_HybridVoicePipeline_nativeGetTotalMemory(
    JNIEnv* env, jclass clazz, jlong handle) {

    auto* pipeline = reinterpret_cast<rac::voice::HybridVoicePipeline*>(handle);
    return pipeline ? static_cast<jlong>(pipeline->get_total_memory_allocated()) : 0;
}

JNIEXPORT jlong JNICALL
Java_com_runanywhere_voice_HybridVoicePipeline_nativeGetMemoryUsed(
    JNIEnv* env, jclass clazz, jlong handle) {

    auto* pipeline = reinterpret_cast<rac::voice::HybridVoicePipeline*>(handle);
    return pipeline ? static_cast<jlong>(pipeline->get_memory_used()) : 0;
}

// =============================================================================
// Model Management
// =============================================================================

JNIEXPORT jboolean JNICALL
Java_com_runanywhere_voice_HybridVoicePipeline_nativeReloadAsr(
    JNIEnv* env, jclass clazz, jlong handle, jstring modelPath) {

    auto* pipeline = reinterpret_cast<rac::voice::HybridVoicePipeline*>(handle);
    if (!pipeline || !modelPath) {
        return JNI_FALSE;
    }

    const char* path = env->GetStringUTFChars(modelPath, nullptr);
    bool result = pipeline->reload_asr_model(path);
    env->ReleaseStringUTFChars(modelPath, path);

    return result ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT jboolean JNICALL
Java_com_runanywhere_voice_HybridVoicePipeline_nativeReloadLlm(
    JNIEnv* env, jclass clazz, jlong handle, jstring modelPath) {

    auto* pipeline = reinterpret_cast<rac::voice::HybridVoicePipeline*>(handle);
    if (!pipeline || !modelPath) {
        return JNI_FALSE;
    }

    const char* path = env->GetStringUTFChars(modelPath, nullptr);
    bool result = pipeline->reload_llm_model(path);
    env->ReleaseStringUTFChars(modelPath, path);

    return result ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT jboolean JNICALL
Java_com_runanywhere_voice_HybridVoicePipeline_nativeReloadTts(
    JNIEnv* env, jclass clazz, jlong handle, jstring modelPath) {

    auto* pipeline = reinterpret_cast<rac::voice::HybridVoicePipeline*>(handle);
    if (!pipeline || !modelPath) {
        return JNI_FALSE;
    }

    const char* path = env->GetStringUTFChars(modelPath, nullptr);
    bool result = pipeline->reload_tts_model(path);
    env->ReleaseStringUTFChars(modelPath, path);

    return result ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT void JNICALL
Java_com_runanywhere_voice_HybridVoicePipeline_nativeSetSystemPrompt(
    JNIEnv* env, jclass clazz, jlong handle, jstring prompt) {

    auto* pipeline = reinterpret_cast<rac::voice::HybridVoicePipeline*>(handle);
    if (!pipeline || !prompt) {
        return;
    }

    const char* promptChars = env->GetStringUTFChars(prompt, nullptr);
    pipeline->set_system_prompt(promptChars);
    env->ReleaseStringUTFChars(prompt, promptChars);
}

}  // extern "C"
