package com.arm.aichat.internal;

import android.content.Context;
import android.util.Log;

import dalvik.annotation.optimization.FastNative;

import java.io.File;
import java.io.IOException;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.atomic.AtomicReference;

/**
 * JNI wrapper for the llama.cpp library providing Android-friendly access to large language models.
 * <p>
 * (Converted to Java) This class implements a singleton pattern. All operations are executed
 * on a dedicated single-threaded ExecutorService to ensure thread safety with the underlying C++ code.
 */
public class InferenceEngineImpl {

    private static final String TAG = InferenceEngineImpl.class.getSimpleName();

    private static volatile InferenceEngineImpl instance;

    // --- Native JNI Methods (DO NOT MODIFY, strictly maps to ai_chat.cpp) ---
    @FastNative private native void init(String nativeLibDir);
    @FastNative private native int load(String modelPath);
    @FastNative private native int prepare();
    @FastNative private native String systemInfo();
    @FastNative private native String benchModel(int pp, int tg, int pl, int nr);
    @FastNative private native int processSystemPrompt(String systemPrompt);
    @FastNative private native int processUserPrompt(String userPrompt, int predictLength);
    @FastNative private native String generateNextToken();
    @FastNative private native void unload();
    @FastNative private native void shutdown();
    // -------------------------------------------------------------------------

    // 重新定义的对外状态枚举
    public enum State {
        UNINITIALIZED, INITIALIZING, INITIALIZED, LOADING_MODEL, MODEL_READY,
        PROCESSING_SYSTEM_PROMPT, PROCESSING_USER_PROMPT, GENERATING, BENCHMARKING,
        UNLOADING_MODEL, ERROR
    }

    // 回调接口定义
    public interface ActionCallback {
        void onSuccess();
        void onError(Exception e);
    }

    public interface GenerationCallback {
        void onToken(String token);
        void onComplete();
        void onError(Exception e);
    }

    public interface BenchCallback {
        void onResult(String result);
        void onError(Exception e);
    }

    private final String nativeLibDir;
    private final AtomicReference<State> state = new AtomicReference<>(State.UNINITIALIZED);
    private volatile boolean readyForSystemPrompt = false;
    private volatile boolean cancelGeneration = false;

    /**
     * Single-threaded executor replaces Kotlin's Dispatchers.IO.limitedParallelism(1)
     * 保证所有 Native 调用在同一个子线程中串行执行，防止 C++ 崩溃
     */
    private final ExecutorService llamaExecutor = Executors.newSingleThreadExecutor();

    private InferenceEngineImpl(String nativeLibDir) {
        this.nativeLibDir = nativeLibDir;
        
        // 初始化 Native 库
        llamaExecutor.execute(() -> {
            try {
                if (state.get() != State.UNINITIALIZED) {
                    throw new IllegalStateException("Cannot load native library in " + state.get());
                }
                state.set(State.INITIALIZING);
                Log.i(TAG, "Loading native library...");
                System.loadLibrary("ai-chat");
                init(this.nativeLibDir);
                state.set(State.INITIALIZED);
                Log.i(TAG, "Native library loaded! System info: \n" + systemInfo());
            } catch (Exception e) {
                Log.e(TAG, "Failed to load native library", e);
                state.set(State.ERROR);
            }
        });
    }

    /**
     * 获取单例
     */
    public static InferenceEngineImpl getInstance(Context context) {
        if (instance == null) {
            synchronized (InferenceEngineImpl.class) {
                if (instance == null) {
                    String nativeLibDir = context.getApplicationInfo().nativeLibraryDir;
                    if (nativeLibDir == null || nativeLibDir.trim().isEmpty()) {
                        throw new IllegalArgumentException("Expected a valid native library path!");
                    }
                    try {
                        Log.i(TAG, "Instantiating InferenceEngineImpl...");
                        instance = new InferenceEngineImpl(nativeLibDir);
                    } catch (UnsatisfiedLinkError e) {
                        Log.e(TAG, "Failed to load native library from " + nativeLibDir, e);
                        throw e;
                    }
                }
            }
        }
        return instance;
    }

    public State getState() {
        return state.get();
    }
    
    public void cancelGeneration() {
        this.cancelGeneration = true;
    }

    /**
     * 异步加载模型
     */
    public void loadModel(String pathToModel, ActionCallback callback) {
        llamaExecutor.execute(() -> {
            try {
                if (state.get() != State.INITIALIZED) {
                    throw new IllegalStateException("Cannot load model in " + state.get() + "!");
                }

                Log.i(TAG, "Checking access to model file... \n" + pathToModel);
                File file = new File(pathToModel);
                if (!file.exists()) throw new IllegalArgumentException("File not found");
                if (!file.isFile()) throw new IllegalArgumentException("Not a valid file");
                if (!file.canRead()) throw new IllegalArgumentException("Cannot read file");

                Log.i(TAG, "Loading model... \n" + pathToModel);
                readyForSystemPrompt = false;
                state.set(State.LOADING_MODEL);
                
                int loadRes = load(pathToModel);
                if (loadRes != 0) {
                    // 原代码抛出了自定义的 UnsupportedArchitectureException，这里用通用异常代替
                    throw new RuntimeException("Unsupported Architecture or Load Failed. Code: " + loadRes);
                }
                
                int prepRes = prepare();
                if (prepRes != 0) {
                    throw new IOException("Failed to prepare resources");
                }
                
                Log.i(TAG, "Model loaded!");
                readyForSystemPrompt = true;
                cancelGeneration = false;
                state.set(State.MODEL_READY);
                
                if (callback != null) callback.onSuccess();
            } catch (Exception e) {
                Log.e(TAG, (e.getMessage() != null ? e.getMessage() : "Error loading model") + "\n" + pathToModel, e);
                state.set(State.ERROR);
                if (callback != null) callback.onError(e);
            }
        });
    }

    /**
     * 异步设置 System Prompt
     */
    public void setSystemPrompt(String prompt, ActionCallback callback) {
        llamaExecutor.execute(() -> {
            try {
                if (prompt == null || prompt.trim().isEmpty()) {
                    throw new IllegalArgumentException("Cannot process empty system prompt!");
                }
                if (!readyForSystemPrompt) {
                    throw new IllegalStateException("System prompt must be set ** RIGHT AFTER ** model loaded!");
                }
                if (state.get() != State.MODEL_READY) {
                    throw new IllegalStateException("Cannot process system prompt in " + state.get() + "!");
                }

                Log.i(TAG, "Sending system prompt...");
                readyForSystemPrompt = false;
                state.set(State.PROCESSING_SYSTEM_PROMPT);
                
                int result = processSystemPrompt(prompt);
                if (result != 0) {
                    throw new RuntimeException("Failed to process system prompt: " + result);
                }
                
                Log.i(TAG, "System prompt processed! Awaiting user prompt...");
                state.set(State.MODEL_READY);
                if (callback != null) callback.onSuccess();
            } catch (Exception e) {
                state.set(State.ERROR);
                if (callback != null) callback.onError(e);
            }
        });
    }

    /**
     * 异步发送 User Prompt 并通过回调接收 Token (替换了 Kotlin 的 Flow)
     */
    public void sendUserPrompt(String message, int predictLength, GenerationCallback callback) {
        llamaExecutor.execute(() -> {
            try {
                if (message == null || message.isEmpty()) {
                    throw new IllegalArgumentException("User prompt discarded due to being empty!");
                }
                if (state.get() != State.MODEL_READY) {
                    throw new IllegalStateException("User prompt discarded due to: " + state.get());
                }

                Log.i(TAG, "Sending user prompt...");
                readyForSystemPrompt = false;
                state.set(State.PROCESSING_USER_PROMPT);

                int result = processUserPrompt(message, predictLength);
                if (result != 0) {
                    Log.e(TAG, "Failed to process user prompt: " + result);
                    if (callback != null) callback.onError(new RuntimeException("Failed to process user prompt: " + result));
                    return;
                }

                Log.i(TAG, "User prompt processed. Generating assistant prompt...");
                state.set(State.GENERATING);
                
                while (!cancelGeneration) {
                    String utf8token = generateNextToken();
                    if (utf8token == null) break;
                    
                    if (!utf8token.isEmpty() && callback != null) {
                        callback.onToken(utf8token);
                    }
                }
                
                if (cancelGeneration) {
                    Log.i(TAG, "Assistant generation aborted per requested.");
                } else {
                    Log.i(TAG, "Assistant generation complete. Awaiting user prompt...");
                }
                
                state.set(State.MODEL_READY);
                if (callback != null) callback.onComplete();

            } catch (Exception e) {
                Log.e(TAG, "Error during generation!", e);
                state.set(State.ERROR);
                if (callback != null) callback.onError(e);
            }
        });
    }

    /**
     * 异步 Benchmark
     */
    public void bench(int pp, int tg, int pl, int nr, BenchCallback callback) {
        llamaExecutor.execute(() -> {
            try {
                if (state.get() != State.MODEL_READY) {
                    throw new IllegalStateException("Benchmark request discarded due to: " + state.get());
                }
                Log.i(TAG, "Start benchmark (pp: " + pp + ", tg: " + tg + ", pl: " + pl + ", nr: " + nr + ")");
                readyForSystemPrompt = false; 
                state.set(State.BENCHMARKING);
                
                String result = benchModel(pp, tg, pl, nr);
                
                state.set(State.MODEL_READY);
                if (callback != null) callback.onResult(result);
            } catch (Exception e) {
                if (callback != null) callback.onError(e);
            }
        });
    }

    /**
     * 卸载模型或重置错误状态 (阻塞式清理，替代 runBlocking)
     */
    public void cleanUp() {
        cancelGeneration = true;
        Future<?> future = llamaExecutor.submit(() -> {
            State currentState = state.get();
            if (currentState == State.MODEL_READY) {
                Log.i(TAG, "Unloading model and free resources...");
                readyForSystemPrompt = false;
                state.set(State.UNLOADING_MODEL);
                
                unload();
                
                state.set(State.INITIALIZED);
                Log.i(TAG, "Model unloaded!");
            } else if (currentState == State.ERROR) {
                Log.i(TAG, "Resetting error states...");
                state.set(State.INITIALIZED);
                Log.i(TAG, "States reset!");
            } else {
                throw new IllegalStateException("Cannot unload model in " + currentState);
            }
        });

        try {
            future.get(); // 阻塞当前线程等待清理完成
        } catch (Exception e) {
            Log.e(TAG, "Error during cleanUp", e);
        }
    }

    /**
     * 彻底销毁
     */
    public void destroy() {
        cancelGeneration = true;
        Future<?> future = llamaExecutor.submit(() -> {
            readyForSystemPrompt = false;
            State currentState = state.get();
            
            if (currentState == State.UNINITIALIZED) {
                // Do nothing
            } else if (currentState == State.INITIALIZED) {
                shutdown();
            } else {
                unload();
                shutdown();
            }
        });

        try {
            future.get(); // 阻塞等待 C++ 端安全释放
        } catch (Exception e) {
            Log.e(TAG, "Error during destroy", e);
        } finally {
            llamaExecutor.shutdown(); // 关闭线程池，替代 coroutine scope cancel
        }
    }
}