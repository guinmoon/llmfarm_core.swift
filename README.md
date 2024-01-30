# LLMFarm_core.swift
LLMFarm_core swift library to work with large language models (LLM). It allows you to load different LLMs with certain parameters.<br>
Based on [ggml](https://github.com/ggerganov/ggml) and [llama.cpp](https://github.com/ggerganov/llama.cpp) by [Georgi Gerganov](https://github.com/ggerganov).

Also used sources from:
* [rwkv.cpp](https://github.com/saharNooby/rwkv.cpp) by [saharNooby](https://github.com/saharNooby).
* [Mia](https://github.com/byroneverson/Mia) by [byroneverson](https://github.com/byroneverson).

# Features

- [x] MacOS (13+)
- [x] iOS (16+)
- [x] Various inferences
- [x] Various sampling methods
- [x] Metal ([dont work](https://github.com/ggerganov/llama.cpp/issues/2407#issuecomment-1699544808) on intel Mac)
- [x] Model setting templates
- [x] LoRA adapters support ([read more](https://github.com/guinmoon/LLMFarm/blob/main/lora.md))
- [x] LoRA train support
- [ ] LoRA export as model support
- [ ] Restore context state (now only chat history) 

# Inferences

- [x] [LLaMA](https://arxiv.org/abs/2302.13971) <img src="dist/metal-96x96_2x.png" width="16px" heigth="16px">
- [x] [GPTNeoX](https://huggingface.co/docs/transformers/model_doc/gpt_neox)
- [x] [Replit](https://huggingface.co/replit/replit-code-v1-3b)
- [x] [GPT2](https://huggingface.co/docs/transformers/model_doc/gpt2) + [Cerebras](https://arxiv.org/abs/2304.03208) <img src="dist/metal-96x96_2x.png" width="16px" heigth="16px">
- [x] [Starcoder(Santacoder)](https://huggingface.co/bigcode/santacoder) <img src="dist/metal-96x96_2x.png" width="16px" heigth="16px">
- [x] [RWKV](https://huggingface.co/docs/transformers/model_doc/rwkv) (20B tokenizer)
- [x] [Falcon](https://github.com/cmp-nct/ggllm.cpp) <img src="dist/metal-96x96_2x.png" width="16px" heigth="16px">
- [x] [MPT](https://huggingface.co/guinmoon/mpt-7b-storywriter-GGUF) <img src="dist/metal-96x96_2x.png" width="16px" heigth="16px">
- [x] [Bloom](https://huggingface.co/guinmoon/bloomz-1b7-gguf) <img src="dist/metal-96x96_2x.png" width="16px" heigth="16px">
- [x] [StableLM-3b-4e1t](https://huggingface.co/stabilityai/stablelm-3b-4e1t) <img src="dist/metal-96x96_2x.png" width="16px" heigth="16px">
- [x] [Qwen](https://huggingface.co/Qwen/Qwen-7B) <img src="dist/metal-96x96_2x.png" width="16px" heigth="16px">
- [x] [Yi models](https://huggingface.co/models?search=01-ai/Yi) <img src="dist/metal-96x96_2x.png" width="16px" heigth="16px">
- [x] [Deepseek models](https://huggingface.co/models?search=deepseek-ai/deepseek) <img src="dist/metal-96x96_2x.png" width="16px" heigth="16px">
- [x] [Mixtral MoE](https://huggingface.co/models?search=mistral-ai/Mixtral) <img src="dist/metal-96x96_2x.png" width="16px" heigth="16px">
- [x] [PLaMo-13B](https://github.com/ggerganov/llama.cpp/pull/3557) <img src="dist/metal-96x96_2x.png" width="16px" heigth="16px">
  
Note: For *Falcon, Alpaca, GPT4All, Chinese LLaMA / Alpaca and Chinese LLaMA-2 / Alpaca-2, Vigogne (French), Vicuna, Koala, OpenBuddy (Multilingual), Pygmalion/Metharme, WizardLM, Baichuan 1 & 2 + derivations, Aquila 1 & 2, Mistral AI v0.1, Refact, Persimmon 8B, MPT, Bloom* select `llama inferece` in model settings.

# Sampling methods
- [x] Temperature (temp, tok-k, top-p)
- [x] [Tail Free Sampling (TFS)](https://www.trentonbricken.com/Tail-Free-Sampling/)
- [x] [Locally Typical Sampling](https://arxiv.org/abs/2202.00666)
- [x] [Mirostat](https://arxiv.org/abs/2007.14966)
- [x] Greedy
- [x] Grammar (dont work for GPTNeoX, GPT-2, RWKV)
- [ ] Classifier-Free Guidance


# Installation
```
git clone https://github.com/guinmoon/llmfarm_core.swift
```

## Swift Package Manager

Add `llmfarm_core` to your project using Xcode (File > Add Packages...) or by adding it to your project's `Package.swift` file:

```swift
dependencies: [
  .package(url: "https://github.com/guinmoon/llmfarm_core.swift")
]
```

## Build and Debug 

To Debug `llmfarm_core` package, do not forget to comment `.unsafeFlags(["-Ofast"])` in `Package.swift`.
Don't forget that the debug version is slower than the release version.

To build with `QKK_64` support uncomment `.unsafeFlags(["-DGGML_QKK_64"])` in `Package.swift`.

# Usage

## [More examples in the Demo Project](/DemoProject)

### Example generate output from a prompt


```swift
import Foundation
import llmfarm_core

let maxOutputLength = 256
var total_output = 0

func mainCallback(_ str: String, _ time: Double) -> Bool {
    print("\(str)",terminator: "")
    total_output += str.count
    if(total_output>maxOutputLength){
        return true
    }
    return false
}



let ai = AI(_modelPath: "/Users/guinmoon/dev/alpaca_llama_etc/llama-2-7b-chat-q4_K_M.gguf",_chatName: "chat")
var params:ModelAndContextParams = .default
params.promptFormat = .Custom
params.custom_prompt_format = """
SYSTEM: You are a helpful, respectful and honest assistant.
USER: State the meaning of life
ASSISTANT:
"""
var input_text = "State the meaning of life"
params.use_metal = true

_ = try? ai.loadModel(ModelInference.LLama_gguf,contextParams: params)

let output = try? ai.model.predict(input_text, mainCallback)


```

# Projects based on this library
 * ## [LLM Farm](https://github.com/guinmoon/LLMFarm)
App to run LLaMA and other large language models locally on iOS and MacOS.