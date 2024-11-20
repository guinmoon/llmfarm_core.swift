**The code in this repository is under constant revision/refactoring. You could say that I am learning Swift as I develop LLMFarm. So don't expect too much from the code you'll find here. 
If you have any useful comments on the code, its style or architecture, I will be glad to hear them.**

# LLMFarm_core.swift
LLMFarm_core swift library to work with large language models (LLM). It allows you to load different LLMs with certain parameters.<br>
Based on [ggml](https://github.com/ggerganov/ggml) and [llama.cpp](https://github.com/ggerganov/llama.cpp) by [Georgi Gerganov](https://github.com/ggerganov).

# Features

- [x] MacOS (13+)
- [x] iOS (16+)
- [x] Various inferences
- [x] Various sampling methods
- [x] Metal ([dont work](https://github.com/ggerganov/llama.cpp/issues/2407#issuecomment-1699544808) on intel Mac)
- [x] Model setting templates
- [ ] LoRA adapters support ([read more](https://github.com/guinmoon/LLMFarm/blob/main/lora.md))
- [ ] LoRA train support
- [ ] LoRA export as model support
- [ ] Restore context state (now only chat history) 

# Inferences

See full list [here](https://github.com/ggerganov/llama.cpp).
  
# Sampling methods
- [x] Temperature (temp, tok-k, top-p)
- [x] [Tail Free Sampling (TFS)](https://www.trentonbricken.com/Tail-Free-Sampling/)
- [x] [Locally Typical Sampling](https://arxiv.org/abs/2202.00666)
- [x] [Mirostat](https://arxiv.org/abs/2007.14966)
- [x] Greedy
- [x] Grammar 
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

# Usage

## [See examples in the Demo Project](/DemoProject)

## Also used sources from:
* [rwkv.cpp](https://github.com/saharNooby/rwkv.cpp) by [saharNooby](https://github.com/saharNooby).
* [Mia](https://github.com/byroneverson/Mia) by [byroneverson](https://github.com/byroneverson).
* [swift-markdown-ui](https://github.com/gonzalezreal/swift-markdown-ui) by [gonzalezreal](https://github.com/gonzalezreal)

## Projects based on this library
 * ### [LLM Farm](https://github.com/guinmoon/LLMFarm)
App to run LLaMA and other large language models locally on iOS and MacOS.
