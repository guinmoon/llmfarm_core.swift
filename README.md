# LLMFarm_core.swift
LLMFarm_core swift library to work with large language models (LLM). It allows you to load different LLMs with certain parameters.<br>
Based on [ggml](https://github.com/ggerganov/ggml) and [llama.cpp](https://github.com/ggerganov/llama.cpp) by [Georgi Gerganov](https://github.com/ggerganov).

Also used sources from:
* [rwkv.cpp](https://github.com/saharNooby/rwkv.cpp) by [saharNooby](https://github.com/saharNooby).
* [Mia](https://github.com/byroneverson/Mia) by [byroneverson](https://github.com/byroneverson).

## Features

- [x] MacOS (13+)
- [x] iOS (16+)
- [x] Various inferences
- [x] Metal for llama inference (MacOS and iOS)
- [x] Model setting templates
- [x] Sampling from llama.cpp for other inference
- [ ] classifier-free guidance sampling from llama.cpp 
- [ ] Other tokenizers support
- [ ] Restore context state (now only chat history) 
- [ ] Metal for other inference

## Inferences

- [x] [LLaMA](https://arxiv.org/abs/2302.13971)
- [x] [GPTNeoX](https://huggingface.co/docs/transformers/model_doc/gpt_neox)
- [x] [Replit](https://huggingface.co/replit/replit-code-v1-3b)
- [x] [GPT2](https://huggingface.co/docs/transformers/model_doc/gpt2) + [Cerebras](https://arxiv.org/abs/2304.03208)
- [x] [Starcoder(Santacoder)](https://huggingface.co/bigcode/santacoder)
- [x] [RWKV](https://huggingface.co/docs/transformers/model_doc/rwkv) (20B tokenizer)
- [ ] [Falcon](https://github.com/cmp-nct/ggllm.cpp)


# Installation
```
git clone https://github.com/guinmoon/llmfarm_core.swift
```

## Installation

### Swift Package Manager

Add `llmfarm_core` to your project using Xcode (File > Add Packages...) or by adding it to your project's `Package.swift` file:

```swift
dependencies: [
  .package(url: "https://github.com/guinmoon/llmfarm_core.swift")
]
```

## Usage

### Swift library

Example generate output from a prompt:

```swift
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

let ai = AI(_modelPath: "orca-mini-3b.ggmlv3.q4_1.bin",_chatName: "chat")
var params:ModelContextParams = .default
params.use_metal = true

try? ai.loadModel(ModelInference.LLamaInference,contextParams: params)
ai.model.promptFormat = .LLaMa

let output = try? ai.model.predict(prompt, mainCallback)

```


#### Configuration


```swift

```