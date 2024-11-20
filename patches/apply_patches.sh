#!/bin/bash

#Apply LLMFarm patches
patch ../Sources/llmfarm_core_cpp/llama.cpp/ggml/include/ggml.h < ggml.h.patch # ok
echo ggml.h patch appied
patch ../Sources/llmfarm_core_cpp/llama.cpp/ggml/src/ggml.c < ggml.c.patch # ok
echo ggml.c patch appied
patch ../Sources/llmfarm_core_cpp/llama.cpp/ggml/src/ggml-backend-reg.cpp < ggml-backend-reg.cpp.patch # ok
echo ggml-backend-reg.cpp patch appied
patch ../Sources/llmfarm_core_cpp/llama.cpp/examples/llava/clip.h < clip.h.patch
echo clip.h patch appied
# patch ../Sources/llmfarm_core_cpp/llama.cpp/examples/llava/llava.cpp < llava.cpp.patch
# echo llava.cpp patch appied
patch ../Sources/llmfarm_core_cpp/llama.cpp/examples/llava/llava-cli.cpp < llava-cli.cpp.patch
echo llava-cli.cpp.cpp patch appied
patch ../Sources/llmfarm_core_cpp/llama.cpp/examples/llava/clip.cpp < clip.cpp.patch
echo clip.cpp patch appied
# patch ../Sources/llmfarm_core_cpp/llama.cpp/ggml/src/llamafile/sgemm.cpp < sgemm.cpp.patch
# echo sgemm.cpp patch appied

