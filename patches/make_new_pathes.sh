#!/bin/bash

#make new pathes
diff -u ./update/ggml/include/ggml.h ../Sources/llmfarm_core_cpp/llama.cpp/ggml/include/ggml.h > ./new_patches/ggml.h.patch
diff -u ./update/ggml/src/ggml.c ../Sources/llmfarm_core_cpp/llama.cpp/ggml/src/ggml.c > ./new_patches/ggml.c.patch
diff -u ./update/ggml/src/ggml-backend.cpp ../Sources/llmfarm_core_cpp/llama.cpp/ggml/src/ggml-backend.cpp > ./new_patches/ggml-backend.cpp.patch
diff -u ./update/ggml/src/llamafile/sgemm.cpp ../Sources/llmfarm_core_cpp/llama.cpp/ggml/src/llamafile/sgemm.cpp > ./new_patches/sgemm.cpp.patch
diff -u ./update/examples/llava/llava.h ../Sources/llmfarm_core_cpp/spm-headers/llava.h > ./new_patches/llava.h.patch
diff -u ./update/examples/llava/llava.cpp ../Sources/llmfarm_core_cpp/llama.cpp/examples/llava/llava.cpp > ./new_patches/llava.cpp.patch
diff -u ./update/examples/llava/llava-cli.cpp ../Sources/llmfarm_core_cpp/llama.cpp/examples/llava/llava-cli.cpp > ./new_patches/llava-cli.cpp.patch
diff -u ./update/examples/llava/clip.cpp ../Sources/llmfarm_core_cpp/llama.cpp/examples/llava/clip.cpp > ./new_patches/clip.cpp.patch
diff -u ./update/examples/llava/clip.h ../Sources/llmfarm_core_cpp/llama.cpp/examples/llava/clip.h > ./new_patches/clip.h.patch
diff -u ./update/examples/export-lora/export-lora.cpp ../Sources/llmfarm_core_cpp/llama.cpp/examples/export-lora/export-lora.cpp > ./export-lora/export-lora.cpp.patch