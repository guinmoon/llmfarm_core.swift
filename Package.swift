// swift-tools-version: 5.6
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "llmfarm_core",
    platforms: [.macOS(.v11),.iOS(.v15)],
    products: [
        .library(
            name: "llmfarm_core",
//            type: .dynamic,
            targets: ["llmfarm_core"]),
//        .library(
//            name: "llmfarm_core_cpp",
////            type: .dynamic,
//            targets: ["llmfarm_core_cpp"]),
    ],
    dependencies: [
        // Dependencies declare other packages that this package depends on.
    ],
    targets: [
        .target(
              name: "llmfarm_core",
              dependencies: ["llmfarm_core_cpp"],
              path: "Sources/llmfarm_core"),
        .target(
            name: "llmfarm_core_cpp",
            sources: ["ggml/ggml.c","exception_helper.cpp","ggml/ggml-quants.c","ggml/ggml-alloc.c","ggml/ggml-backend.c","ggml/ggml-metal.m",
                      "ggml/common.cpp","ggml/sampling.cpp","ggml/train.cpp","ggml/ggml-blas.cpp","ggml/build-info.cpp",
                      "llava/llava.cpp","llava/clip.cpp","llava/llava-cli.cpp","llama/unicode.cpp","llama/unicode-data.cpp","ggml/sgemm.cpp",
                      "gpt_helpers.cpp","gpt_spm.cpp","package_helper.m","ggml/grammar-parser.cpp","exception_helper_objc.mm","ggml/common_old.cpp",
                      "ggml/json-schema-to-grammar.cpp","finetune/finetune.cpp","finetune/export-lora.cpp","llama/llama.cpp",
                      "ggml/ggml_d925ed.c","ggml/ggml_d925ed-alloc.c","ggml/ggml_d925ed-metal.m","rwkv/rwkv.cpp",
                      "ggml/ggml_dadbed9.c","ggml/k_quants_dadbed9.c","ggml/ggml-alloc_dadbed9.c","ggml/ggml-metal_dadbed9.m",
                      "gptneox/gptneox.cpp","gpt2/gpt2.cpp","replit/replit.cpp","starcoder/starcoder.cpp","llama/llama_dadbed9.cpp"
                      ],
            resources: [
                .copy("tokenizers"),
                .process("ggml-metal.metal"),
                .copy("metal")
            ],
            publicHeadersPath: "spm-headers",
            cSettings: [
                .define("SWIFT_PACKAGE"),
                .define("GGML_USE_ACCELERATE"),
                .define("ACCELERATE_NEW_LAPACK"),
                .define("ACCELERATE_LAPACK_ILP64"),
                .define("GGML_USE_METAL"),
                .define("GGML_USE_BLAS"),
//                .define("_DARWIN_C_SOURCE"),
                .define("GGML_USE_LLAMAFILE"),
                .define("GGML_METAL_NDEBUG"),
                .define("NDEBUG"),
//                .define("GGML_METAL_NDEBUG", .when(configuration: .release)),
//                .define("NDEBUG", .when(configuration: .release)),
//                .unsafeFlags(["-Ofast"], .when(configuration: .release)), 
                .unsafeFlags(["-O3"]),
//                .unsafeFlags(["-O3"], .when(configuration: .release)),
                // .unsafeFlags(["-mfma","-mfma","-mavx","-mavx2","-mf16c","-msse3","-mssse3"]), //for Intel CPU
                .unsafeFlags(["-march=native","-mtune=native"],.when(platforms: [.macOS])),
//                .unsafeFlags(["-mcpu=apple-a14"],.when(platforms: [.iOS])),// use at your own risk, I've noticed more responsive work on 12 pro max
                .unsafeFlags(["-pthread"]),
                .unsafeFlags(["-fno-objc-arc"]),
//                .unsafeFlags(["-fPIC"]),
                .unsafeFlags(["-Wno-shorten-64-to-32"]),
//                .unsafeFlags(["-fno-finite-math-only"], .when(configuration: .release)),
                .unsafeFlags(["-w"]),    // ignore all warnings

                
            ],
            linkerSettings: [
                .linkedFramework("Foundation"),
                .linkedFramework("Accelerate"),
                .linkedFramework("Metal"),
                .linkedFramework("MetalKit"),
                .linkedFramework("MetalPerformanceShaders"),
            ]
        ),
        
    ],
    // cLanguageStandard: .c99,
    cxxLanguageStandard: .cxx11
)

// c++ -std=c++11 -fPIC -O3 -Wall -Wextra -Wpedantic -Wcast-qual -Wno-unused-function -Wmissing-declarations -Wmissing-noreturn -pthread  -march=native -mtune=native -Wunreachable-code-break -Wunreachable-code-return -Wmissing-prototypes -Wextra-semi -Iggml/include -Iggml/src -Iinclude -Isrc -Icommon -D_XOPEN_SOURCE=600 -D_DARWIN_C_SOURCE -DNDEBUG -DGGML_USE_ACCELERATE -DGGML_USE_BLAS -DACCELERATE_NEW_LAPACK -DACCELERATE_LAPACK_ILP64 -DGGML_USE_LLAMAFILE -DGGML_USE_METAL -DGGML_METAL_EMBED_LIBRARY  -c src/llama.cpp -o src/llama.o
