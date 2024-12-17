// swift-tools-version: 5.6
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription



var sources = [ "llama.cpp/ggml/src/ggml.c",
                "llama.cpp/ggml/src/ggml-quants.c",
                "llama.cpp/ggml/src/ggml-alloc.c",
                "llama.cpp/ggml/src/ggml-backend.cpp",
                "llama.cpp/ggml/src/ggml-threading.cpp",
                "llama.cpp/ggml/src/ggml-backend-reg.cpp",
                "llama.cpp/ggml/src/ggml-metal/ggml-metal.m",
                "llama.cpp/ggml/src/ggml-blas/ggml-blas.cpp",
                "llama.cpp/ggml/src/ggml-aarch64.c",
                "llama.cpp/ggml/src/ggml-cpu/ggml-cpu-aarch64.c",
                "llama.cpp/ggml/src/ggml-cpu/ggml-cpu.c",
                "llama.cpp/ggml/src/ggml-cpu/ggml-cpu.cpp",
                "llama.cpp/ggml/src/ggml-cpu/ggml-cpu-quants.c",
                "llama.cpp/ggml/src/ggml-cpu/ggml-cpu-traits.cpp",
                "llama.cpp/ggml/src/ggml-cpu/llamafile/sgemm.cpp",
                "llama.cpp/src/llama.cpp",
                "llama.cpp/src/unicode.cpp",
                "llama.cpp/src/unicode-data.cpp",
                "llama.cpp/src/llama-grammar.cpp",
                "llama.cpp/src/llama-vocab.cpp",
                "llama.cpp/src/llama-sampling.cpp",
                "llama.cpp/common/common.cpp",
                "llama.cpp/common/log.cpp",
                "llama.cpp/common/arg.cpp",
                "llama.cpp/common/json-schema-to-grammar.cpp",
                "llama.cpp/common/sampling.cpp",
                // "llama.cpp/common/train.cpp",
                "llama.cpp/examples/llava/llava.cpp",
                "llama.cpp/examples/llava/clip.cpp",
                "llama.cpp/examples/llava/llava-cli.cpp",
                // "llama.cpp/examples/export-lora/export-lora.cpp",                
                "gpt_spm.cpp",
                "package_helper.m",
                "exception_helper_objc.mm",
                "exception_helper.cpp",                   
                // "ggml_legacy/ggml_d925ed.c","ggml_legacy/ggml_d925ed-alloc.c","ggml_legacy/ggml_d925ed-metal.m","rwkv/rwkv.cpp",
                // "ggml_legacy/ggml_dadbed9.c","ggml_legacy/k_quants_dadbed9.c","ggml_legacy/ggml-alloc_dadbed9.c","ggml_legacy/ggml-metal_dadbed9.m",
                // "gptneox/gptneox.cpp","gpt2/gpt2.cpp","replit/replit.cpp","starcoder/starcoder.cpp","llama_legacy/llama_dadbed9.cpp",
                // "ggml_legacy/common_old.cpp",      
                // "ggml_legacy/build-info.cpp",          
                // "finetune/finetune.cpp",                        
                ]

var sources_legacy = [ 
                "ggml_legacy/ggml_d925ed.c","ggml_legacy/ggml_d925ed-alloc.c","ggml_legacy/ggml_d925ed-metal.m","rwkv/rwkv.cpp",
                "ggml_legacy/ggml_dadbed9.c","ggml_legacy/k_quants_dadbed9.c","ggml_legacy/ggml-alloc_dadbed9.c","ggml_legacy/ggml-metal_dadbed9.m",
                "gptneox/gptneox.cpp","gpt2/gpt2.cpp","replit/replit.cpp","starcoder/starcoder.cpp","llama_legacy/llama_dadbed9.cpp",
                "ggml_legacy/common_old.cpp",      
                "ggml_legacy/build-info.cpp",                                                  
                ]

var cSettings: [CSetting] =  [
                .define("SWIFT_PACKAGE"),
                .define("GGML_USE_ACCELERATE"),
                .define("GGML_BLAS_USE_ACCELERATE"),
                .define("ACCELERATE_NEW_LAPACK"),
                .define("ACCELERATE_LAPACK_ILP64"),
                .define("GGML_USE_BLAS"),
//                .define("_DARWIN_C_SOURCE"),
                .define("GGML_USE_LLAMAFILE"),
                .define("GGML_METAL_NDEBUG"),
                .define("NDEBUG"),
                .define("GGML_USE_CPU"),
                .define("GGML_USE_METAL"),
                
//                .define("GGML_METAL_NDEBUG", .when(configuration: .release)),
//                .define("NDEBUG", .when(configuration: .release)),
                .unsafeFlags(["-Ofast"], .when(configuration: .release)), 
//                .unsafeFlags(["-O3"]),
                .unsafeFlags(["-O3"], .when(configuration: .debug)),
                 .unsafeFlags(["-mfma","-mfma","-mavx","-mavx2","-mf16c","-msse3","-mssse3"]), //for Intel CPU
//                .unsafeFlags(["-march=native","-mtune=native"],.when(platforms: [.macOS])),
//                .unsafeFlags(["-mcpu=apple-a14"],.when(platforms: [.iOS])),// use at your own risk, I've noticed more responsive work on 12 pro max
                .unsafeFlags(["-pthread"]),
                .unsafeFlags(["-fno-objc-arc"]),
//                .unsafeFlags(["-fPIC"]),
                .unsafeFlags(["-Wno-shorten-64-to-32"]),
                .unsafeFlags(["-fno-finite-math-only"], .when(configuration: .release)),
                .unsafeFlags(["-w"]),    // ignore all warnings

                .headerSearchPath("llama.cpp/common"),
                .headerSearchPath("llama.cpp/ggml/include"),
                .headerSearchPath("llama.cpp/ggml/src"),
                .headerSearchPath("llama.cpp/ggml/src/ggml-cpu"),
                
            ]



var linkerSettings: [LinkerSetting] = [
                .linkedFramework("Foundation"),
                .linkedFramework("Accelerate"),
                .linkedFramework("Metal"),
                .linkedFramework("MetalKit"),
                .linkedFramework("MetalPerformanceShaders"),
                ]

var resources: [Resource] = [
                // .copy("tokenizers"),
                .process("llama.cpp/ggml/src/ggml-metal.metal"),
                // .copy("metal")
            ]

let package = Package(
    name: "llmfarm_core",
    platforms: [.macOS(.v11),.iOS(.v15)],
    products: [
        .library(
            name: "llmfarm_core",
//           type: .dynamic,
            targets: ["llmfarm_core"]),
//       .library(
//           name: "llmfarm_core_cpp",
//           targets: ["llmfarm_core_cpp"]),
    //    .library(
    //        name: "llmfarm_core_cpp_legacy",
    //        targets: ["llmfarm_core_cpp_legacy"]),
    ],
    dependencies: [
        // Dependencies declare other packages that this package depends on.
    ],
    targets: [
        .target(
              name: "llmfarm_core",
              dependencies: ["llmfarm_core_cpp"/*,"llmfarm_core_cpp_legacy"*/],
              path: "Sources/llmfarm_core"),
        .target(
            name: "llmfarm_core_cpp",
            sources: sources,
            resources: resources,
            publicHeadersPath: "spm-headers",
            cSettings:cSettings,
            linkerSettings: linkerSettings
        ),
        // .target(
        //     name: "llmfarm_core_cpp_legacy",
        //     sources: sources_legacy,
        //     resources: resources,
        //     publicHeadersPath: "spm-headers",
        //     cSettings:cSettings,
        //     linkerSettings: linkerSettings
        // ),
        
    ],
    // cLanguageStandard: .c99,
//    cxxLanguageStandard: .cxx11
    cxxLanguageStandard: .cxx17
)

// c++ -std=c++11 -fPIC -O3 -Wall -Wextra -Wpedantic -Wcast-qual -Wno-unused-function -Wmissing-declarations -Wmissing-noreturn -pthread  -march=native -mtune=native -Wunreachable-code-break -Wunreachable-code-return -Wmissing-prototypes -Wextra-semi -Iggml/include -Iggml/src -Iinclude -Isrc -Icommon -D_XOPEN_SOURCE=600 -D_DARWIN_C_SOURCE -DNDEBUG -DGGML_USE_ACCELERATE -DGGML_USE_BLAS -DACCELERATE_NEW_LAPACK -DACCELERATE_LAPACK_ILP64 -DGGML_USE_LLAMAFILE -DGGML_USE_METAL -DGGML_METAL_EMBED_LIBRARY  -c src/llama.cpp -o src/llama.o
