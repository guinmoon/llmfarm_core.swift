// swift-tools-version: 5.6
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "llmfarm_core",
    platforms: [.macOS(.v11),.iOS(.v15)],
    products: [
        .library(
            name: "llmfarm_core",
            targets: ["llmfarm_core"]),
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
            sources: ["ggml/ggml.m","exception_helper.mm","ggml/ggml-quants.m","ggml/ggml-alloc.m","ggml/ggml-backend.m","ggml/ggml-metal.m","ggml/common.mm",
                      "gpt_helpers.mm","gpt_spm.mm","package_helper.m","grammar-parser.mm","exception_helper_objc.mm",
                      "ggml/train.mm","finetune/finetune.mm","finetune/export-lora.mm","llama/llama.mm",
                      "ggml/ggml_d925ed.m","ggml/ggml_d925ed-alloc.m","ggml/ggml_d925ed-metal.m","rwkv/rwkv.mm",
                      "ggml/ggml_dadbed9.m","ggml/k_quants_dadbed9.m","ggml/ggml-alloc_dadbed9.m","ggml/ggml-metal_dadbed9.m",
                      "gptneox/gptneox.mm","gpt2/gpt2.mm","replit/replit.mm","starcoder/starcoder.mm","llama/llama_dadbed9.mm"
                      ],
            resources: [
                .copy("tokenizers"),
                .copy("metal")
            ],
            publicHeadersPath: "spm-headers",
            cSettings: [
                .define("SWIFT_PACKAGE"),
                .define("GGML_USE_ACCELERATE"),
                .define("ACCELERATE_NEW_LAPACK"),
                .define("ACCELERATE_LAPACK_ILP64"),
                .define("GGML_USE_METAL"),
//                .define("HAVE_BUGGY_APPLE_LINKER"),
//                .define("GGML_METAL_NDEBUG"),
                .define("NDEBUG"),
                //.define("GGML_QKK_64"), // Dont forget to comment this if you dont use QKK_64
                .unsafeFlags(["-Ofast"]), //comment this if you need to Debug llama_cpp
                .unsafeFlags(["-mfma","-mfma","-mavx","-mavx2","-mf16c","-msse3","-mssse3"]), //for Intel CPU
                .unsafeFlags(["-pthread"]),
                .unsafeFlags(["-fno-objc-arc"]),
                .unsafeFlags(["-Wno-shorten-64-to-32"]),
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
    cxxLanguageStandard: .cxx20
)

