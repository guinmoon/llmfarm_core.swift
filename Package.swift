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
        // .package(url: /* package url */, from: "1.0.0"),
    ],
    targets: [
        .target(
              name: "llmfarm_core",
              dependencies: ["llmfarm_core_cpp"],
              path: "Sources/llmfarm_core"),
        .target(
            name: "llmfarm_core_cpp",
            sources: ["ggml/ggml.c","exception_helper.cpp","ggml/ggml-quants.c","ggml/ggml-alloc.c","ggml/ggml-backend.c","ggml/ggml-metal.m","ggml/common.cpp",
                      "gpt_helpers.cpp","gpt_spm.cpp","package_helper.m","grammar-parser.cpp","exception_helper_objc.mm",
                      "ggml/train.cpp","finetune/finetune.cpp","llama/llama.cpp",
                      "ggml/ggml_d925ed.c","ggml/ggml_d925ed-alloc.c","ggml/ggml_d925ed-metal.m","rwkv/rwkv.cpp",
                      "ggml/ggml_dadbed9.c","ggml/k_quants_dadbed9.c","ggml/ggml-alloc_dadbed9.c","ggml/ggml-metal_dadbed9.m",
                      "gptneox/gptneox.cpp","gpt2/gpt2.cpp","replit/replit.cpp","starcoder/starcoder.cpp","llama/llama_dadbed9.cpp"
                      ],
            resources: [
                .copy("tokenizers"),
                .copy("metal")
            ],
            publicHeadersPath: "spm-headers",
            //            I'm not sure about some of the flags, please correct it's wrong.
            cSettings: [
                .unsafeFlags(["-Ofast"]), //comment this if you need to Debug llama_cpp                
//                .unsafeFlags(["-O3"]),
                .unsafeFlags(["-DNDEBUG"]),
//                .unsafeFlags(["-mfma","-mfma","-mavx","-mavx2","-mf16c","-msse3","-mssse3"]), //for Intel CPU
                .unsafeFlags(["-DHAVE_BUGGY_APPLE_LINKER"]),
                .unsafeFlags(["-DGGML_METAL_NDEBUG"]),
                .unsafeFlags(["-DGGML_USE_ACCELERATE"]),
                .unsafeFlags(["-DACCELERATE_NEW_LAPACK"]),
                .unsafeFlags(["-DACCELERATE_LAPACK_ILP64"]),
                .unsafeFlags(["-DGGML_USE_METAL"]),
                .unsafeFlags(["-DSWIFT_PACKAGE"]),
                .unsafeFlags(["-pthread"]),
                .unsafeFlags(["-fno-objc-arc"]),
                .unsafeFlags(["-Wno-shorten-64-to-32"]),
                .define("GGML_USE_ACCELERATE"),
//                .unsafeFlags(["-fsanitize=thread"]),
                .unsafeFlags(["-w"]),    // ignore all warnings
                //                .unsafeFlags(["-DGGML_QKK_64"]), // Dont forget to comment this if you dont use QKK_64
                
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

