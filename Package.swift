// swift-tools-version: 5.6
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription


//
var sources = [
                "package_helper.m",
                "exception_helper_objc.mm",
                "exception_helper.cpp"
                ]


let package = Package(
    name: "llmfarm_core",
    platforms: [.macOS(.v11),.iOS(.v15)],
    products: [
        .library(
            name: "llmfarm_core",
//           type: .dynamic,
            targets: ["llmfarm_core"]),
       .library(
           name: "llmfarm_core_cpp",
           targets: ["llmfarm_core_cpp"]),

    ],
    dependencies: [
      
        // Dependencies declare other packages that this package depends on.
    ],
    targets: [
            .binaryTarget(
                name: "llama",
//                path: "../llama_cpu.xcframework"
                path: "./llama.cpp/build-apple/llama.xcframework"
            ),
            .target(
                name: "llmfarm_core",
                dependencies: ["llama","llmfarm_core_cpp"],
                path: "Sources/llmfarm_core",
                linkerSettings: [
                    .linkedFramework("Accelerate")
                ]
            ),
            .target(
                    name: "llmfarm_core_cpp",
                    sources: sources,
                    publicHeadersPath: "spm-headers"
                ),
        ],
    // cLanguageStandard: .c99,
//    cxxLanguageStandard: .cxx11
    cxxLanguageStandard: .cxx20
)

// c++ -std=c++11 -fPIC -O3 -Wall -Wextra -Wpedantic -Wcast-qual -Wno-unused-function -Wmissing-declarations -Wmissing-noreturn -pthread  -march=native -mtune=native -Wunreachable-code-break -Wunreachable-code-return -Wmissing-prototypes -Wextra-semi -Iggml/include -Iggml/src -Iinclude -Isrc -Icommon -D_XOPEN_SOURCE=600 -D_DARWIN_C_SOURCE -DNDEBUG -DGGML_USE_ACCELERATE -DGGML_USE_BLAS -DACCELERATE_NEW_LAPACK -DACCELERATE_LAPACK_ILP64 -DGGML_USE_LLAMAFILE -DGGML_USE_METAL -DGGML_METAL_EMBED_LIBRARY  -c src/llama.cpp -o src/llama.o
