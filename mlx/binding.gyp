{
  "targets": [
    {
      "target_name": "mlx_server",
      "sources": [
        "src/mlx_server.cpp"
      ],
      "include_dirs": [
        "<!@(node -p \"require('node-addon-api').include\")",
        "/opt/homebrew/include",
        "/usr/local/include"
      ],
      "libraries": [
        "-L/opt/homebrew/lib",
        "-L/usr/local/lib",
        "-lmlx"
      ],
      "cflags!": [ "-fno-exceptions" ],
      "cflags_cc!": [ "-fno-exceptions" ],
      "xcode_settings": {
        "GCC_ENABLE_CPP_EXCEPTIONS": "YES",
        "CLANG_CXX_LIBRARY": "libc++",
        "MACOSX_DEPLOYMENT_TARGET": "10.13",
        "OTHER_CPLUSPLUSFLAGS": [
          "-std=c++17"
        ],
        "LD_RUNPATH_SEARCH_PATHS": [
          "/opt/homebrew/lib",
          "/usr/local/lib"
        ]
      },
      "defines": [ "NAPI_DISABLE_CPP_EXCEPTIONS" ]
    }
  ]
}

