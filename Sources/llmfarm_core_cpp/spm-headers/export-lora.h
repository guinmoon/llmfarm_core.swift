#pragma once
//
//  export-lora.h
//
//
//  Created by guinmoon on 31.10.2023.
//



#ifdef __cplusplus
extern "C" {
#endif

int export_lora_main(int argc, char ** argv, bool(*swift_callback)(double));
#define LLAMA_FILE_MAGIC_GGLA 0x67676c61u // 'ggla'

#ifdef __cplusplus
}
#endif
