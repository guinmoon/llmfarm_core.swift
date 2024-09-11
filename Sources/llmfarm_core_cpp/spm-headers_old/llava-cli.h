#pragma once
//
//  finetune.h
//
//
//  Created by guinmoon on 31.10.2023.
//



#ifdef __cplusplus
extern "C" {
#endif

int run_llava(int argc, char ** argv, bool(*swift_callback)(const char*));


#ifdef __cplusplus
}
#endif
