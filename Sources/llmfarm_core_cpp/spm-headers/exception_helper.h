//
//  exception_helper.h.hpp
//  
//
//  Created by guinmoon on 22.09.2023.
//
#pragma once

#ifdef __cplusplus
#define EXTERNC extern "C"
#else
#define EXTERNC
#endif

EXTERNC void throw_exception(const char* description);

#undef EXTERNC

