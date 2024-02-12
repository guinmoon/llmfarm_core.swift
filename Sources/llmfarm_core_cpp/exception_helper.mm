//
//  exception_helper.h.cpp
//  
//
//  Created by guinmoon on 22.09.2023.
//

#include "exception_helper.h"
#include <stdexcept>

void throw_exception(const char* description){
    throw std::invalid_argument(description);
}
