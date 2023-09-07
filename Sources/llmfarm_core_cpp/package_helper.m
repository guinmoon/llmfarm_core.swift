#import "spm-headers/package_helper.h"
#import <Foundation/Foundation.h>
#include <sys/utsname.h>

NSString * get_core_bundle_path(){
    NSString *path = [SWIFTPM_MODULE_BUNDLE resourcePath];
    return  path;
}


NSString *Get_Machine_Hardware_Name(void) {
    struct utsname sysinfo;
    int retVal = uname(&sysinfo);
    if (EXIT_SUCCESS != retVal) return nil;
    
    return [NSString stringWithUTF8String:sysinfo.machine];
}


