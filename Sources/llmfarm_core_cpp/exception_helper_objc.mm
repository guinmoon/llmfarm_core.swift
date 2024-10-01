//
//  exc_helper_objc.m
//  
//
//  Created by guinmoon on 08.10.2023.
//

#import <Foundation/Foundation.h>
#import "exception_helper_objc.h"

#include <exception>
#include <sys/utsname.h>

@implementation ExceptionCather
+ (BOOL)catchException:(noEscape void(^)(void))tryBlock error:(__autoreleasing NSError **)error {
    try {
        tryBlock();
        return YES;
    }
    catch(NSException* e) {
        NSLog(@"%@", e.reason);
        *error = [[NSError alloc] initWithDomain:e.name code:-1 userInfo:e.userInfo];
        return NO;
    }
    catch (std::exception& e) {
        NSString* what = [NSString stringWithUTF8String: e.what()];
        NSDictionary* userInfo = @{NSLocalizedDescriptionKey : what};
        *error = [[NSError alloc] initWithDomain:@"cpp_exception" code:-2 userInfo:userInfo];
        return NO;
    }
    catch(...) {
        NSDictionary* userInfo = @{NSLocalizedDescriptionKey:@"Other C++ exception"};
        *error = [[NSError alloc] initWithDomain:@"cpp_exception" code:-3 userInfo:userInfo];
        return NO;
    }
}
@end

// NSString * get_core_bundle_path(){
//     NSString *path = [SWIFTPM_MODULE_BUNDLE resourcePath];
//     return  path;
// }


// NSString *Get_Machine_Hardware_Name(void) {
//     struct utsname sysinfo;
//     int retVal = uname(&sysinfo);
//     if (EXIT_SUCCESS != retVal) return nil;
//     return [NSString stringWithUTF8String:sysinfo.machine];
// }

