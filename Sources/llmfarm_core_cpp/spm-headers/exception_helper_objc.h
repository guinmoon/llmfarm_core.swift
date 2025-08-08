//
//  exc_helper_objc.h
//
//
//  Created by guinmoon on 08.10.2023.
//

#ifndef exc_helper_objc
#define exc_helper_objc

#import <Foundation/Foundation.h>

#define noEscape __attribute__((noescape))

@interface ExceptionCather : NSObject
+ (BOOL)catchException:(noEscape void(^)(void))tryBlock error:(__autoreleasing NSError **)error;
@end

// NSString * get_core_bundle_path();

// NSString * Get_Machine_Hardware_Name(void);

// bool setSignalHandler();

#endif /* exc_helper_objc */
