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

#endif /* exc_helper_objc */
