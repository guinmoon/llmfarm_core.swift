#import <Foundation/Foundation.h>

NSString * get_core_bundle_path();

NSString * Get_Machine_Hardware_Name(void);

bool setSignalHandler();

NS_INLINE NSException * _Nullable tryBlock(void(^_Nonnull tryBlock)(void)) {
    @try {
        tryBlock();
        return nil;
    }
    @catch (NSException *exception) {
        return exception;
    }
    NSException *exception = [NSException exceptionWithName:@"Custom Exception" reason:@"Custom Reason" userInfo:@{@"C++": @"Exc"}];
    return exception;
}
