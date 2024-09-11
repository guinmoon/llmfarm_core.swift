#import <Foundation/Foundation.h>


NSString * get_core_bundle_path();

NSString * Get_Machine_Hardware_Name(void);

bool setSignalHandler();

//NS_INLINE NSException * _Nullable tryBlock(void(^_Nonnull tryBlock)(void)) {
//    @try {
//        tryBlock();
//    }
//    @catch (NSException *exception) {
//        return exception;
//    }
//    @catch (...) {
////        fprintf(stderr,"EXC: %d",exc);
//        NSException *exception = [NSException exceptionWithName:@"C++ Exception" reason:@"GGML_ASSERT" userInfo:@{@"C++": @"Exc"}];
//        return exception;
//    }
//    
//    return nil;
//}
//

