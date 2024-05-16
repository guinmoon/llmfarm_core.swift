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

//void handleSignal(int sig) {
//    if (sig == SIGTERM) {
//        // Caught a SIGTERM
//    }
//    if (sig == SIGABRT){
//        printf("SIGABRT");
//    }
//    /*
//      SIGTERM is a clear directive to quit, so we exit
//      and return the signal number for us to inspect if we desire.
//      We can actually omit the exit(), and everything
//      will still build normally.
//
//      If you Force Quit the application, it will still eventually
//      exit, suggesting a follow-up SIGKILL is sent.
//    */
//    exit(sig);
//}
//
///**
// This will let us set a handler for a specific signal (SIGTERM in this case)
// */
//bool setSignalHandler() {
//    if (signal(SIGTERM, handleSignal) == SIG_ERR) {
//        NSLog(@"Failed to set a signal handler.");
//        return  true;
//    } else {
//        NSLog(@"Successfully set a signal handler.");
//        return false;
//    }
//}
