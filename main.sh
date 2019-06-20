scons Werror=1 -j8 debug=0 asserts=1 neon=1 opencl=1 embed_kernels=1 os=android arch=arm64-v8a
aarch64-linux-android-gcc -o libOpenCL.so -Iinclude -shared opencl-1.2-stubs/opencl_stubs.c -fPIC -shared
aarch64-linux-android-g++ examples/vgg16_model_arm_compute_library_NEON.cpp test_helpers/Utils.cpp -I. -Iinclude -std=c++11 -larm_compute-static -lOpenCL -L/home/zhoupeilin/vgg16-by-ARM-Compute-Library/build/arm_compute -L. -o main -static-libstdc++ -pie
adb push main /data/local/tmp/
adb shell /data/local/tmp/main
