# Lift

###### Build Instructions

```bash
git clone --recursive https://github.com/GoncaloFDS/Lift
```

> Install CUDA 10.1 and Optix 7.0 
*  Install Directories should be specified on cmake or you can set system variables

> Open the directory with cmake and generate the build files for your ide/build system 
 * currently supported on visual studio and clion
 
 `If you are using msvc 16.x the realease build is broken (black screen)`
>

##### On Windows, add cl.exe to your system PATH, so that nvcc can compile the cuda files
