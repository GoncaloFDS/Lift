# Lift

Build Instructions

```bash
git clone --recursive https://github.com/GoncaloFDS/Lift
```

> Install CUDA 10.1 and Optix 7.0 ( Install Directories should be specified on premake5.lua file)
>
> Use vckpg to get assimp

```bash
> git clone https://github.com/Microsoft/vcpkg
> cd vcpkg
> .\bootsrap-vcpkg.bat
> .\vcpkg integrate install
> .\vcpkg install assimp:x64-windows
```

> Generate visual studio solution  

```bash
> .\GenerateProjectsVS2019.bat
or
> .\GenerateProjectsVS2017.bat
```
