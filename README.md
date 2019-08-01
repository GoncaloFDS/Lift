# Lift

Build Instructions
```
git clone --recursive https://github.com/GoncaloFDS/Lift
```
> Install CUDA and Optix ( Install Directories should be specified on premake5.lua file)

> Use vckpg to get assimp
```
> git clone https://github.com/Microsoft/vcpkg
> cd vcpkg
> .\bootsrap-vcpkg.bat
> .\vcpkg integrate install
> .\vcpkg install assimp:x64-windows
```
> Generate visual studio solution 

```
> .\GenerateProjectsVS2019.bat
or
> .\GenerateProjectsVS2017.bat
```