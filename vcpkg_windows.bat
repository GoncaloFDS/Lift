mkdir build
cd build || goto :error
git clone https://github.com/Microsoft/vcpkg.git vcpkg.windows || goto :error
cd vcpkg.windows || goto :error
git checkout 2019.10 || goto :error
call bootstrap-vcpkg.bat || goto :error

vcpkg.exe install ^
	boost-program-options:x64-windows-static ^
	glfw3:x64-windows-static ^
	|| goto :error

cd ..
cd ..

exit /b


:error
echo Failed with error #%errorlevel%.
exit /b %errorlevel%
