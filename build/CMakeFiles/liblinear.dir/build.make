# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.5

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/huaizhi/Lazy_Boy_Diary/cross_compilation_android/dav

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/huaizhi/Lazy_Boy_Diary/cross_compilation_android/dav/build

# Include any dependencies generated for this target.
include CMakeFiles/liblinear.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/liblinear.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/liblinear.dir/flags.make

CMakeFiles/liblinear.dir/3rdparty/liblinear/daxpy.c.o: CMakeFiles/liblinear.dir/flags.make
CMakeFiles/liblinear.dir/3rdparty/liblinear/daxpy.c.o: ../3rdparty/liblinear/daxpy.c
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/huaizhi/Lazy_Boy_Diary/cross_compilation_android/dav/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building C object CMakeFiles/liblinear.dir/3rdparty/liblinear/daxpy.c.o"
	/usr/bin/cc  $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -o CMakeFiles/liblinear.dir/3rdparty/liblinear/daxpy.c.o   -c /home/huaizhi/Lazy_Boy_Diary/cross_compilation_android/dav/3rdparty/liblinear/daxpy.c

CMakeFiles/liblinear.dir/3rdparty/liblinear/daxpy.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/liblinear.dir/3rdparty/liblinear/daxpy.c.i"
	/usr/bin/cc  $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/huaizhi/Lazy_Boy_Diary/cross_compilation_android/dav/3rdparty/liblinear/daxpy.c > CMakeFiles/liblinear.dir/3rdparty/liblinear/daxpy.c.i

CMakeFiles/liblinear.dir/3rdparty/liblinear/daxpy.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/liblinear.dir/3rdparty/liblinear/daxpy.c.s"
	/usr/bin/cc  $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/huaizhi/Lazy_Boy_Diary/cross_compilation_android/dav/3rdparty/liblinear/daxpy.c -o CMakeFiles/liblinear.dir/3rdparty/liblinear/daxpy.c.s

CMakeFiles/liblinear.dir/3rdparty/liblinear/daxpy.c.o.requires:

.PHONY : CMakeFiles/liblinear.dir/3rdparty/liblinear/daxpy.c.o.requires

CMakeFiles/liblinear.dir/3rdparty/liblinear/daxpy.c.o.provides: CMakeFiles/liblinear.dir/3rdparty/liblinear/daxpy.c.o.requires
	$(MAKE) -f CMakeFiles/liblinear.dir/build.make CMakeFiles/liblinear.dir/3rdparty/liblinear/daxpy.c.o.provides.build
.PHONY : CMakeFiles/liblinear.dir/3rdparty/liblinear/daxpy.c.o.provides

CMakeFiles/liblinear.dir/3rdparty/liblinear/daxpy.c.o.provides.build: CMakeFiles/liblinear.dir/3rdparty/liblinear/daxpy.c.o


CMakeFiles/liblinear.dir/3rdparty/liblinear/ddot.c.o: CMakeFiles/liblinear.dir/flags.make
CMakeFiles/liblinear.dir/3rdparty/liblinear/ddot.c.o: ../3rdparty/liblinear/ddot.c
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/huaizhi/Lazy_Boy_Diary/cross_compilation_android/dav/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building C object CMakeFiles/liblinear.dir/3rdparty/liblinear/ddot.c.o"
	/usr/bin/cc  $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -o CMakeFiles/liblinear.dir/3rdparty/liblinear/ddot.c.o   -c /home/huaizhi/Lazy_Boy_Diary/cross_compilation_android/dav/3rdparty/liblinear/ddot.c

CMakeFiles/liblinear.dir/3rdparty/liblinear/ddot.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/liblinear.dir/3rdparty/liblinear/ddot.c.i"
	/usr/bin/cc  $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/huaizhi/Lazy_Boy_Diary/cross_compilation_android/dav/3rdparty/liblinear/ddot.c > CMakeFiles/liblinear.dir/3rdparty/liblinear/ddot.c.i

CMakeFiles/liblinear.dir/3rdparty/liblinear/ddot.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/liblinear.dir/3rdparty/liblinear/ddot.c.s"
	/usr/bin/cc  $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/huaizhi/Lazy_Boy_Diary/cross_compilation_android/dav/3rdparty/liblinear/ddot.c -o CMakeFiles/liblinear.dir/3rdparty/liblinear/ddot.c.s

CMakeFiles/liblinear.dir/3rdparty/liblinear/ddot.c.o.requires:

.PHONY : CMakeFiles/liblinear.dir/3rdparty/liblinear/ddot.c.o.requires

CMakeFiles/liblinear.dir/3rdparty/liblinear/ddot.c.o.provides: CMakeFiles/liblinear.dir/3rdparty/liblinear/ddot.c.o.requires
	$(MAKE) -f CMakeFiles/liblinear.dir/build.make CMakeFiles/liblinear.dir/3rdparty/liblinear/ddot.c.o.provides.build
.PHONY : CMakeFiles/liblinear.dir/3rdparty/liblinear/ddot.c.o.provides

CMakeFiles/liblinear.dir/3rdparty/liblinear/ddot.c.o.provides.build: CMakeFiles/liblinear.dir/3rdparty/liblinear/ddot.c.o


CMakeFiles/liblinear.dir/3rdparty/liblinear/dnrm2.c.o: CMakeFiles/liblinear.dir/flags.make
CMakeFiles/liblinear.dir/3rdparty/liblinear/dnrm2.c.o: ../3rdparty/liblinear/dnrm2.c
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/huaizhi/Lazy_Boy_Diary/cross_compilation_android/dav/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building C object CMakeFiles/liblinear.dir/3rdparty/liblinear/dnrm2.c.o"
	/usr/bin/cc  $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -o CMakeFiles/liblinear.dir/3rdparty/liblinear/dnrm2.c.o   -c /home/huaizhi/Lazy_Boy_Diary/cross_compilation_android/dav/3rdparty/liblinear/dnrm2.c

CMakeFiles/liblinear.dir/3rdparty/liblinear/dnrm2.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/liblinear.dir/3rdparty/liblinear/dnrm2.c.i"
	/usr/bin/cc  $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/huaizhi/Lazy_Boy_Diary/cross_compilation_android/dav/3rdparty/liblinear/dnrm2.c > CMakeFiles/liblinear.dir/3rdparty/liblinear/dnrm2.c.i

CMakeFiles/liblinear.dir/3rdparty/liblinear/dnrm2.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/liblinear.dir/3rdparty/liblinear/dnrm2.c.s"
	/usr/bin/cc  $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/huaizhi/Lazy_Boy_Diary/cross_compilation_android/dav/3rdparty/liblinear/dnrm2.c -o CMakeFiles/liblinear.dir/3rdparty/liblinear/dnrm2.c.s

CMakeFiles/liblinear.dir/3rdparty/liblinear/dnrm2.c.o.requires:

.PHONY : CMakeFiles/liblinear.dir/3rdparty/liblinear/dnrm2.c.o.requires

CMakeFiles/liblinear.dir/3rdparty/liblinear/dnrm2.c.o.provides: CMakeFiles/liblinear.dir/3rdparty/liblinear/dnrm2.c.o.requires
	$(MAKE) -f CMakeFiles/liblinear.dir/build.make CMakeFiles/liblinear.dir/3rdparty/liblinear/dnrm2.c.o.provides.build
.PHONY : CMakeFiles/liblinear.dir/3rdparty/liblinear/dnrm2.c.o.provides

CMakeFiles/liblinear.dir/3rdparty/liblinear/dnrm2.c.o.provides.build: CMakeFiles/liblinear.dir/3rdparty/liblinear/dnrm2.c.o


CMakeFiles/liblinear.dir/3rdparty/liblinear/dscal.c.o: CMakeFiles/liblinear.dir/flags.make
CMakeFiles/liblinear.dir/3rdparty/liblinear/dscal.c.o: ../3rdparty/liblinear/dscal.c
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/huaizhi/Lazy_Boy_Diary/cross_compilation_android/dav/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building C object CMakeFiles/liblinear.dir/3rdparty/liblinear/dscal.c.o"
	/usr/bin/cc  $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -o CMakeFiles/liblinear.dir/3rdparty/liblinear/dscal.c.o   -c /home/huaizhi/Lazy_Boy_Diary/cross_compilation_android/dav/3rdparty/liblinear/dscal.c

CMakeFiles/liblinear.dir/3rdparty/liblinear/dscal.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/liblinear.dir/3rdparty/liblinear/dscal.c.i"
	/usr/bin/cc  $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/huaizhi/Lazy_Boy_Diary/cross_compilation_android/dav/3rdparty/liblinear/dscal.c > CMakeFiles/liblinear.dir/3rdparty/liblinear/dscal.c.i

CMakeFiles/liblinear.dir/3rdparty/liblinear/dscal.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/liblinear.dir/3rdparty/liblinear/dscal.c.s"
	/usr/bin/cc  $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/huaizhi/Lazy_Boy_Diary/cross_compilation_android/dav/3rdparty/liblinear/dscal.c -o CMakeFiles/liblinear.dir/3rdparty/liblinear/dscal.c.s

CMakeFiles/liblinear.dir/3rdparty/liblinear/dscal.c.o.requires:

.PHONY : CMakeFiles/liblinear.dir/3rdparty/liblinear/dscal.c.o.requires

CMakeFiles/liblinear.dir/3rdparty/liblinear/dscal.c.o.provides: CMakeFiles/liblinear.dir/3rdparty/liblinear/dscal.c.o.requires
	$(MAKE) -f CMakeFiles/liblinear.dir/build.make CMakeFiles/liblinear.dir/3rdparty/liblinear/dscal.c.o.provides.build
.PHONY : CMakeFiles/liblinear.dir/3rdparty/liblinear/dscal.c.o.provides

CMakeFiles/liblinear.dir/3rdparty/liblinear/dscal.c.o.provides.build: CMakeFiles/liblinear.dir/3rdparty/liblinear/dscal.c.o


CMakeFiles/liblinear.dir/3rdparty/liblinear/linear.cpp.o: CMakeFiles/liblinear.dir/flags.make
CMakeFiles/liblinear.dir/3rdparty/liblinear/linear.cpp.o: ../3rdparty/liblinear/linear.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/huaizhi/Lazy_Boy_Diary/cross_compilation_android/dav/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building CXX object CMakeFiles/liblinear.dir/3rdparty/liblinear/linear.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/liblinear.dir/3rdparty/liblinear/linear.cpp.o -c /home/huaizhi/Lazy_Boy_Diary/cross_compilation_android/dav/3rdparty/liblinear/linear.cpp

CMakeFiles/liblinear.dir/3rdparty/liblinear/linear.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/liblinear.dir/3rdparty/liblinear/linear.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/huaizhi/Lazy_Boy_Diary/cross_compilation_android/dav/3rdparty/liblinear/linear.cpp > CMakeFiles/liblinear.dir/3rdparty/liblinear/linear.cpp.i

CMakeFiles/liblinear.dir/3rdparty/liblinear/linear.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/liblinear.dir/3rdparty/liblinear/linear.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/huaizhi/Lazy_Boy_Diary/cross_compilation_android/dav/3rdparty/liblinear/linear.cpp -o CMakeFiles/liblinear.dir/3rdparty/liblinear/linear.cpp.s

CMakeFiles/liblinear.dir/3rdparty/liblinear/linear.cpp.o.requires:

.PHONY : CMakeFiles/liblinear.dir/3rdparty/liblinear/linear.cpp.o.requires

CMakeFiles/liblinear.dir/3rdparty/liblinear/linear.cpp.o.provides: CMakeFiles/liblinear.dir/3rdparty/liblinear/linear.cpp.o.requires
	$(MAKE) -f CMakeFiles/liblinear.dir/build.make CMakeFiles/liblinear.dir/3rdparty/liblinear/linear.cpp.o.provides.build
.PHONY : CMakeFiles/liblinear.dir/3rdparty/liblinear/linear.cpp.o.provides

CMakeFiles/liblinear.dir/3rdparty/liblinear/linear.cpp.o.provides.build: CMakeFiles/liblinear.dir/3rdparty/liblinear/linear.cpp.o


CMakeFiles/liblinear.dir/3rdparty/liblinear/tron.cpp.o: CMakeFiles/liblinear.dir/flags.make
CMakeFiles/liblinear.dir/3rdparty/liblinear/tron.cpp.o: ../3rdparty/liblinear/tron.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/huaizhi/Lazy_Boy_Diary/cross_compilation_android/dav/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Building CXX object CMakeFiles/liblinear.dir/3rdparty/liblinear/tron.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/liblinear.dir/3rdparty/liblinear/tron.cpp.o -c /home/huaizhi/Lazy_Boy_Diary/cross_compilation_android/dav/3rdparty/liblinear/tron.cpp

CMakeFiles/liblinear.dir/3rdparty/liblinear/tron.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/liblinear.dir/3rdparty/liblinear/tron.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/huaizhi/Lazy_Boy_Diary/cross_compilation_android/dav/3rdparty/liblinear/tron.cpp > CMakeFiles/liblinear.dir/3rdparty/liblinear/tron.cpp.i

CMakeFiles/liblinear.dir/3rdparty/liblinear/tron.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/liblinear.dir/3rdparty/liblinear/tron.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/huaizhi/Lazy_Boy_Diary/cross_compilation_android/dav/3rdparty/liblinear/tron.cpp -o CMakeFiles/liblinear.dir/3rdparty/liblinear/tron.cpp.s

CMakeFiles/liblinear.dir/3rdparty/liblinear/tron.cpp.o.requires:

.PHONY : CMakeFiles/liblinear.dir/3rdparty/liblinear/tron.cpp.o.requires

CMakeFiles/liblinear.dir/3rdparty/liblinear/tron.cpp.o.provides: CMakeFiles/liblinear.dir/3rdparty/liblinear/tron.cpp.o.requires
	$(MAKE) -f CMakeFiles/liblinear.dir/build.make CMakeFiles/liblinear.dir/3rdparty/liblinear/tron.cpp.o.provides.build
.PHONY : CMakeFiles/liblinear.dir/3rdparty/liblinear/tron.cpp.o.provides

CMakeFiles/liblinear.dir/3rdparty/liblinear/tron.cpp.o.provides.build: CMakeFiles/liblinear.dir/3rdparty/liblinear/tron.cpp.o


# Object files for target liblinear
liblinear_OBJECTS = \
"CMakeFiles/liblinear.dir/3rdparty/liblinear/daxpy.c.o" \
"CMakeFiles/liblinear.dir/3rdparty/liblinear/ddot.c.o" \
"CMakeFiles/liblinear.dir/3rdparty/liblinear/dnrm2.c.o" \
"CMakeFiles/liblinear.dir/3rdparty/liblinear/dscal.c.o" \
"CMakeFiles/liblinear.dir/3rdparty/liblinear/linear.cpp.o" \
"CMakeFiles/liblinear.dir/3rdparty/liblinear/tron.cpp.o"

# External object files for target liblinear
liblinear_EXTERNAL_OBJECTS =

libliblinear.a: CMakeFiles/liblinear.dir/3rdparty/liblinear/daxpy.c.o
libliblinear.a: CMakeFiles/liblinear.dir/3rdparty/liblinear/ddot.c.o
libliblinear.a: CMakeFiles/liblinear.dir/3rdparty/liblinear/dnrm2.c.o
libliblinear.a: CMakeFiles/liblinear.dir/3rdparty/liblinear/dscal.c.o
libliblinear.a: CMakeFiles/liblinear.dir/3rdparty/liblinear/linear.cpp.o
libliblinear.a: CMakeFiles/liblinear.dir/3rdparty/liblinear/tron.cpp.o
libliblinear.a: CMakeFiles/liblinear.dir/build.make
libliblinear.a: CMakeFiles/liblinear.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/huaizhi/Lazy_Boy_Diary/cross_compilation_android/dav/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_7) "Linking CXX static library libliblinear.a"
	$(CMAKE_COMMAND) -P CMakeFiles/liblinear.dir/cmake_clean_target.cmake
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/liblinear.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/liblinear.dir/build: libliblinear.a

.PHONY : CMakeFiles/liblinear.dir/build

CMakeFiles/liblinear.dir/requires: CMakeFiles/liblinear.dir/3rdparty/liblinear/daxpy.c.o.requires
CMakeFiles/liblinear.dir/requires: CMakeFiles/liblinear.dir/3rdparty/liblinear/ddot.c.o.requires
CMakeFiles/liblinear.dir/requires: CMakeFiles/liblinear.dir/3rdparty/liblinear/dnrm2.c.o.requires
CMakeFiles/liblinear.dir/requires: CMakeFiles/liblinear.dir/3rdparty/liblinear/dscal.c.o.requires
CMakeFiles/liblinear.dir/requires: CMakeFiles/liblinear.dir/3rdparty/liblinear/linear.cpp.o.requires
CMakeFiles/liblinear.dir/requires: CMakeFiles/liblinear.dir/3rdparty/liblinear/tron.cpp.o.requires

.PHONY : CMakeFiles/liblinear.dir/requires

CMakeFiles/liblinear.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/liblinear.dir/cmake_clean.cmake
.PHONY : CMakeFiles/liblinear.dir/clean

CMakeFiles/liblinear.dir/depend:
	cd /home/huaizhi/Lazy_Boy_Diary/cross_compilation_android/dav/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/huaizhi/Lazy_Boy_Diary/cross_compilation_android/dav /home/huaizhi/Lazy_Boy_Diary/cross_compilation_android/dav /home/huaizhi/Lazy_Boy_Diary/cross_compilation_android/dav/build /home/huaizhi/Lazy_Boy_Diary/cross_compilation_android/dav/build /home/huaizhi/Lazy_Boy_Diary/cross_compilation_android/dav/build/CMakeFiles/liblinear.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/liblinear.dir/depend
