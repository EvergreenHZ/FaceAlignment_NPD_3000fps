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
include CMakeFiles/lib3000fps.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/lib3000fps.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/lib3000fps.dir/flags.make

CMakeFiles/lib3000fps.dir/src/lbf/rf.cpp.o: CMakeFiles/lib3000fps.dir/flags.make
CMakeFiles/lib3000fps.dir/src/lbf/rf.cpp.o: ../src/lbf/rf.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/huaizhi/Lazy_Boy_Diary/cross_compilation_android/dav/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/lib3000fps.dir/src/lbf/rf.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/lib3000fps.dir/src/lbf/rf.cpp.o -c /home/huaizhi/Lazy_Boy_Diary/cross_compilation_android/dav/src/lbf/rf.cpp

CMakeFiles/lib3000fps.dir/src/lbf/rf.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/lib3000fps.dir/src/lbf/rf.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/huaizhi/Lazy_Boy_Diary/cross_compilation_android/dav/src/lbf/rf.cpp > CMakeFiles/lib3000fps.dir/src/lbf/rf.cpp.i

CMakeFiles/lib3000fps.dir/src/lbf/rf.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/lib3000fps.dir/src/lbf/rf.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/huaizhi/Lazy_Boy_Diary/cross_compilation_android/dav/src/lbf/rf.cpp -o CMakeFiles/lib3000fps.dir/src/lbf/rf.cpp.s

CMakeFiles/lib3000fps.dir/src/lbf/rf.cpp.o.requires:

.PHONY : CMakeFiles/lib3000fps.dir/src/lbf/rf.cpp.o.requires

CMakeFiles/lib3000fps.dir/src/lbf/rf.cpp.o.provides: CMakeFiles/lib3000fps.dir/src/lbf/rf.cpp.o.requires
	$(MAKE) -f CMakeFiles/lib3000fps.dir/build.make CMakeFiles/lib3000fps.dir/src/lbf/rf.cpp.o.provides.build
.PHONY : CMakeFiles/lib3000fps.dir/src/lbf/rf.cpp.o.provides

CMakeFiles/lib3000fps.dir/src/lbf/rf.cpp.o.provides.build: CMakeFiles/lib3000fps.dir/src/lbf/rf.cpp.o


CMakeFiles/lib3000fps.dir/src/lbf/common.cpp.o: CMakeFiles/lib3000fps.dir/flags.make
CMakeFiles/lib3000fps.dir/src/lbf/common.cpp.o: ../src/lbf/common.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/huaizhi/Lazy_Boy_Diary/cross_compilation_android/dav/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/lib3000fps.dir/src/lbf/common.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/lib3000fps.dir/src/lbf/common.cpp.o -c /home/huaizhi/Lazy_Boy_Diary/cross_compilation_android/dav/src/lbf/common.cpp

CMakeFiles/lib3000fps.dir/src/lbf/common.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/lib3000fps.dir/src/lbf/common.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/huaizhi/Lazy_Boy_Diary/cross_compilation_android/dav/src/lbf/common.cpp > CMakeFiles/lib3000fps.dir/src/lbf/common.cpp.i

CMakeFiles/lib3000fps.dir/src/lbf/common.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/lib3000fps.dir/src/lbf/common.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/huaizhi/Lazy_Boy_Diary/cross_compilation_android/dav/src/lbf/common.cpp -o CMakeFiles/lib3000fps.dir/src/lbf/common.cpp.s

CMakeFiles/lib3000fps.dir/src/lbf/common.cpp.o.requires:

.PHONY : CMakeFiles/lib3000fps.dir/src/lbf/common.cpp.o.requires

CMakeFiles/lib3000fps.dir/src/lbf/common.cpp.o.provides: CMakeFiles/lib3000fps.dir/src/lbf/common.cpp.o.requires
	$(MAKE) -f CMakeFiles/lib3000fps.dir/build.make CMakeFiles/lib3000fps.dir/src/lbf/common.cpp.o.provides.build
.PHONY : CMakeFiles/lib3000fps.dir/src/lbf/common.cpp.o.provides

CMakeFiles/lib3000fps.dir/src/lbf/common.cpp.o.provides.build: CMakeFiles/lib3000fps.dir/src/lbf/common.cpp.o


CMakeFiles/lib3000fps.dir/src/lbf/lbf.cpp.o: CMakeFiles/lib3000fps.dir/flags.make
CMakeFiles/lib3000fps.dir/src/lbf/lbf.cpp.o: ../src/lbf/lbf.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/huaizhi/Lazy_Boy_Diary/cross_compilation_android/dav/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/lib3000fps.dir/src/lbf/lbf.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/lib3000fps.dir/src/lbf/lbf.cpp.o -c /home/huaizhi/Lazy_Boy_Diary/cross_compilation_android/dav/src/lbf/lbf.cpp

CMakeFiles/lib3000fps.dir/src/lbf/lbf.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/lib3000fps.dir/src/lbf/lbf.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/huaizhi/Lazy_Boy_Diary/cross_compilation_android/dav/src/lbf/lbf.cpp > CMakeFiles/lib3000fps.dir/src/lbf/lbf.cpp.i

CMakeFiles/lib3000fps.dir/src/lbf/lbf.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/lib3000fps.dir/src/lbf/lbf.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/huaizhi/Lazy_Boy_Diary/cross_compilation_android/dav/src/lbf/lbf.cpp -o CMakeFiles/lib3000fps.dir/src/lbf/lbf.cpp.s

CMakeFiles/lib3000fps.dir/src/lbf/lbf.cpp.o.requires:

.PHONY : CMakeFiles/lib3000fps.dir/src/lbf/lbf.cpp.o.requires

CMakeFiles/lib3000fps.dir/src/lbf/lbf.cpp.o.provides: CMakeFiles/lib3000fps.dir/src/lbf/lbf.cpp.o.requires
	$(MAKE) -f CMakeFiles/lib3000fps.dir/build.make CMakeFiles/lib3000fps.dir/src/lbf/lbf.cpp.o.provides.build
.PHONY : CMakeFiles/lib3000fps.dir/src/lbf/lbf.cpp.o.provides

CMakeFiles/lib3000fps.dir/src/lbf/lbf.cpp.o.provides.build: CMakeFiles/lib3000fps.dir/src/lbf/lbf.cpp.o


# Object files for target lib3000fps
lib3000fps_OBJECTS = \
"CMakeFiles/lib3000fps.dir/src/lbf/rf.cpp.o" \
"CMakeFiles/lib3000fps.dir/src/lbf/common.cpp.o" \
"CMakeFiles/lib3000fps.dir/src/lbf/lbf.cpp.o"

# External object files for target lib3000fps
lib3000fps_EXTERNAL_OBJECTS =

liblib3000fps.a: CMakeFiles/lib3000fps.dir/src/lbf/rf.cpp.o
liblib3000fps.a: CMakeFiles/lib3000fps.dir/src/lbf/common.cpp.o
liblib3000fps.a: CMakeFiles/lib3000fps.dir/src/lbf/lbf.cpp.o
liblib3000fps.a: CMakeFiles/lib3000fps.dir/build.make
liblib3000fps.a: CMakeFiles/lib3000fps.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/huaizhi/Lazy_Boy_Diary/cross_compilation_android/dav/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Linking CXX static library liblib3000fps.a"
	$(CMAKE_COMMAND) -P CMakeFiles/lib3000fps.dir/cmake_clean_target.cmake
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/lib3000fps.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/lib3000fps.dir/build: liblib3000fps.a

.PHONY : CMakeFiles/lib3000fps.dir/build

CMakeFiles/lib3000fps.dir/requires: CMakeFiles/lib3000fps.dir/src/lbf/rf.cpp.o.requires
CMakeFiles/lib3000fps.dir/requires: CMakeFiles/lib3000fps.dir/src/lbf/common.cpp.o.requires
CMakeFiles/lib3000fps.dir/requires: CMakeFiles/lib3000fps.dir/src/lbf/lbf.cpp.o.requires

.PHONY : CMakeFiles/lib3000fps.dir/requires

CMakeFiles/lib3000fps.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/lib3000fps.dir/cmake_clean.cmake
.PHONY : CMakeFiles/lib3000fps.dir/clean

CMakeFiles/lib3000fps.dir/depend:
	cd /home/huaizhi/Lazy_Boy_Diary/cross_compilation_android/dav/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/huaizhi/Lazy_Boy_Diary/cross_compilation_android/dav /home/huaizhi/Lazy_Boy_Diary/cross_compilation_android/dav /home/huaizhi/Lazy_Boy_Diary/cross_compilation_android/dav/build /home/huaizhi/Lazy_Boy_Diary/cross_compilation_android/dav/build /home/huaizhi/Lazy_Boy_Diary/cross_compilation_android/dav/build/CMakeFiles/lib3000fps.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/lib3000fps.dir/depend

