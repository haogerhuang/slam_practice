# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

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
CMAKE_SOURCE_DIR = /home/pinhao/pinhao/slam_practice/visual_odemetry

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/pinhao/pinhao/slam_practice/visual_odemetry

# Include any dependencies generated for this target.
include CMakeFiles/output.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/output.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/output.dir/flags.make

CMakeFiles/output.dir/main.cpp.o: CMakeFiles/output.dir/flags.make
CMakeFiles/output.dir/main.cpp.o: main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/pinhao/pinhao/slam_practice/visual_odemetry/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/output.dir/main.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/output.dir/main.cpp.o -c /home/pinhao/pinhao/slam_practice/visual_odemetry/main.cpp

CMakeFiles/output.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/output.dir/main.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/pinhao/pinhao/slam_practice/visual_odemetry/main.cpp > CMakeFiles/output.dir/main.cpp.i

CMakeFiles/output.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/output.dir/main.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/pinhao/pinhao/slam_practice/visual_odemetry/main.cpp -o CMakeFiles/output.dir/main.cpp.s

# Object files for target output
output_OBJECTS = \
"CMakeFiles/output.dir/main.cpp.o"

# External object files for target output
output_EXTERNAL_OBJECTS =

output: CMakeFiles/output.dir/main.cpp.o
output: CMakeFiles/output.dir/build.make
output: /usr/local/lib/libopencv_gapi.so.4.5.3
output: /usr/local/lib/libopencv_highgui.so.4.5.3
output: /usr/local/lib/libopencv_ml.so.4.5.3
output: /usr/local/lib/libopencv_objdetect.so.4.5.3
output: /usr/local/lib/libopencv_photo.so.4.5.3
output: /usr/local/lib/libopencv_stitching.so.4.5.3
output: /usr/local/lib/libopencv_video.so.4.5.3
output: /usr/local/lib/libopencv_videoio.so.4.5.3
output: /usr/local/lib/libpcl_surface.so
output: /usr/local/lib/libpcl_keypoints.so
output: /usr/local/lib/libpcl_tracking.so
output: /usr/local/lib/libpcl_recognition.so
output: /usr/local/lib/libpcl_stereo.so
output: /usr/local/lib/libpcl_outofcore.so
output: /usr/local/lib/libpcl_people.so
output: /usr/lib/x86_64-linux-gnu/libboost_system.so
output: /usr/lib/x86_64-linux-gnu/libboost_filesystem.so
output: /usr/lib/x86_64-linux-gnu/libboost_date_time.so
output: /usr/lib/x86_64-linux-gnu/libboost_iostreams.so
output: /usr/lib/x86_64-linux-gnu/libboost_serialization.so
output: /usr/lib/x86_64-linux-gnu/libboost_regex.so
output: /usr/lib/x86_64-linux-gnu/libqhull_r.so
output: /usr/lib/libOpenNI.so
output: /usr/lib/libOpenNI2.so
output: /usr/lib/x86_64-linux-gnu/libvtkChartsCore-7.1.so.7.1p.1
output: /usr/lib/x86_64-linux-gnu/libvtkInfovisCore-7.1.so.7.1p.1
output: /usr/lib/x86_64-linux-gnu/libfreetype.so
output: /usr/lib/x86_64-linux-gnu/libz.so
output: /usr/lib/x86_64-linux-gnu/libjpeg.so
output: /usr/lib/x86_64-linux-gnu/libpng.so
output: /usr/lib/x86_64-linux-gnu/libtiff.so
output: /usr/lib/x86_64-linux-gnu/libexpat.so
output: /usr/lib/x86_64-linux-gnu/libvtkIOGeometry-7.1.so.7.1p.1
output: /usr/lib/x86_64-linux-gnu/libvtkIOLegacy-7.1.so.7.1p.1
output: /usr/lib/x86_64-linux-gnu/libvtkIOPLY-7.1.so.7.1p.1
output: /usr/lib/x86_64-linux-gnu/libvtkRenderingLOD-7.1.so.7.1p.1
output: /usr/lib/x86_64-linux-gnu/libvtkViewsContext2D-7.1.so.7.1p.1
output: /usr/lib/x86_64-linux-gnu/libvtkRenderingOpenGL2-7.1.so.7.1p.1
output: /usr/lib/x86_64-linux-gnu/libflann_cpp_s.a
output: /usr/local/lib/libopencv_dnn.so.4.5.3
output: /usr/local/lib/libopencv_imgcodecs.so.4.5.3
output: /usr/local/lib/libopencv_calib3d.so.4.5.3
output: /usr/local/lib/libopencv_features2d.so.4.5.3
output: /usr/local/lib/libopencv_flann.so.4.5.3
output: /usr/local/lib/libopencv_imgproc.so.4.5.3
output: /usr/local/lib/libopencv_core.so.4.5.3
output: /usr/local/lib/libpcl_registration.so
output: /usr/local/lib/libpcl_segmentation.so
output: /usr/local/lib/libpcl_features.so
output: /usr/local/lib/libpcl_filters.so
output: /usr/local/lib/libpcl_sample_consensus.so
output: /usr/local/lib/libpcl_ml.so
output: /usr/local/lib/libpcl_visualization.so
output: /usr/local/lib/libpcl_search.so
output: /usr/local/lib/libpcl_kdtree.so
output: /usr/local/lib/libpcl_io.so
output: /usr/local/lib/libpcl_octree.so
output: /usr/local/lib/libpcl_common.so
output: /usr/lib/x86_64-linux-gnu/libvtkRenderingContext2D-7.1.so.7.1p.1
output: /usr/lib/x86_64-linux-gnu/libvtkViewsCore-7.1.so.7.1p.1
output: /usr/lib/x86_64-linux-gnu/libvtkInteractionWidgets-7.1.so.7.1p.1
output: /usr/lib/x86_64-linux-gnu/libvtkFiltersModeling-7.1.so.7.1p.1
output: /usr/lib/x86_64-linux-gnu/libvtkInteractionStyle-7.1.so.7.1p.1
output: /usr/lib/x86_64-linux-gnu/libvtkFiltersExtraction-7.1.so.7.1p.1
output: /usr/lib/x86_64-linux-gnu/libvtkFiltersStatistics-7.1.so.7.1p.1
output: /usr/lib/x86_64-linux-gnu/libvtkImagingFourier-7.1.so.7.1p.1
output: /usr/lib/x86_64-linux-gnu/libvtkalglib-7.1.so.7.1p.1
output: /usr/lib/x86_64-linux-gnu/libvtkFiltersHybrid-7.1.so.7.1p.1
output: /usr/lib/x86_64-linux-gnu/libvtkImagingGeneral-7.1.so.7.1p.1
output: /usr/lib/x86_64-linux-gnu/libvtkImagingSources-7.1.so.7.1p.1
output: /usr/lib/x86_64-linux-gnu/libvtkImagingHybrid-7.1.so.7.1p.1
output: /usr/lib/x86_64-linux-gnu/libvtkRenderingAnnotation-7.1.so.7.1p.1
output: /usr/lib/x86_64-linux-gnu/libvtkRenderingFreeType-7.1.so.7.1p.1
output: /usr/lib/x86_64-linux-gnu/libfreetype.so
output: /usr/lib/x86_64-linux-gnu/libvtkImagingColor-7.1.so.7.1p.1
output: /usr/lib/x86_64-linux-gnu/libvtkRenderingVolume-7.1.so.7.1p.1
output: /usr/lib/x86_64-linux-gnu/libvtkIOXML-7.1.so.7.1p.1
output: /usr/lib/x86_64-linux-gnu/libvtkIOXMLParser-7.1.so.7.1p.1
output: /usr/lib/x86_64-linux-gnu/libvtkIOCore-7.1.so.7.1p.1
output: /usr/lib/x86_64-linux-gnu/libvtkImagingCore-7.1.so.7.1p.1
output: /usr/lib/x86_64-linux-gnu/libvtkRenderingCore-7.1.so.7.1p.1
output: /usr/lib/x86_64-linux-gnu/libvtkCommonColor-7.1.so.7.1p.1
output: /usr/lib/x86_64-linux-gnu/libvtkFiltersGeometry-7.1.so.7.1p.1
output: /usr/lib/x86_64-linux-gnu/libvtkFiltersSources-7.1.so.7.1p.1
output: /usr/lib/x86_64-linux-gnu/libvtkFiltersGeneral-7.1.so.7.1p.1
output: /usr/lib/x86_64-linux-gnu/libvtkCommonComputationalGeometry-7.1.so.7.1p.1
output: /usr/lib/x86_64-linux-gnu/libvtkFiltersCore-7.1.so.7.1p.1
output: /usr/lib/x86_64-linux-gnu/libvtkIOImage-7.1.so.7.1p.1
output: /usr/lib/x86_64-linux-gnu/libvtkCommonExecutionModel-7.1.so.7.1p.1
output: /usr/lib/x86_64-linux-gnu/libvtkCommonDataModel-7.1.so.7.1p.1
output: /usr/lib/x86_64-linux-gnu/libvtkCommonTransforms-7.1.so.7.1p.1
output: /usr/lib/x86_64-linux-gnu/libvtkCommonMisc-7.1.so.7.1p.1
output: /usr/lib/x86_64-linux-gnu/libvtkCommonMath-7.1.so.7.1p.1
output: /usr/lib/x86_64-linux-gnu/libvtkCommonSystem-7.1.so.7.1p.1
output: /usr/lib/x86_64-linux-gnu/libvtkCommonCore-7.1.so.7.1p.1
output: /usr/lib/x86_64-linux-gnu/libvtksys-7.1.so.7.1p.1
output: /usr/lib/x86_64-linux-gnu/libvtkDICOMParser-7.1.so.7.1p.1
output: /usr/lib/x86_64-linux-gnu/libvtkmetaio-7.1.so.7.1p.1
output: /usr/lib/x86_64-linux-gnu/libz.so
output: /usr/lib/x86_64-linux-gnu/libGLEW.so
output: /usr/lib/x86_64-linux-gnu/libSM.so
output: /usr/lib/x86_64-linux-gnu/libICE.so
output: /usr/lib/x86_64-linux-gnu/libX11.so
output: /usr/lib/x86_64-linux-gnu/libXext.so
output: /usr/lib/x86_64-linux-gnu/libXt.so
output: CMakeFiles/output.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/pinhao/pinhao/slam_practice/visual_odemetry/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable output"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/output.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/output.dir/build: output

.PHONY : CMakeFiles/output.dir/build

CMakeFiles/output.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/output.dir/cmake_clean.cmake
.PHONY : CMakeFiles/output.dir/clean

CMakeFiles/output.dir/depend:
	cd /home/pinhao/pinhao/slam_practice/visual_odemetry && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/pinhao/pinhao/slam_practice/visual_odemetry /home/pinhao/pinhao/slam_practice/visual_odemetry /home/pinhao/pinhao/slam_practice/visual_odemetry /home/pinhao/pinhao/slam_practice/visual_odemetry /home/pinhao/pinhao/slam_practice/visual_odemetry/CMakeFiles/output.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/output.dir/depend
