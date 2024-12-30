# CMakeList.txt : CMake project for ParcoDeliverable2, include source and define
# project specific logic here.
#

# Add source to this project's executable.
add_executable (ParcoDeliverable2 "ParcoDeliverable2.cpp" "ParcoDeliverable2.h" "Utils.cpp" "Seq.cpp" "Parallel.cpp")

if (CMAKE_VERSION VERSION_GREATER 3.16)
  set_property(TARGET ParcoDeliverable2 PROPERTY CXX_STANDARD 20)
else()
  set_property(TARGET ParcoDeliverable2 PROPERTY CXX_STANDARD 11)
endif()

if(WIN32) 
  target_include_directories(ParcoDeliverable2 PUBLIC $ENV{MSMPI_INC})
  #build only for 64 bits
  message(STATUS "MPI Lib path: $ENV{MSMPI_INC}/../Lib/x64/")
  target_link_directories(ParcoDeliverable2 PUBLIC "$ENV{MSMPI_INC}/../Lib/x64/")
  target_link_libraries(ParcoDeliverable2 PUBLIC "msmpi")
  if(MSVC)
	message(STATUS "Enable /W4")
	target_compile_options(ParcoDeliverable2 PUBLIC "/W4")
  endif()
else()
  target_link_libraries(ParcoDeliverable2 PUBLIC "mpi" "mpi_cxx")
endif()

if(NOT MSVC)
	target_compile_options(ParcoDeliverable2 PUBLIC "-Wall" "-msse4.1")
endif()

#target_compile_options(ParcoDeliverable2 PUBLIC "/fsanitize=address" "/DEBUG")
#target_link_options(ParcoDeliverable2 PUBLIC "/DEBUG")

# TODO: Add tests and install targets if needed.
