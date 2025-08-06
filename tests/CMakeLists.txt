find_package(PkgConfig REQUIRED)
#find_library(CUFFT_LIB cufft HINTS ${CUDA_TOOLKIT_ROOT_DIR}/lib64)
pkg_check_modules(CNPY REQUIRED cnpy)


#================TestBurst===============
add_executable(TestBurst TestBurst.cpp)
target_include_directories(TestBurst PRIVATE
    ${CNPY_INCLUDE_DIRS}
)
target_compile_options(TestBurst PRIVATE
    ${CNPY_CFLAGS_OTHER}
)
target_link_libraries(TestBurst
    PRIVATE
        handheld_lib           # now linking against the shared library
        ${CNPY_LIBRARIES}
        ${CUDA_LIBRARIES}
)
target_link_options(TestBurst PRIVATE
    ${CNPY_LDFLAGS_OTHER}
)
add_test(NAME TestBurst 
    COMMAND TestBurst
    #run this binary with the source tree as cwd
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}  
  )

#================TestNoiseModel===============
add_executable(TestNoiseModel TestNoiseModel.cpp)
target_include_directories(TestNoiseModel PRIVATE
    ${CNPY_INCLUDE_DIRS}
)
target_compile_options(TestNoiseModel PRIVATE
    ${CNPY_CFLAGS_OTHER}
)
target_link_libraries(TestNoiseModel
    PRIVATE
        handheld_lib           # now linking against the shared library
        ${CNPY_LIBRARIES}
        ${CUDA_LIBRARIES}
)
target_link_options(TestNoiseModel PRIVATE
    ${CNPY_LDFLAGS_OTHER}
)
add_test(NAME TestNoiseModel 
    COMMAND TestNoiseModel
    #run this binary with the source tree as cwd
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}  
  )

#================TestNoiseModel===============
add_executable(TestGetParams TestGetParams.cpp)
target_link_libraries(TestGetParams
    PRIVATE
        handheld_lib           # now linking against the shared library
)
add_test(NAME TestGetParams 
    COMMAND TestGetParams
    #run this binary with the source tree as cwd
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}  
  )
#================TestGreyConverter===============
add_executable(TestGreyConverter TestGreyConverter.cpp)
target_link_libraries(TestGreyConverter
    PRIVATE
        handheld_lib           # now linking against the shared library
        #${CUFFT_LIB}  # ensures cufftPlan2d, cufftExecC2C, etc. are resolved
        #${CUDA_LIBRARIES}        # if needed for other CUDA runtime symbols
        CUDA::cufft      # the CUFFT library
        CUDA::cudart     # the CUDA runtime
)
add_test(NAME TestGreyConverter 
    COMMAND TestGreyConverter
    #run this binary with the source tree as cwd
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}  
  )
#================TestPyramid===============
add_executable(TestPyramid TestPyramid.cpp)
target_link_libraries(TestPyramid
    PRIVATE
        handheld_lib           # now linking against the shared library
        CUDA::cufft      # the CUFFT library
        CUDA::cudart     # the CUDA runtime
)
add_test(NAME TestPyramid 
    COMMAND TestPyramid
    #run this binary with the source tree as cwd
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}  
  )
#================TestBlockMatcher===============
add_executable(TestBlockMatcher TestBlockMatcher.cpp)
set_source_files_properties(TestBlockMatcher.cpp PROPERTIES LANGUAGE CUDA)
target_link_libraries(TestBlockMatcher
  PRIVATE 
    handheld_lib
    CUDA::cudart
)
add_test(NAME TestBlockMatcher 
    COMMAND TestBlockMatcher
    #run this binary with the source tree as cwd
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}  
  )