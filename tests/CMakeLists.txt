find_package(PkgConfig REQUIRED)
pkg_check_modules(CNPY REQUIRED cnpy)

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
