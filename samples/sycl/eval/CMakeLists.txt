add_executable(Eval_Map_Mandelbrot map_mandelbrot.cpp)
target_link_libraries(Eval_Map_Mandelbrot ${CMAKE_THREAD_LIBS_INIT})

add_executable(Eval_Map_Mandelbrot_Pure map_mandelbrot_pure.cpp)

add_executable(Eval_Reduce_Sequence reduce_sequence.cpp)
target_link_libraries(Eval_Reduce_Sequence ${CMAKE_THREAD_LIBS_INIT})

add_executable(Eval_Reduce_Sequence_Pure reduce_sequence_pure.cpp)

add_executable(Simple_Map simple_map.cpp)
target_link_libraries(Simple_Map ${CMAKE_THREAD_LIBS_INIT})
