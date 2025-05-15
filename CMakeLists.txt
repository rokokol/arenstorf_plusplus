cmake_minimum_required(VERSION 3.30)
project(matplotlibcpp_test)

set(CMAKE_CXX_STANDARD 23)
add_compile_options(-mavx)
add_executable(matplotlibcpp_test main.cpp
        rung.h
        satellite.h)


#=====//НУЖНО ДЛЯ РАБОТЫ С <matplotlibcpp.h>//=====
# Ищем Python3 и NumPy
find_package(Python3 COMPONENTS Interpreter Development NumPy REQUIRED)

# Подключаем заголовки Python и NumPy
target_include_directories(matplotlibcpp_test PRIVATE
        ${Python3_INCLUDE_DIRS}
        ${Python3_NumPy_INCLUDE_DIRS}
)

# Линкуемся с Python
target_link_libraries(matplotlibcpp_test PRIVATE
        Python3::Python
)
