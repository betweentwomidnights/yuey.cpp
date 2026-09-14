if(NOT DEFINED INPUT OR NOT DEFINED OUTPUT OR NOT DEFINED SYMBOL)
    message(FATAL_ERROR "embed_resource.cmake needs INPUT, OUTPUT, and SYMBOL")
endif()

file(READ "${INPUT}" bytes HEX)
string(LENGTH "${bytes}" hex_length)
math(EXPR byte_count "${hex_length} / 2")
set(initializer "")
if(byte_count GREATER 0)
    math(EXPR last_byte "${byte_count} - 1")
    foreach(index RANGE 0 ${last_byte})
        math(EXPR offset "${index} * 2")
        string(SUBSTRING "${bytes}" ${offset} 2 byte)
        string(APPEND initializer "0x${byte},")
        math(EXPR column "(${index} + 1) % 20")
        if(column EQUAL 0)
            string(APPEND initializer "\n")
        endif()
    endforeach()
endif()

get_filename_component(output_directory "${OUTPUT}" DIRECTORY)
file(MAKE_DIRECTORY "${output_directory}")
file(WRITE "${OUTPUT}"
    "#pragma once\n"
    "#include <cstddef>\n"
    "namespace yue2::ui {\n"
    "inline constexpr unsigned char ${SYMBOL}[] = {${initializer}};\n"
    "inline constexpr std::size_t ${SYMBOL}_size = ${byte_count};\n"
    "} // namespace yue2::ui\n")
