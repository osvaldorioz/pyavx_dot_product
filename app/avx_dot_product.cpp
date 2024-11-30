#include <immintrin.h> // Para AVX intrinsics
#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

//c++ -O2 -mavx -shared -std=c++20 -fPIC $(python3.12 -m pybind11 --includes) avx_dot_product.cpp -o avx_dot_product$(python3.12-config --extension-suffix)

namespace py = pybind11;

// Función para calcular el producto punto usando AVX
float avx_dot_product(const std::vector<float>& vec1, const std::vector<float>& vec2) {
    if (vec1.size() != vec2.size()) {
        throw std::runtime_error("Los vectores deben tener la misma longitud.");
    }

    size_t size = vec1.size();
    size_t avx_size = size / 8 * 8; // Tamaño alineado con bloques de 8 elementos (256 bits)

    __m256 sum = _mm256_setzero_ps(); // Inicializar suma vectorial

    // Calcular el producto punto usando AVX
    for (size_t i = 0; i < avx_size; i += 8) {
        __m256 v1 = _mm256_loadu_ps(&vec1[i]);
        __m256 v2 = _mm256_loadu_ps(&vec2[i]);
        __m256 prod = _mm256_mul_ps(v1, v2);
        sum = _mm256_add_ps(sum, prod);
    }

    // Reducir los 8 valores del registro AVX a un único escalar
    alignas(32) float temp[8];
    _mm256_store_ps(temp, sum);
    float result = temp[0] + temp[1] + temp[2] + temp[3] + temp[4] + temp[5] + temp[6] + temp[7];

    // Sumar los elementos restantes (si el tamaño no es múltiplo de 8)
    for (size_t i = avx_size; i < size; ++i) {
        result += vec1[i] * vec2[i];
    }

    return result;
}

// Función envoltura para aceptar arreglos NumPy
py::array_t<float> py_avx_dot_product(py::array_t<float> array1, py::array_t<float> array2) {
    // Verificar la forma de los arreglos
    if (array1.size() != array2.size()) {
        throw std::runtime_error("Los arreglos deben tener el mismo tamaño.");
    }

    // Convertir los arreglos NumPy en vectores std
    auto buf1 = array1.request();
    auto buf2 = array2.request();
    const float* ptr1 = static_cast<float*>(buf1.ptr);
    const float* ptr2 = static_cast<float*>(buf2.ptr);

    std::vector<float> vec1(ptr1, ptr1 + buf1.size);
    std::vector<float> vec2(ptr2, ptr2 + buf2.size);

    // Llamar a la función de producto punto
    float result = avx_dot_product(vec1, vec2);

    // Retornar el resultado como un escalar NumPy
    return py::array_t<float>(result);
}

// Registrar el módulo Pybind11
PYBIND11_MODULE(avx_dot_product, m) {
    m.doc() = "Módulo para producto punto vectorizado con AVX"; // Documentación
    m.def("dot_product", &py_avx_dot_product, "Calcula el producto punto usando AVX.");
}
