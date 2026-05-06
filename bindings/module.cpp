#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/complex.h>
#include <pybind11/stl.h>
#include "math_utils.h"
#include "poly.h"
#include "ckks_types.h"
#include "encoder.h"
#include "keygen.h"
#include "encryptor.h"
#include "decryptor.h"
#include "evaluator.h"

namespace py = pybind11;

using CoeffArray = py::array_t<uint64_t,
                               py::array::c_style | py::array::forcecast>;

static Poly poly_from_array(uint32_t N, uint64_t q, const CoeffArray& arr) {
    py::buffer_info buf = arr.request();
    if (buf.ndim != 1)
        throw std::invalid_argument("coeffs must be a 1-D array");
    if (static_cast<uint32_t>(buf.shape[0]) != N)
        throw std::invalid_argument("coeffs length must equal N");
    const auto* ptr = static_cast<const uint64_t*>(buf.ptr);
    return Poly(N, q, std::vector<uint64_t>(ptr, ptr + N));
}

// Zero-copy uint64 view of a Poly's coefficient buffer.
// `base` is the Python object that owns the Poly; NumPy holds a ref to it
// so the buffer stays alive as long as the array view lives.
static py::array_t<uint64_t> poly_coeffs_view(py::object base) {
    Poly& poly = base.cast<Poly&>();
    return py::array_t<uint64_t>(
        {static_cast<py::ssize_t>(poly.N)},
        {static_cast<py::ssize_t>(sizeof(uint64_t))},
        poly.coeffs.data(),
        base);
}

PYBIND11_MODULE(ckks_python, m) {
    m.doc() = "CKKS homomorphic encryption for Federated Learning";

    // -------------------------------------------------------------------------
    // Prime generation utility
    // -------------------------------------------------------------------------
    m.def("generate_prime", &generate_proth_prime,
          py::arg("bit_size"), py::arg("N"),
          "Return a prime q with exactly bit_size bits satisfying q ≡ 1 (mod 2*N).\n"
          "Searches downward from 2^bit_size-1. Raises RuntimeError if none found.");

    // -------------------------------------------------------------------------
    // Poly
    // -------------------------------------------------------------------------
    py::class_<Poly>(m, "Poly")
        .def(py::init<uint32_t, uint64_t>(),
             py::arg("N"), py::arg("q"))
        .def(py::init([](uint32_t N, uint64_t q, const CoeffArray& arr) {
                 return poly_from_array(N, q, arr);
             }),
             py::arg("N"), py::arg("q"), py::arg("coeffs"))
        .def("__add__", &Poly::operator+)
        .def_property("coeffs",
            [](py::object self) -> py::array_t<uint64_t> {
                return poly_coeffs_view(self);
            },
            [](Poly& self, const CoeffArray& arr) {
                py::buffer_info buf = arr.request();
                if (buf.ndim != 1 ||
                    static_cast<uint32_t>(buf.shape[0]) != self.N)
                    throw std::invalid_argument(
                        "coeffs must be a 1-D array of length N");
                const auto* src = static_cast<const uint64_t*>(buf.ptr);
                std::copy(src, src + self.N, self.coeffs.begin());
            })
        .def_readonly("N", &Poly::N)
        .def_readonly("q", &Poly::q);

    // -------------------------------------------------------------------------
    // Plaintext
    // -------------------------------------------------------------------------
    py::class_<Plaintext>(m, "Plaintext")
        .def(py::init([](Poly p) { return Plaintext(std::move(p)); }),
             py::arg("poly"))
        // Zero-copy view: the lambda captures the Plaintext py::object as the
        // array base so NumPy keeps it alive while the view exists.
        .def_property_readonly("poly",
            [](py::object self) -> py::array_t<uint64_t> {
                Plaintext& pt = self.cast<Plaintext&>();
                return py::array_t<uint64_t>(
                    {static_cast<py::ssize_t>(pt.poly.N)},
                    {static_cast<py::ssize_t>(sizeof(uint64_t))},
                    pt.poly.coeffs.data(),
                    self);
            });

    // -------------------------------------------------------------------------
    // Ciphertext
    // -------------------------------------------------------------------------
    py::class_<Ciphertext>(m, "Ciphertext")
        .def(py::init([](Poly c0, Poly c1) {
                 return Ciphertext(std::move(c0), std::move(c1));
             }),
             py::arg("c0"), py::arg("c1"))
        .def_property_readonly("c0",
            [](py::object self) -> py::array_t<uint64_t> {
                Ciphertext& ct = self.cast<Ciphertext&>();
                return py::array_t<uint64_t>(
                    {static_cast<py::ssize_t>(ct.c0.N)},
                    {static_cast<py::ssize_t>(sizeof(uint64_t))},
                    ct.c0.coeffs.data(),
                    self);
            })
        .def_property_readonly("c1",
            [](py::object self) -> py::array_t<uint64_t> {
                Ciphertext& ct = self.cast<Ciphertext&>();
                return py::array_t<uint64_t>(
                    {static_cast<py::ssize_t>(ct.c1.N)},
                    {static_cast<py::ssize_t>(sizeof(uint64_t))},
                    ct.c1.coeffs.data(),
                    self);
            })
        .def("to_bytes",
            [](const Ciphertext& self) -> py::bytes {
                auto v = self.to_bytes();
                return py::bytes(reinterpret_cast<const char*>(v.data()),
                                 v.size());
            },
            "Serialize to a Python bytes object (little-endian, self-describing).")
        .def_static("from_bytes",
            [](py::bytes b) -> Ciphertext {
                std::string s(b);
                return Ciphertext::from_bytes(
                    reinterpret_cast<const uint8_t*>(s.data()), s.size());
            },
            py::arg("data"),
            "Deserialize from a Python bytes object.");

    // -------------------------------------------------------------------------
    // SecretKey
    // -------------------------------------------------------------------------
    py::class_<SecretKey>(m, "SecretKey")
        // Zero-copy view of the internal ternary polynomial.
        .def_property_readonly("s",
            [](py::object self) -> py::array_t<uint64_t> {
                SecretKey& sk = self.cast<SecretKey&>();
                return py::array_t<uint64_t>(
                    {static_cast<py::ssize_t>(sk.s.N)},
                    {static_cast<py::ssize_t>(sizeof(uint64_t))},
                    sk.s.coeffs.data(),
                    self);
            })
        .def("to_bytes",
            [](const SecretKey& self) -> py::bytes {
                auto v = self.to_bytes();
                return py::bytes(reinterpret_cast<const char*>(v.data()),
                                 v.size());
            },
            "Serialize to a Python bytes object (little-endian, self-describing).")
        .def_static("from_bytes",
            [](py::bytes b) -> SecretKey {
                std::string s(b);
                return SecretKey::from_bytes(
                    reinterpret_cast<const uint8_t*>(s.data()), s.size());
            },
            py::arg("data"),
            "Deserialize from a Python bytes object.");

    // -------------------------------------------------------------------------
    // PublicKey
    // -------------------------------------------------------------------------
    py::class_<PublicKey>(m, "PublicKey")
        .def_property_readonly("pk0",
            [](py::object self) -> py::array_t<uint64_t> {
                PublicKey& pk = self.cast<PublicKey&>();
                return py::array_t<uint64_t>(
                    {static_cast<py::ssize_t>(pk.pk0.N)},
                    {static_cast<py::ssize_t>(sizeof(uint64_t))},
                    pk.pk0.coeffs.data(),
                    self);
            })
        .def_property_readonly("pk1",
            [](py::object self) -> py::array_t<uint64_t> {
                PublicKey& pk = self.cast<PublicKey&>();
                return py::array_t<uint64_t>(
                    {static_cast<py::ssize_t>(pk.pk1.N)},
                    {static_cast<py::ssize_t>(sizeof(uint64_t))},
                    pk.pk1.coeffs.data(),
                    self);
            })
        .def("to_bytes",
            [](const PublicKey& self) -> py::bytes {
                auto v = self.to_bytes();
                return py::bytes(reinterpret_cast<const char*>(v.data()),
                                 v.size());
            },
            "Serialize to a Python bytes object (little-endian, self-describing).")
        .def_static("from_bytes",
            [](py::bytes b) -> PublicKey {
                std::string s(b);
                return PublicKey::from_bytes(
                    reinterpret_cast<const uint8_t*>(s.data()), s.size());
            },
            py::arg("data"),
            "Deserialize from a Python bytes object.");

    // -------------------------------------------------------------------------
    // Encoder
    // -------------------------------------------------------------------------
    py::class_<Encoder>(m, "Encoder")
        .def(py::init<uint32_t, uint64_t, double>(),
             py::arg("N"), py::arg("q"), py::arg("scale"))
        // Real-valued overload: float64 array → Poly.
        .def("encode",
            [](const Encoder& self,
               py::array_t<double, py::array::c_style | py::array::forcecast> arr)
               -> Poly {
                py::buffer_info buf = arr.request();
                if (buf.ndim != 1)
                    throw std::invalid_argument("input must be a 1-D array");
                const auto* ptr = static_cast<const double*>(buf.ptr);
                std::vector<std::complex<double>> slots(buf.shape[0]);
                for (py::ssize_t i = 0; i < buf.shape[0]; ++i)
                    slots[i] = ptr[i];
                return self.encode(slots);
            }, py::arg("z"),
            "Encode a real-valued float64 array into a Poly.")
        // Complex-valued overload: complex128 array → Poly.
        .def("encode",
            [](const Encoder& self,
               py::array_t<std::complex<double>,
                           py::array::c_style | py::array::forcecast> arr)
               -> Poly {
                py::buffer_info buf = arr.request();
                if (buf.ndim != 1)
                    throw std::invalid_argument("input must be a 1-D array");
                const auto* ptr = static_cast<const std::complex<double>*>(buf.ptr);
                return self.encode(
                    std::vector<std::complex<double>>(ptr, ptr + buf.shape[0]));
            }, py::arg("z"),
            "Encode a complex128 array into a Poly.")
        // decode: returns a 1-D complex128 NumPy array of length N/2.
        .def("decode",
            [](const Encoder& self, const Poly& p) {
                auto slots = self.decode(p);
                py::array_t<std::complex<double>> out(
                    {static_cast<py::ssize_t>(slots.size())});
                auto* dst = static_cast<std::complex<double>*>(out.request().ptr);
                std::copy(slots.begin(), slots.end(), dst);
                return out;
            }, py::arg("p"))
        .def_property_readonly("slots",  &Encoder::slots)
        .def_property_readonly("degree", &Encoder::degree);

    // -------------------------------------------------------------------------
    // KeyGenerator  (non-copyable — pybind11 uses unique_ptr holder)
    // -------------------------------------------------------------------------
    py::class_<KeyGenerator>(m, "KeyGenerator")
        .def(py::init<uint32_t, uint64_t, double, uint64_t>(),
             py::arg("N"), py::arg("q"),
             py::arg("sigma") = 3.2,
             py::arg("seed")  = 0)
        .def("gen_secret_key",
             &KeyGenerator::gen_secret_key,
             py::arg("hamming_weight") = 0)
        .def("gen_public_key",
             &KeyGenerator::gen_public_key,
             py::arg("sk"));

    // -------------------------------------------------------------------------
    // Encryptor  (non-copyable)
    // -------------------------------------------------------------------------
    py::class_<Encryptor>(m, "Encryptor")
        .def(py::init<uint32_t, uint64_t, const PublicKey&, double, uint64_t>(),
             py::arg("N"), py::arg("q"), py::arg("pk"),
             py::arg("sigma") = 3.2,
             py::arg("seed")  = 0)
        .def("encrypt",
             &Encryptor::encrypt,
             py::arg("pt"));

    // -------------------------------------------------------------------------
    // Decryptor  (non-copyable)
    // -------------------------------------------------------------------------
    py::class_<Decryptor>(m, "Decryptor")
        .def(py::init<uint32_t, uint64_t, const SecretKey&>(),
             py::arg("N"), py::arg("q"), py::arg("sk"))
        .def("decrypt",
             &Decryptor::decrypt,
             py::arg("ct"));

    // -------------------------------------------------------------------------
    // Evaluator
    // -------------------------------------------------------------------------
    py::class_<Evaluator>(m, "Evaluator")
        .def(py::init<uint32_t, uint64_t>(),
             py::arg("N"), py::arg("q"))
        .def("add_inplace",
             &Evaluator::add_inplace,
             py::arg("ct1"), py::arg("ct2"),
             "ct1 += ct2 in-place (coefficient-wise mod q).")
        .def("add",
             &Evaluator::add,
             py::arg("ct1"), py::arg("ct2"),
             "Returns a new Ciphertext equal to ct1 + ct2.");
}
