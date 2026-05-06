add_executable(test_math_utils test_math_utils.cpp)
target_link_libraries(test_math_utils PRIVATE ckks_core)
add_test(NAME test_math_utils COMMAND test_math_utils)

add_executable(test_poly test_poly.cpp)
target_link_libraries(test_poly PRIVATE ckks_core)
add_test(NAME test_poly COMMAND test_poly)

add_executable(test_ckks_ops test_ckks_ops.cpp)
target_link_libraries(test_ckks_ops PRIVATE ckks_core)
add_test(NAME test_ckks_ops COMMAND test_ckks_ops)

add_executable(test_sampler test_sampler.cpp)
target_link_libraries(test_sampler PRIVATE ckks_core)
add_test(NAME test_sampler COMMAND test_sampler)

add_executable(test_encoder test_encoder.cpp)
target_link_libraries(test_encoder PRIVATE ckks_core)
add_test(NAME test_encoder COMMAND test_encoder)

add_executable(test_ntt test_ntt.cpp)
target_link_libraries(test_ntt PRIVATE ckks_core)
add_test(NAME test_ntt COMMAND test_ntt)

add_executable(test_keygen test_keygen.cpp)
target_link_libraries(test_keygen PRIVATE ckks_core)
add_test(NAME test_keygen COMMAND test_keygen)

add_executable(test_encryption test_encryption.cpp)
target_link_libraries(test_encryption PRIVATE ckks_core)
add_test(NAME test_encryption COMMAND test_encryption)

add_executable(test_homomorphic_add test_homomorphic_add.cpp)
target_link_libraries(test_homomorphic_add PRIVATE ckks_core)
add_test(NAME test_homomorphic_add COMMAND test_homomorphic_add)

# Serialization round-trip tests — Ciphertext and PublicKey to/from bytes.
add_executable(test_serialization test_serialization.cpp)
target_link_libraries(test_serialization PRIVATE ckks_core)
add_test(NAME test_serialization COMMAND test_serialization)

# FL aggregation stress test — 90 clients, Δ = 2^40, 90 homomorphic additions.
add_executable(test_fl_aggregation ../test_fl_aggregation.cpp)
target_link_libraries(test_fl_aggregation PRIVATE ckks_core)
add_test(NAME test_fl_aggregation COMMAND test_fl_aggregation)

# Prime generation — bit-length, congruence, and primality checks.
add_executable(test_prime_gen test_prime_gen.cpp)
target_link_libraries(test_prime_gen PRIVATE ckks_core)
add_test(NAME test_prime_gen COMMAND test_prime_gen)

# Noise characterisation tool — lives at tests/test_noise.cpp (one level up).
# Prints the decryption noise infinity norm to stdout; always exits 0.
add_executable(test_noise ../test_noise.cpp)
target_link_libraries(test_noise PRIVATE ckks_core)
add_test(NAME test_noise COMMAND test_noise)
