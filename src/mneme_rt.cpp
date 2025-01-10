#include <iostream>

extern "C" {
void __mneme_register_fatbin(const char *fatbin_wrapper,
                             size_t fatbin_wrapper_size, const char *hash) {
  // This is a dummy function that should never be called from here. Since we LD
  // preload it.
  std::cout << "Loading binary on address " << std::hex
            << (void *)fatbin_wrapper << std::dec << " and size "
            << fatbin_wrapper_size << "\n";
}
}
