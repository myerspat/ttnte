#pragma once

namespace ttnte::utils {

enum class MPITag : int {
  DEFAULT = 0,

  // Initialization
  PATCHMETA = 1,
  PATCHPAYLOAD = 2,
};

}
