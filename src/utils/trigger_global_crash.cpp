#include "ttnte/parallel/parallel_context.hpp"
#include "ttnte/utils/trigger_global_crach.hpp"
#include <iostream>

namespace ttnte::utils {

void trigger_global_crash(const std::string& error_msg)
{
  const auto& context = parallel::ParallelContext::instance();

  std::cerr << "\nFATAL ERROR on MPI Rank " << context.rank()
            << " with message:\n"
            << error_msg << "\nInitiating global MPI_Abort\n\n";

  context.mpi_abort();
}

} // namespace ttnte::utils
