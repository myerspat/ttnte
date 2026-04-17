#include "ttnte/parallel/mpi_types.hpp"
#include <mpi.h>

namespace ttnte::parallel {

int std2mpi<int32_t>::value()
{
  return MPI_Type_c2f(MPI_INT32_T);
}

int std2mpi<int64_t>::value()
{
  return MPI_Type_c2f(MPI_INT64_T);
}

int std2mpi<float>::value()
{
  return MPI_Type_c2f(MPI_FLOAT);
}

int std2mpi<double>::value()
{
  return MPI_Type_c2f(MPI_DOUBLE);
}

int std2mpi<std::byte>::value()
{
  return MPI_Type_c2f(MPI_BYTE);
}

} // namespace ttnte::parallel
