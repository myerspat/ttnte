import cupy as cu
import numpy as np
from quimb.tensor import Tensor, TensorNetwork


class GPUManager(object):
    def __init__(self, safety_factor=0.9, device_id=0):
        # Get device instance
        self._device = cu.cuda.Device(device_id)

        # Factor to ensure we never overflow
        self._safety_factor = safety_factor

        # Determine available memory
        self.reset_free_memory()

    # =============================================================
    # Methods

    def reset_free_memory(self):
        # Free blocks
        cu.get_default_memory_pool().free_all_blocks()

        # Available memory
        self._free_memory, self._total_memory = self._device.mem_info

        # Apply factor
        self._free_memory *= self._safety_factor

    def check_available(self, mem=0, num_elements=0, types=np.float32, *arrays):
        # Sum the memory of given arrays
        total_memory = mem

        if num_elements != 0:
            if isinstance(num_elements, int):
                num_elements = [num_elements]
                types = [types]

            for num_element, t in zip(num_elements, types):
                total_memory += num_element * np.dtype(t).itemsize

        if arrays:
            total_memory += self._calc_array_memory(arrays)

        return self._free_memory > total_memory

    def _calc_array_memory(self, array):
        if isinstance(array, np.ndarray) or isinstance(array, cu.ndarray):
            return array.nbytes

        elif isinstance(array, Tensor):
            return array.data.nbytes

        elif isinstance(array, TensorNetwork):
            return sum([self._calc_array_memory(a.data) for a in array])

        elif isinstance(array, list) or isinstance(array, tuple):
            return sum([self._calc_array_memory(a) for a in array])

        else:
            raise RuntimeError(f"Given array type ({type(array)}) is not supported")

    def to_gpu(self, arrays):
        """Send arrays to GPU."""
        # Calculate memory removed
        self._free_memory -= self._calc_array_memory(arrays)

        # Check we did not oversubscribe memory
        if self._free_memory < 0:
            raise RuntimeError("Not enough memory for given arrays")

        return self._send_or_get_arrays(arrays, lambda a: cu.array(a))

    def from_gpu(self, arrays):
        """Get arrays from GPU."""
        self._free_memory += self._calc_array_memory(arrays)
        return self._send_or_get_arrays(arrays, lambda a: a.get())

    def _send_or_get_arrays(self, arrays, modifier):
        with self._device:
            if isinstance(arrays, np.ndarray) or isinstance(arrays, cu.ndarray):
                arrays = modifier(arrays)

            elif isinstance(arrays, Tensor):
                arrays.modify(data=modifier(arrays.data))

            elif isinstance(arrays, TensorNetwork):
                for array in arrays:
                    array.modify(data=modifier(array.data))

            elif isinstance(arrays, list) or isinstance(arrays, tuple):
                arrays = [self._send_or_get_arrays(array, modifier) for array in arrays]

            else:
                raise RuntimeError(
                    f"Given array type ({type(arrays)}) is not supported"
                )

            return arrays
