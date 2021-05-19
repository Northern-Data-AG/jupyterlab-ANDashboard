# requires rocm!
import subprocess
import logging
import re
from typing import List


logging.basicConfig(
    filename="/var/log/amd_gpu.log",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)


class AmdGpuProperties:
    def __init__(self, bash=subprocess):
        self.bash = bash
        self.gpus = self.get_gpu_count()

    def get_gpu_count(self) -> int:
        """Get the number of working GPUs."""
        try:
            output = self.bash.run("rocm-smi", capture_output=True).stdout
            if "ROCm System Management Interface" in output:
                gpus_count = [int(num[-1]) for num in re.findall(r'\n\d{1,2}', str(output))]
                return len(gpus_count)
            else:
                raise ValueError("No GPUs have been found!")
        except ValueError as v_err:
            logger.error(v_err)
        except IndexError as i_err:
            logger.error("Unexpected return message while parsing ROCm output -", i_err)
        except FileNotFoundError as f_err:
            logger.error("ROCm is not installed -", f_err)

    def get_gpu_clock_freq(self, flag="-g") -> List[int]:
        """Return the clock frequence of each GPU in Mhz."""
        try:
            output = self.bash.run(["rocm-smi", flag], capture_output=True).stdout
            if "ROCm System Management Interface" in output:
                freq = [int(num[1:-1]) for num in re.findall(r'\([0-9]*M', str(output))]
                return freq
            else:
                raise LookupError("Unexpected return message while parsing ROCm output - clock frequence not found")
        except LookupError as l_err:
            logger.error(l_err)
        except IndexError as i_err:
            logger.error("Unexpected return message while parsing ROCm output -", i_err)

    def get_gpu_mem_use(self, flag="--showmemuse") -> List[int]:
        """Return the current memory usage of each GPU in %."""
        try:
            output = self.bash.run(["rocm-smi", flag], capture_output=True).stdout
            if "ROCm System Management Interface" in output:
                memUse = [int(num[0:-1]) for num in re.findall(r'\d{1,3}\n', str(output))]
                return memUse
            else:
                raise LookupError("Unexpected return message while parsing ROCm output - memory usage not found")
        except LookupError as l_err:
            logger.error(l_err)
        except IndexError as i_err:
            logger.error("Unexpected return message while parsing ROCm output -", i_err)
    
    def get_gpu_pcie_bandwith(self, flag="-b") -> List[float]:
        """Return the estimated maximum PCIe bandwith in MB/s."""
        try:
            output = self.bash.run(["rocm-smi", flag], capture_output=True).stdout      # takes a few seconds
            if "ROCm System Management Interface" in output:
                pcie_use = [float(num[1:-1]) for num in re.findall(r' [0-9]*\.[0-9]*\n', str(output))]
                return pcie_use
            else:
                raise LookupError("Unexpected return message while parsing ROCm output - PCIe bandwith not found")
        except LookupError as l_err:
            logger.error(l_err)
        except IndexError as i_err:
            logger.error("Unexpected return message while parsing ROCm output -", i_err)

    def get_gpu_voltage(self, flag="--showvoltage") -> List[int]:
        """Return the current Voltage per GPU in mV."""
        try:
            output = self.bash.run(["rocm-smi", flag], capture_output=True).stdout      # takes a few seconds
            if "ROCm System Management Interface" in output:
                voltage = [int(num[:-1]) for num in re.findall(r' [0-9]*\n', str(output))]
                return voltage
            else:
                raise LookupError("Unexpected return message while parsing ROCm output - Voltage not found")
        except LookupError as l_err:
            logger.error(l_err)
        except IndexError as i_err:
            logger.error("Unexpected return message while parsing ROCm output -", i_err)
