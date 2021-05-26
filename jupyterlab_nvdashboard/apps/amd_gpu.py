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
            output = self.bash.run("rocm-smi", capture_output=True).stdout.decode("utf-8")
            if "ROCm System Management Interface" in output:
                gpus_count = [int(num) for num in re.findall(r'[0-9]* ', output) if num != " "]
                if len(gpus_count) == 0:
                    return -1
                else:
                    return len(gpus_count)
            else:
                return -1
        except IndexError as i_err:
            logger.error("Unexpected return message while parsing ROCm output -", i_err)
        except FileNotFoundError as f_err:
            logger.error("ROCm is not installed -", f_err)

    def get_gpu_utilization(self, flag="-u") -> List[int]:
        """Return the utilization of each GPU in %."""
        try:
            output = self.bash.run(["rocm-smi", flag], capture_output=True).stdout.decode("utf-8")
            if "ROCm System Management Interface" in output:
                util = [int(num[:-1]) for num in re.findall(r' [0-9]*\n', output)]
                return util
            else:
                return [-1 for _ in range(self.gpus)]
        except IndexError as i_err:
            logger.error("Unexpected return message while parsing ROCm output -", i_err)

    def get_gpu_clock_freq(self, flag="-g") -> List[int]:
        """Return the clock frequence of each GPU in Mhz."""
        try:
            output = self.bash.run(["rocm-smi", flag], capture_output=True).stdout.decode("utf-8")
            if "ROCm System Management Interface" in output:
                freq = [int(num[1:]) for num in re.findall(r'\([0-9]*', output)]
                return freq
            else:
                return [-1 for _ in range(self.gpus)]
        except IndexError as i_err:
            logger.error("Unexpected return message while parsing ROCm output -", i_err)

    def get_gpu_mem_use(self, flag="--showmemuse") -> List[int]:
        """Return the current memory usage of each GPU in %."""
        try:
            output = self.bash.run(["rocm-smi", flag], capture_output=True).stdout.decode("utf-8")
            if "ROCm System Management Interface" in output:
                memUse = [int(num[:-1]) for num in re.findall(r' [0-9]*\n', output)]
                return memUse
            else:
                return [-1 for _ in range(self.gpus)]
        except IndexError as i_err:
            logger.error("Unexpected return message while parsing ROCm output -", i_err)
    
    def get_gpu_pcie_bandwith(self, flag="-b") -> List[float]:
        """Return the estimated maximum PCIe bandwith in MB/s."""
        try:
            output = self.bash.run(["rocm-smi", flag], capture_output=True).stdout.decode("utf-8")
            if "ROCm System Management Interface" in output:
                pcie_use = [float(num[:-1]) for num in re.findall(r' [0-9]*\.[0-9]*\n', output)]
                return pcie_use
            else:
                return [-1.0 for _ in range(self.gpus)]
        except IndexError as i_err:
            logger.error("Unexpected return message while parsing ROCm output -", i_err)

    def get_gpu_voltage(self, flag="--showvoltage") -> List[int]:
        """Return the current Voltage per GPU in mV."""
        try:
            output = self.bash.run(["rocm-smi", flag], capture_output=True).stdout.decode("utf-8")
            if "ROCm System Management Interface" in output:
                voltage = [int(num[:-1]) for num in re.findall(r' [0-9]*\n', output)]
                return voltage
            else:
                return [-1 for _ in range(self.gpus)]
        except IndexError as i_err:
            logger.error("Unexpected return message while parsing ROCm output -", i_err)
