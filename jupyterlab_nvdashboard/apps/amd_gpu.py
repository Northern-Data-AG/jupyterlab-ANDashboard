# requires rocm!
import subprocess
import logging
import re


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

    def get_gpu_count(self) -> None:
        """Get the number of working GPUs."""
        try:
            output = self.bash.run("rocm-smi", capture_output=True).stdout
            if "ROCm System Management Interface" in output:
                gpus_count = [int(num[-1]) for num in re.findall(r'\n\d{1,2}', str(output))]
                self.gpus = len(gpus_count)
            else:
                self.gpus = 0

        except Exception:
            logger.error("An error has occured while searching for the number of GPUs!")

    def get_gpu_clock_freq(self, flag="-g") -> list[int]:
        """Return the clock frequence of each GPU in Mhz."""
        try:
            output = self.bash.run(["rocm-smi", flag], capture_output=True).stdout
            if "ROCm System Management Interface" in output:
                freq = [int(num[1:-1]) for num in re.findall(r'\([0-9]*M', str(output))]
                return freq
            else:
                return [0 for _ in range(self.gpus)]

        except Exception:
            logger.error("An error has occured while searching for the clock frequence!")

    def get_gpu_mem_use(self, flag="--showmemuse") -> list[int]:
        """Return the current memory usage of each GPU in %."""
        try:
            output = self.bash.run(["rocm-smi", flag], capture_output=True).stdout
            if "ROCm System Management Interface" in output:
                memUse = [int(num[0:-1]) for num in re.findall(r'\d{1,3}\n', str(output))]
                return memUse
            else:
                return [0 for _ in range(self.gpus)]

        except Exception:
            logger.error("An error has occured while searching for the memory usage!")
    
    def get_gpu_pcie_bandwith(self, flag="-b") -> list[float]:
        """Return the estimated maximum PCIe bandwith in MB/s."""
        try:
            output = self.bash.run(["rocm-smi", flag], capture_output=True).stdout      # takes a few seconds
            if "ROCm System Management Interface" in output:
                pcie_use = [float(num[1:-1]) for num in re.findall(r' [0-9]*\.[0-9]*\n', str(output))]
                return pcie_use
            else:
                return [0.0 for _ in range(self.gpus)]

        except Exception:
            logger.error("An error has occured while searching for the PCIe bandwith!")

    def get_gpu_voltage(self, flag="--showvoltage") -> list[int]:
        """Return the current Voltage per GPU in mV."""
        try:
            output = self.bash.run(["rocm-smi", flag], capture_output=True).stdout      # takes a few seconds
            if "ROCm System Management Interface" in output:
                voltage = [int(num[:-1]) for num in re.findall(r' [0-9]*\n', str(output))]
                return voltage
            else:
                return [0 for _ in range(self.gpus)]

        except Exception:
            logger.error("An error has occured while searching for the GPU voltage!")
