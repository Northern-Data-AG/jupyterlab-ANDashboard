import pytest
from unittest.mock import MagicMock

from apps import AmdGpuProperties as amd

@pytest.mark.parametrize(
    "test_input,expected_val",
    [(b"\n\n======================= ROCm System Management Interface =======================\n\
        ================================= Concise Info =================================\n\
        GPU  Temp   AvgPwr  SCLK    MCLK    Fan   Perf  PwrCap  VRAM%  GPU% \n\
        0    20.0c  14.0W   925Mhz  350Mhz  0.0%  auto  225.0W    0%   0%   \n\
        1    22.0c  19.0W   930Mhz  350Mhz  0.0%  auto  225.0W    0%   0%   \n\
        2    18.0c  17.0W   930Mhz  350Mhz  0.0%  auto  225.0W    0%   0%   \n\
        3    19.0c  24.0W   930Mhz  350Mhz  0.0%  auto  225.0W    0%   0%   \n\
        4    22.0c  17.0W   930Mhz  350Mhz  0.0%  auto  225.0W    0%   0%   \n\
        5    22.0c  15.0W   930Mhz  350Mhz  0.0%  auto  225.0W    0%   0%   \n\
        6    20.0c  18.0W   930Mhz  350Mhz  0.0%  auto  225.0W    0%   0%   \n\
        7    22.0c  16.0W   930Mhz  350Mhz  0.0%  auto  225.0W    0%   0%   \n\
        ================================================================================\n\
        ============================= End of ROCm SMI Log ==============================\n",
        8
    ),
    (b"\n\n======================= ROCm System Management Interface =======================\n\
        ================================= Concise Info =================================\n\
        GPU  Temp   AvgPwr  SCLK    MCLK    Fan   Perf  PwrCap  VRAM%  GPU% \n\
        0    20.0c  14.0W   925Mhz  350Mhz  0.0%  auto  225.0W    0%   0%   \n\
        1    22.0c  19.0W   930Mhz  350Mhz  0.0%  auto  225.0W    0%   0%   \n\
        2    18.0c  17.0W   930Mhz  350Mhz  0.0%  auto  225.0W    0%   0%   \n\
        3    19.0c  24.0W   930Mhz  350Mhz  0.0%  auto  225.0W    0%   0%   \n\
        4    22.0c  17.0W   930Mhz  350Mhz  0.0%  auto  225.0W    0%   0%   \n\
        5    22.0c  15.0W   930Mhz  350Mhz  0.0%  auto  225.0W    0%   0%   \n\
        6    20.0c  18.0W   930Mhz  350Mhz  0.0%  auto  225.0W    0%   0%   \n\
        ================================================================================\n\
        ============================= End of ROCm SMI Log ==============================\n",
        7
    ),(b"\n\n======================= ROCm System Management Interface =======================\n\
        ================================= Concise Info =================================\n\
        \n\
        ================================================================================\n\
        ============================= End of ROCm SMI Log ==============================\n",
        -1
    ),
    (b"rocm-smi: command not found", -1),
    (b"-sh: rocm-smi: not found", -1)
    ])
def test_get_gpu_count(test_input, expected_val):
    bash = MagicMock()
    bash.run().stdout = test_input
    gpus = amd.gpus
    assert gpus == expected_val
    bash.run.assert_called_with("rocm-smi", capture_output=True)

@pytest.mar.parameterize(
    "test_input,expected_sum,expected_len",
    [(b"\n\n======================= ROCm System Management Interface =======================\n\
        ============================== % time GPU is busy ==============================\n\
        GPU[0]\t\t: GPU use (%): 0\n\
        GPU[1]\t\t: GPU use (%): 0\n\
        GPU[2]\t\t: GPU use (%): 0\n\
        GPU[3]\t\t: GPU use (%): 0\n\
        GPU[4]\t\t: GPU use (%): 0\n\
        GPU[5]\t\t: GPU use (%): 0\n\
        GPU[6]\t\t: GPU use (%): 0\n\
        GPU[7]\t\t: GPU use (%): 0\n\
        ================================================================================\n\
        ============================= End of ROCm SMI Log ==============================\n",
        0,
        8
    ),
    (b"\n\n======================= ROCm System Management Interface =======================\n\
        ============================== % time GPU is busy ==============================\n\
        \n\
        ================================================================================\n\
        ============================= End of ROCm SMI Log ==============================\n",
        0,
        0
    ),
    (b"\n\n======================= ROCm System Management Interface =======================\n\
        ============================== % time GPU is busy ==============================\n\
        GPU[0]\t\t: GPU use (%): 100\n\
        GPU[1]\t\t: GPU use (%): 50\n\
        GPU[2]\t\t: GPU use (%): 50\n\
        GPU[3]\t\t: GPU use (%): 45\n\
        GPU[4]\t\t: GPU use (%): 0\n\
        GPU[5]\t\t: GPU use (%): 0\n\
        GPU[6]\t\t: GPU use (%): 90\n\
        ================================================================================\n\
        ============================= End of ROCm SMI Log ==============================\n",
        335,
        7
    ),
    (b"rocm-smi: command not found", 0, 0),
    (b"-sh: rocm-smi: not found", 0, 0)
    ])
def test_get_gpu_utilization(test_input, expected_sum, expected_len):
    bash = MagicMock()
    bash.run().stdout = test_input
    gpus = amd.gpus
    gpu_util = amd.get_gpu_utilization()
    assert len(gpu_util) == expected_len
    assert sum(gpu_util) == expected_sum
    bash.run.assert_called_with(["rocm-smi", "-u"], capture_output=True)

@pytest.mark.parameterize(
    "test_input,expected_sum,expected_len",
    [(b"\n\n======================= ROCm System Management Interface =======================\n\
        ========================== Current clock frequencies ===========================\n\
        GPU[0]\t\t: sclk clock level: 0 (925Mhz)\n\
        GPU[1]\t\t: sclk clock level: 1 (930Mhz)\n\
        GPU[2]\t\t: sclk clock level: 1 (930Mhz)\n\
        GPU[3]\t\t: sclk clock level: 1 (930Mhz)\n\
        GPU[4]\t\t: sclk clock level: 1 (930Mhz)\n\
        GPU[5]\t\t: sclk clock level: 1 (930Mhz)\n\
        GPU[6]\t\t: sclk clock level: 1 (930Mhz)\n\
        GPU[7]\t\t: sclk clock level: 1 (930Mhz)\n\
        ================================================================================\n\
        ============================= End of ROCm SMI Log ==============================\n",
        7435,
        8
    ),
    (b"\n\n======================= ROCm System Management Interface =======================\n\
        ========================== Current clock frequencies ===========================\n\
        GPU[0]\t\t: sclk clock level: 0 (925Mhz)\n\
        GPU[1]\t\t: sclk clock level: 1 (930Mhz)\n\
        GPU[2]\t\t: sclk clock level: 1 (1930Mhz)\n\
        GPU[3]\t\t: sclk clock level: 1 (930Mhz)\n\
        GPU[4]\t\t: sclk clock level: 1 (930Mhz)\n\
        GPU[5]\t\t: sclk clock level: 1 (0Mhz)\n\
        GPU[6]\t\t: sclk clock level: 1 (930Mhz)\n\
        GPU[7]\t\t: sclk clock level: 1 (10930Mhz)\n\
        ================================================================================\n\
        ============================= End of ROCm SMI Log ==============================\n",
        17505,
        8
    ),
    (b"\n\n======================= ROCm System Management Interface =======================\n\
        ========================== Current clock frequencies ===========================\n\
        GPU[0]\t\t: sclk clock level: 0 (925Mhz)\n\
        GPU[1]\t\t: sclk clock level: 1 (930Mhz)\n\
        GPU[2]\t\t: sclk clock level: 1 (1930Mhz)\n\
        GPU[3]\t\t: sclk clock level: 1 (930Mhz)\n\
        GPU[4]\t\t: sclk clock level: 1 (930Mhz)\n\
        GPU[5]\t\t: sclk clock level: 1 (0Mhz)\n\
        ================================================================================\n\
        ============================= End of ROCm SMI Log ==============================\n",
        5645,
        6
    ),
    (b"rocm-smi: command not found", 0, 0),
    (b"-sh: rocm-smi: not found", 0, 0)
    ])
def test_get_gpu_clock_freq(test_input, expected_sum, expected_len):
    bash = MagicMock()
    bash.run().stdout = test_input
    gpus = amd.gpus
    gpu_freq = amd.get_gpu_clock_freq()
    assert len(gpu_freq) == expected_len
    assert sum(gpu_freq) == expected_sum
    bash.run.assert_called_with(["rocm-smi", "-g"], capture_output=True)

@pytest.mark.parametrize(
    "test_input,expected_sum,expected_len",
    [(b"\n\n======================= ROCm System Management Interface =======================\n\
        ================================= Concise Info =================================\n\
        GPU  Temp   AvgPwr  SCLK    MCLK    Fan   Perf  PwrCap  VRAM%  GPU%  \n\
        0    20.0c  16.0W   925Mhz  350Mhz  0.0%  auto  225.0W   100%  8%    \n\
        1    23.0c  19.0W   925Mhz  350Mhz  0.0%  auto  225.0W    0%   8%    \n\
        2    18.0c  18.0W   925Mhz  350Mhz  0.0%  auto  225.0W    0%   8%    \n\
        3    20.0c  23.0W   925Mhz  350Mhz  0.0%  auto  225.0W    0%   8%    \n\
        4    22.0c  17.0W   925Mhz  350Mhz  0.0%  auto  225.0W    4%   8%    \n\
        5    22.0c  15.0W   925Mhz  350Mhz  0.0%  auto  225.0W    0%   8%    \n\
        6    21.0c  19.0W   925Mhz  350Mhz  0.0%  auto  225.0W    0%   8%    \n\
        7    21.0c  15.0W   925Mhz  350Mhz  0.0%  auto  225.0W    0%   8%    \n\
        ================================================================================\n\
        ============================= End of ROCm SMI Log ==============================\n",
        104,
        8
    ),
    (b"\n\n======================= ROCm System Management Interface =======================\n\
        ================================= Concise Info =================================\n\
        GPU  Temp   AvgPwr  SCLK    MCLK    Fan   Perf  PwrCap  VRAM%  GPU%  \n\
        0    20.0c  16.0W   925Mhz  350Mhz  0.0%  auto  225.0W    0%   0%    \n\
        1    23.0c  19.0W   925Mhz  350Mhz  0.0%  auto  225.0W    0%   0%    \n\
        2    20.0c  23.0W   925Mhz  350Mhz  0.0%  auto  225.0W    0%   0%    \n\
        3    22.0c  17.0W   925Mhz  350Mhz  0.0%  auto  225.0W    0%   0%    \n\
        4    22.0c  15.0W   925Mhz  350Mhz  0.0%  auto  225.0W    0%   0%    \n\
        5    21.0c  19.0W   925Mhz  350Mhz  0.0%  auto  225.0W    0%   0%    \n\
        6    21.0c  15.0W   925Mhz  350Mhz  0.0%  auto  225.0W    0%   0%    \n\
        ================================================================================\n\
        ============================= End of ROCm SMI Log ==============================\n",
        0,
        7
    ),
    (b"rocm-smi: command not found", 0, 0),
    (b"-sh: rocm-smi: not found", 0, 0)
    ])
def test_get_gpu_vram_use(test_input, expected_sum, expected_len):
    bash = MagicMock()
    bash.run().stdout = test_input
    gpus = amd.gpus
    gpu_vram = amd.get_gpu_vram_use()
    assert len(gpu_vram) == expected_len
    assert sum(gpu_vram) == expected_sum
    bash.run.assert_called_with(["rocm-smi", "--showmemuse"], capture_output=True)

@pytest.mark.parametrize(
    "test_input,expected_sum,expected_len",
    [(b"\n\n======================= ROCm System Management Interface =======================\n\
        =========================== Measured PCIe Bandwidth ============================\n\
        GPU[0]\t\t: Estimated maximum PCIe bandwidth over the last second (MB/s): 0.000\n\
        GPU[1]\t\t: Estimated maximum PCIe bandwidth over the last second (MB/s): 0.000\n\
        GPU[2]\t\t: Estimated maximum PCIe bandwidth over the last second (MB/s): 0.000\n\
        GPU[3]\t\t: Estimated maximum PCIe bandwidth over the last second (MB/s): 0.000\n\
        GPU[4]\t\t: Estimated maximum PCIe bandwidth over the last second (MB/s): 0.000\n\
        GPU[5]\t\t: Estimated maximum PCIe bandwidth over the last second (MB/s): 0.000\n\
        GPU[6]\t\t: Estimated maximum PCIe bandwidth over the last second (MB/s): 0.000\n\
        GPU[7]\t\t: Estimated maximum PCIe bandwidth over the last second (MB/s): 0.000\n\
        ================================================================================\n\
        ============================= End of ROCm SMI Log ==============================\n",
        0.0,
        8
    ),
    (b"\n\n======================= ROCm System Management Interface =======================\n\
        =========================== Measured PCIe Bandwidth ============================\n\
        GPU[0]\t\t: Estimated maximum PCIe bandwidth over the last second (MB/s): 500.000\n\
        GPU[1]\t\t: Estimated maximum PCIe bandwidth over the last second (MB/s): 0.071\n\
        GPU[2]\t\t: Estimated maximum PCIe bandwidth over the last second (MB/s): 0.999\n\
        GPU[3]\t\t: Estimated maximum PCIe bandwidth over the last second (MB/s): 1501.002\n\
        GPU[4]\t\t: Estimated maximum PCIe bandwidth over the last second (MB/s): 0.000\n\
        GPU[5]\t\t: Estimated maximum PCIe bandwidth over the last second (MB/s): 0.000\n\
        GPU[6]\t\t: Estimated maximum PCIe bandwidth over the last second (MB/s): 0.000\n\
        GPU[7]\t\t: Estimated maximum PCIe bandwidth over the last second (MB/s): 0.500\n\
        ================================================================================\n\
        ============================= End of ROCm SMI Log ==============================\n",
        2002.5720000000001,
        8
    )(b"\n\n======================= ROCm System Management Interface =======================\n\
        =========================== Measured PCIe Bandwidth ============================\n\
        GPU[0]\t\t: Estimated maximum PCIe bandwidth over the last second (MB/s): 500.000\n\
        GPU[1]\t\t: Estimated maximum PCIe bandwidth over the last second (MB/s): 0.071\n\
        GPU[2]\t\t: Estimated maximum PCIe bandwidth over the last second (MB/s): 0.999\n\
        GPU[3]\t\t: Estimated maximum PCIe bandwidth over the last second (MB/s): 1501.002\n\
        GPU[4]\t\t: Estimated maximum PCIe bandwidth over the last second (MB/s): 0.000\n\
        GPU[5]\t\t: Estimated maximum PCIe bandwidth over the last second (MB/s): 0.000\n\
        GPU[6]\t\t: Estimated maximum PCIe bandwidth over the last second (MB/s): 0.500\n\
        ================================================================================\n\
        ============================= End of ROCm SMI Log ==============================\n",
        2002.5720000000001,
        7
    ),
    (b"rocm-smi: command not found", 0, 0),
    (b"-sh: rocm-smi: not found", 0, 0)
    ])
def test_get_gpu_pcie_use(test_input, expected_sum, expected_len):
    bash = MagicMock()
    bash.run().stdout = test_input
    gpus = amd.gpus
    gpu_pcie = amd.get_pcie_use()
    assert len(gpu_pcie) == expected_len
    assert sum(gpu_pcie) == expected_sum
    bash.run.assert_called_with(["rocm-smi", "-b"], capture_output=True)

@pytest.mark.parametrize(
    "test_input,expected_sum,expected_len",
    [(b"\n\n======================= ROCm System Management Interface =======================\n\
        =============================== Current voltage ================================\n\
        GPU[0]\t\t: Voltage (mV): 737\n\
        GPU[1]\t\t: Voltage (mV): 737\n\
        GPU[2]\t\t: Voltage (mV): 737\n\
        GPU[3]\t\t: Voltage (mV): 737\n\
        GPU[4]\t\t: Voltage (mV): 737\n\
        GPU[5]\t\t: Voltage (mV): 737\n\
        GPU[6]\t\t: Voltage (mV): 737\n\
        GPU[7]\t\t: Voltage (mV): 737\n\
        ================================================================================\n\
        ============================= End of ROCm SMI Log ==============================\n",
        5896,
        8
    ),
    (b"\n\n======================= ROCm System Management Interface =======================\n\
        =============================== Current voltage ================================\n\
        GPU[0]\t\t: Voltage (mV): 200000\n\
        GPU[1]\t\t: Voltage (mV): 0\n\
        GPU[2]\t\t: Voltage (mV): 1000\n\
        GPU[3]\t\t: Voltage (mV): 200000\n\
        GPU[4]\t\t: Voltage (mV): 200000\n\
        GPU[5]\t\t: Voltage (mV): 200000\n\
        GPU[6]\t\t: Voltage (mV): 200000\n\
        GPU[7]\t\t: Voltage (mV): 50000\n\
        ================================================================================\n\
        ============================= End of ROCm SMI Log ==============================\n",
        1051000,
        8
    ),
    (b"\n\n======================= ROCm System Management Interface =======================\n\
        =============================== Current voltage ================================\n\
        ================================================================================\n\
        ============================= End of ROCm SMI Log ==============================\n",
        0,
        0
    ),
    (b"rocm-smi: command not found", 0, 0),
    (b"-sh: rocm-smi: not found", 0, 0)
    ])
def test_get_gpu_voltage(test_input, expected_sum, expected_len):
    bash = MagicMock()
    bash.run().stdout = test_input
    gpus = amd.gpus
    gpu_volt = amd.get_gpu_voltage()
    assert len(gpu_volt) == expected_len
    assert sum(gpu_volt) == expected_sum
    bash.run.assert_called_with(["rocm-smi", "--showvoltage"], capture_output=True)
