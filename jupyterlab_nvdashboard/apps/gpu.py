from bokeh.plotting import figure, ColumnDataSource
from bokeh.models import DataRange1d, NumeralTickFormatter, BasicTicker
from bokeh.layouts import column
from bokeh.models.mappers import LinearColorMapper
from bokeh.palettes import all_palettes
import time
import subprocess
import logging
import re
from typing import List
from statistics import mean

from jupyterlab_nvdashboard.utils import format_bytes


logging.basicConfig(
    filename="amd_gpu.log",
    level=logging.ERROR,
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





KB = 1e3
MB = KB * KB
GB = MB * KB
amd = AmdGpuProperties()


ngpus = amd.get_gpu_count()
gpu_handles = [i for i in range(ngpus)]

def gpu(doc):
    fig = figure(title="GPU Utilization", sizing_mode="stretch_both", x_range=[0, 100])

    def get_utilization():
        return amd.get_gpu_utilization()

    gpu = get_utilization()
    y = list(range(len(gpu)))
    source = ColumnDataSource({"right": y, "gpu": gpu})
    mapper = LinearColorMapper(palette=all_palettes["RdYlBu"][4], low=0, high=100)

    fig.hbar(
        source=source,
        y="right",
        right="gpu",
        height=0.8,
        color={"field": "gpu", "transform": mapper},
    )

    fig.toolbar_location = None

    doc.title = "GPU Utilization [%]"
    doc.add_root(fig)

    def cb():
        source.data.update({"gpu": get_utilization()})

    doc.add_periodic_callback(cb, 1000)


def gpu_mem(doc):
    fig = figure(title="GPU Memory Utilization", sizing_mode="stretch_both", x_range=[0, 100])

    def get_utilization():
        return amd.get_gpu_utilization()

    gpu = get_utilization()
    y = list(range(len(gpu)))
    source = ColumnDataSource({"right": y, "memory": gpu})
    mapper = LinearColorMapper(palette=all_palettes["RdYlBu"][4], low=0, high=100)

    fig.hbar(
        source=source,
        y="right",
        right="memory",
        height=0.8,
        color={"field": "memory", "transform": mapper},
    )

    fig.toolbar_location = None

    doc.title = "GPU Memory Utilization [%]"
    doc.add_root(fig)

    def cb():
        source.data.update({"memory": get_utilization()})

    doc.add_periodic_callback(cb, 1000)

def gpu_clock_frequency(doc):
   
    fig = figure(title="GPU Clock Frequency [Mhz]", sizing_mode="stretch_both", y_range=[0, 2200])

    frequency = amd.get_gpu_clock_freq()
    print(frequency)
    left = list(range(len(frequency)))
    right = [l + 0.8 for l in left]

    source = ColumnDataSource({"left": left, "right": right, "frequency": frequency})
    mapper = LinearColorMapper(palette=all_palettes["RdYlBu"][4], low=0, high=4000)

    fig.quad(
        source=source,
        left="left",
        right="right",
        bottom=0,
        top="frequency",
        color={"field": "frequency", "transform": mapper},
    )

    doc.title = "GPU Clock Frequency"
    doc.add_root(fig)

    def cb():
        source.data.update({"frequency": frequency})

    doc.add_periodic_callback(cb, 1000)





def gpu_resource_timeline(doc):

    memory_list = amd.getGPUMemUse()
    gpu_mem_max = max(memory_list) * (1024 * 1024)
    gpu_mem_sum = sum(memory_list)

    # Shared X Range for all plots
    x_range = DataRange1d(follow="end", follow_interval=20000, range_padding=0)
    tools = "reset,xpan,xwheel_zoom"

    item_dict = {
        "time": [],
        "gpu-total": [],
        "memory-total": [],
        "rx-total": [],
        "tx-total": [],
    }
    for i in range(ngpus):
        item_dict["gpu-" + str(i)] = []
        item_dict["memory-" + str(i)] = []

    source = ColumnDataSource(item_dict)

    def _get_color(ind):
        color_list = [
            "blue",
            "red",
            "green",
            "black",
            "brown",
            "cyan",
            "orange",
            "pink",
            "purple",
            "gold",
        ]
        return color_list[ind % len(color_list)]

    memory_fig = figure(
        title="Memory Utilization (per Device) [B]",
        sizing_mode="stretch_both",
        x_axis_type="datetime",
        y_range=[0, gpu_mem_max],
        x_range=x_range,
        tools=tools,
    )
    for i in range(ngpus):
        memory_fig.line(
            source=source, x="time", y="memory-" + str(i), color=_get_color(i)
        )
    memory_fig.yaxis.formatter = NumeralTickFormatter(format="0.0 b")

    gpu_fig = figure(
        title="GPU Utilization (per Device) [%]",
        sizing_mode="stretch_both",
        x_axis_type="datetime",
        y_range=[0, 100],
        x_range=x_range,
        tools=tools,
    )
    for i in range(ngpus):
        gpu_fig.line(source=source, x="time", y="gpu-" + str(i), color=_get_color(i))

    tot_fig = figure(
        title="Total Utilization [%]",
        sizing_mode="stretch_both",
        x_axis_type="datetime",
        y_range=[0, 100],
        x_range=x_range,
        tools=tools,
    )
    tot_fig.line(
        source=source, x="time", y="gpu-total", color="blue", legend="Total-GPU"
    )
    tot_fig.line(
        source=source, x="time", y="memory-total", color="red", legend="Total-Memory"
    )
    tot_fig.legend.location = "top_left"

    pci_fig = figure(
        title="Total PCI Throughput [B/s]",
        sizing_mode="stretch_both",
        x_axis_type="datetime",
        x_range=x_range,
        tools=tools,
    )
    pci_fig.line(source=source, x="time", y="tx-total", color="blue", legend="TX")
    pci_fig.line(source=source, x="time", y="rx-total", color="red", legend="RX")
    pci_fig.yaxis.formatter = NumeralTickFormatter(format="0.0 b")
    pci_fig.legend.location = "top_left"

    doc.title = "Resource Timeline"
    doc.add_root(
        column(gpu_fig, memory_fig, tot_fig, pci_fig, sizing_mode="stretch_both")
    )

    last_time = time.time()

    def cb():
        nonlocal last_time
        now = time.time()
        src_dict = {"time": [now * 1000]}
        gpu_tot = 0
        mem_tot = 0
        tx_tot = 0
        rx_tot = 0
        for i in range(ngpus):
            gpu = amd.get_gpu_utilization()
            mem = amd.get_gpu_mem_use()
            tx = 0
            rx = 0
            gpu_tot += gpu
            mem_tot += mem / (1024 * 1024)
            rx_tot += rx
            tx_tot += tx
            src_dict["gpu-" + str(i)] = [gpu]
            src_dict["memory-" + str(i)] = [mem]
        src_dict["gpu-total"] = [gpu_tot / ngpus]
        src_dict["memory-total"] = [(mem_tot / gpu_mem_sum) * 100]
        src_dict["tx-total"] = [tx_tot]
        src_dict["rx-total"] = [rx_tot]

        source.stream(src_dict, 1000)

        last_time = now

    doc.add_periodic_callback(cb, 200)
