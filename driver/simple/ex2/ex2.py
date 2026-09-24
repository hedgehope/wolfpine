from framework.device.clock import Clock
from framework.device.device import Device, DeviceMemory
from framework.device.reset import Reset
from framework.memory.memory import DRAM
from framework.memory.memory_map import AddressRange, MemoryMap
from framework.pe.rv.rv32 import RV32I
from framework.util.conversion import conv_to_bytes, conv_to_uint32

# Read in binary executable (sets 10 to location 0x512)
with open("main.bin", "rb") as file:
    data = file.read()

# Create DRAM
dram = DRAM(4096)

# Create memory map and set DRAM into here
mem_map = MemoryMap()
dram_range = AddressRange(0x0, 4096)
mem_map[dram_range] = dram

# Create device memory and write executable into this
dm = DeviceMemory(mem_map)
dm.write(0x0, data)

# Create CPU
cpu = RV32I(0x0, [dm], snoop=True)
# GCC assumed sp is initalised by the preamble, we don't have that here
# so therefore set it ourselves
cpu.getRegisterFile()["sp"].write(conv_to_bytes(0x256))

# Set the output memory to be zero, this is because the compiler has
# generated an sb, so otherwise there could be garbage
dram.write(0x512, conv_to_bytes(0))

# Create a clock
clock = Clock([cpu])

# Create a reset
reset = Reset([cpu])

# Create a device
device = Device(dm, [clock], [reset])

# Reset the device and run the clock for 100 iterations
device.reset()
device.run(10)

# Now check the result at location 0x512
rval = dram.read(0x512, 4)
assert conv_to_uint32(rval) == 10

print("Completed successfully")
