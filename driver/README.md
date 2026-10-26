# Driver for the simulator

The benefit of Python is that we can very easily write scripts that drive parts of the simulator, making it trivial to experiment with different parts. There are two groups of drivers provided here:

* [wormhole](https://github.com/hedgehope/wolfpine/tree/main/driver/wormhole) which is a simple Wormhole (one DRAM tile, one Tensix tile) and a range of code kernels that will run on this. The `server/` entry point runs the simulator behind UMD's simulation backend, so an unmodified tt-metal host program drives it over the wire — no tt-metal version is pinned.
* [simple](https://github.com/hedgehope/wolfpine/tree/main/driver/simple) which is a range of examples driving individual components, building up to codes running on the RV32IM CPU. Helped to develop and test some of the foundational building blocks.
* [naiad](https://github.com/hedgehope/wolfpine/tree/main/driver/naiad) which is a set of progressive, self-checking C programs for the Naiad RISC-V reference SoC (`framework/naiad`): console and exit codes, the CLINT timer, DMA, counters, an AME GEMM, and several harts. `make -C driver/naiad run` builds and runs them against their designs.
