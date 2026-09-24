from abc import ABC, abstractmethod

from framework.device.clock import Clockable
from framework.device.reset import Resetable
from framework.memory.memory import MemorySpace


class PEMemory(MemorySpace):
    def __init__(self, memory_map, safe=True, snoop_addresses=None):
        super().__init__(memory_map, safe, snoop_addresses)


class ProcessingElement(Clockable, Resetable, ABC):
    class PEStall:
        pass

    @abstractmethod
    def start(self):
        raise NotImplementedError()

    @abstractmethod
    def stop(self):
        raise NotImplementedError()

    @abstractmethod
    def getRegisterFile(self):
        raise NotImplementedError()
