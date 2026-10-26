"""HTIF: Spike's host-target interface, on the ELF's ``tohost``/``fromhost`` words.

The block has no window of its own. ``tohost`` and ``fromhost`` are ordinary
64-bit words inside whatever memory the program was linked into; on load the
HTIF looks them up in the symbol table and asks the SoC memory to call it back
after every store that touches ``tohost`` (``NaiadMemory.add_write_intercept``).
The store lands first, so the word reads back like plain memory.

A ``tohost`` value is ``device << 56 | cmd << 48 | payload``, as in Spike:

* device 0, LSB set: exit with code ``value >> 1``;
* device 0, cmd 0, LSB clear: ``payload`` points at a ``magic_mem`` block of
  eight 64-bit words ``[syscall, a0, a1, ...]``; ``SYS_write`` (64) and
  ``SYS_exit`` (93) are proxied, the return value goes back in word 0;
* device 1, cmd 1: console putchar of ``payload & 0xff``.

Every handled command clears ``tohost``. A proxied syscall is also acked
through ``fromhost`` (non-zero), so software waits on ``fromhost`` and then
clears it; a console write is not acked, exactly as Spike's ``bcd`` device
does not, so software waits only for ``tohost`` to clear before the next one.

With several harts there is still one ``tohost``, and the **first exit command
from any hart ends the run** (as on Spike): :meth:`NaiadSoC.halt` stops every
core in the cycle the store lands. Each command is handled inside the store
that rings it, so on Naiad two harts can never overwrite each other's command;
the platform library still sends each command under a lock, because Spike
polls ``tohost`` and would lose one. Console bytes therefore interleave per
character at worst, and ``naiad_printf`` / ``naiad_puts`` hold a console lock
for a whole string. By the generated crt0's convention hart 0's return from
``main`` is the exit; other harts that return park.

The doorbell is the store that covers ``tohost``'s *low* byte. An RV64 core
writes the whole word with one ``sd``; an RV32 core writes the high word first
and the low word last (``naiad_platform.h`` does this), and a lone ``sw`` to the
low word is enough when the high word is zero, as it is for an exit.
"""

import sys

from framework.ip.base import IPBlock

SYS_WRITE = 64
SYS_EXIT = 93
_PAYLOAD_MASK = (1 << 48) - 1


class HTIFError(Exception):
    pass


class HTIF(IPBlock):
    mapped = False
    PARAMS = frozenset({"tohost_symbol", "fromhost_symbol"})

    def __init__(self, spec):
        super().__init__(spec)
        self.tohost_symbol = spec.params.get("tohost_symbol", "tohost")
        self.fromhost_symbol = spec.params.get("fromhost_symbol", "fromhost")
        self.tohost = None
        self.fromhost = None
        #: Where console bytes go as they arrive; None to only capture them.
        self.console_stream = sys.stdout
        self.reset()

    def reset(self):
        self.console = bytearray()
        self.exit_code = None

    @property
    def exited(self):
        return self.exit_code is not None

    def on_load(self, image):
        try:
            self.tohost = image.symbol(self.tohost_symbol)
            self.fromhost = image.symbol(self.fromhost_symbol)
        except KeyError as e:
            raise HTIFError(
                f"{e.args[0]}. The HTIF needs '{self.tohost_symbol}' and '{self.fromhost_symbol}'; "
                f"link with the generated crt0.S, which defines them."
            ) from None
        self.soc.memory.add_write_intercept(self.tohost, 8, self._on_tohost_write)

    # -- the doorbell ------------------------------------------------------

    def _read64(self, addr):
        return int.from_bytes(self.soc.memory.read(addr, 8), "little")

    def _write64(self, addr, value):
        self.soc.memory.write_raw(addr, (value & ((1 << 64) - 1)).to_bytes(8, "little"))

    def _on_tohost_write(self, addr, size):
        if not addr <= self.tohost < addr + size:
            return
        value = self._read64(self.tohost)
        if value == 0:
            return
        device, cmd, payload = value >> 56, (value >> 48) & 0xFF, value & _PAYLOAD_MASK
        response = None
        if device == 0 and value & 1:
            self._exit(value >> 1)
        elif device == 0 and cmd == 0:
            response = self._syscall(payload)
        elif device == 1 and cmd == 1:
            self._putchar(payload & 0xFF)
        else:
            raise HTIFError(
                f"unsupported HTIF command {hex(value)} (device {device}, cmd {cmd})"
            )
        self._write64(self.tohost, 0)
        if response is not None and not self.exited:
            self._write64(self.fromhost, response)

    def _exit(self, code):
        self.exit_code = code
        self.soc.halt()

    def _putchar(self, ch):
        self.console.append(ch)
        stream = self.console_stream
        if stream is not None:
            stream.write(chr(ch))
            if ch == 0x0A:
                stream.flush()

    def _syscall(self, magic_mem):
        args = [self._read64(magic_mem + 8 * i) for i in range(4)]
        num = args[0]
        if num == SYS_EXIT:
            self._exit(args[1])
            return 0
        if num == SYS_WRITE:
            fd, buf, length = args[1], args[2], args[3]
            if fd not in (1, 2):
                raise HTIFError(f"SYS_write to fd {fd}: only stdout/stderr are proxied")
            for ch in self.soc.memory.read(buf, length):
                self._putchar(ch)
            self._write64(magic_mem, length)
            return 1
        raise HTIFError(f"unsupported proxied syscall {num}")

    def counters(self):
        return {"chars": len(self.console)}

    # Never mapped, so never dispatched to; the words live in program memory.
    def read(self, addr, size):
        raise IndexError(f"{self.name}: the HTIF has no window of its own")

    def write(self, addr, value, size=None):
        raise IndexError(f"{self.name}: the HTIF has no window of its own")
