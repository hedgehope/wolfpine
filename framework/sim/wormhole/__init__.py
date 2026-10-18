"""The Wormhole simulator device: the directory ``TT_METAL_SIMULATOR`` points at.

UMD launches ``run.sh`` from here, which starts the wire-bridge server
(``python -m framework.sim.wormhole``); ``soc_descriptor.yaml`` and
``cluster_descriptor.yaml`` are read by both the tt-metal host and ``coords``.
"""
