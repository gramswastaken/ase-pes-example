import gc
from copy import deepcopy
from pathlib import Path

import ase.build
import numpy as np
import torch
from ase import Atom, Atoms
from ase.constraints import FixAtoms, FixedLine
from ase.filters import FrechetCellFilter
from ase.optimize import FIRE2, LBFGS
from upet.ase import UPETCalculator


def _default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def build_slab(
    unit: Atoms,
    basis,
    superlattice_dims: tuple[int, int, int],
    vacuum: float = 20.0,
) -> Atoms:
    surface_cell = ase.build.cut(unit, a=basis[0], b=basis[1], c=basis[2])
    slab = surface_cell.repeat(superlattice_dims)
    slab.center(vacuum=vacuum, axis=2)
    slab.set_pbc((True, True, True))
    return slab


def _make_probe_system(
    slab: Atoms,
    probe_symbol: str,
    probe_coords: tuple[float, float, float],
) -> Atoms:
    probe_sys = deepcopy(slab)
    probe_sys.append(Atom(symbol=probe_symbol, position=probe_coords))
    constraints = list(probe_sys.constraints)
    constraints.append(FixedLine(indices=[-1], direction=(0, 0, 1)))
    probe_sys.set_constraint(constraints)
    return probe_sys


def compute_pes(
    slab: Atoms,
    probe_symbol: str,
    probe_height: float,
    fixed_z_cutoff_A: float,
    probe_spacing: tuple[float, float],
    extra_overshoot: tuple[float, float],
    checkpoint_force_eVpA: float,
    thr_force_eVpA: float,
    *,
    device: str | None = None,
    output_root: str | Path = ".",
) -> None:
    if device is None:
        device = _default_device()

    root = Path(output_root)
    fire_traj_dir = root / "traj-files" / "abc-fire"
    bfgs_traj_dir = root / "traj-files" / "bfgs"
    fire_restart_dir = root / "restart-files" / "abc-fire"
    bfgs_restart_dir = root / "restart-files" / "bfgs"
    fire_log_dir = root / "log-files" / "abc-fire"
    bfgs_log_dir = root / "log-files" / "bfgs"
    for directory in (
        fire_traj_dir,
        bfgs_traj_dir,
        fire_restart_dir,
        bfgs_restart_dir,
        fire_log_dir,
        bfgs_log_dir,
    ):
        directory.mkdir(exist_ok=True, parents=True)

    slab = slab.copy()
    pos = slab.get_positions()[:, 2]
    minz = float(np.min(pos))
    maxz = float(np.max(pos))
    z_fixed = minz + fixed_z_cutoff_A
    fixed_idxs = [i for i, p in enumerate(pos) if p < z_fixed]
    slab.set_constraint(FixAtoms(indices=fixed_idxs))

    x, y, _ = slab.cell.lengths()
    xrange = int(np.ceil((x + extra_overshoot[0]) / probe_spacing[0]))
    yrange = int(np.ceil((y + extra_overshoot[1]) / probe_spacing[1]))

    probe_systems: list[Atoms] = []
    for xi in range(xrange):
        xp = probe_spacing[0] * xi
        for yi in range(yrange):
            yp = probe_spacing[1] * yi
            probe_systems.append(
                _make_probe_system(
                    slab=slab,
                    probe_symbol=probe_symbol,
                    probe_coords=(xp, yp, maxz + probe_height),
                )
            )

    for i, probe_system in enumerate(probe_systems):
        probe_system.calc = UPETCalculator(
            model="pet-mad-m", version="1.6.0", device=device
        )
        opt_fire = FIRE2(
            probe_system,
            trajectory=str(fire_traj_dir / f"{i:03d}.traj"),
            restart=str(fire_restart_dir / f"{i:03d}.json"),
            logfile=str(fire_log_dir / f"log-{i:03d}.log"),
        )
        opt_fire.run(fmax=checkpoint_force_eVpA)

        opt_bfgs = LBFGS(
            probe_system,
            trajectory=str(bfgs_traj_dir / f"{i:03d}.traj"),
            restart=str(bfgs_restart_dir / f"{i:03d}.json"),
            logfile=str(bfgs_log_dir / f"log-{i:03d}.log"),
        )
        opt_bfgs.run(fmax=thr_force_eVpA)
        probe_system.get_potential_energy()
        # Delete and force garbage collection to prevent running out of memory.
        # The gc will do it normally but this forces one to unload before the next loads.
        # My 3080 has 10GB vram, this will eat like 7GB.
        probe_system.calc = None
        del probe_system
        gc.collect()


def optimize_bulk(
    bulk: Atoms,
    *,
    device: str | None = None,
    output_root: str | Path = ".",
) -> Atoms:
    if device is None:
        device = _default_device()
    root = Path(output_root)
    root.mkdir(exist_ok=True, parents=True)
    work_sys = deepcopy(bulk)
    work_sys.calc = UPETCalculator(model="pet-mad-m", version="1.6.0", device=device)
    filtered = FrechetCellFilter(work_sys)
    opt = LBFGS(
        filtered,
        trajectory=str(root / "bulk-opt.traj"),
        logfile=str(root / "bfgs.log"),
    )
    opt.run(fmax=0.001, steps=300)
    work_sys.calc = None
    return work_sys


def main() -> None:
    bulk = ase.build.bulk("GaAs", "zincblende", cubic=True, a=5.65)
    opt_bulk = optimize_bulk(bulk)
    slab = build_slab(
        opt_bulk,
        superlattice_dims=(2, 4, 6),
        basis=((0.5, 0.5, 0), (-0.5, 0.5, 0), (0, 0, 1)),
    )
    compute_pes(slab, "Ga", 5, 7, (1, 1), (2, 2), 0.1, 0.01)
