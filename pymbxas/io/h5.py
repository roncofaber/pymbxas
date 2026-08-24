#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HDF5 persistence primitives for pymbxas.
"""

import json
import logging

import h5py
import numpy as np
from ase.io import jsonio

logger = logging.getLogger(__name__)

#%%

SCHEMA_VERSION = 1

KIND_CALCULATION = "calculation"
KIND_SPECTRA     = "spectra"
KIND_SPECTRAS    = "spectras"

_COMPRESS_MIN_BYTES = 65536
_GZIP_LEVEL         = 4


def write_array(group, key, value):
    arr = np.ascontiguousarray(value)
    if arr.nbytes >= _COMPRESS_MIN_BYTES:
        group.create_dataset(key, data=arr, compression="gzip",
                             compression_opts=_GZIP_LEVEL)
    else:
        group.create_dataset(key, data=arr)
    return


def write_str(group, key, text):
    group.create_dataset(key, data=text)
    return


def read_str(group, key):
    raw = group[key][()]
    return raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)


def write_text(group, key, text):
    buf = np.frombuffer((text or "").encode("utf-8"), dtype=np.uint8)
    write_array(group, key, buf)
    return


def read_text(group, key):
    return group[key][()].tobytes().decode("utf-8")


def write_json(group, key, obj):
    write_str(group, key, json.dumps(obj))
    return


def read_json(group, key):
    return json.loads(read_str(group, key))


def write_structure(group, key, atoms):
    write_str(group, key, jsonio.encode(atoms))
    return


def read_structure(group, key):
    return jsonio.decode(read_str(group, key))


def read_attr_str(obj, key):
    raw = obj.attrs[key]
    return raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)


def stamp(f, kind):
    import pymbxas
    f.attrs["kind"]            = kind
    f.attrs["schema_version"]  = SCHEMA_VERSION
    f.attrs["pymbxas_version"] = pymbxas.__version__
    return


def check_schema(f, expected_kind):
    if "schema_version" not in f.attrs:
        raise ValueError("{} is not a pymbxas HDF5 file: no schema_version attribute".format(
            f.filename))

    version = int(f.attrs["schema_version"])
    if version > SCHEMA_VERSION:
        raise ValueError("{} has schema version {}, this pymbxas reads up to {}".format(
            f.filename, version, SCHEMA_VERSION))

    kind = read_attr_str(f, "kind")
    if kind != expected_kind:
        raise ValueError("{} holds a '{}' payload, expected '{}'".format(
            f.filename, kind, expected_kind))
    return


def open_read(path, expected_kind):
    path = str(path)

    if path.endswith(".pkl"):
        raise ValueError(
            "pymbxas 0.6.0 removed pickle support, so {} cannot be loaded. "
            "Re-run the calculation to produce an HDF5 file.".format(path))

    if not h5py.is_hdf5(path):
        raise ValueError("{} is not an HDF5 file".format(path))

    f = h5py.File(path, "r")
    try:
        check_schema(f, expected_kind)
    except Exception:
        f.close()
        raise
    logger.debug("Opened %s HDF5 file for reading: %s", expected_kind, path)
    return f


def read_lazy_field(source, name):
    path, key = source
    with h5py.File(path, "r") as f:
        scf = f[key]["scf"]
        if name not in scf:
            raise AttributeError(name)
        return scf[name][()]


def create(path, kind):
    f = h5py.File(path, "w")
    stamp(f, kind)
    logger.debug("Created %s HDF5 file: %s", kind, path)
    return f


def append(path):
    return h5py.File(path, "a")


def open_plain(path, mode="r"):
    return h5py.File(path, mode)


def write_snapshot(group, data):
    data = data.to_cpu()

    write_str(group, "mol", data.mol.dumps())

    scf = group.create_group("scf")
    scf.create_dataset("e_tot", data=float(data.e_tot))
    write_array(scf, "nelec", np.asarray(data.nelec, dtype=np.int64))
    for name in ("mo_coeff", "mo_occ", "mo_energy"):
        write_array(scf, name, np.asarray(getattr(data, name)))

    mo_coeff_del = getattr(data, "mo_coeff_del", None)
    if mo_coeff_del is not None:
        write_array(scf, "mo_coeff_del", np.asarray(mo_coeff_del))
    logger.debug("Wrote SCF snapshot to group %s", group.name)
    return


def read_snapshot(path, key, lazy=False):
    from pyscf import gto
    from pymbxas.io.data import pyscf_data

    with h5py.File(path, "r") as f:
        group = f[key]
        mol   = gto.loads(read_str(group, "mol"))
        scf   = group["scf"]
        e_tot = float(scf["e_tot"][()])
        nelec = tuple(int(x) for x in scf["nelec"][()])

        if lazy:
            mol.verbose = 0
            return pyscf_data.from_h5_source(mol, e_tot, nelec, str(path), key)

        arrays = {name: scf[name][()] for name in
                  ("mo_coeff", "mo_occ", "mo_energy") if name in scf}
        if "mo_coeff_del" in scf:
            arrays["mo_coeff_del"] = scf["mo_coeff_del"][()]

    mol.verbose = 0
    return pyscf_data.from_arrays(mol, e_tot, nelec, **arrays)
