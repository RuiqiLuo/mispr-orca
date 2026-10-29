"""Define the chemical contexts handed to user-provided ML models.

A context bundles everything a model might need to predict one property: the
RDKit graph, atom/bond tables, charge, multiplicity and an (unoptimized)
geometry. Atom numbering is always that of the (hydrogen-explicit) RDKit
molecule, which is verified to match the pymatgen molecule that the DFT
workflows operate on.
"""

import logging

from copy import deepcopy
from dataclasses import replace, dataclass

import numpy as np

try:
    from rdkit import Chem
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "RDKit is required by mispr.ml; install it with `pip install mispr[chem]`"
    ) from e

from pymatgen.io.babel import BabelMolAdaptor
from pymatgen.core.structure import Molecule
from pymatgen.analysis.graphs import MoleculeGraph
from pymatgen.analysis.local_env import OpenBabelNN

from mispr.gaussian.utilities.mol import process_mol, link_molecules
from mispr.gaussian.firetasks.geo_transformation import BreakMolecule

__author__ = "Ruiqi Luo"
__status__ = "Development"
__date__ = "2026_10_29"
__version__ = "0.0.5"

logger = logging.getLogger(__name__)


@dataclass
class MolContext:
    """One molecule (parent molecule, fragment or complex).

    ``rdmol`` is hydrogen-explicit and atom ``i`` of ``rdmol`` is atom ``i`` of
    ``atoms``/``coords``. For a fragment these are fragment-local numbers;
    ``parent_idx[i]`` is the number of the same atom in the parent molecule.

    ``charge`` and ``multiplicity`` are authoritative. The formal charges and
    radical electrons stored on a fragment's ``rdmol`` only reflect the
    electron assignment of the homolytic split and must not be read as the
    fragment's actual electronic state.

    Attributes:
        smiles (str): User input for a parent molecule; ``Chem.MolToSmiles``
            of ``rdmol`` for fragments and complexes.
        rdmol (Chem.Mol): Hydrogen-explicit RDKit molecule.
        atoms (list): ``(index, element)`` for every atom.
        bonds (list): ``(i, j, bond order, is aromatic, is in ring)`` with
            ``i < j``.
        charge (int): Total charge.
        multiplicity (int): Spin multiplicity.
        coords (np.ndarray): ``(N, 3)`` cartesian coordinates in Angstrom; for
            a parent molecule the initial, unoptimized structure produced by
            ``process_mol``.
        parent_idx (list, optional): For fragments, atom numbers in the parent.
    """

    smiles: str
    rdmol: Chem.Mol
    atoms: list
    bonds: list
    charge: int
    multiplicity: int
    coords: np.ndarray
    parent_idx: list = None


@dataclass
class BondContext:
    """One bond of a parent molecule and the fragments formed by breaking it (BDE); one fragment when a ring is opened."""

    mol: MolContext
    bond: tuple
    fragments: list


@dataclass
class PairContext:
    """Two molecules and the complex formed by linking them (binding energy); the complex lists all atoms of ``mol_a`` followed by all atoms of ``mol_b``."""

    mol_a: MolContext
    mol_b: MolContext
    site_a: int
    site_b: int
    bond_order: int
    complex: MolContext


@dataclass
class RedoxStepContext:
    """One oxidation/reduction step; ``end`` has the same molecular graph as ``start`` and differs only in charge and multiplicity."""

    start: MolContext
    end: MolContext
    process: str
    phase: str
    vertical: bool


def _multiplicity(elements, coords, charge):
    """Spin multiplicity pymatgen assigns for the given charge; same rule as the DFT workflows."""
    mol = Molecule(elements, coords)
    mol.set_charge_and_spin(charge)
    return mol.spin_multiplicity


def to_pymatgen(ctx):
    """Convert a ``MolContext`` to a pymatgen ``Molecule`` carrying its charge and multiplicity."""
    return Molecule(
        [el for _, el in ctx.atoms],
        ctx.coords,
        charge=ctx.charge,
        spin_multiplicity=ctx.multiplicity,
    )


def _atom_table(rdmol):
    return [(a.GetIdx(), a.GetSymbol()) for a in rdmol.GetAtoms()]


def _bond_table(rdmol):
    return [
        (
            *sorted((b.GetBeginAtomIdx(), b.GetEndAtomIdx())),
            b.GetBondTypeAsDouble(),
            b.GetIsAromatic(),
            b.IsInRing(),
        )
        for b in rdmol.GetBonds()
    ]


def _derived_context(rdmol, charge, coords, parent_idx=None):
    """MolContext for a fragment/complex: SMILES, atom/bond tables and multiplicity are derived from ``rdmol``."""
    atoms = _atom_table(rdmol)
    return MolContext(
        smiles=Chem.MolToSmiles(rdmol),
        rdmol=rdmol,
        atoms=atoms,
        bonds=_bond_table(rdmol),
        charge=charge,
        multiplicity=_multiplicity([el for _, el in atoms], coords, charge),
        coords=coords,
        parent_idx=parent_idx,
    )


def _smiles_to_pymatgen(smiles, charge):
    """Initial 3D structure of a SMILES string as a pymatgen molecule.

    ``process_mol("get_from_str", ..., str_type="smi")`` returns heavy atoms
    only, all at the origin, and does not read the charge from the SMILES, so
    hydrogens are added and a 3D structure is generated here with OpenBabel;
    ``process_mol("get_from_mol", ...)`` then sets charge and spin as in the DFT
    workflows.
    """
    adaptor = BabelMolAdaptor.from_str(smiles, "smi")
    adaptor.add_hydrogen()
    adaptor.make3d()
    return process_mol("get_from_mol", adaptor.pymatgen_mol, charge=charge)


def _build_mol_context(smiles, charge=None):
    """Build the parent ``MolContext`` and return it with the pymatgen molecule it was checked against."""
    rdmol = Chem.MolFromSmiles(smiles)
    if rdmol is None:
        raise ValueError(f"RDKit could not parse SMILES: {smiles}")
    rdmol = Chem.AddHs(rdmol)
    if charge is None:
        charge = Chem.GetFormalCharge(rdmol)
    pmg_mol = _smiles_to_pymatgen(smiles, charge)

    atoms = _atom_table(rdmol)
    if [el for _, el in atoms] != [str(s.specie) for s in pmg_mol]:
        raise ValueError(
            f"atom order of RDKit and pymatgen molecules differ for {smiles}; "
            "atom numbering cannot be guaranteed"
        )
    bonds = _bond_table(rdmol)
    rd_edges = sorted({(i, j) for i, j, *_ in bonds})
    graph = MoleculeGraph.with_local_env_strategy(pmg_mol, OpenBabelNN())
    pmg_edges = sorted({tuple(sorted((i, j))) for i, j, _ in graph.graph.edges})
    if rd_edges != pmg_edges:
        raise ValueError(
            f"bonds perceived by RDKit and by OpenBabel differ for {smiles}; "
            "atom numbering cannot be guaranteed"
        )

    ctx = MolContext(
        smiles=smiles,
        rdmol=rdmol,
        atoms=atoms,
        bonds=bonds,
        charge=pmg_mol.charge,
        multiplicity=pmg_mol.spin_multiplicity,
        coords=np.array(pmg_mol.cart_coords),
    )
    return ctx, pmg_mol


def build_mol_context(smiles, charge=None):
    """Build the context of a parent molecule from a SMILES string.

    Args:
        smiles (str): SMILES of the molecule.
        charge (int, optional): Total charge; if ``None``, the formal charge
            implied by the SMILES is used.

    Returns:
        MolContext

    Raises:
        ValueError: If the SMILES cannot be parsed, or if the atom order or the
            bonds of the RDKit molecule differ from those of the pymatgen
            molecule built from the SMILES.
    """
    return _build_mol_context(smiles, charge)[0]


def _split(mol_ctx, bond, ring):
    """Break ``bond`` homolytically; return ``(fragment mols, atom mappings)`` or ``None`` if sanitization fails."""
    i, j = bond
    rw = Chem.RWMol(mol_ctx.rdmol)
    if ring:
        Chem.Kekulize(rw, clearAromaticFlags=True)
    rw.RemoveBond(i, j)
    for idx in (i, j):
        atom = rw.GetAtomWithIdx(idx)
        atom.SetNoImplicit(True)
        atom.SetNumRadicalElectrons(atom.GetNumRadicalElectrons() + 1)
    try:
        Chem.SanitizeMol(rw)
    except Exception as e:
        logger.warning(f"could not sanitize fragments of bond {bond}; skipping: {e}")
        return None
    mapping = []
    frags = Chem.GetMolFrags(rw, asMols=True, fragsMolAtomMapping=mapping)
    return list(frags), [list(m) for m in mapping]


def build_bond_contexts(
    mol_ctx, ref_charge, fragment_charges=None, bonds=None, open_rings=False
):
    """Build one ``BondContext`` per bond and fragment charge combination.

    Charge combinations come from ``BreakMolecule._define_charges``, as in the
    DFT workflow. Ring bonds are skipped unless ``open_rings`` is set, in which
    case opening the ring yields a single fragment (all atoms) with charge
    ``ref_charge``; its coordinates are those of the parent (no ring-opening
    optimization). A bond whose fragments fail sanitization is skipped with a
    warning.

    The fragments' ``rdmol`` formal charges and radicals only reflect the
    homolytic split; ``MolContext.charge``/``multiplicity`` are authoritative.

    Args:
        mol_ctx (MolContext): Parent molecule.
        ref_charge (int): Charge of the parent molecule.
        fragment_charges (list, optional): Additional fragment charges.
        bonds (list, optional): Bonds to break, as atom index pairs; used in the
            given order and direction (the fragment containing ``bond[0]`` comes
            first). Defaults to all bonds of ``mol_ctx``.
        open_rings (bool, optional): Whether to open ring bonds.

    Returns:
        list: ``BondContext`` objects.
    """
    if bonds is None:
        bonds = sorted((i, j) for i, j, *_ in mol_ctx.bonds)
    # _define_charges extends the list it is given in place
    _, charge_pairs, _ = BreakMolecule._define_charges(
        ref_charge, list(fragment_charges or [])
    )

    contexts = []
    for bond in bonds:
        bond = tuple(bond)
        rd_bond = mol_ctx.rdmol.GetBondBetweenAtoms(*bond)
        if rd_bond is None:
            raise ValueError(f"atoms {bond[0]} and {bond[1]} are not bonded")
        ring = rd_bond.IsInRing()
        if ring and not open_rings:
            logger.info(
                "encountered a ring bond; should set open_ring to True to open the ring"
            )
            continue
        split = _split(mol_ctx, bond, ring)
        if split is None:
            continue
        frags, mapping = split
        if ring:
            pairs = [(ref_charge,)]
        else:
            if bond[0] not in mapping[0]:
                frags, mapping = frags[::-1], mapping[::-1]
            pairs = charge_pairs
        for charges in pairs:
            fragments = [
                _derived_context(
                    deepcopy(frag),
                    charge,
                    mol_ctx.coords[parent_idx],
                    parent_idx=parent_idx,
                )
                for frag, parent_idx, charge in zip(frags, mapping, charges)
            ]
            contexts.append(BondContext(mol=mol_ctx, bond=bond, fragments=fragments))
    return contexts


def build_pair_context(smiles_a, smiles_b, site_a, site_b, bond_order=1):
    """Build the context of two molecules and the complex formed by linking ``site_a`` of the first to ``site_b`` of the second.

    Geometry, atom order and total charge follow ``link_molecules`` (no atom is
    removed or changed, no bond is added to ``complex.rdmol``).

    Args:
        smiles_a (str): SMILES of the first molecule.
        smiles_b (str): SMILES of the second molecule.
        site_a (int): Atom number in the first molecule.
        site_b (int): Atom number in the second molecule.
        bond_order (int, optional): Bond order assumed for the link.

    Returns:
        PairContext
    """
    a, pmg_a = _build_mol_context(smiles_a)
    b, pmg_b = _build_mol_context(smiles_b)
    linked = link_molecules(pmg_a, site_a, pmg_b, site_b, bond_order)
    rdmol = Chem.CombineMols(a.rdmol, b.rdmol)
    cplx = _derived_context(rdmol, a.charge + b.charge, np.array(linked.cart_coords))
    return PairContext(
        mol_a=a,
        mol_b=b,
        site_a=site_a,
        site_b=site_b,
        bond_order=bond_order,
        complex=cplx,
    )


def build_redox_species(mol_ctx, ref_charge, electrons):
    """Return a copy of ``mol_ctx`` whose electron count changed by ``electrons`` relative to the reference state (charge ``ref_charge - electrons``; multiplicity by the usual rule)."""
    charge = ref_charge - electrons
    return replace(
        mol_ctx,
        rdmol=Chem.Mol(mol_ctx.rdmol),
        coords=mol_ctx.coords.copy(),
        charge=charge,
        multiplicity=_multiplicity(
            [el for _, el in mol_ctx.atoms], mol_ctx.coords, charge
        ),
    )


def build_redox_step_context(
    mol_ctx, ref_charge, e_start, e_end, process, phase, vertical
):
    """Build the ``RedoxStepContext`` of the step from electron change ``e_start`` to ``e_end`` (``0`` is the reference state; oxidation steps have negative values)."""
    return RedoxStepContext(
        start=build_redox_species(mol_ctx, ref_charge, e_start),
        end=build_redox_species(mol_ctx, ref_charge, e_end),
        process=process,
        phase=phase,
        vertical=vertical,
    )
