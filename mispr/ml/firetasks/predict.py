"""Define firetasks that predict molecular properties with a user-provided ML model.

Each task builds the chemical context of its property (see
``mispr.ml.utilities.context``), loads the model (see
``mispr.ml.utilities.model``), calls ``model.predict(ctx)`` and stores a
document in the same collection and format as the corresponding DFT workflow,
with ``"method": "ml"`` added. No training is done here.
"""

import os
import json
import time
import logging
import datetime

from fireworks.core.firework import FWAction, FiretaskBase
from fireworks.utilities.fw_utilities import explicit_serialize
from fireworks.utilities.fw_serializers import DATETIME_HANDLER

from mispr import __version__ as mispr_version
from mispr.gaussian.firetasks import parse_outputs
from mispr.ml.utilities.model import load_model
from mispr.ml.utilities.context import (
    to_pymatgen,
    build_mol_context,
    build_pair_context,
    build_bond_contexts,
    build_redox_step_context,
)
from mispr.gaussian.utilities.files import bibtex_parser
from mispr.gaussian.utilities.metadata import get_chem_schema
from mispr.gaussian.utilities.db_utilities import get_db
from mispr.gaussian.firetasks.parse_outputs import FARAD, HARTREE_TO_EV, HARTREE_TO_KJ

__author__ = "Ruiqi Luo"
__status__ = "Development"
__date__ = "2026_10_29"
__version__ = "0.0.5"

logger = logging.getLogger(__name__)

# same indices as the DFT *toDB firetasks
INDEX_LIST = [
    ("formula_alphabetical", 1),
    ("smiles", 1),
    ("inchi", 1),
    ("chemsys", 1),
    ("functional", 1),
    ("basis", 1),
    ("tag", 1),
]


def _chem_fields(pmg_mol):
    schema = get_chem_schema(pmg_mol)
    return {
        "molecule": pmg_mol.as_dict(),
        "smiles": schema["smiles"],
        "inchi": schema["inchi"],
        "formula_alphabetical": schema["formula_alphabetical"],
        "chemsys": schema["chemsys"],
    }


def _base_doc(task, fw_spec, pmg_mol, model, start_time):
    """Fields shared by all ML documents."""
    doc = _chem_fields(pmg_mol)
    doc.update(
        {
            "tag": fw_spec["tag"],
            "state": "successful",
            "wall_time (s)": time.time() - start_time,
            "version": mispr_version,
            "last_updated": datetime.datetime.utcnow(),
            "method": "ml",
            "model": {
                "entry": task["model"]["entry"],
                "kwargs": task["model"].get("kwargs", {}),
                "metadata": getattr(model, "metadata", None),
            },
        }
    )
    return doc


def _finish(task, doc, collection, filename):
    """Merge additional fields and save the document as requested."""
    if task.get("additional_prop_doc_fields"):
        doc.update(task["additional_prop_doc_fields"])
    if task.get("save_to_db"):
        get_db(task.get("db")).insert_property(collection, doc, INDEX_LIST)
    if task.get("save_to_file"):
        with open(filename, "w") as f:
            f.write(json.dumps(doc, default=DATETIME_HANDLER))
    logger.info(f"{collection} ML prediction complete")


@explicit_serialize
class PredictBDE(FiretaskBase):
    """Predict bond dissociation energies with an ML model and enter them into the database.

    The model's ``predict(ctx)`` receives a ``BondContext`` (one bond and one
    fragment charge combination) and returns a ``float``: the BDE in eV,
    i.e. the sum of the fragment enthalpies minus the enthalpy of the parent
    molecule (the DFT workflow uses enthalpies). A call that raises is logged
    and skipped; if no value is obtained, nothing is saved.

    Args:
        mol (str): SMILES of the molecule.
        model (dict): ``{"entry": "package.module:ClassName", "kwargs": {...}}``.
        ref_charge (int, optional): Charge of the molecule; defaults to 0.
        fragment_charges (list, optional): Additional fragment charges.
        bonds (list, optional): Bonds to break; defaults to all bonds.
        open_rings (bool, optional): Whether to open ring bonds.
        db (str or dict, optional): Database credentials.
        save_to_db (bool, optional): Whether to insert the document into the
            ``bde`` collection.
        save_to_file (bool, optional): Whether to save it to "bde.json".
        additional_prop_doc_fields (dict, optional): Fields added to the document.
    """

    required_params = ["mol", "model"]
    optional_params = [
        "ref_charge",
        "fragment_charges",
        "bonds",
        "open_rings",
        "db",
        "save_to_db",
        "save_to_file",
        "additional_prop_doc_fields",
    ]

    def run_task(self, fw_spec):
        start_time = time.time()
        ref_charge = self.get("ref_charge", 0)
        # bonds are lists once they have gone through the FireWorks spec
        bonds = [tuple(b) for b in self["bonds"]] if self.get("bonds") else None
        mol_ctx = build_mol_context(self["mol"], charge=ref_charge)
        contexts = build_bond_contexts(
            mol_ctx,
            ref_charge,
            self.get("fragment_charges"),
            bonds,
            self.get("open_rings", False),
        )
        model = load_model(self["model"])

        fragments = {}
        frag_ids = {}
        bde_results = {}
        for ctx in contexts:
            frag_docs = []
            for frag in ctx.fragments:
                schema = get_chem_schema(to_pymatgen(frag))
                frag_docs.append((frag, schema))
            label = "+".join(
                f"{schema['formula_alphabetical']}(q={frag.charge})"
                for frag, schema in frag_docs
            )
            results = bde_results.setdefault(str(list(ctx.bond)), {})
            try:
                value = float(model.predict(ctx))
            except Exception as e:
                logger.warning(f"Prediction failed for bond {ctx.bond}, {label}: {e}")
                continue
            results[label] = value
            for frag, schema in frag_docs:
                key = (frag.smiles, frag.charge)
                if key not in frag_ids:
                    frag_ids[key] = f"frag_{len(frag_ids)}"
                    fragments[frag_ids[key]] = {
                        "fragment": to_pymatgen(frag).as_dict(),
                        "smiles": schema["smiles"],
                        "inchi": schema["inchi"],
                        "formula_alphabetical": schema["formula_alphabetical"],
                        "chemsys": schema["chemsys"],
                    }

        if not any(bde_results.values()):
            logger.warning("No BDE was predicted; nothing saved")
            return FWAction()

        doc = _base_doc(self, fw_spec, to_pymatgen(mol_ctx), model, start_time)
        doc.update({"fragments": fragments, "bde_eV": bde_results})
        _finish(self, doc, "bde", "bde.json")
        return FWAction()


@explicit_serialize
class PredictESP(FiretaskBase):
    """Predict atomic partial charges with an ML model and enter them into the database.

    The model's ``predict(ctx)`` receives a ``MolContext`` and returns a
    ``list[float]`` with one charge per atom, in atom-number order.

    Args:
        mol (str): SMILES of the molecule.
        model (dict): ``{"entry": "package.module:ClassName", "kwargs": {...}}``.
        db (str or dict, optional): Database credentials.
        save_to_db (bool, optional): Whether to insert the document into the
            ``esp`` collection.
        save_to_file (bool, optional): Whether to save it to "esp.json".
        additional_prop_doc_fields (dict, optional): Fields added to the document.
    """

    required_params = ["mol", "model"]
    optional_params = [
        "db",
        "save_to_db",
        "save_to_file",
        "additional_prop_doc_fields",
    ]

    def run_task(self, fw_spec):
        start_time = time.time()
        mol_ctx = build_mol_context(self["mol"])
        model = load_model(self["model"])
        charges = [float(q) for q in model.predict(mol_ctx)]
        if len(charges) != len(mol_ctx.atoms):
            raise ValueError(
                f"model returned {len(charges)} charges for "
                f"{len(mol_ctx.atoms)} atoms"
            )

        doc = _base_doc(self, fw_spec, to_pymatgen(mol_ctx), model, start_time)
        doc.update(
            {
                "esp": charges,
                "esp_by_atom": {
                    f"{el}{i}": q for (i, el), q in zip(mol_ctx.atoms, charges)
                },
            }
        )
        _finish(self, doc, "esp", "esp.json")
        return FWAction()


@explicit_serialize
class PredictBindingEnergy(FiretaskBase):
    """Predict a binding energy with an ML model and enter it into the database.

    The model's ``predict(ctx)`` receives a ``PairContext`` and returns a
    ``float``: the binding energy in eV, i.e. E(complex) - E(a) - E(b) (the DFT
    workflow uses electronic energies).

    Args:
        mol (list): Two SMILES.
        index (list): Atom numbers at which the molecules bind, one per molecule.
        model (dict): ``{"entry": "package.module:ClassName", "kwargs": {...}}``.
        bond_order (int, optional): Bond order of the link; defaults to 1.
        db (str or dict, optional): Database credentials.
        save_to_db (bool, optional): Whether to insert the document into the
            ``binding_energy`` collection.
        save_to_file (bool, optional): Whether to save it to "binding_energy.json".
        additional_prop_doc_fields (dict, optional): Fields added to the document.
    """

    required_params = ["mol", "index", "model"]
    optional_params = [
        "bond_order",
        "db",
        "save_to_db",
        "save_to_file",
        "additional_prop_doc_fields",
    ]

    def run_task(self, fw_spec):
        start_time = time.time()
        site_a, site_b = self["index"]
        ctx = build_pair_context(
            self["mol"][0],
            self["mol"][1],
            site_a,
            site_b,
            self.get("bond_order", 1),
        )
        model = load_model(self["model"])
        value = float(model.predict(ctx))

        doc = _base_doc(self, fw_spec, to_pymatgen(ctx.complex), model, start_time)
        doc["be_eV"] = {
            "sites": [site_a, len(ctx.mol_a.atoms) + site_b],
            "atoms": [ctx.mol_a.atoms[site_a][1], ctx.mol_b.atoms[site_b][1]],
            "value": value,
        }
        _finish(self, doc, "binding_energy", "binding_energy.json")
        return FWAction()


@explicit_serialize
class PredictIPEA(FiretaskBase):
    """Predict ionization potentials / electron affinities (redox potentials) with an ML model and enter them into the database.

    The model's ``predict(ctx)`` receives a ``RedoxStepContext`` and returns a
    ``float``: G(end) - G(start) in eV for the species themselves, *without*
    the electron term. The electron Gibbs free energy ``gibbs_elec`` is added
    here, and the potentials are derived as in ``IPEAtoDB``.

    Steps follow the non-PCET branch of ``IPEAtoDB``: for each state
    (cation: oxidation, electron change ``-n``; anion: reduction, ``+n``) and
    phase, the total step ``0 -> sign*n``, plus, unless ``single_step``, the
    steps ``sign*i -> sign*(i+1)``.

    Args:
        mol (str): SMILES of the molecule.
        ref_charge (int): Charge of the reference species.
        model (dict): ``{"entry": "package.module:ClassName", "kwargs": {...}}``.
        single_step (bool, optional): Only the total step; defaults to False.
        vertical (bool, optional): Whether vertical IP/EA are requested; passed
            to the model and stored in the document.
        num_electrons (int, optional): Number of electrons transferred; defaults to 1.
        states (list, optional): Defaults to ["cation", "anion"].
        phases (list, optional): Defaults to ["gas", "solution"].
        electrode_potentials (dict, optional): Additional reference electrodes;
            hydrogen, magnesium and lithium are always included.
        gibbs_elec (float, optional): Electron Gibbs free energy in Hartree.
        db (str or dict, optional): Database credentials.
        save_to_db (bool, optional): Whether to insert the document into the
            ``ip_ea`` collection.
        save_to_file (bool, optional): Whether to save it to "ip_ea.json".
        additional_prop_doc_fields (dict, optional): Fields added to the document.
    """

    required_params = ["mol", "ref_charge", "model"]
    optional_params = [
        "single_step",
        "vertical",
        "num_electrons",
        "states",
        "phases",
        "electrode_potentials",
        "gibbs_elec",
        "db",
        "save_to_db",
        "save_to_file",
        "additional_prop_doc_fields",
    ]

    def run_task(self, fw_spec):
        start_time = time.time()
        ref_charge = self["ref_charge"]
        single_step = self.get("single_step", False)
        vertical = self.get("vertical", False)
        num_electrons = self.get("num_electrons", 1)
        states = self.get("states") or ["cation", "anion"]
        phases = self.get("phases") or ["gas", "solution"]
        gibbs_elec = self.get("gibbs_elec", -0.001378786)

        data_dir = os.path.join(os.path.dirname(parse_outputs.__file__), "../data")
        ref_potentials = {
            "hydrogen": {
                "potential": 4.43,
                "ref": bibtex_parser("h_pot.bib", data_dir),
            },
            "magnesium": {
                "potential": 2.375,
                "ref": bibtex_parser("mg_pot.bib", data_dir),
            },
            "lithium": {
                "potential": 1.40,
                "ref": bibtex_parser("li_pot.bib", data_dir),
            },
        }
        electrode_potentials = {
            **ref_potentials,
            **(self.get("electrode_potentials") or {}),
        }

        mol_ctx = build_mol_context(self["mol"], charge=ref_charge)
        model = load_model(self["model"])

        def _name_(e1, e2):
            return f"{e1}e_{e2}e"

        def _gibbs_to_redox(s, g, e_count):
            return -s * g / ((1 / HARTREE_TO_KJ) * e_count * FARAD)

        gibbs_dict = {}
        redox_dict = {}
        for state in states:
            process = "oxidation" if state == "cation" else "reduction"
            sign = 1 if state == "anion" else -1
            gibbs_dict[process] = {}
            redox_dict[process] = {}
            steps = [(0, sign * num_electrons)]
            if not single_step:
                steps += [(sign * i, sign * (i + 1)) for i in range(num_electrons)]
            for phase in phases:
                gibbs_dict[process][phase] = {}
                redox_dict[process][phase] = {}
                for e_start, e_end in steps:
                    name = _name_(e_start, e_end)
                    if name in gibbs_dict[process][phase]:
                        continue
                    count = abs(e_end - e_start)
                    ctx = build_redox_step_context(
                        mol_ctx, ref_charge, e_start, e_end, process, phase, vertical
                    )
                    d_g_species = float(model.predict(ctx))
                    gibbs = d_g_species / HARTREE_TO_EV - sign * gibbs_elec * count
                    gibbs_dict[process][phase][name] = gibbs * HARTREE_TO_EV
                    redox = _gibbs_to_redox(sign, gibbs, count)
                    redox_dict[process][phase][name] = {"raw": redox}
                    for key, value in electrode_potentials.items():
                        redox_dict[process][phase][name][key] = (
                            redox - value["potential"]
                        )

        doc = _base_doc(self, fw_spec, to_pymatgen(mol_ctx), model, start_time)
        doc.update(
            {
                "num_electrons": num_electrons,
                "phase": phases,
                "steps": "single" if single_step else "multi",
                "vertical": vertical,
                "pcet": False,
                "gibbs_elec": gibbs_elec * HARTREE_TO_EV,
                "gibbs": gibbs_dict,
                "potentials": redox_dict,
                "standard_potentials": electrode_potentials,
            }
        )
        _finish(self, doc, "ip_ea", "ip_ea.json")
        return FWAction()
