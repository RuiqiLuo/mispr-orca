"""Define the bond dissociation energy workflow, using an ML model instead of DFT.

Mirrors ``mispr.orca.workflows.base.bde.get_bde``: the energies are predicted by
a user-provided model (see ``mispr.ml.utilities.model``) and stored in the same
``bde`` collection, with ``"method": "ml"``.
"""

import os

from fireworks import Firework, Workflow

from mispr.ml.utilities.model import validate_model_spec
from mispr.ml.firetasks.predict import PredictBDE
from mispr.ml.utilities.context import to_pymatgen, build_mol_context
from mispr.gaussian.utilities.metadata import get_mol_formula

__author__ = "Ruiqi Luo"
__status__ = "Development"
__date__ = "2026_10_29"
__version__ = "0.0.5"

WORKFLOW_KWARGS = Workflow.__init__.__code__.co_varnames


def get_bde_ml(
    mol_operation_type,
    mol,
    ref_charge=0,
    fragment_charges=None,
    bonds=None,
    open_rings=False,
    db=None,
    name="bde_calculation_ml",
    working_dir=None,
    model=None,
    **kwargs,
):
    """Define a workflow that predicts bond dissociation energies with an ML model.

    The model's ``predict(ctx)`` is called once per bond and fragment charge
    combination with a ``BondContext`` and must return the BDE in eV (see
    ``mispr.ml.firetasks.predict.PredictBDE``).

    Args:
        mol_operation_type (str): Must be ``"get_from_str"``; only SMILES are
            accepted.
        mol (str): SMILES of the molecule.
        ref_charge (int, optional): Charge of the molecule; defaults to 0.
        fragment_charges (list, optional): Additional fragment charges.
        bonds (list, optional): Bonds to break; defaults to all bonds.
        open_rings (bool, optional): Whether to open ring bonds instead of
            skipping them; defaults to ``False``.
        db (str or dict, optional): Database credentials.
        name (str, optional): Name of the workflow; defaults to
            "bde_calculation_ml".
        working_dir (str, optional): Working directory; defaults to the current
            working directory.
        model (dict): ``{"entry": "package.module:ClassName", "kwargs": {...}}``.
        kwargs (keyword arguments): Additional kwargs passed to ``PredictBDE``
            and ``Workflow`` (e.g. ``tag``); others are ignored.

    Returns:
        Workflow

    """
    if mol_operation_type != "get_from_str":
        raise NotImplementedError("ML workflows currently accept SMILES only")
    if model is None:
        raise ValueError("a model must be provided")
    validate_model_spec(model)
    working_dir = working_dir or os.getcwd()

    mol_ctx = build_mol_context(mol, charge=ref_charge)
    label = get_mol_formula(to_pymatgen(mol_ctx))

    fw = Firework(
        PredictBDE(
            mol=mol,
            model=model,
            ref_charge=ref_charge,
            fragment_charges=fragment_charges,
            bonds=bonds,
            open_rings=open_rings,
            db=db,
            **{
                i: j
                for i, j in kwargs.items()
                if i in PredictBDE.required_params + PredictBDE.optional_params
            },
        ),
        name="{}-{}".format(label, "bde_prediction"),
        spec={
            "tag": kwargs.get("tag", "unknown"),
            "_launch_dir": os.path.join(working_dir, label),
        },
    )

    return Workflow(
        [fw],
        name="{}_{}".format(label, name),
        **{i: j for i, j in kwargs.items() if i in WORKFLOW_KWARGS},
    )
