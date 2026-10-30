"""Define the binding energy workflow, using an ML model instead of DFT.

Mirrors ``mispr.orca.workflows.base.binding_energy.get_binding_energies``: the
energy is predicted by a user-provided model (see ``mispr.ml.utilities.model``)
and stored in the same ``binding_energy`` collection, with ``"method": "ml"``.
"""

import os

from fireworks import Firework, Workflow

from mispr.ml.utilities.model import validate_model_spec
from mispr.ml.firetasks.predict import PredictBindingEnergy
from mispr.ml.utilities.context import to_pymatgen, build_pair_context
from mispr.gaussian.utilities.metadata import get_mol_formula

WORKFLOW_KWARGS = Workflow.__init__.__code__.co_varnames


def get_binding_energies_ml(
    mol_operation_type,
    mol,
    index,
    bond_order=1,
    db=None,
    name="binding_energy_calculation_ml",
    working_dir=None,
    model=None,
    **kwargs,
):
    """Define a workflow that predicts the binding energy of two molecules with an ML model.

    The model's ``predict(ctx)`` is called once with a ``PairContext`` and must
    return the binding energy in eV (see
    ``mispr.ml.firetasks.predict.PredictBindingEnergy``).

    Args:
        mol_operation_type (list): Two-element list; both must be
            ``"get_from_str"`` (only SMILES are accepted).
        mol (list): Two SMILES.
        index (list): Atom numbers at which the two molecules bind, one per
            molecule.
        bond_order (int, optional): Bond order of the link; defaults to 1.
        db (str or dict, optional): Database credentials.
        name (str, optional): Name of the workflow; defaults to
            "binding_energy_calculation_ml".
        working_dir (str, optional): Working directory; defaults to the current
            working directory.
        model (dict): ``{"entry": "package.module:ClassName", "kwargs": {...}}``.
        kwargs (keyword arguments): Additional kwargs passed to
            ``PredictBindingEnergy`` and ``Workflow`` (e.g. ``tag``); others are
            ignored.

    Returns:
        Workflow

    """
    if len(mol_operation_type) != 2 or any(
        i != "get_from_str" for i in mol_operation_type
    ):
        raise NotImplementedError("ML workflows currently accept SMILES only")
    if model is None:
        raise ValueError("a model must be provided")
    validate_model_spec(model)
    working_dir = working_dir or os.getcwd()

    pair = build_pair_context(mol[0], mol[1], index[0], index[1], bond_order)
    label_a = get_mol_formula(to_pymatgen(pair.mol_a))
    label_b = get_mol_formula(to_pymatgen(pair.mol_b))
    label = "_".join([label_a, label_b])

    fw = Firework(
        PredictBindingEnergy(
            mol=list(mol),
            index=list(index),
            model=model,
            bond_order=bond_order,
            db=db,
            **{
                i: j
                for i, j in kwargs.items()
                if i
                in PredictBindingEnergy.required_params
                + PredictBindingEnergy.optional_params
            },
        ),
        name="{}-{}".format(label, "binding_energy_prediction"),
        spec={
            "tag": kwargs.get("tag", "unknown"),
            "_launch_dir": os.path.join(working_dir, label),
        },
    )

    return Workflow(
        [fw],
        name="{}_{}".format(name, label),
        **{i: j for i, j in kwargs.items() if i in WORKFLOW_KWARGS},
    )
