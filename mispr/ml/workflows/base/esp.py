"""Define the electrostatic partial charge workflow, using an ML model instead of DFT.

Mirrors ``mispr.orca.workflows.base.esp.get_esp_charges``: the charges are
predicted by a user-provided model (see ``mispr.ml.utilities.model``) and stored
in the same ``esp`` collection, with ``"method": "ml"``.
"""

import os

from fireworks import Firework, Workflow

from mispr.ml.utilities.model import validate_model_spec
from mispr.ml.firetasks.predict import PredictESP
from mispr.ml.utilities.context import to_pymatgen, build_mol_context
from mispr.gaussian.utilities.metadata import get_mol_formula

WORKFLOW_KWARGS = Workflow.__init__.__code__.co_varnames


def get_esp_charges_ml(
    mol_operation_type,
    mol,
    db=None,
    name="esp_charges_calculation_ml",
    working_dir=None,
    model=None,
    **kwargs,
):
    """Define a workflow that predicts atomic partial charges with an ML model.

    The model's ``predict(ctx)`` is called once with a ``MolContext`` and must
    return one charge per atom (see ``mispr.ml.firetasks.predict.PredictESP``).

    Args:
        mol_operation_type (str): Must be ``"get_from_str"``; only SMILES are
            accepted.
        mol (str): SMILES of the molecule.
        db (str or dict, optional): Database credentials.
        name (str, optional): Name of the workflow; defaults to
            "esp_charges_calculation_ml".
        working_dir (str, optional): Working directory; defaults to the current
            working directory.
        model (dict): ``{"entry": "package.module:ClassName", "kwargs": {...}}``.
        kwargs (keyword arguments): Additional kwargs passed to ``PredictESP``
            and ``Workflow`` (e.g. ``tag``); others are ignored.

    Returns:
        tuple:
            - Workflow
            - str: Label of the molecule (e.g. "H2O").

    """
    if mol_operation_type != "get_from_str":
        raise NotImplementedError("ML workflows currently accept SMILES only")
    if model is None:
        raise ValueError("a model must be provided")
    validate_model_spec(model)
    working_dir = working_dir or os.getcwd()

    mol_ctx = build_mol_context(mol)
    label = get_mol_formula(to_pymatgen(mol_ctx))

    fw = Firework(
        PredictESP(
            mol=mol,
            model=model,
            db=db,
            **{
                i: j
                for i, j in kwargs.items()
                if i in PredictESP.required_params + PredictESP.optional_params
            },
        ),
        name="{}-{}".format(label, "esp_prediction"),
        spec={
            "tag": kwargs.get("tag", "unknown"),
            "_launch_dir": os.path.join(working_dir, label),
        },
    )

    return (
        Workflow(
            [fw],
            name=name,
            **{i: j for i, j in kwargs.items() if i in WORKFLOW_KWARGS},
        ),
        label,
    )
