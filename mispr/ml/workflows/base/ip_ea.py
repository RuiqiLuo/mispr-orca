"""Define the redox potentials workflow, using an ML model instead of DFT.

Mirrors ``mispr.orca.workflows.base.ip_ea.get_ip_ea`` for direct electron
transfer (no PCET): the free energy differences are predicted by a
user-provided model (see ``mispr.ml.utilities.model``) and stored in the same
``ip_ea`` collection, with ``"method": "ml"``.
"""

import os

from fireworks import Firework, Workflow

from mispr.ml.utilities.model import validate_model_spec
from mispr.ml.firetasks.predict import PredictIPEA
from mispr.ml.utilities.context import to_pymatgen, build_mol_context
from mispr.gaussian.utilities.files import bibtex_parser
from mispr.gaussian.utilities.metadata import get_mol_formula

WORKFLOW_KWARGS = Workflow.__init__.__code__.co_varnames


def get_ip_ea_ml(
    mol_operation_type,
    mol,
    ref_charge,
    single_step=False,
    vertical=False,
    pcet=False,
    num_electrons=1,
    states=None,
    phases=None,
    electrode_potentials=None,
    gibbs_elec=-0.001378786,
    db=None,
    name="ip_ea_calculation_ml",
    working_dir=None,
    model=None,
    **kwargs,
):
    """Define a workflow that predicts ionization potentials and electron affinities with an ML model.

    The model's ``predict(ctx)`` is called once per step with a
    ``RedoxStepContext`` and must return G(end) - G(start) in eV for the species
    only, without the electron term (see
    ``mispr.ml.firetasks.predict.PredictIPEA``).

    Args:
        mol_operation_type (str): Must be ``"get_from_str"``; only SMILES are
            accepted.
        mol (str): SMILES of the molecule.
        ref_charge (int): Charge of the reference species.
        single_step (bool, optional): Only the total electron transfer step;
            defaults to ``False``.
        vertical (bool, optional): Whether vertical IP/EA are requested; passed
            to the model; defaults to ``False``.
        pcet (bool, optional): Not supported; must be ``False``.
        num_electrons (int, optional): Number of electrons transferred; defaults
            to 1.
        states (list, optional): Defaults to ["cation", "anion"].
        phases (list, optional): Defaults to ["gas", "solution"].
        electrode_potentials (dict, optional): Additional reference electrodes,
            as in ``mispr.orca.workflows.base.ip_ea.get_ip_ea``.
        gibbs_elec (float, optional): Electron Gibbs free energy in Hartree.
        db (str or dict, optional): Database credentials.
        name (str, optional): Name of the workflow; defaults to
            "ip_ea_calculation_ml".
        working_dir (str, optional): Working directory; defaults to the current
            working directory.
        model (dict): ``{"entry": "package.module:ClassName", "kwargs": {...}}``.
        kwargs (keyword arguments): Additional kwargs passed to ``PredictIPEA``
            and ``Workflow`` (e.g. ``tag``); others are ignored.

    Returns:
        Workflow

    """
    if mol_operation_type != "get_from_str":
        raise NotImplementedError("ML workflows currently accept SMILES only")
    if pcet:
        raise NotImplementedError("PCET is not supported by the ML workflows")
    if model is None:
        raise ValueError("a model must be provided")
    validate_model_spec(model)
    working_dir = working_dir or os.getcwd()

    if states is None:
        states = ["cation", "anion"]
    if phases is None:
        phases = ["gas", "solution"]
    if not states:
        raise ValueError("states list is empty")
    for state in states:
        if state.lower() not in ["cation", "anion"]:
            raise ValueError(
                "The provided states are not supported. Supported"
                " ones are reference, cation, and/or anion."
            )
    if not phases:
        raise ValueError("phases list is empty")
    for phase in phases:
        if phase.lower() not in ["gas", "solution"]:
            raise ValueError(
                "The provided phases are not supported. Supported"
                " ones are gas and/or solution."
            )

    if electrode_potentials:
        electrode_potentials = {
            k.lower(): {i.lower(): j for i, j in v.items()}
            if isinstance(v, dict)
            else v
            for k, v in electrode_potentials.items()
        }
        for k, v in electrode_potentials.items():
            if type(v) != dict or "potential" and "ref" not in v:
                raise KeyError(
                    "Standard electrode potential dict should "
                    "contain potential and ref keys."
                )
            electrode_potentials[k]["ref"] = bibtex_parser(v["ref"], working_dir)

    mol_ctx = build_mol_context(mol, charge=ref_charge)
    label = get_mol_formula(to_pymatgen(mol_ctx))

    fw = Firework(
        PredictIPEA(
            mol=mol,
            ref_charge=ref_charge,
            model=model,
            single_step=single_step,
            vertical=vertical,
            num_electrons=num_electrons,
            states=states,
            phases=phases,
            electrode_potentials=electrode_potentials,
            gibbs_elec=gibbs_elec,
            db=db,
            **{
                i: j
                for i, j in kwargs.items()
                if i in PredictIPEA.required_params + PredictIPEA.optional_params
            },
        ),
        name="{}-{}".format(label, "ip_ea_prediction"),
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
