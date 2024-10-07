import json
import logging
import math
import time
from copy import deepcopy
from typing import List, Set

import wandb
from carbs import (
    CARBS,
    LinearSpace,
    LogitSpace,
    LogSpace,
    ObservationInParam,
    SuggestionInBasic,
    Param,
)

logger = logging.getLogger("wandb_carbs")
# logger.setLevel(logging.DEBUG)

class WandbCarbs:
    def __init__(self, carbs: CARBS, wandb_run = None):
        """
        Initialize WandbCarbs with a CARBS instance and optionally a wandb run.

        Args:
            carbs (CARBS): The CARBS instance to use for suggestions.
            wandb_run (wandb.Run, optional): The wandb run to use. If None, uses the current run.
        """
        self._wandb_run = wandb_run or wandb.run
        self._sweep_id = self._wandb_run.sweep_id
        self._api = wandb.Api()

        self._carbs = carbs
        self._carbs._set_seed(int(time.time()))
        self._num_observations = 0
        self._num_failures = 0

        assert self._wandb_run.summary.get("carbs.state") is None, \
            f"Run {self._wandb_run.name} already has carbs state"

        self._wandb_run.summary.update({"carbs.state": "initializing"})
        self._load_runs()
        self._suggestion = self._carbs.suggest().suggestion

        wandb_config = self._transform_suggestion(deepcopy(self._suggestion))
        del wandb_config["suggestion_uuid"]
        self._wandb_run.config.__dict__["_locked"] = {}
        self._wandb_run.config.update(wandb_config, allow_val_change=True)
        self._wandb_run.summary.update({"carbs.state": "running"})

    def record_observation(self, objective: float, cost: float, allow_update: bool = False):
        """
        Record an observation for the current run.

        Args:
            objective (float): The objective value to record.
            cost (float): The cost value to record.
            allow_update (bool, optional): If True, allows updating even if the run is not in "running" state.
        """
        if not allow_update:
            assert self._wandb_run.summary["carbs.state"] == "running", \
                f"Run is not running, cannot record observation {self._wandb_run.summary}"

        self._wandb_run.summary.update({
            "carbs.objective": objective,
            "carbs.cost": cost,
            "carbs.state": "success"
        })
        logger.info(f"Recording observation ({objective}, {cost}) for {self._wandb_run.name}")

    def record_failure(self):
        """
        Record a failure for the current run.
        """
        logger.info(f"Recording failure for {self._wandb_run.name}")
        self._wandb_run.summary.update({"carbs.state": "failure"})


    def suggest(self):
        """
        Get the current suggestion.

        Returns:
            dict: The current suggestion.
        """
        return self._transform_suggestion(deepcopy(self._suggestion))

    def _transform_suggestion(self, suggestion):
        return suggestion

    def _load_runs(self):
        logger.info(f"Loading previous runs from sweep {self._sweep_id}")
        runs = self._api.runs(
            path=f"{self._wandb_run.entity}/{self._wandb_run.project}",
            filters={
                "sweep": self._sweep_id,
                "summary_metrics.carbs.state": {"$exists": True},
                "id": {"$ne": self._wandb_run.id}
            },
            order="+created_at"
        )

        for run in runs:
            self._update_carbs_from_run(run)

        logger.info(f"Initialized CARBS with {self._num_observations} observations" +
                    f" and {self._num_failures} failures")

    def _update_carbs_from_run(self, run):
        if run.summary["carbs.state"] == "initializing":
            return

        suggestion = self._suggestion_from_run(run)
        self._carbs._remember_suggestion(
            suggestion,
            SuggestionInBasic(self._carbs._param_space_real_to_basic_space_real(suggestion)),
            run.id
        )

        if run.summary["carbs.state"] == "running":
            return

        objective = run.summary.get("carbs.objective", 0)
        cost = run.summary.get("carbs.cost", 0)

        if run.summary["carbs.state"] == "failure":
            self._num_failures += 1
        else:
            self._num_observations += 1

        logger.debug(
            f"Observation {run.name} " +
            f"{objective} / {cost} " +
            f"failure: {run.summary['carbs.state'] == 'failure'} " +
            json.dumps(suggestion, indent=2)
        )
        self._carbs.observe(ObservationInParam(
            input=suggestion,
            output=objective,
            cost=cost,
            is_failure=run.summary["carbs.state"] == "failure"
        ))


    def _suggestion_from_run(self, run):
        suggestion = {
            param.name: run.config.get(param.name, param.search_center)
            for param in self._carbs.params
        }
        suggestion["suggestion_uuid"] = run.id
        return suggestion



class Pow2WandbCarbs(WandbCarbs):
    """
    A subclass of WandbCarbs that handles parameters that should be treated as powers of 2.

    This class extends WandbCarbs to support parameters that are internally represented as
    exponents but should be presented as powers of 2 externally.

    Attributes:
        pow2_params (Set[str]): A set of parameter names that should be treated as powers of 2.

    """

    def __init__(self, carbs: CARBS, pow2_params: Set[str], wandb_run = None):
        """
        Initialize the Pow2WandbCarbs instance.

        Args:
            carbs (CARBS): The CARBS instance to use for optimization.
            pow2_params (Set[str]): A set of parameter names to be treated as powers of 2.
            wandb_run: The Weights & Biases run object (optional).
        """
        self.pow2_params = pow2_params or set()
        super().__init__(carbs, wandb_run)

    def _transform_suggestion(self, suggestion):
        for param in self._carbs.params:
            if param.name in self.pow2_params:
                suggestion[param.name] = 2 ** suggestion[param.name]
        return suggestion

    def _suggestion_from_run(self, run):
        suggestion = super()._suggestion_from_run(run)
        for param in self._carbs.params:
            if param.name in self.pow2_params:
                suggestion[param.name] = int(math.log2(suggestion[param.name]))
        return suggestion

def create_sweep(sweep_name: str, wandb_entity: str, wandb_project: str, carb_params: List[Param]):
    """
    Create a new wandb sweep based on CARBS parameters.

    Args:
        sweep_name (str): The name of the sweep.
        wandb_entity (str): The wandb entity (username or team name).
        wandb_project (str): The wandb project name.
        carb_params (List[Param]): The CARBS parameter spaces.

    Returns:
        str: The ID of the created sweep.
    """
    sweep_id = wandb.sweep(
        sweep=_wandb_sweep_cfg_from_carbs_params(sweep_name, carb_params),
        project=wandb_project,
        entity=wandb_entity,
    )
    return sweep_id

def _wandb_sweep_cfg_from_carbs_params(name, carb_params: List[Param]):
    wandb_sweep_cfg = {
        "method": "bayes",
        "metric": {
            "goal": "maximize",
            "name": "eval_metric",
        },
        "parameters": {},
        "name": name,
    }
    for param in carb_params:
        wandb_sweep_cfg["parameters"][param.name] = {
            "min": param.space.min,
            "max": param.space.max,
            "distribution": _wandb_distribution(param),
        }
    return wandb_sweep_cfg

def _wandb_distribution(param: Param):
    if isinstance(param.space, LogSpace):
        return "log_uniform_values"
    elif isinstance(param.space, LogitSpace):
        return "uniform"
    elif isinstance(param.space, LinearSpace):
        if param.space.is_integer:
            return "int_uniform"
        else:
            return "uniform"
