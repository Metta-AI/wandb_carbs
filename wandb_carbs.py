import wandb
from carbs import CARBS
import logging
import json
from carbs import (
    ObservationInParam,
    Param,
    LogSpace,
    LogitSpace,
    LinearSpace,
)
from typing import List


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
        self._carbs._set_seed(hash(self._sweep_id) % (2**32))

        assert self._wandb_run.summary.get("carbs.state") is None, \
            f"Run {self._wandb_run.name} already has carbs state"

        self._wandb_run.summary.update({"carbs.state": "running"})
        self._load_runs()
        self._suggestion = self._carbs.suggest().suggestion
        logger.info(f"Making suggestion for {self._wandb_run.name}: {json.dumps(self._suggestion, indent=2)}")
        self._wandb_run.config.__dict__["_locked"] = {}
        self._wandb_run.config.update(self._suggestion, allow_val_change=True)

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
        return self._suggestion

    def rewrite_carbs_suggestion(self, suggestion):
        """
        Rewrite the CARBS suggestion before it is applied to the config.

        Args:
            suggestion (dict): The original suggestion from CARBS.

        Returns:
            dict: The rewritten suggestion.
        """
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
            self._process_run(run)

    def _process_run(self, run):
        if run.summary["carbs.state"] == "running":
            suggestion = self._carbs.suggest().suggestion
            logger.debug(f"Suggestion: {json.dumps(suggestion, indent=2)}")
        else:
            suggestion = self._suggestion_from_run(run)
            objective = run.summary.get("carbs.objective", 0)
            cost = run.summary.get("carbs.cost", 0)
            self._carbs.observe(
                ObservationInParam(
                    input=suggestion,
                    output=objective,
                    cost=cost,
                    is_failure=run.summary["carbs.state"] == "failure"
                )
            )
            logger.debug(
                f"Observation {run.name} " +
                f"{objective} / {cost} " +
                f"failure: {run.summary['carbs.state'] == 'failure'} " +
                json.dumps(suggestion, indent=2)
            )

    def _suggestion_from_run(self, run):
        return {
            param.name: run.config.get(param.name, param.search_center)
            for param in self._carbs.params
        }

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
