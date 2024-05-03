from typing import Any, Dict, Optional
import numpy as np
import pandas as pd
import pycaret
from unittest.mock import patch
from scipy.optimize import shgo
from pycaret.internal.meta_estimators import (
    CustomProbabilityThresholdClassifier
)
from typing import Any, Dict, Optional
from shap import sample
from explainerdashboard import ClassifierExplainer, ExplainerDashboard
from pycaret.utils.generic import get_label_encoder

import plotly.express as px


def get_metric_by_name_or_id(pc, name_or_id: str, metrics: Optional[Any] = None):
    """
    Gets a metric from get_metrics() by name or index.
    """
    if metrics is None:
        metrics = pc.oop.get_all_metric_containers(globals())
    metric = None
    try:
        metric = metrics[name_or_id]
        return metric
    except Exception:
        pass

    try:
        metric = next(
            v for k, v in metrics.items() if name_or_id in (v.display_name, v.name)
        )
        return metric
    except Exception:
        pass

    return metric


def custom_optimize_threshold(
    pc: pycaret.classification,
    estimator,
    optimize: str = "Accuracy",
    return_data: bool = False,
    plot_kwargs: Optional[dict] = None,
    verbose: bool = True,
    **shgo_kwargs,
):
    """
    This function optimizes probability threshold for a trained classifier. It
    uses the SHGO optimizer from ``scipy`` to optimize for the given metric.
    This function will display a plot of the performance metrics at each probability
    threshold checked by the optimizer and returns the best model based on the metric
    defined under ``optimize`` parameter.


    Example
    -------
    >>> from pycaret.datasets import get_data
    >>> juice = get_data('juice')
    >>> experiment_name = setup(data = juice,  target = 'Purchase')
    >>> lr = create_model('lr')
    >>> best_lr_threshold = optimize_threshold(lr)


    Parameters
    ----------
    estimator : object
        A trained model object should be passed as an estimator.


    optimize : str, default = 'Accuracy'
        Metric to be used for selecting best model.


    return_data :  bool, default = False
        When set to True, data used for visualization is also returned.


    plot_kwargs :  dict, default = {} (empty dict)
        Dictionary of arguments passed to the visualizer class.


    verbose: bool, default = True
        Whether to print out messages at end of every iteration or not.


    **shgo_kwargs:
        Kwargs to pass to ``scipy.optimize.shgo``.


    Returns
    -------
    Trained Model


    Warnings
    --------
    - This function does not support multiclass classification problems.
    """

    function_params_str = ", ".join([f"{k}={v}" for k, v in locals().items()])

    # import libraries
    seed = pc.get_config("seed")
    np.random.seed(seed)

    """
        ERROR HANDLING STARTS HERE
        """

    # check predict_proba value
    if type(estimator) is not list:
        if not hasattr(estimator, "predict_proba"):
            raise TypeError(
                "Estimator doesn't support predict_proba function and cannot be used in optimize_threshold()."
            )

    if "func" in shgo_kwargs or "bounds" in shgo_kwargs or "args" in shgo_kwargs:
        raise ValueError(
            "shgo_kwargs cannot contain 'func', 'bounds' or 'args'.")

    shgo_kwargs.setdefault("sampling_method", "sobol")
    shgo_kwargs.setdefault("options", {})
    shgo_kwargs.setdefault("minimizer_kwargs", {})
    shgo_kwargs["minimizer_kwargs"].setdefault("options", {})
    shgo_kwargs["minimizer_kwargs"]["options"].setdefault("ftol", 1e-3)
    shgo_kwargs.setdefault("n", 50)
    shgo_kwargs["options"].setdefault("maxiter", 1)
    shgo_kwargs["options"].setdefault("f_tol", 1e-3)

    # checking optimize parameter
    optimize = get_metric_by_name_or_id(pc, optimize)
    if optimize is None:
        raise ValueError(
            "Optimize method not supported. See docstring for list of available parameters."
        )
    direction = -1 if optimize.greater_is_better else 1
    optimize = optimize.display_name

    """
        ERROR HANDLING ENDS HERE
        """

    # get estimator name
    # model_name = self._get_model_name(estimator)
    model_name = type(estimator)

    # defines empty list
    results_df = []
    models = {}

    def objective(x, *args):
        probability_threshold = x[0]
        model = pc.create_model(
            estimator,
            verbose=False,
            probability_threshold=probability_threshold,
        )
        model_results = (
            pc.pull(pop=True)
            .reset_index()
            .drop(columns=["Split"], errors="ignore")
            .set_index(["Fold"])
            .loc[["Mean"]]
        )
        model_results["probability_threshold"] = probability_threshold
        # add column model to model_results
        models[str(probability_threshold)] = model

        results_df.append(model_results)
        value = model_results[optimize].values[0]
        msg = f"Threshold: {probability_threshold}. {optimize}: {value}"
        if verbose:
            print(msg)
        return value * direction

    # This is necessary to make sure the sampler has a
    # deterministic seed.
    class FixedRandom(np.random.RandomState):
        def __init__(self_, seed=None) -> None:  # noqa
            super().__init__(seed)

    with patch("numpy.random.RandomState", FixedRandom):
        result = shgo(objective, ((0, 1),), **shgo_kwargs)

    message = (
        "optimization loop finished successfully. "
        f"Best threshold: {result.x[0]} with {optimize}={result.fun*direction}"
    )
    if verbose:
        print(message)

    results_concat = pd.concat(results_df, axis=0)
    results_concat = results_concat.sort_values("probability_threshold")
    results_concat_melted = results_concat.melt(
        id_vars=["probability_threshold"],
        value_vars=list(results_concat.columns[:-1]),
    )
    best_model_by_metric = models[str(result.x[0])]

    assert isinstance(best_model_by_metric,
                      CustomProbabilityThresholdClassifier)
    assert best_model_by_metric.probability_threshold == result.x[0]

    title = f"{model_name} Probability Threshold Optimization (default = 0.5)"
    plot_kwargs = plot_kwargs or {}
    fig = px.line(
        results_concat_melted,
        x="probability_threshold",
        y="value",
        title=title,
        color="variable",
        **plot_kwargs,
    )
    fig.show()

    if return_data:
        return (results_concat_melted, best_model_by_metric)
    else:

        return best_model_by_metric


def custom_dashboard(
    pc,
    estimator,
    display_format: str = "dash",
    categorical_columns = [],
    dashboard_kwargs: Optional[Dict[str, Any]] = None,
    run_kwargs: Optional[Dict[str, Any]] = None,
    **kwargs,
):
    """
    This function generates the interactive dashboard for a trained model. The
    dashboard is implemented using ExplainerDashboard (explainerdashboard.readthedocs.io)


    Example
    -------
    >>> from pycaret.datasets import get_data
    >>> juice = get_data('juice')
    >>> from pycaret.classification import *
    >>> exp_name = setup(data = juice,  target = 'Purchase')
    >>> lr = create_model('lr')
    >>> dashboard(lr)


    estimator: scikit-learn compatible object
        Trained model object


    display_format: str, default = 'dash'
        Render mode for the dashboard. The default is set to ``dash`` which will
        render a dashboard in browser. There are four possible options:

        - 'dash' - displays the dashboard in browser
        - 'inline' - displays the dashboard in the jupyter notebook cell.
        - 'jupyterlab' - displays the dashboard in jupyterlab pane.
        - 'external' - displays the dashboard in a separate tab. (use in Colab)


    dashboard_kwargs: dict, default = {} (empty dict)
        Dictionary of arguments passed to the ``ExplainerDashboard`` class.


    run_kwargs: dict, default = {} (empty dict)
        Dictionary of arguments passed to the ``run`` method of ``ExplainerDashboard``.

    **kwargs:
        Additional keyword arguments to pass to the ``ClassifierExplainer`` or
        ``RegressionExplainer`` class.


    Returns:
        ExplainerDashboard
    """

    dashboard_kwargs = dashboard_kwargs or {}
    run_kwargs = run_kwargs or {}

    le = get_label_encoder(pc.get_config("pipeline"))
    if le:
        labels_ = list(le.classes_)
    else:
        labels_ = None

    seed = pc.get_config("seed")
    # Replacing chars which dash doesn't accept for column name `.` , `{`, `}`
    X_test_df = sample(pc.get_config(
        'X_test_transformed').copy(), 1000, random_state=seed)
    X_test_df.columns = [
        col.replace(".", "__").replace("{", "__").replace("}", "__")
        for col in X_test_df.columns
    ]

    y_test_df = sample(pc.get_config(
        'y_test_transformed').copy(), 1000, random_state=seed)

    onehotencoded = categorical_columns.copy()
    onehotencoded.remove("Gender")
    explainer = ClassifierExplainer(
        model=estimator,
        X=X_test_df,
        y=y_test_df,
        labels=labels_,
        n_jobs=10,
        cats=onehotencoded,
        **kwargs,
    )

    explainer_dashboard = ExplainerDashboard(
        explainer, mode=display_format, **dashboard_kwargs
    )

    explainer_dashboard.run(**run_kwargs)
    return explainer_dashboard
