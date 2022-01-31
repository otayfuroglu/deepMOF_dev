
import schnetpack as spk
import torch


def get_loss_fn(argsDict):
    derivative =  argsDict.get("derivative")#Return None if Dictionary key is not available
    contributions = argsDict.get("stress")
    stress = argsDict.get("negative_dr")

    # simple loss function for training on property only
    if derivative is None and contributions is None and stress is None:
        return simple_loss_fn(argsDict)

    # loss function with tradeoff weights
    #  if type(argsDict.get("rho")) == float:
    rho = dict(property=argsDict.get("rho"), derivative=1 - argsDict.get("rho"))
    #  else:
    #      rho = dict()
    #      rho["property"] = (
    #          1.0 if "property" not in args.rho.keys() else args.rho["property"]
    #      )
    #      if derivative is not None:
    #          rho["derivative"] = (
    #              1.0 if "derivative" not in args.rho.keys() else args.rho["derivative"]
    #          )
    #      if contributions is not None:
    #          rho["contributions"] = (
    #              1.0
    #              if "contributions" not in args.rho.keys()
    #              else args.rho["contributions"]
    #          )
    #      if stress is not None:
    #          rho["stress"] = (
    #              1.0 if "stress" not in args.rho.keys() else args.rho["stress"]
    #          )
    #      # type cast of rho values
    #      for key in rho.keys():
    #          rho[key] = float(rho[key])
    #      # norm rho values
    #      norm = sum(rho.values())
    #      for key in rho.keys():
    #          rho[key] = rho[key] / norm
    property_names = dict(
        property=argsDict.get("property"),
        derivative=derivative,
        contributions=contributions,
        stress=stress,
    )
    return tradeoff_loss_fn(rho, property_names)


def simple_loss_fn(argsDict):
    def loss(batch, result):
        diff = batch[argsDict.get("property")] - result[argsDict.get("property")]
        diff = diff ** 2
        err_sq = torch.mean(diff)
        return err_sq

    return loss


def tradeoff_loss_fn(rho, property_names):
    def loss(batch, result):
        err = 0.0
        for prop, tradeoff_weight in rho.items():
            # TODO: contributions should not be here
            diff = batch[property_names[prop]] - result[property_names[prop]]
            diff = diff ** 2
            err += tradeoff_weight * torch.mean(diff)

        return err

    return loss


def get_metrics(argsDict):
    # setup property metrics
    metrics = [
        spk.train.metrics.MeanAbsoluteError(argsDict.get("property"), argsDict.get("property")),
        spk.train.metrics.RootMeanSquaredError(argsDict.get("property"), argsDict.get("property")),
    ]

    # add metrics for derivative
    derivative = argsDict.get("derivative")
    if derivative is not None:
        metrics += [
            spk.train.metrics.MeanAbsoluteError(
                derivative, derivative, element_wise=True
            ),
            spk.train.metrics.RootMeanSquaredError(
                derivative, derivative, element_wise=True
            ),
        ]

    # Add stress metric
    stress = argsDict.get("stress")
    if stress is not None:
        metrics += [
            spk.train.metrics.MeanAbsoluteError(stress, stress, element_wise=True),
            spk.train.metrics.RootMeanSquaredError(stress, stress, element_wise=True),
        ]

    return metrics

def testTrainingHelper():

    argsDict = {"derivative": "forces",
                "stress": "stress",
                "negative_dr": True,
                "property": "energy",
                "rho": 0.01,
               }

    get_loss_fn(argsDict)
    get_metrics(argsDict)
#  testTrainingHelper()
