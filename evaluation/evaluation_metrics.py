class ConfusionMatrix:
    def __init__(self, tn, fp, fn, tp):
        self.tn = tn
        self.fp = fp
        self.fn = fn
        self.tp = tp

    def print_matrix(self):
        print(f'True negatives: {self.tn}')
        print(f'True positives: {self.tp}')
        print(f'False negatives: {self.fn}')
        print(f'False positives: {self.fp}')
        print(f'Sum: {self.tn + self.tp + self.fn + self.fp}')


class EvaluationMetrics:
    def __init__(self, cm_invalid, cm_valid, cm_land):
        self.cm_invalid = cm_invalid
        self.cm_valid = cm_valid
        self.cm_land = cm_land

        self.iou_invalid = self.get_iou(self.cm_invalid)
        self.iou_valid = self.get_iou(self.cm_valid)
        self.iou_land = self.get_iou(self.cm_land)

        self.mean_iou = (self.iou_invalid + self.iou_valid + self.iou_land) / 3

        self.precision_invalid = self.get_precision(self.cm_invalid)
        self.precision_valid = self.get_precision(self.cm_valid)
        self.precision_land = self.get_precision(self.cm_land)

        self.recall_invalid = self.get_recall(self.cm_invalid)
        self.recall_valid = self.get_recall(self.cm_valid)
        self.recall_land = self.get_recall(self.cm_land)

        self.specificity_invalid = self.get_specificity(self.cm_invalid)
        self.specificity_valid = self.get_specificity(self.cm_valid)
        self.specificity_land = self.get_specificity(self.cm_land)

        self.f1_invalid = self.get_f1(self.cm_invalid)
        self.f1_valid = self.get_f1(self.cm_valid)
        self.f1_land = self.get_f1(self.cm_land)

        self.accuracy_invalid = self.get_accuracy(self.cm_invalid)
        self.accuracy_valid = self.get_accuracy(self.cm_valid)
        self.accuracy_land = self.get_accuracy(self.cm_land)

    def get_iou(self, cm: ConfusionMatrix) -> float:
        return cm.tp / (cm.tp + cm.fp + cm.fn)

    def get_precision(self, cm: ConfusionMatrix) -> float:
        return cm.tp / (cm.tp + cm.fp)

    def get_specificity(self, cm: ConfusionMatrix) -> float:
        return cm.tn / (cm.tn + cm.fp)

    def get_recall(self, cm: ConfusionMatrix) -> float:
        return cm.tp / (cm.tp + cm.fn)

    def get_f1(self, cm: ConfusionMatrix) -> float:
        return 2 * cm.tp / (2 * cm.tp + cm.fp + cm.fn)

    def get_accuracy(self, cm: ConfusionMatrix) -> float:
        return (cm.tp + cm.tn) / (cm.tp + cm.tn + cm.fp + cm.fn)

    def print_metrics(self):
        print(f"mean iou index: {self.mean_iou}")

        print(f"iou invalid index: {self.iou_invalid}")
        print(f"iou valid index: {self.iou_valid}")
        print(f"iou land index: {self.iou_land}\n")

        print(f"precision_invalid: {self.precision_invalid}")
        print(f"precision_valid: {self.precision_valid}")
        print(f"precision_land: {self.precision_land}\n")

        print(f"recall_invalid: {self.recall_invalid}")
        print(f"recall_valid: {self.recall_valid}")
        print(f"recall_land: {self.recall_land}\n")

        print(f"specificity_invalid: {self.specificity_invalid}")
        print(f"specificity_valid: {self.specificity_valid}")
        print(f"specificity_land: {self.specificity_land}\n")

        print(f"f1_invalid: {self.f1_invalid}")
        print(f"f1_valid: {self.f1_valid}")
        print(f"f1_land: {self.f1_land}\n")

        print(f"accuracy_invalid: {self.accuracy_invalid}")
        print(f"accuracy_valid: {self.accuracy_valid}")
        print(f"accuracy_land: {self.accuracy_land}\n")
