from LunarLearn.data.synthetic import make_classification


def make_imbalanced_classification(
    n_samples=1000,
    n_features=20,
    n_classes=2,
    imbalance_ratio=0.1,   # minority fraction for binary; for multiclass use class_weights instead
    class_weights=None,
    **kwargs
):
    """
    Convenience wrapper around make_classification.
    - For binary: imbalance_ratio is minority class fraction.
    - For multiclass: pass class_weights explicitly.
    """
    if class_weights is None:
        if n_classes == 2:
            r = float(imbalance_ratio)
            if r <= 0 or r >= 0.5:
                raise ValueError("For binary, imbalance_ratio should be in (0, 0.5)")
            class_weights = [1.0 - r, r]
        else:
            raise ValueError("For n_classes > 2, provide class_weights explicitly")

    return make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_classes=n_classes,
        class_weights=class_weights,  # handled via kwargs below
        **kwargs
    )